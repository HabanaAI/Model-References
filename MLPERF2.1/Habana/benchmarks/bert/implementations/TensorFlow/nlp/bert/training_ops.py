###############################################################################
# Copyright (C) 2021-2022 Habana Labs, Ltd. an Intel Company
###############################################################################

"""Functions responsible for creating training operation.
    They are based on optimization.py, but reduce the number of switch/merge operations.
    This implementation will be used instead of original optimization.py
    if flag loop_unrolling_for_train_op in run_pretraining.py is set to True."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

try:
  import horovod.tensorflow as hvd
except ImportError:
  hvd = None

from optimization import AdamWeightDecayOptimizer
from optimization import LAMBOptimizer


def make_optimizer(global_step,
                   init_lr,
                   num_train_steps,
                   num_warmup_steps,
                   num_accumulation_steps,
                   allreduce_post_accumulation,
                   init_loss_scale,
                   manual_fp16,
                   use_fp16,
                   use_tpu,
                   optimizer_type="lamb",
                   weight_decay_rate=0.01,
                   beta_1=0.9,
                   beta_2=0.999,
                   epsilon=1e-6,
                   power=0.5):
  # avoid step change in learning rate at end of warmup phase
  if optimizer_type == "adam":
    power = 1.0
    decayed_learning_rate_at_crossover_point = init_lr * (
        (1.0 - float(num_warmup_steps) / float(num_train_steps))**power)
  else:
    power = power
    decayed_learning_rate_at_crossover_point = init_lr
  adjusted_init_lr = init_lr * (init_lr /
                                decayed_learning_rate_at_crossover_point)
  print(f"decayed_learning_rate_at_crossover_point = {decayed_learning_rate_at_crossover_point},")
  print(f"adjusted_init_lr = {adjusted_init_lr}")
  learning_rate = tf.constant(value=adjusted_init_lr,
                              shape=[],
                              dtype=tf.float32)
  # Implements linear decay of the learning rate.
  learning_rate = tf.compat.v1.train.polynomial_decay(
      learning_rate,
      # First update global_step, then apply_grad, thus we use global_step-1.
      global_step - 1,
      num_train_steps,
      end_learning_rate=0.0,
      power=power,
      cycle=False)
  # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
  # learning rate will be `global_step/num_warmup_steps * init_lr`.
  if num_warmup_steps:
    global_steps_int = tf.cast(global_step, tf.int32)
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)
    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)
    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = init_lr * warmup_percent_done
    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = ((1.0 - is_warmup) * learning_rate +
                     is_warmup * warmup_learning_rate)
  if optimizer_type in ["lamb", "sharded_lamb"]:
    print("Initializing LAMB Optimizer")
    optimizer = LAMBOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=weight_decay_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
  else:
    print("Initializing ADAM Weight Decay Optimizer")
    # It is recommended that you use this optimizer for fine tuning, since this
    # is how the model was trained (note that the Adam m/v variables are NOT
    # loaded from init_checkpoint.)
    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
  if hvd is not None and hvd.is_initialized() and not allreduce_post_accumulation:
    optimizer = hvd.DistributedOptimizer(optimizer, sparse_as_dense=True)
  if use_fp16:
    loss_scaler = tf.train.experimental.DynamicLossScale(
        initial_loss_scale=init_loss_scale,
        increment_period=1000,
        multiplier=2.0)
    optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(
        optimizer, loss_scaler)
    loss_scale_value = tf.identity(loss_scaler(), name="loss_scale")
  if manual_fp16:
    assert False, "No support for ExponentialUpdateLossScaleManager and LossScaleOptimizer in TF2.0"
    loss_scale_manager = tf.contrib.mixed_precision.ExponentialUpdateLossScaleManager(
        init_loss_scale=init_loss_scale,
        incr_every_n_steps=1000,
        decr_every_n_nan_or_inf=2,
        decr_ratio=0.5)
    optimizer = tf.contrib.mixed_precision.LossScaleOptimizer(
        optimizer, loss_scale_manager)
  if use_tpu:
    optimizer = tf.compat.v1.tpu.CrossShardOptimizer(optimizer)
  return optimizer


def create_training_ops(global_step,
                        loss,
                        init_lr,
                        num_train_steps,
                        num_warmup_steps,
                        manual_fp16=False,
                        use_fp16=False,
                        num_accumulation_steps=1,
                        optimizer_type="adam",
                        allreduce_post_accumulation=False,
                        init_loss_scale=2**32,
                        weight_decay_rate=0.01,
                        beta_1=0.9,
                        beta_2=0.999,
                        epsilon=1e-6,
                        power=0.5,
                        use_tpu=False):
  if optimizer_type == "sharded_lamb":
      assert num_accumulation_steps > 1 and allreduce_post_accumulation, \
      "Sharded LAMB can be used only when num_accumulation_steps > 1 and allreduce_post_accumulation is True"


  optimizer = make_optimizer(global_step, init_lr, num_train_steps,
                             num_warmup_steps, num_accumulation_steps,
                             allreduce_post_accumulation, init_loss_scale,
                             manual_fp16, use_fp16, use_tpu, optimizer_type,
                             weight_decay_rate, beta_1, beta_2, epsilon, power)

  local_step = tf.compat.v1.placeholder(tf.int32, shape=(), name='local_step')
  ops_list = []
  if num_accumulation_steps > 1:
    reset = reset_op(local_step, optimizer, loss, num_accumulation_steps,
                     manual_fp16, use_fp16, allreduce_post_accumulation)
    accum = accum_op(local_step, optimizer, loss, num_accumulation_steps,
                     manual_fp16, use_fp16, allreduce_post_accumulation)
    update = update_op(local_step, global_step, optimizer, optimizer_type,
                       loss, num_accumulation_steps, manual_fp16, use_fp16,
                       allreduce_post_accumulation)

    ops_list.append(reset)
    for _ in range(num_accumulation_steps - 2):
      ops_list.append(accum)
  else:
    update = train_op_without_accumulation(optimizer, loss, global_step,
                                           use_fp16, manual_fp16, allreduce_post_accumulation)
  return ops_list, update


def train_op_without_accumulation(optimizer, loss, global_step, use_fp16,
                                  manual_fp16, allreduce_post_accumulation):
  trainable_variables = tf.compat.v1.trainable_variables()
  grads_and_vars = optimizer.compute_gradients(
      loss, trainable_variables, gate_gradients=tf.compat.v1.train.Optimizer.GATE_NONE)
  grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
  grads, trainable_variables = list(zip(*grads_and_vars))
  finite_grads_mask = [tf.reduce_all(input_tensor=tf.math.is_finite(g)) for g in grads]
  all_are_finite = tf.reduce_all(input_tensor=finite_grads_mask) if use_fp16 or manual_fp16 else tf.constant(True, dtype=tf.bool)

  # This is how the model was pre-trained.
  # ensure global norm is a finite number
  # to prevent clip_by_global_norm from having a hizzy fit.
  (clipped_grads, _) = tf.clip_by_global_norm(
      grads,
      clip_norm=1.0,
      use_norm=tf.cond(pred=all_are_finite,
                       true_fn=lambda: tf.linalg.global_norm(grads),
                       false_fn=lambda: tf.constant(1.0)))
  new_global_step = tf.cond(pred=all_are_finite,
                            true_fn=lambda: global_step + 1,
                            false_fn=lambda: global_step)
  new_global_step = tf.identity(new_global_step, name='step_update')
  with tf.control_dependencies([global_step.assign(new_global_step)]):
    if allreduce_post_accumulation and hvd is not None and hvd.is_initialized():
      clipped_grads = [hvd.allreduce(tf.convert_to_tensor(value=clipped_grad), op=hvd.Sum) for clipped_grad in clipped_grads]
    train_op = optimizer.apply_gradients(list(zip(clipped_grads, trainable_variables)),
                                         global_step=global_step)
  return train_op


def reset_op(local_step, optimizer, loss, num_accumulation_steps, manual_fp16,
             use_fp16, allreduce_post_accumulation):
  # grads_and_vars = optimizer.compute_gradients(loss * 1.0 / num_accumulation_steps, tvars)
  # to match mlcomm ref we need to clip before scaling
  trainable_variables = tf.compat.v1.trainable_variables()
  grads_and_vars = optimizer.compute_gradients(
      loss, trainable_variables, gate_gradients=tf.compat.v1.train.Optimizer.GATE_NONE)
  accum_vars = []
  for tvar in trainable_variables:
    var_name = tvar.name.split(":")[0] + "/accum"
    with tf.compat.v1.variable_scope(var_name, reuse=tf.compat.v1.AUTO_REUSE):
      accum_vars.append(
          tf.compat.v1.get_variable(
              name=var_name,
              shape=tvar.shape.as_list(),
              dtype=tf.bfloat16,
              trainable=False)
      )
  grads_and_vars_and_accums = [(gv[0], gv[1], accum_vars[i])
                               for i, gv in enumerate(grads_and_vars)
                               if gv[0] is not None]
  grads, trainable_variables, accum_vars = list(zip(*grads_and_vars_and_accums))
  # This is how the model was pre-trained.
  # ensure global norm is a finite number
  # to prevent clip_by_global_norm from having a hizzy fit.
  finite_grads_mask = [tf.reduce_all(input_tensor=tf.math.is_finite(g)) for g in grads]
  all_are_finite = tf.reduce_all(input_tensor=finite_grads_mask) if use_fp16 or manual_fp16 else tf.constant(True, dtype=tf.bool)
  (clipped_grads, _) = tf.clip_by_global_norm(
      grads,
      clip_norm=1.0,
      use_norm=tf.cond(pred=all_are_finite,
                       true_fn=lambda: tf.linalg.global_norm(grads),
                       false_fn=lambda: tf.constant(1.0)))
  # divide grad by acc_steps before accumulating
  accum_vars = [
      accum_vars[i].assign(tf.cast(grad, accum_vars[i].dtype))
      for i, grad in enumerate(clipped_grads)
  ]
  # required for MLPerfHook
  update_step = tf.math.equal(local_step % num_accumulation_steps,
                              0,
                              name="update_step")
  if allreduce_post_accumulation and hvd is not None and hvd.is_initialized():
    accum_vars = [
        hvd.allreduce(tf.convert_to_tensor(value=accum_var) * 1.0 /
                      num_accumulation_steps,
                      op=hvd.Sum) if isinstance(accum_var, tf.IndexedSlices)
        else hvd.allreduce(accum_var * 1.0 / num_accumulation_steps, op=hvd.Sum)
        for accum_var in accum_vars
    ]
  return tf.group(*accum_vars, update_step)


def accum_op(local_step, optimizer, loss, num_accumulation_steps, manual_fp16,
             use_fp16, allreduce_post_accumulation):
  # grads_and_vars = optimizer.compute_gradients(loss * 1.0 / num_accumulation_steps, tvars)
  # to match mlcomm ref we need to clip before scaling
  trainable_variables = tf.compat.v1.trainable_variables()
  grads_and_vars = optimizer.compute_gradients(
      loss, trainable_variables, gate_gradients=tf.compat.v1.train.Optimizer.GATE_NONE)
  accum_vars = []
  for tvar in trainable_variables:
    var_name = tvar.name.split(":")[0] + "/accum"
    with tf.compat.v1.variable_scope(var_name, reuse=tf.compat.v1.AUTO_REUSE):
      accum_vars.append(
          tf.compat.v1.get_variable(
              name=var_name,
              shape=tvar.shape.as_list(),
              dtype=tf.bfloat16,
              trainable=False)
      )
  grads_and_vars_and_accums = [(gv[0], gv[1], accum_vars[i])
                               for i, gv in enumerate(grads_and_vars)
                               if gv[0] is not None]
  grads, trainable_variables, accum_vars = list(zip(*grads_and_vars_and_accums))
  # This is how the model was pre-trained.
  # ensure global norm is a finite number
  # to prevent clip_by_global_norm from having a hizzy fit.
  finite_grads_mask = [tf.reduce_all(input_tensor=tf.math.is_finite(g)) for g in grads]
  all_are_finite = tf.reduce_all(input_tensor=finite_grads_mask) if use_fp16 or manual_fp16 else tf.constant(True, dtype=tf.bool)
  (clipped_grads, _) = tf.clip_by_global_norm(
      grads,
      clip_norm=1.0,
      use_norm=tf.cond(pred=all_are_finite,
                       true_fn=lambda: tf.linalg.global_norm(grads),
                       false_fn=lambda: tf.constant(1.0)))
  # divide grad by acc_steps before accumulating
  accum_vars = [
      accum_vars[i].assign_add(tf.cast(grad, accum_vars[i].dtype))
      for i, grad in enumerate(clipped_grads)
  ]
  # required for MLPerfHook
  update_step = tf.math.equal(local_step % num_accumulation_steps,
                              0,
                              name="update_step")
  if allreduce_post_accumulation and hvd is not None and hvd.is_initialized():
    accum_vars = [
        hvd.allreduce(tf.convert_to_tensor(value=accum_var) * 1.0 /
                      num_accumulation_steps,
                      op=hvd.Sum) if isinstance(accum_var, tf.IndexedSlices)
        else hvd.allreduce(accum_var * 1.0 / num_accumulation_steps, op=hvd.Sum)
        for accum_var in accum_vars
    ]
  return tf.group(*accum_vars, update_step)


def update_op(local_step, global_step, optimizer, optimizer_type, loss, num_accumulation_steps,
              manual_fp16, use_fp16, allreduce_post_accumulation):
  # grads_and_vars = optimizer.compute_gradients(loss * 1.0 / num_accumulation_steps, tvars)
  # to match mlcomm ref we need to clip before scaling
  trainable_variables = tf.compat.v1.trainable_variables()
  grads_and_vars = optimizer.compute_gradients(
      loss, trainable_variables, gate_gradients=tf.compat.v1.train.Optimizer.GATE_NONE)
  batch_finite = tf.compat.v1.get_variable(
      name="batch_finite",
      shape=[],
      dtype=tf.bool,
      trainable=False,
      initializer=tf.compat.v1.ones_initializer)
  accum_vars = []
  for tvar in trainable_variables:
    var_name = tvar.name.split(":")[0] + "/accum"
    with tf.compat.v1.variable_scope(var_name, reuse=tf.compat.v1.AUTO_REUSE):
      accum_vars.append(
          tf.compat.v1.get_variable(
              name=var_name,
              shape=tvar.shape.as_list(),
              dtype=tf.bfloat16,
              trainable=False)
      )
  grads_and_vars_and_accums = [(gv[0], gv[1], accum_vars[i])
                               for i, gv in enumerate(grads_and_vars)
                               if gv[0] is not None]
  grads, trainable_variables, accum_vars = list(zip(*grads_and_vars_and_accums))
  finite_grads_mask = [tf.reduce_all(input_tensor=tf.math.is_finite(g)) for g in grads]
  all_are_finite = tf.reduce_all(input_tensor=finite_grads_mask) if use_fp16 or manual_fp16 else tf.constant(True, dtype=tf.bool)
  batch_finite = batch_finite.assign(
      tf.math.logical_and(batch_finite, all_are_finite))
  # This is how the model was pre-trained.
  # ensure global norm is a finite number
  # to prevent clip_by_global_norm from having a hizzy fit.
  (clipped_grads, _) = tf.clip_by_global_norm(
      grads,
      clip_norm=1.0,
      use_norm=tf.cond(pred=all_are_finite,
                       true_fn=lambda: tf.linalg.global_norm(grads),
                       false_fn=lambda: tf.constant(1.0)))
  # divide grad by acc_steps before accumulating
  accum_vars = [
      accum_vars[i].assign_add(tf.cast(grad, accum_vars[i].dtype))
      for i, grad in enumerate(clipped_grads)
  ]
  new_global_step = global_step + 1
  new_global_step = tf.identity(new_global_step, name='step_update')
  global_step = global_step.assign(new_global_step)
  # required for MLPerfHook
  update_step = tf.math.equal(local_step % num_accumulation_steps,
                              0,
                              name="update_step")

  def update(accum_vars):
    with tf.control_dependencies([global_step]):
      if allreduce_post_accumulation and hvd is not None and hvd.is_initialized():
        if optimizer_type == "sharded_lamb":
          accum_vars = [tf.convert_to_tensor(value=accum_var) * 1.0 / num_accumulation_steps if isinstance(accum_var, tf.IndexedSlices)
                        else accum_var * 1.0 / num_accumulation_steps for accum_var in accum_vars]
          apply_gradients_fn = optimizer.apply_distributed_gradients

        else:
          accum_vars = [
              hvd.allreduce(tf.convert_to_tensor(value=accum_var) * 1.0 /
                            num_accumulation_steps,
                            op=hvd.Sum) if isinstance(accum_var, tf.IndexedSlices)
              else hvd.allreduce(accum_var * 1.0 / num_accumulation_steps,
                                op=hvd.Sum) for accum_var in accum_vars
          ]
          apply_gradients_fn = optimizer.apply_gradients
      else:
        apply_gradients_fn = optimizer.apply_gradients

      return tf.group(apply_gradients_fn(list(zip(accum_vars, trainable_variables)),
                                    global_step=global_step), update_step)

  return update(accum_vars)
