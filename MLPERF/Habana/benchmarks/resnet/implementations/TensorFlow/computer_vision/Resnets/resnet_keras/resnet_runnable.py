# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# List of changes:
# - added profiling callbacks support

# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company

"""Runs a ResNet model on the ImageNet dataset using custom training loops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict, Optional, Text

import tensorflow as tf
from tensorflow.python.framework import ops

from TensorFlow.common.modeling import performance
from TensorFlow.common.training import grad_utils
from TensorFlow.common.training import standard_runnable
from TensorFlow.common.training import utils
from TensorFlow.utils.flags import core as flags_core
from TensorFlow.computer_vision.common import imagenet_preprocessing
from TensorFlow.computer_vision.Resnets.resnet_keras import common
from TensorFlow.computer_vision.Resnets.resnet_keras import resnet_model
from TensorFlow.computer_vision.Resnets.resnet_keras.common import get_global_batch_size
from TensorFlow.common.horovod_helpers import hvd, horovod_enabled

class ResnetRunnable(standard_runnable.StandardTrainable,
                     standard_runnable.StandardEvaluable):
  """Implements the training and evaluation APIs for Resnet model."""

  def __init__(self, flags_obj, time_callback, train_steps, epoch_steps, profiler_callback,mlperf_mlloger,mlperf_mllog):
    standard_runnable.StandardTrainable.__init__(self,
                                                 flags_obj.use_tf_while_loop,
                                                 flags_obj.use_tf_function)
    standard_runnable.StandardEvaluable.__init__(self,
                                                 flags_obj.use_tf_function)

    self.strategy = tf.distribute.get_strategy()
    self.flags_obj = flags_obj
    self.dtype = flags_core.get_tf_dtype(flags_obj)
    self.time_callback = time_callback
    self.profiler_callback = profiler_callback
    self.first_step = True


    self.mlperf_mlloger, self.mlperf_mllog = mlperf_mlloger,mlperf_mllog
    # Input pipeline related
    batch_size = flags_obj.batch_size
    if batch_size % self.strategy.num_replicas_in_sync != 0:
      raise ValueError(
          'Batch size must be divisible by number of replicas : {}'.format(
              self.strategy.num_replicas_in_sync))

    # As auto rebatching is not supported in
    # `experimental_distribute_datasets_from_function()` API, which is
    # required when cloning dataset to multiple workers in eager mode,
    # we use per-replica batch size.
    self.batch_size = int(batch_size / self.strategy.num_replicas_in_sync)

    if self.flags_obj.use_synthetic_data:
      self.input_fn = self.get_synth_input_fn()
    else:
      self.input_fn = imagenet_preprocessing.input_fn

    self.model = resnet_model.resnet50(
        num_classes=imagenet_preprocessing.NUM_CLASSES,
        batch_size=flags_obj.batch_size,
        use_l2_regularizer=not flags_obj.single_l2_loss_op)

    self.use_lars_optimizer = self.flags_obj.optimizer == 'LARS'

    self.optimizer = common.get_optimizer(flags_obj,
                                          get_global_batch_size(flags_obj.batch_size),
                                          train_steps,mlperf_mlloger,mlperf_mllog)
    # Make sure iterations variable is created inside scope.
    self.global_step = self.optimizer.iterations
    self.train_steps = train_steps

    self.one_hot = False
    self.label_smoothing = flags_obj.label_smoothing
    if self.label_smoothing and self.label_smoothing > 0:
      self.one_hot = True

    use_graph_rewrite = flags_obj.fp16_implementation == 'graph_rewrite'
    if use_graph_rewrite and not flags_obj.use_tf_function:
      raise ValueError('--fp16_implementation=graph_rewrite requires '
                       '--use_tf_function to be true')
    self.optimizer = performance.configure_optimizer(
        self.optimizer,
        use_float16=self.dtype == tf.float16,
        use_graph_rewrite=use_graph_rewrite,
        loss_scale=flags_core.get_loss_scale(flags_obj, default_for_fp16=128))

    if self.flags_obj.report_accuracy_metrics:
      self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
      if self.one_hot:
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(
            'train_accuracy', dtype=tf.float32)
      else:
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            'train_accuracy', dtype=tf.float32)
      self.test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    else:
      self.train_loss = None
      self.train_accuracy = None
      self.test_loss = None

    self.dist_eval = flags_obj.dist_eval
    self.profile = flags_obj.profile

    if self.one_hot:
      self.test_accuracy = tf.keras.metrics.CategoricalAccuracy(
          'test_accuracy', dtype=tf.float32)
    else:
      self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
          'test_accuracy', dtype=tf.float32)
    self.eval_accuracy = 0

    self.checkpoint = tf.train.Checkpoint(
        model=self.model, optimizer=self.optimizer)

    self.local_loss_mean = tf.keras.metrics.Mean("local_loss_min", dtype=tf.float32)

    # Handling epochs.
    self.epoch_steps = epoch_steps
    self.epoch_helper = utils.EpochHelper(epoch_steps, self.global_step)

  def get_synth_input_fn(self, is_training):
    return common.get_synth_input_fn(
          height=imagenet_preprocessing.DEFAULT_IMAGE_SIZE,
          width=imagenet_preprocessing.DEFAULT_IMAGE_SIZE,
          num_channels=imagenet_preprocessing.NUM_CHANNELS,
          num_classes=imagenet_preprocessing.NUM_CLASSES,
          dtype=common.get_dl_type(self.flags_obj),
          drop_remainder=is_training,
          experimental_preloading=self.flags_obj.experimental_preloading)

  def build_train_dataset(self, synthetic=False):
    """See base class."""
    return utils.make_distributed_dataset(
        self.strategy,
        self.get_synth_input_fn(True) if synthetic else self.input_fn,
        is_training=True,
        data_dir=self.flags_obj.data_dir,
        batch_size=self.batch_size,
        parse_record_fn=imagenet_preprocessing.parse_record,
        datasets_num_private_threads=self.flags_obj
        .datasets_num_private_threads,
        dtype=common.get_dl_type(self.flags_obj),
        drop_remainder=True,
        dataset_cache=self.flags_obj.dataset_cache,
        experimental_preloading=self.flags_obj.experimental_preloading)

  def build_synthetic_train_dataset(self):
    return self.build_train_dataset(synthetic=True)

  def build_eval_dataset(self, synthetic=False):
    """See base class."""
    return utils.make_distributed_dataset(
        self.strategy,
        self.get_synth_input_fn(False) if synthetic else self.input_fn,
        is_training=False,
        data_dir=self.flags_obj.data_dir,
        batch_size=self.batch_size,
        parse_record_fn=imagenet_preprocessing.parse_record,
        dtype=common.get_dl_type(self.flags_obj),
        dataset_cache=self.flags_obj.dataset_cache,
        experimental_preloading=self.flags_obj.experimental_preloading)

  def build_synthetic_eval_dataset(self):
    return self.build_eval_dataset(synthetic=True)

  def get_prediction_loss(self, labels, logits, training=True):
    if self.one_hot:
      return tf.keras.losses.categorical_crossentropy(
          labels, logits, label_smoothing=self.label_smoothing)
    else:
      return tf.keras.losses.sparse_categorical_crossentropy(labels, logits)

  def train_loop_begin(self):
    """See base class."""
    # Reset all metrics
    if self.train_loss:
      self.train_loss.reset_states()
    if self.train_accuracy:
      self.train_accuracy.reset_states()

    self._epoch_begin()
    self.time_callback.on_batch_begin(self.epoch_helper.batch_index)
    if self.profiler_callback is not None:
      self.profiler_callback.on_batch_begin(self.epoch_helper.batch_index)

  def train_step(self, iterator):
    """See base class."""

    def step_fn(inputs):
      """Function to run on the device."""
      images, labels = inputs
      if self.one_hot:
        labels = tf.cast(labels, tf.int32)
        labels = tf.one_hot(labels, 1001)
        labels = tf.squeeze(labels)

      with tf.GradientTape() as tape:
        logits = self.model(images, training=True)

        prediction_loss = self.get_prediction_loss(labels, logits)

        loss = tf.reduce_sum(prediction_loss) * (1.0 /
                                                 self.flags_obj.batch_size)

        if not self.use_lars_optimizer:
          num_replicas = self.strategy.num_replicas_in_sync

          if self.flags_obj.single_l2_loss_op:
            l2_loss = self.flags_obj.weight_decay * tf.add_n([
                tf.nn.l2_loss(v)
                for v in self.model.trainable_variables
                if ('bn' not in v.name)
            ])

            loss += (l2_loss / num_replicas)
          else:
            loss += (tf.reduce_sum(self.model.losses) / num_replicas)

      if horovod_enabled():
        tape = hvd.DistributedGradientTape(tape)
        grads = tape.gradient(loss, self.model.trainable_variables)
        grads_and_vars = zip(grads, self.model.trainable_variables)

        self.optimizer.apply_gradients(
          grads_and_vars, experimental_aggregate_gradients=False)

        tf.cond(self.global_step == 1,
          lambda: hvd.broadcast_variables(self.model.variables + self.optimizer.variables(),
                                          root_rank=0),
          lambda: tf.constant(True))
      else:
        grad_utils.minimize_using_explicit_allreduce(
          tape, self.optimizer, loss, self.model.trainable_variables)

      if self.flags_obj.modeling:
          sess = tf.compat.v1.Session()
          # pbtxt generation
          tf.io.write_graph(sess.graph.as_graph_def(add_shapes=True), self.flags_obj.model_dir, 'graph.pbtxt')
          # meta graph generation
          tf.compat.v1.train.export_meta_graph(filename='checkpoint_model.meta', meta_info_def=None, graph_def=None, saver_def=None, collection_list=None, as_text=False, graph=None, export_scope=None, clear_devices=False, clear_extraneous_savers=False, strip_default_attrs=False, save_debug_info=False)

      if self.train_loss:
        self.train_loss.update_state(loss)
      if self.train_accuracy:
        self.train_accuracy.update_state(labels, logits)

    self.strategy.run(step_fn, args=(next(iterator),))

  def train_loop_end(self):
    """See base class."""
    metrics = dict()
    if self.train_loss:
      metrics['train_loss'] = self.train_loss.result()
    if self.train_accuracy:
      metrics['train_accuracy'] = self.train_accuracy.result()
    self.time_callback.on_batch_end(self.epoch_helper.batch_index - 1)
    if self.profiler_callback is not None:
      self.profiler_callback.on_batch_end(self.epoch_helper.batch_index - 1)
    self._epoch_end()
    return metrics

  def eval_begin(self):
    """See base class."""
    if self.test_loss:
      self.test_loss.reset_states()
    self.test_accuracy.reset_states()
    epoch_num = int(self.epoch_helper.current_epoch)
    self.mlperf_mlloger.start(
        key=self.mlperf_mllog.constants.EVAL_START, value=None, metadata={'epoch_num': epoch_num + 1})

  def eval_step(self, iterator):
    """See base class."""

    def step_fn(inputs):
      """Function to run on the device."""
      images, labels = inputs
      if self.one_hot:
        labels = tf.cast(labels, tf.int32)
        labels = tf.one_hot(labels, 1001)
        labels = tf.squeeze(labels)

      logits = self.model(images, training=False)
      loss = self.get_prediction_loss(labels, logits, training=False)
      loss = tf.reduce_sum(loss) * (1.0 / self.flags_obj.batch_size)
      if self.test_loss:
        self.test_loss.update_state(loss)
      self.test_accuracy.update_state(labels, logits)

    self.strategy.run(step_fn, args=(next(iterator),))

  def eval_end(self):
    """See base class."""
    epoch_num = int(self.epoch_helper.current_epoch)
    self.mlperf_mlloger.end(
        key=self.mlperf_mllog.constants.EVAL_STOP, value=None, metadata={'epoch_num': epoch_num + 1})

    local_hit = self.test_accuracy.total
    local_count = self.test_accuracy.count

    global_hit = local_hit
    global_count = local_count
    if horovod_enabled() and self.dist_eval:
        global_hit = hvd.allreduce(local_hit, op=hvd.Sum)
        global_count = hvd.allreduce(local_count, op=hvd.Sum)
    global_accuracy = float(global_hit / global_count)

    # assign to self
    self.test_accuracy.total.assign(global_hit)
    self.test_accuracy.count.assign(global_count)

    eval_accuracy = global_accuracy
    self.eval_accuracy = eval_accuracy
    self.mlperf_mlloger.event(
        key=self.mlperf_mllog.constants.EVAL_ACCURACY, value=eval_accuracy, metadata={'epoch_num': epoch_num + 1})

    first_epoch_num = max(epoch_num - self.flags_obj.epochs_between_evals + 1, 0)
    epoch_count = self.flags_obj.epochs_between_evals
    if first_epoch_num == 0:
      epoch_count = self.flags_obj.eval_offset_epochs
      if epoch_count == 0:
        epoch_count = self.flags_obj.epochs_between_evals
    self.mlperf_mlloger.end(
        key=self.mlperf_mllog.constants.BLOCK_STOP,
        value=None,
        metadata={
            'first_epoch_num': first_epoch_num + 1,
            'epoch_count': epoch_count
        })

    past_threshold = False
    if self.flags_obj.target_accuracy is not None:
      past_threshold = eval_accuracy >= self.flags_obj.target_accuracy
      if ( horovod_enabled() and (not self.dist_eval) ):
          past_threshold = hvd.allreduce(tf.cast(past_threshold, tf.float32),
                  op=hvd.Sum) > 0

    continue_training = True
    if past_threshold:
      continue_training = False
    elif ( (not self.profile) and eval_accuracy <= 0.002):
      continue_training = False
    elif self.global_step.numpy() < self.train_steps:
      self.mlperf_mlloger.start(
          key=self.mlperf_mllog.constants.BLOCK_START,
          value=None,
          metadata={
              'first_epoch_num': epoch_num + 2,
              'epoch_count': self.flags_obj.epochs_between_evals
          })

    metrics = {
        'test_accuracy': eval_accuracy,
        'continue_training': continue_training,
    }
    if self.test_loss:
      metrics['test_loss'] = self.test_loss.result()
    return metrics

  def warmup(self,
             num_steps: Optional[tf.Tensor]) -> Optional[Dict[Text, tf.Tensor]]:
    """Implements device warmup with multiple steps.

    This loop runs the input pipeline on synthetic data before training, thereby
    allowing tf.function tracing before the dataset is accessed.

    Args:
      num_steps: A guideline for how many training steps to run. Note that it is
        up to the model what constitutes a "step" (this may involve more than
        one update to model parameters, e.g. if training a GAN).

    Returns:
      The function may return a dictionary of `Tensors`, which will be
      written to logs and as TensorBoard summaries.
    """
    self.warmup_train_iter = tf.nest.map_structure(
                              iter, self.build_synthetic_train_dataset())
    self.warmup_eval_iter = tf.nest.map_structure(
                              iter, self.build_synthetic_eval_dataset())

    if self.train_loop_fn is None:
      train_fn = self.train_step
      if self.use_tf_while_loop:
        self.train_loop_fn = utils.create_tf_while_loop_fn(train_fn)
      else:
        if self.use_tf_function:
          train_fn = tf.function(train_fn)
        self.train_loop_fn = utils.create_loop_fn(train_fn)

    if self.eval_loop_fn is None:
      eval_fn = self.eval_step
      if self.eval_use_tf_function:
        eval_fn = tf.function(eval_fn)
      self.eval_loop_fn = utils.create_loop_fn(eval_fn)

    self.train_loop_fn(self.warmup_train_iter, num_steps)
    self.eval_loop_fn(self.warmup_eval_iter, num_steps)
    return self.warmup_loop_end()

  def warmup_loop_end(self):
    """See base class."""
    # Reset the state
    self.model.reset_states()
    tf.keras.backend.set_value(self.optimizer.iterations, 0)

  def _epoch_begin(self):
    if self.epoch_helper.epoch_begin():
      self.time_callback.on_epoch_begin(self.epoch_helper.current_epoch)
      if self.profiler_callback is not None:
        self.profiler_callback.on_epoch_begin(self.epoch_helper.current_epoch)

  def _epoch_end(self):
    if self.epoch_helper.epoch_end():
      self.time_callback.on_epoch_end(self.epoch_helper.current_epoch)
      if self.profiler_callback is not None:
        self.profiler_callback.on_epoch_end(self.epoch_helper.current_epoch)
