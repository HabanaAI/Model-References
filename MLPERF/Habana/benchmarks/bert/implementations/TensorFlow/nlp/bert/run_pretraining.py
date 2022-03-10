# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors.
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

"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import TensorFlow.nlp.bert.modeling as modeling
import TensorFlow.nlp.bert.optimization as optimization
import tensorflow as tf
import glob
from TensorFlow.nlp.bert.utils.utils import LogEvalRunHook
import TensorFlow.nlp.bert.utils.dllogger_class as dllogger_class
from dllogger import Verbosity
import math
import numbers
import numpy as np
from tensorflow.core.protobuf import rewriter_config_pb2

from TensorFlow.common.tb_utils import ExamplesPerSecondEstimatorHook, write_hparams_v1
from TensorFlow.common.str2bool import condition_env_var
from TensorFlow.common.debug import dump_callback
from TensorFlow.common.utils import RangeTFProfilerHook

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, '..', '..'))


flags = tf.compat.v1.flags

FLAGS = flags.FLAGS

from tensorflow.python.framework.ops import Tensor
from typing import Union


def init_flags():
  ## Required parameters
  flags.DEFINE_string(
      "bert_config_file", None,
      "The config json file corresponding to the pre-trained BERT model. "
      "This specifies the model architecture.")


  flags.DEFINE_integer("samples_between_eval", 150000, "MLPerf Evaluation frequency in samples.")

  flags.DEFINE_float("stop_threshold", 0.720, "MLperf Mask LM accuracy target")

  flags.DEFINE_integer("samples_start_eval", 3000000, " Required samples to start evaluation for MLPerf.")

  flags.DEFINE_bool("enable_device_warmup", False, " Enable device warmup for MLPerf.")

  flags.DEFINE_string(
      "input_files_dir", None,
      "Directory with input files, comma separated or single directory.")

  flags.DEFINE_string(
      "eval_files_dir", None,
      "Directory with eval files, comma separated or single directory. ")

  flags.DEFINE_string(
      "output_dir", None,
      "The output directory where the model checkpoints will be written.")

  ## Other parameters
  flags.DEFINE_string(
      "dllog_path", "/results/bert_dllog.json",
      "filename where dllogger writes to")

  flags.DEFINE_string(
      "init_checkpoint", None,
      "Initial checkpoint (usually from a pre-trained BERT model).")

  flags.DEFINE_string(
      "eval_checkpoint_path", None,
      "eval checkpoint path.")

  flags.DEFINE_bool(
      'is_dist_eval_enabled', False,  'IF true enable distributed evaluation')

  flags.DEFINE_string(
      "optimizer_type", "lamb",
      "Optimizer used for training - LAMB or ADAM")

  flags.DEFINE_integer(
      "max_seq_length", 512,
      "The maximum total input sequence length after WordPiece tokenization. "
      "Sequences longer than this will be truncated, and sequences shorter "
      "than this will be padded. Must match data generation.")

  flags.DEFINE_integer(
      "max_predictions_per_seq", 80,
      "Maximum number of masked LM predictions per sequence. "
      "Must match data generation.")

  flags.DEFINE_bool("do_train", False, "Whether to run training.")

  flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

  flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

  flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

  flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

  flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

  flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

  flags.DEFINE_integer("save_checkpoints_steps", 1000,
                      "How often to save the model checkpoint.")

  flags.DEFINE_bool("checkpoint_all_workers", False,
                    "Whether to save checkpoint from all workers.")

  flags.DEFINE_integer("save_summary_steps", 1,
                       "How often to save the summary data.")

  flags.DEFINE_integer("display_loss_steps", 10,
                      "How often to print loss")

  flags.DEFINE_integer("iterations_per_loop", 1000,
                      "How many steps to make in each estimator call.")

  flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

  flags.DEFINE_integer("num_accumulation_steps", 1,
                      "Number of accumulation steps before gradient update."
                        "Global batch size = num_accumulation_steps * train_batch_size")

  flags.DEFINE_bool("allreduce_post_accumulation", False, "Whether to all reduce after accumulation of N steps or after each step")

  flags.DEFINE_bool(
      "verbose_logging", False,
      "If true, all of the trainable parameters are printed")

  flags.DEFINE_bool("horovod", False, "Whether to use Horovod for multi-gpu runs")

  flags.DEFINE_bool("report_loss", True, "Whether to report total loss during training.")

  flags.DEFINE_bool("manual_fp16", False, "Whether to use fp32 or fp16 arithmetic on GPU. "
                                          "Manual casting is done instead of using AMP")

  flags.DEFINE_bool("amp", True, "Whether to enable AMP ops. When false, uses TF32 on A100 and FP32 on V100 GPUS.")
  flags.DEFINE_bool("use_xla", True, "Whether to enable XLA JIT compilation.")
  flags.DEFINE_integer("init_loss_scale", 2**32, "Initial value of loss scale if mixed precision training")
  flags.DEFINE_bool("deterministic_run", False, "If set run will be deterministic (set random seed, read dataset in single thread, disable dropout)")
  flags.DEFINE_bool("enable_habana_backend", False, "Whether to enable habana backend.")
  flags.DEFINE_bool("enable_packed_data_mode", False, "Whether to enable training with packed data. "
                                                      "If enabled, packed data should be provided to input_files_dir.")
  flags.DEFINE_integer("avg_seq_per_pack", 1, "Average number of sequences per packed sample.")
  flags.DEFINE_bool("compute_lm_loss_per_seq", False, "Whether to compute lm loss per sample in packed record.")

  flags.DEFINE_float("beta_1", 0.9, "LAMB_BETA_1")
  flags.DEFINE_float("beta_2", 0.999, "LAMB_BETA_2")
  flags.DEFINE_float("epsilon", 1e-6, "EPSILON")
  flags.DEFINE_float("weight_decay_rate", 0.01, "LAMB_WEIGHT_DECAY_RATE")
  flags.DEFINE_float("power", 1.0, "LAMB_LEARNING_RATE_DECAY_POLY_POWER")

  flags.DEFINE_bool(
      'use_lightweight_checkpoint', False,  'If true use lightweight checkpoint')

  flags.DEFINE_integer("tf_profiler_begin_iter", 0, "First iteration to profile")
  flags.DEFINE_integer("tf_profiler_num_iter", 0, "Number of iterations to profile")

def get_mllog_mlloger():
    from mlperf_logging import mllog

    str_hvd_rank = str(hvd.rank()) if horovod_enabled() else "0"
    mllogger = mllog.get_mllogger()
    filenames = os.path.normpath(FLAGS.output_dir) + "/result_rank_" + str_hvd_rank + ".txt"
    mllog.config(filename=filenames)
    workername = "worker" + str_hvd_rank
    mllog.config(
            default_namespace = workername,
            default_stack_offset = 1,
            default_clear_line = False,
            root_dir = os.path.normpath(
           os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")))

    return mllogger, mllog


def past_stop_threshold(stop_threshold, eval_metric):
  """Return a boolean representing whether a model should be stopped.

  Args:
    stop_threshold: float, the threshold above which a model should stop
      training.
    eval_metric: float, the current value of the relevant metric to check.

  Returns:
    True if training should stop, False otherwise.

  Raises:
    ValueError: if either stop_threshold or eval_metric is not a number
  """
  if stop_threshold is None:
    return False

  if not isinstance(stop_threshold, numbers.Number):
    raise ValueError("Threshold for checking stop conditions must be a number.")
  if not isinstance(eval_metric, numbers.Number):
    raise ValueError("Eval metric being checked against stop conditions "
                     "must be a number.")

  if eval_metric >= stop_threshold:
    tf.compat.v1.logging.info(
        "Stop threshold of {} was passed with metric value {}.".format(
            stop_threshold, eval_metric))
    return True

  return False

#_NUM_EXAMPLES_NAME = "num_examples"

# report samples/sec, total loss and learning rate during training
class _LogSessionRunHook(tf.estimator.SessionRunHook):
  def __init__(self, global_batch_size, num_accumulation_steps, dllogging, display_every=10,
               save_ckpt_steps=1000, report_loss=True, hvd_rank=-1):
    self.global_batch_size = global_batch_size
    self.display_every = display_every
    self.save_ckpt_steps = save_ckpt_steps
    self.hvd_rank = hvd_rank
    self.num_accumulation_steps = num_accumulation_steps
    self.dllogging = dllogging
    self.report_loss = report_loss
    self.skip_iters = 6

  def after_create_session(self, session, coord):
    self.elapsed_secs = 0.0 #elapsed seconds between every print
    self.count = 0 # number of global steps between every print
    self.all_count = 0 #number of steps (including accumulation) between every print
    self.loss = 0.0 # accumulation of loss in each step between every print

    self.total_time = 0.0 # total time taken to train (excluding warmup + ckpt saving steps)
    self.step_time = 0.0 # time taken per step
    self.init_global_step = session.run(tf.compat.v1.train.get_global_step()) # training starts at init_global_step
    self.skipped = 0

  def before_run(self, run_context):
    if horovod_enabled() and hvd_rank() != 0:
      return
    self.t0 = time.time()
    if self.num_accumulation_steps <= 1:
        if FLAGS.manual_fp16 or FLAGS.amp:
            return tf.estimator.SessionRunArgs(
                fetches=['step_update:0', 'total_loss:0',
                         'learning_rate:0', 'nsp_loss:0',
                         'mlm_loss:0', 'loss_scale:0'])
        else:
            return tf.estimator.SessionRunArgs(
                fetches=['step_update:0', 'total_loss:0',
                         'learning_rate:0', 'nsp_loss:0',
                         'mlm_loss:0'])
    else:
        if FLAGS.manual_fp16 or FLAGS.amp:
            return tf.estimator.SessionRunArgs(
                fetches=['step_update:0', 'update_step:0', 'total_loss:0',
                         'learning_rate:0', 'nsp_loss:0',
                         'mlm_loss:0', 'loss_scale:0'])
        else:
          return tf.estimator.SessionRunArgs(
              fetches=['step_update:0', 'update_step:0', 'total_loss:0',
                       'learning_rate:0', 'nsp_loss:0',
                       'mlm_loss:0'])

  def after_run(self, run_context, run_values):
    if horovod_enabled() and hvd_rank() != 0:
      return
    run_time = time.time() - self.t0

    if self.num_accumulation_steps <=1:
        if FLAGS.manual_fp16 or FLAGS.amp:
            self.global_step, total_loss, lr, nsp_loss, mlm_loss, loss_scaler = run_values.results
        else:
            self.global_step, total_loss, lr, nsp_loss, mlm_loss = run_values. \
                results
        update_step = True
    else:
        if FLAGS.manual_fp16 or FLAGS.amp:
          self.global_step, update_step, total_loss, lr, nsp_loss, mlm_loss, loss_scaler = run_values.results
        else:
          self.global_step, update_step, total_loss, lr, nsp_loss, mlm_loss = run_values.\
              results

    self.elapsed_secs += run_time
    self.step_time += run_time

    print_step = self.global_step + 1 # One-based index for printing.
    self.loss += total_loss
    self.all_count += 1
    if update_step:

        self.count += 1

        # Removing first six steps after every checkpoint save from timing
        if (self.global_step - self.init_global_step) % max(1, self.save_ckpt_steps) < self.skip_iters:
          print("Skipping time record for ", self.global_step, " due to checkpoint-saving/warmup overhead")
          self.skipped += 1
        else:
          self.total_time += self.step_time

        self.step_time = 0.0 #Reset Step Time

        if (print_step == 1 or print_step % self.display_every == 0):
            dt = self.elapsed_secs / self.count
            sent_per_sec = self.global_batch_size / dt
            avg_loss_step = self.loss / self.all_count
            if self.hvd_rank >= 0 and FLAGS.report_loss:
              if FLAGS.manual_fp16 or FLAGS.amp:
                self.dllogging.logger.log(step=(print_step),
                                     data={"Rank": int(self.hvd_rank), "throughput_train": float(sent_per_sec),
                                           "mlm_loss":float(mlm_loss), "nsp_loss":float(nsp_loss),
                                           "total_loss":float(total_loss), "avg_loss_step":float(avg_loss_step),
                                           "learning_rate": str(lr), "loss_scaler":int(loss_scaler)},
                                     verbosity=Verbosity.DEFAULT)
              else:
                self.dllogging.logger.log(step=int(print_step),
                                     data={"Rank": int(self.hvd_rank), "throughput_train": float(sent_per_sec),
                                           "mlm_loss":float(mlm_loss), "nsp_loss":float(nsp_loss),
                                           "total_loss":float(total_loss), "avg_loss_step":float(avg_loss_step),
                                           "learning_rate": str(lr)},
                                     verbosity=Verbosity.DEFAULT)
            else:
              if FLAGS.manual_fp16 or FLAGS.amp:
                self.dllogging.logger.log(step=int(print_step),
                                     data={"throughput_train": float(sent_per_sec),
                                           "mlm_loss":float(mlm_loss), "nsp_loss":float(nsp_loss),
                                           "total_loss":float(total_loss), "avg_loss_step":float(avg_loss_step),
                                           "learning_rate": str(lr), "loss_scaler":int(loss_scaler)},
                                     verbosity=Verbosity.DEFAULT)
              else:
                self.dllogging.logger.log(step=int(print_step),
                                     data={"throughput_train": float(sent_per_sec),
                                           "mlm_loss":float(mlm_loss), "nsp_loss":float(nsp_loss),
                                           "total_loss":float(total_loss), "avg_loss_step":float(avg_loss_step),
                                           "learning_rate": str(lr)},
                                     verbosity=Verbosity.DEFAULT)

            self.elapsed_secs = 0.0
            self.count = 0
            self.loss = 0.0
            self.all_count = 0



train_op_name = None
class MLPerfHook(tf.estimator.SessionRunHook):
  def __init__(self, global_batch_size, num_accumulation_steps, num_train_steps, samples_between_eval,
               weight_decay_rate, beta_1, beta_2, epsilon, power, enable_device_warmup):
    '''
       global_batch_size = train_batch_size * num_accumulation_steps * num_of_devices
       num_train_steps = each step consumes global_batch_size samples
       samples_between_eval = total samples in each block
    '''
    mllogger, mllog = get_mllog_mlloger()
    mllogger.event(key=mllog.constants.CACHE_CLEAR)
    mllogger.start(key=mllog.constants.INIT_START)
    mllogger.event(key=mllog.constants.GLOBAL_BATCH_SIZE, value=global_batch_size*FLAGS.avg_seq_per_pack)
    mllogger.event(key=mllog.constants.TRAIN_SAMPLES, value=global_batch_size * FLAGS.num_train_steps)
    mllogger.event(key=mllog.constants.MAX_SEQUENCE_LENGTH, value=FLAGS.max_seq_length)
    mllogger.event(key='max_predictions_per_seq', value=FLAGS.max_predictions_per_seq)
    mllogger.event(key=mllog.constants.GRADIENT_ACCUMULATION_STEPS, value=FLAGS.num_accumulation_steps)
    mllogger.event(key=mllog.constants.OPT_LR_TRAINING_STEPS, value=FLAGS.num_train_steps)
    mllogger.event(key=mllog.constants.NUM_WARMUP_STEPS, value=FLAGS.num_warmup_steps)
    mllogger.event(key=mllog.constants.OPT_LR_WARMUP_STEPS, value=FLAGS.num_warmup_steps)
    mllogger.event(key=mllog.constants.START_WARMUP_STEP, value=0)
    mllogger.event(key=mllog.constants.OPT_BASE_LR, value=FLAGS.learning_rate)
    mllogger.event(key=mllog.constants.EVAL_SAMPLES, value=10000)
    mllogger.event(key=mllog.constants.OPT_LAMB_BETA_1, value=beta_1)
    mllogger.event(key=mllog.constants.OPT_LAMB_BETA_2, value=beta_2)
    mllogger.event(key=mllog.constants.OPT_LAMB_LR_DECAY_POLY_POWER, value=power)
    mllogger.event(key=mllog.constants.OPT_LAMB_WEIGHT_DECAY, value=weight_decay_rate)
    mllogger.event(key="opt_epsilon", value=epsilon)
    mllogger.start(key=mllog.constants.INIT_STOP)

    self.mllogger = mllogger
    self.mllog = mllog
    self.chpt_timestamp_dict={}
    self.run_start_timestamp=None
    self.checkpoint_timestamp_dict={}
    self.block_stop_timestamp_dict={}

    num_steps_between_eval = math.ceil(samples_between_eval / global_batch_size)
    n_loops = math.ceil(num_train_steps / num_steps_between_eval)
    schedule = [num_steps_between_eval  for _ in range(int(n_loops))]
    schedule[-1] = num_train_steps  - sum(schedule[:-1])

    self.num_accumulation_steps = num_accumulation_steps
    self.num_steps_between_eval = num_steps_between_eval
    self.schedule = schedule
    self.cycle_index = 0
    self.count = 0 # global step counter
    self.block_started = False

    self.enable_device_warmup = enable_device_warmup
    self.warmup_step = False
    self.session = None
    self.state_dict = None
    self.variable_names = None
    self.variable_name_to_assigner_input1_name = None
    self.variable_assigners = None


  def after_create_session(self, session, coord):
    if self.enable_device_warmup:
      self.session = session
      self.warmup_step = True
      graph = session.graph
      variables = list(filter(lambda op: op.type=='VarHandleOp', graph.get_operations()))
      variable_names = [op.name for op in variables]
      variable_readers = [name + '/Read/ReadVariableOp:0' for name in variable_names]
      variable_assigners = [name + '/Assign' for name in variable_names]
      variable_assigners_input1_name = [graph.get_operation_by_name(name + '/Assign').inputs[1].name for name in variable_names]
      variable_name_to_assigner_input1_name = dict(zip(variable_names, variable_assigners_input1_name))

      # save state_dict
      state_dict = dict(zip(variable_names, variable_readers))
      state_dict = session.run(fetches=state_dict)
      self.state_dict = state_dict
      self.variable_names = variable_names
      self.variable_name_to_assigner_input1_name = variable_name_to_assigner_input1_name
      self.variable_assigners = variable_assigners
    else:
      self.mllogger.start(key=self.mllog.constants.RUN_START)
      self.run_start_timestamp=time.time()

  def before_run(self, run_context):
    if self.warmup_step:
      if self.num_accumulation_steps <= 1:
        return None
      else:
        return tf.estimator.SessionRunArgs(fetches='update_step:0')  # update_step
    else:
      if self.block_started == False:
        #self.checkpoint_timestamp_dict[self.cycle_index]=int(time.time()*1e3)
        self.mllogger.start(key=self.mllog.constants.BLOCK_START, value=self.cycle_index + 1, metadata={self.mllog.constants.FIRST_EPOCH_NUM: int(self.cycle_index * self.num_steps_between_eval), self.mllog.constants.EPOCH_COUNT: int(self.num_steps_between_eval)})
        self.block_started = True

      if self.num_accumulation_steps <= 1:
        return tf.estimator.SessionRunArgs(fetches=['step_update:0']) # global_step
      else:
        return tf.estimator.SessionRunArgs(fetches=['step_update:0', 'update_step:0']) # global_step, update_step

  def after_run(self, run_context, run_values):
    if self.warmup_step:
      if self.num_accumulation_steps <= 1:
        update_step = True
      else:
        update_step = run_values.results

      if update_step:
        self.warmup_step = False

        # restore data loader iterator
        self.session.run(self.session.graph.get_operation_by_name('MakeIterator'))

        # load state_dict
        feed_dict = dict()
        for key in self.variable_names:
          feed_dict[self.variable_name_to_assigner_input1_name[key]] = self.state_dict[key]
        self.session.run(fetches=self.variable_assigners, feed_dict=feed_dict)

        self.mllogger.start(key=self.mllog.constants.RUN_START)
        self.run_start_timestamp = time.time()
    else:
      if self.num_accumulation_steps <=1:
        self.global_step = run_values.results
        update_step = True
      else:
        self.global_step, update_step = run_values.results

      if update_step:
        self.count += 1

      if self.count >= self.schedule[self.cycle_index]:
        self.mllogger.end(key=self.mllog.constants.BLOCK_STOP, value=self.cycle_index + 1, metadata={self.mllog.constants.FIRST_EPOCH_NUM: int(self.cycle_index * self.num_steps_between_eval)})
        self.chpt_timestamp_dict[self.cycle_index + 1]=time.time()
        self.checkpoint_timestamp_dict[self.cycle_index + 1]=int(time.time()*1e3)
        self.block_stop_timestamp_dict[self.cycle_index + 1]=time.time()
        self.cycle_index += 1
        self.count = 0
        self.block_started = False


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps,
                     use_one_hot_embeddings, weight_decay_rate, beta_1, beta_2, epsilon, power):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.compat.v1.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.compat.v1.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_ids = features["masked_lm_ids"]
    masked_lm_weights = features["masked_lm_weights"]
    next_sentence_labels = features["next_sentence_labels"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    positions = None
    next_sentence_positions = None
    next_sentence_weights = None
    if FLAGS.enable_packed_data_mode and is_training: # only training should work in packed mode
      positions = features["positions"]
      next_sentence_positions = features["next_sentence_positions"]
      next_sentence_weights = features["next_sentence_weights"]


    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
        compute_type=tf.float16 if FLAGS.manual_fp16 else tf.float32,
        enable_packed_data_mode=FLAGS.enable_packed_data_mode,
        positions=positions,
        next_sentence_positions=next_sentence_positions,
        use_habana_layer_norm=FLAGS.enable_habana_backend)

    (masked_lm_loss,
     masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
         bert_config, model.get_sequence_output(), model.get_embedding_table(),
         masked_lm_positions, masked_lm_ids,
         masked_lm_weights,
         enable_packed_data_mode=FLAGS.enable_packed_data_mode,
         is_training=is_training,
         avg_seq_per_pack=FLAGS.avg_seq_per_pack,
         compute_lm_loss_per_seq=FLAGS.compute_lm_loss_per_seq)

    (next_sentence_loss, next_sentence_example_loss,
     next_sentence_log_probs) = get_next_sentence_output(
         bert_config, model.get_pooled_output(), next_sentence_labels,
         enable_packed_data_mode=FLAGS.enable_packed_data_mode,
         is_training=is_training,
         weights=next_sentence_weights)

    masked_lm_loss = tf.identity(masked_lm_loss, name="mlm_loss")
    next_sentence_loss = tf.identity(next_sentence_loss, name="nsp_loss")
    total_loss = masked_lm_loss + next_sentence_loss
    total_loss = tf.identity(total_loss, name='total_loss')

    tvars = tf.compat.v1.trainable_variables()

    initialized_variable_names = {}
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

      tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)

    if FLAGS.verbose_logging:
        tf.compat.v1.logging.info("**** Trainable Variables ****")
        for var in tvars:
          init_string = ""
          if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
          tf.compat.v1.logging.info("  %d :: name = %s, shape = %s%s", 0 if horovod_enabled() else hvd.rank(), var.name, var.shape,
                          init_string)

    output_spec = None

    training_hooks = []

    if FLAGS.tf_profiler_begin_iter >= 0 and FLAGS.tf_profiler_num_iter > 0:
      tf_profiler_hook = RangeTFProfilerHook(
        FLAGS.tf_profiler_begin_iter,
        FLAGS.tf_profiler_num_iter,
        "./rank-{}".format(hvd.rank()))
      training_hooks.append(tf_profiler_hook)

    if FLAGS.use_lightweight_checkpoint:
      global_variables = tf.compat.v1.global_variables()

      light_checkpoint_saver =  tf.compat.v1.train.Saver(
        var_list=global_variables,
        sharded=False,
        max_to_keep=19,
        defer_build=False,
        save_relative_paths=True)

      lightweight_checkpoint_saver_hook= tf.estimator.CheckpointSaverHook(
        checkpoint_dir=FLAGS.output_dir,
        save_secs=None,
        save_steps=FLAGS.save_checkpoints_steps if comm_local_rank() == 0 or FLAGS.checkpoint_all_workers else num_train_steps,
        saver=light_checkpoint_saver,
        checkpoint_basename='model.ckpt',
        scaffold=None,
        listeners=None,
        save_graph_def=True
      )
      training_hooks = [lightweight_checkpoint_saver_hook]

    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps,
          FLAGS.manual_fp16, FLAGS.amp, FLAGS.num_accumulation_steps, FLAGS.optimizer_type, FLAGS.allreduce_post_accumulation, FLAGS.init_loss_scale, weight_decay_rate, beta_1, beta_2, epsilon, power)
      global train_op_name
      train_op_name = train_op.name

      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          training_hooks=training_hooks)

    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                    masked_lm_weights, next_sentence_example_loss,
                    next_sentence_log_probs, next_sentence_labels):
        """Computes the loss and accuracy of the model."""
        masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                         [-1, masked_lm_log_probs.shape[-1]])
        masked_lm_predictions = tf.argmax(
            input=masked_lm_log_probs, axis=-1, output_type=tf.int32)
        masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
        masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
        masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
        masked_lm_accuracy = tf.compat.v1.metrics.accuracy(
            labels=masked_lm_ids,
            predictions=masked_lm_predictions,
            weights=masked_lm_weights)
        masked_lm_mean_loss = tf.compat.v1.metrics.mean(
            values=masked_lm_example_loss, weights=masked_lm_weights)

        next_sentence_log_probs = tf.reshape(
            next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
        next_sentence_predictions = tf.argmax(
            input=next_sentence_log_probs, axis=-1, output_type=tf.int32)
        next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
        next_sentence_accuracy = tf.compat.v1.metrics.accuracy(
            labels=next_sentence_labels, predictions=next_sentence_predictions)
        next_sentence_mean_loss = tf.compat.v1.metrics.mean(
            values=next_sentence_example_loss)

        return {
            "masked_lm_accuracy": masked_lm_accuracy,
            "masked_lm_loss": masked_lm_mean_loss,
            "next_sentence_accuracy": next_sentence_accuracy,
            "next_sentence_loss": next_sentence_mean_loss,
        }

      eval_metric_ops = metric_fn(
          masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
          masked_lm_weights, next_sentence_example_loss,
          next_sentence_log_probs, next_sentence_labels
      )
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metric_ops=eval_metric_ops)
    else:
      raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights,
                         enable_packed_data_mode: bool = False,
                         is_training: bool = False,
                         avg_seq_per_pack: int = 1,
                         compute_lm_loss_per_seq: bool = False):
  """Get loss and log probs for the masked LM."""
  input_tensor = modeling.gather_indexes(input_tensor, positions)

  with tf.compat.v1.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.compat.v1.variable_scope("transform"):
      input_tensor = tf.compat.v1.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.compat.v1.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.compat.v1.zeros_initializer())
    logits = tf.matmul(tf.cast(input_tensor, tf.float32), output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    log_probs = tf.nn.log_softmax(logits - tf.reduce_max(logits, keepdims=True, axis=-1), axis=-1)

    #log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(input_tensor=log_probs * one_hot_labels, axis=[-1])
    if enable_packed_data_mode and is_training and compute_lm_loss_per_seq:  # only training should work in packed mode
      label_weights = tf.transpose(tf.one_hot(tf.cast(label_weights, dtype=tf.int32) - 1, depth=3))
      numerator = tf.reduce_sum(input_tensor=label_weights * per_example_loss, axis=[-1])
      denominator = tf.reduce_sum(input_tensor=label_weights, axis=[-1]) + 1e-5
      loss = numerator / denominator
      loss = tf.reduce_sum(loss)
    else:
      if enable_packed_data_mode and is_training:  # only training should work in packed mode
        label_weights = tf.cast(label_weights > 0, dtype=tf.float32)

      numerator = tf.reduce_sum(input_tensor=label_weights * per_example_loss)
      denominator = tf.reduce_sum(input_tensor=label_weights) + 1e-5
      loss = numerator / denominator

  return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels,
                             enable_packed_data_mode: bool = False,
                             is_training: bool = False,
                             weights: Union[Tensor, type(None)] = None):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf.compat.v1.variable_scope("cls/seq_relationship"):
    output_weights = tf.compat.v1.get_variable(
        "output_weights",
        shape=[2, bert_config.hidden_size],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    output_bias = tf.compat.v1.get_variable(
        "output_bias", shape=[2], initializer=tf.compat.v1.zeros_initializer())

    logits = tf.matmul(tf.cast(input_tensor, tf.float32), output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits - tf.reduce_max(logits, keepdims=True, axis=-1), axis=-1)


    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(input_tensor=one_hot_labels * log_probs, axis=-1)
    if enable_packed_data_mode and is_training: # only training should work in packed mode
      weights = tf.reshape(weights, [-1])
      numerator = tf.reduce_sum(input_tensor=weights * per_example_loss)
      denominator = tf.reduce_sum(input_tensor=weights) + 1e-5
      loss = numerator / denominator
    else:
      loss = tf.reduce_mean(input_tensor=per_example_loss)
    return (loss, per_example_loss, log_probs)


def input_fn_builder(input_files,
                     batch_size,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4):
  """Creates an `input_fn` closure to be passed to Estimator."""

  def input_fn():
    """The actual input function."""

    if FLAGS.enable_packed_data_mode and is_training: # only training should work in packed mode
      name_to_features = {
        "input_ids":
          tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask":
          tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids":
          tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "positions":
          tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions":
          tf.io.FixedLenFeature([max_predictions_per_seq + 3], tf.int64),
        "masked_lm_ids":
          tf.io.FixedLenFeature([max_predictions_per_seq + 3], tf.int64),
        "masked_lm_weights":
          tf.io.FixedLenFeature([max_predictions_per_seq + 3], tf.float32),
        "next_sentence_positions":
          tf.io.FixedLenFeature([3], tf.int64),
        "next_sentence_labels":
          tf.io.FixedLenFeature([3], tf.int64),
        "next_sentence_weights":
          tf.io.FixedLenFeature([3], tf.float32),
      }
    else:
      name_to_features = {
          "input_ids":
              tf.io.FixedLenFeature([max_seq_length], tf.int64),
          "input_mask":
              tf.io.FixedLenFeature([max_seq_length], tf.int64),
          "segment_ids":
              tf.io.FixedLenFeature([max_seq_length], tf.int64),
          "masked_lm_positions":
              tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
          "masked_lm_ids":
              tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
          "masked_lm_weights":
              tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32),
          "next_sentence_labels":
              tf.io.FixedLenFeature([1], tf.int64),
      }

    if FLAGS.deterministic_run:
      input_files.sort()
      d = tf.data.TFRecordDataset(input_files)
      if horovod_enabled(): d = d.shard(hvd_size(), hvd_rank())
      d = d.apply(
          tf.data.experimental.map_and_batch(
              lambda record: _decode_record(record, name_to_features),
              batch_size=batch_size,
              num_parallel_calls=1,
              drop_remainder=True))
      return d

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
      if horovod_enabled(): d = d.shard(hvd_size(), hvd_rank())
      d = d.repeat()
      d = d.shuffle(buffer_size=len(input_files))

      # `cycle_length` is the number of parallel files that get read.
      cycle_length = min(num_cpu_threads, len(input_files))

      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = d.apply(
          tf.data.experimental.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=is_training,
              cycle_length=cycle_length))
      d = d.shuffle(buffer_size=100)
    else:
      d = tf.data.TFRecordDataset(input_files)
      # Since we evaluate for a fixed number of steps we don't want to encounter
      # out-of-range exceptions.
      if horovod_enabled() and FLAGS.is_dist_eval_enabled: d = d.shard(hvd_size(), hvd_rank())
      d = d.repeat()

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.
    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True if is_training else False))
    return d

  return input_fn


def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.io.parse_single_example(serialized=record, features=name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.cast(t, dtype=tf.int32)
    example[name] = t

  return example

def main(_):
  if FLAGS.enable_packed_data_mode:
    FLAGS.num_accumulation_steps = int(FLAGS.num_accumulation_steps / FLAGS.avg_seq_per_pack)
    FLAGS.samples_between_eval = int(FLAGS.samples_between_eval / FLAGS.avg_seq_per_pack)

  os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_lazy_compilation=false" #causes memory fragmentation for bert leading to OOM

  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

  dllogging = dllogger_class.dllogger_class(FLAGS.dllog_path)

  if not FLAGS.do_train and not FLAGS.do_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

  # In multi-node scenario, on each of HLSes there must be a checkpoint directly in the output_dir (read by Phase 2).
  # There may be only one worker with comm_local_rank() == 0 on each machine and this worker will put its checkpoints there.
  # All other workers use sub-directories to keep checkpoints.
  if horovod_enabled() and comm_local_rank() != 0:
    master_output_dir = FLAGS.output_dir
    FLAGS.output_dir = os.path.join(FLAGS.output_dir, f'worker_{hvd_rank()}')
  else:
    master_output_dir = FLAGS.output_dir
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  tf.io.gfile.makedirs(FLAGS.output_dir)

  input_files = []
  for input_file_dir in FLAGS.input_files_dir.split(","):
    input_files.extend(tf.io.gfile.glob(os.path.join(input_file_dir, "*")))

  if FLAGS.horovod and len(input_files) < hvd.size():
      tf.compat.v1.logging.warning("Input files count lower then expected. Using single file for OVERFIT test.")
      input_files = [input_files[0] for i in range(hvd.size())]
  if FLAGS.amp and FLAGS.manual_fp16:
      raise ValueError("AMP and Manual Mixed Precision Training are both activated! Error")

  is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2

  # The Scoped Allocator Optimization is enabled by default unless disabled by a flag.
  if condition_env_var('TF_DISABLE_SCOPED_ALLOCATOR', default=False):
    session_config = tf.compat.v1.ConfigProto()
  else:
    from tensorflow.core.protobuf import rewriter_config_pb2  # pylint: disable=import-error

    session_config = tf.compat.v1.ConfigProto()
    session_config.graph_options.rewrite_options.scoped_allocator_optimization = rewriter_config_pb2.RewriterConfig.ON

    enable_op = session_config.graph_options.rewrite_options.scoped_allocator_opts.enable_op
    del enable_op[:]
    enable_op.append("HorovodAllreduce")

  if FLAGS.horovod:
    session_config.gpu_options.visible_device_list = str(hvd.local_rank())
    if hvd.rank() == 0:
      tf.compat.v1.logging.info("***** Configuaration *****")
      for key in FLAGS.__flags.keys():
          tf.compat.v1.logging.info('  {}: {}'.format(key, getattr(FLAGS, key)))
      tf.compat.v1.logging.info("**************************")

#    config.gpu_options.per_process_gpu_memory_fraction = 0.7
  if FLAGS.use_xla:
      session_config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1
      session_config.graph_options.rewrite_options.memory_optimization = rewriter_config_pb2.RewriterConfig.NO_MEM_OPT
      if FLAGS.amp:
        tf.compat.v1.enable_resource_variables()

  run_config = tf.estimator.RunConfig(
      model_dir=FLAGS.output_dir,
      session_config=session_config,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps if comm_local_rank() == 0 or FLAGS.checkpoint_all_workers else None,
      save_checkpoints_secs=None,
      keep_checkpoint_max=19,
      save_summary_steps=FLAGS.save_summary_steps,
      log_step_count_steps=1)

  if FLAGS.optimizer_type == "lamb":
      weight_decay_rate=FLAGS.weight_decay_rate
      beta_1=FLAGS.beta_1
      beta_2=FLAGS.beta_2
      epsilon=FLAGS.epsilon
      power=FLAGS.power

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=FLAGS.num_warmup_steps,
      use_one_hot_embeddings=False, weight_decay_rate=weight_decay_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, power=power)

  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config)

  batch_size_per_node = FLAGS.train_batch_size * FLAGS.num_accumulation_steps

  global_batch_size = (hvd.size() if FLAGS.horovod else 1) * batch_size_per_node
  write_hparams_v1(FLAGS.output_dir, {
    'batch_size': FLAGS.train_batch_size,
    'batch_size_per_pu': FLAGS.train_batch_size,
    'batch_size_per_node': batch_size_per_node,
    'global_batch_size': global_batch_size,
    **{x: getattr(FLAGS, x) for x in FLAGS}
  })

  if FLAGS.do_train:

    training_hooks = []

    train_log_hook = _LogSessionRunHook(
      global_batch_size, FLAGS.num_accumulation_steps, dllogging,
      FLAGS.display_loss_steps, FLAGS.save_checkpoints_steps, FLAGS.report_loss)
    training_hooks.append(train_log_hook)

    training_hooks.append(ExamplesPerSecondEstimatorHook(
      batch_size=batch_size_per_node, output_dir=FLAGS.output_dir,
      extra_metrics={'global_examples/sec': global_batch_size}))

    mlperfhook = MLPerfHook(global_batch_size, FLAGS.num_accumulation_steps, FLAGS.num_train_steps, FLAGS.samples_between_eval,
                                      weight_decay_rate, beta_1, beta_2, epsilon, power, FLAGS.enable_device_warmup)
    training_hooks.append(mlperfhook)

    tf.compat.v1.logging.info("***** Running training *****")
    tf.compat.v1.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    train_input_fn = input_fn_builder(
        input_files=input_files,
        batch_size=FLAGS.train_batch_size,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=True)

    train_start_time = time.time()
    with dump_callback():
      estimator.train(input_fn=train_input_fn, hooks=training_hooks, max_steps=FLAGS.num_train_steps)
    train_time_elapsed = time.time() - train_start_time

    #do offline evaluation right after training for mlperf
    tf.compat.v1.logging.info("***** Running offline evaluation right after training for mlperf *****")
    converged = False
    eval_start_time = time.time()
    mlperf_chpt_timestamp_dict = mlperfhook.chpt_timestamp_dict
    mlperf_run_start_timestamp = mlperfhook.run_start_timestamp
    mlperf_checkpoint_timestamp_dict = mlperfhook.checkpoint_timestamp_dict
    mlperf_mlloger = mlperfhook.mllogger
    mlperf_mllog = mlperfhook.mllog
    mlperf_block_stop_timestamp_dict = mlperfhook.block_stop_timestamp_dict
    num_steps_between_eval = math.ceil(FLAGS.samples_between_eval / global_batch_size)
    print("mlperf_run_start_timestamp={}".format(mlperf_run_start_timestamp))
    print("mlperf_checkpoint_timestamp_dict={}".format(mlperf_checkpoint_timestamp_dict))
    print("mlperf_block_stop_timestamp_dict={}".format(mlperf_block_stop_timestamp_dict))

    chpt_file_path = master_output_dir + "/checkpoint"
    chpt_files = []

    # Barrier in order to avoid races on checkpoint file operations
    hvd.broadcast(0, 0)

    if os.path.isfile(chpt_file_path):
      with open(chpt_file_path, "r") as file:
          for line in file:
            tmp,chpt_step = line.split(":")
            if tmp == 'all_model_checkpoint_paths':
              step = int(chpt_step.strip().split("-")[1].strip('"'))
              if step >0:
                chpt_files.append(master_output_dir + '/'+ chpt_step.strip().strip('"'))

    eval_files = []
    for eval_file_dir in FLAGS.eval_files_dir.split(","):
          eval_files.extend(tf.io.gfile.glob(os.path.join(eval_file_dir, "*")))

    eval_input_fn = input_fn_builder(
        input_files=eval_files,
        batch_size=FLAGS.eval_batch_size,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=False)

    eval_hooks = [LogEvalRunHook(FLAGS.eval_batch_size)]

    if len(chpt_files) == 0:
      tf.compat.v1.logging.info("***** No checkpoints to evaluate *****")

    else:
      if horovod_enabled() and FLAGS.is_dist_eval_enabled:
        tf.compat.v1.logging.info("***** Running offline distributed evaluation for mlperf *****")
        #need to shard the dataset!!!!
        eval_samples = 10000 / hvd_size()
        max_eval_steps = math.ceil(FLAGS.max_eval_steps / hvd_size())
        for ckpt_ind,chpt_path in enumerate(chpt_files):
          print("checkpoint file path={}".format(chpt_path))
          eval_results = estimator.evaluate(
            input_fn=eval_input_fn, steps=max_eval_steps, hooks=eval_hooks, checkpoint_path=chpt_path)

          if FLAGS.stop_threshold:
            partial_eval_masked_lm_accuracy = eval_results["masked_lm_accuracy"] * eval_samples
            print("per rank masked_lm_accuracy={}".format(eval_results["masked_lm_accuracy"]))
            partial_eval_masked_lm_accuracy_FP32=tf.cast(partial_eval_masked_lm_accuracy, tf.float32)
            total_eval_masked_lm_accuracy_FP32 = hvd.allreduce(partial_eval_masked_lm_accuracy_FP32, op=hvd.Sum)
            total_eval_masked_lm_accuracy_FP32 /= 10000.0
            mlperf_mlloger.event(key=mlperf_mllog.constants.EVAL_ACCURACY,value=total_eval_masked_lm_accuracy_FP32.numpy(), time_ms=mlperf_checkpoint_timestamp_dict[ckpt_ind + 1],metadata={'epoch_num': (ckpt_ind + 1)*FLAGS.samples_between_eval*FLAGS.avg_seq_per_pack,'epoch_count': ckpt_ind + 1})
            success = bool(total_eval_masked_lm_accuracy_FP32 >= FLAGS.stop_threshold)
            print("average eval_masked_lm_accuracy_FP32={}".format(total_eval_masked_lm_accuracy_FP32))

            if success:
                mlperf_mlloger.end(key=mlperf_mllog.constants.RUN_STOP,value=total_eval_masked_lm_accuracy_FP32.numpy(), time_ms=mlperf_checkpoint_timestamp_dict[ckpt_ind + 1],metadata={'epoch_num': (ckpt_ind + 1)*FLAGS.samples_between_eval*FLAGS.avg_seq_per_pack,'epoch_count': ckpt_ind + 1,'status': 'success'})
                mlperf_mlloger.event(key=mlperf_mllog.constants.SUBMISSION_BENCHMARK, value=mlperf_mllog.constants.BERT)
                mlperf_mlloger.event(key=mlperf_mllog.constants.SUBMISSION_ORG, value='Habana')
                mlperf_mlloger.event(key=mlperf_mllog.constants.SUBMISSION_DIVISION, value='closed')
                mlperf_mlloger.event(key=mlperf_mllog.constants.SUBMISSION_PLATFORM, value='gaudi-8')
                mlperf_mlloger.event(key=mlperf_mllog.constants.SUBMISSION_STATUS, value='onprem')
                converged = True
                print("converged")
                step = int(chpt_path.strip().split("-")[-1].strip('"'))
                print("step={}".format(step))
                converge_block_idx = int(step / num_steps_between_eval )
                print("converged at step:{}, block:{}".format(step, converge_block_idx))
                break

        eval_time_elapsed = time.time() - eval_start_time
        print("Total offline distributed evaluation time={} seconds".format(eval_time_elapsed))
        if converged:
          total_train_time_secs = (mlperf_block_stop_timestamp_dict[converge_block_idx] - mlperf_run_start_timestamp)
          mlperf_run_stop_timestamp = mlperf_block_stop_timestamp_dict[converge_block_idx] + eval_time_elapsed
          time_to_train_minutes = (total_train_time_secs + eval_time_elapsed) / 60
          print("Total offline distributed evaluation time={} seconds".format(eval_time_elapsed))
          print("Total time-to-train is {} minutes ( = pure training time {} minutes + pure evaluation time  {} minutes), converged in {} blocks ".format(time_to_train_minutes, total_train_time_secs/60, eval_time_elapsed / 60, converge_block_idx))
        else:
          mlperf_mlloger.end(key=mlperf_mllog.constants.RUN_STOP,value=total_eval_masked_lm_accuracy_FP32.numpy(),time_ms=mlperf_checkpoint_timestamp_dict[ckpt_ind + 1],metadata={'epoch_num': (ckpt_ind + 1)*FLAGS.samples_between_eval*FLAGS.avg_seq_per_pack,'epoch_count': ckpt_ind + 1,'status': 'fail'})
      else:
        tf.compat.v1.logging.info("***** Running offline NON-distributed evaluation for mlperf *****")
        for ckpt_ind,chpt_path in enumerate(chpt_files):
          print("checkpoint file path={}".format(chpt_path))
          eval_results = estimator.evaluate(
            input_fn=eval_input_fn, steps=FLAGS.max_eval_steps, hooks=eval_hooks, checkpoint_path=chpt_path)
          mlperf_mlloger.event(key=mlperf_mllog.constants.EVAL_ACCURACY,value=eval_results["masked_lm_accuracy"],time_ms=mlperf_checkpoint_timestamp_dict[ckpt_ind + 1],metadata={'epoch_num': (ckpt_ind + 1)*FLAGS.samples_between_eval*FLAGS.avg_seq_per_pack,'epoch_count': ckpt_ind + 1})

          print("per rank mlm accuracy={}".format(eval_results["masked_lm_accuracy"]))
          if FLAGS.stop_threshold:
                success = bool(eval_results["masked_lm_accuracy"] >= FLAGS.stop_threshold)

          if horovod_enabled():
              past_treshold = tf.cast(past_stop_threshold(
                  FLAGS.stop_threshold, eval_results["masked_lm_accuracy"]), tf.float32)
              global_past_treshold = tf.math.greater(
                  hvd.allreduce(past_treshold, op=hvd.Sum), tf.zeros(1, tf.float32))
              if global_past_treshold.numpy():
                converged = True
                print("converged")
                step = int(chpt_path.strip().split("-")[-1].strip('"'))
                print("step={}".format(step))
                converge_block_idx = int(step / num_steps_between_eval )
                print("converged at step:{}, block:{}".format(step, converge_block_idx))
                break
          else:
              if past_stop_threshold(
                  FLAGS.stop_threshold, eval_results["masked_lm_accuracy"]):
                converged = True
                print("converged")
                step = int(chpt_path.strip().split("-")[-1].strip('"'))
                print("step={}".format(step))
                converge_block_idx = int(step / num_steps_between_eval )
                print("converged at step:{}, block:{}".format(step, converge_block_idx))
                break
        eval_time_elapsed = time.time() - eval_start_time
        print("Total offline non-distributed evaluation time={} seconds".format(eval_time_elapsed))
        if converged:
          total_train_time_secs = (mlperf_block_stop_timestamp_dict[converge_block_idx] - mlperf_run_start_timestamp)
          mlperf_run_stop_timestamp = mlperf_block_stop_timestamp_dict[converge_block_idx] + eval_time_elapsed
          time_to_train_minutes = (total_train_time_secs + eval_time_elapsed) / 60
          mlperf_mlloger.end(key=mlperf_mllog.constants.RUN_STOP,value=eval_results["masked_lm_accuracy"],time_ms=mlperf_checkpoint_timestamp_dict[ckpt_ind + 1],metadata={'epoch_num': (ckpt_ind + 1)*FLAGS.samples_between_eval*FLAGS.avg_seq_per_pack,'epoch_count': ckpt_ind + 1,'status': 'success'})
          print("Total time-to-train is {} minutes ( = pure training time {} minutes + pure evaluation time  {} minutes), converged in {} blocks ".format(time_to_train_minutes, total_train_time_secs/60, eval_time_elapsed / 60, converge_block_idx))
        else:
          mlperf_mlloger.end(key=mlperf_mllog.constants.RUN_STOP,value=eval_results["masked_lm_accuracy"],time_ms=mlperf_checkpoint_timestamp_dict[ckpt_ind + 1],metadata={'epoch_num': (ckpt_ind + 1)*FLAGS.samples_between_eval*FLAGS.avg_seq_per_pack,'epoch_count': ckpt_ind + 1,'status': 'fail'})

  if FLAGS.do_eval:
    converged = False
    num_steps_between_eval = math.ceil(FLAGS.samples_between_eval / global_batch_size)
    eval_start_time = time.time()
  #Stand-alone offline evaluation of multiple checkpoints
    chpt_file_path = master_output_dir + "/checkpoint"
    chpt_files = []
    with open(chpt_file_path, "r") as file:
        for line in file:
          tmp,chpt_step = line.split(":")
          if tmp == 'all_model_checkpoint_paths':
            step = int(chpt_step.strip().split("-")[1].strip('"'))
            if step > 0:
              chpt_files.append(master_output_dir + '/'+ chpt_step.strip().strip('"'))
    eval_files = []
    for eval_file_dir in FLAGS.eval_files_dir.split(","):
          eval_files.extend(tf.io.gfile.glob(os.path.join(eval_file_dir, "*")))

    eval_input_fn = input_fn_builder(
        input_files=eval_files,
        batch_size=FLAGS.eval_batch_size,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=False)

    eval_hooks = [LogEvalRunHook(FLAGS.eval_batch_size)]

    if horovod_enabled() and FLAGS.is_dist_eval_enabled:
      tf.compat.v1.logging.info("***** Running standalone offline distributed evaluation for mlperf *****")

      #need to shard the dataset!!!!
      eval_samples = 10000 / hvd_size()
      max_eval_steps = math.ceil(FLAGS.max_eval_steps / hvd_size())
      for chpt_path in chpt_files:
        print("checkpoint file path={}".format(chpt_path))
        eval_results = estimator.evaluate(
          input_fn=eval_input_fn, steps=max_eval_steps, hooks=eval_hooks, checkpoint_path=chpt_path)

        if FLAGS.stop_threshold:
              partial_eval_masked_lm_accuracy = eval_results["masked_lm_accuracy"] * eval_samples
              print("per rank masked_lm_accuracy={}".format(eval_results["masked_lm_accuracy"]))
              partial_eval_masked_lm_accuracy_FP32=tf.cast(partial_eval_masked_lm_accuracy, tf.float32)
              total_eval_masked_lm_accuracy_FP32 = hvd.allreduce(partial_eval_masked_lm_accuracy_FP32, op=hvd.Sum)
              total_eval_masked_lm_accuracy_FP32 /= 10000.0
              success = bool(total_eval_masked_lm_accuracy_FP32 >= FLAGS.stop_threshold)
              print("average eval_masked_lm_accuracy_FP32={}".format(total_eval_masked_lm_accuracy_FP32))
        if success:
            converged = True
            step = int(chpt_path.strip().split("-")[-1].strip('"'))
            converge_block_idx = int(step / num_steps_between_eval )
            print("converged at step:{}, block:{}".format(step, converge_block_idx))
            break
      eval_time_elapsed = time.time() - eval_start_time
      print("Total stand-alone offline distributed evaluation time={} seconds".format(eval_time_elapsed))
    else:
      tf.compat.v1.logging.info("***** Running standalone offline NON-distributed evaluation for mlperf *****")
      for chpt_path in chpt_files:
        print("checkpoint file path={}".format(chpt_path))
        eval_results = estimator.evaluate(
          input_fn=eval_input_fn, steps=FLAGS.max_eval_steps, hooks=eval_hooks, checkpoint_path=chpt_path)
        print("per rank mlm accuracy={}".format(eval_results["masked_lm_accuracy"]))
        if FLAGS.stop_threshold:
              success = bool(eval_results["masked_lm_accuracy"] >= FLAGS.stop_threshold)

        if horovod_enabled():
            past_treshold = tf.cast(past_stop_threshold(
                FLAGS.stop_threshold, eval_results["masked_lm_accuracy"]), tf.float32)
            global_past_treshold = tf.math.greater(
                hvd.allreduce(past_treshold, op=hvd.Sum), tf.zeros(1, tf.float32))
            if global_past_treshold.numpy():
              converged = True
              step = int(chpt_path.strip().split("-")[-1].strip('"'))
              converge_block_idx = int(step / num_steps_between_eval )
              print("converged at step:{}, block:{}".format(step, converge_block_idx))
              break
        else:
            if past_stop_threshold(
                FLAGS.stop_threshold, eval_results["masked_lm_accuracy"]):
              converged = True
              step = int(chpt_path.strip().split("-")[-1].strip('"'))
              converge_block_idx = int(step / num_steps_between_eval )
              print("converged at step:{}, block:{}".format(step, converge_block_idx))
              break
      eval_time_elapsed = time.time() - eval_start_time
      print("Total stand-alone offline non-distributed evaluation time={} seconds".format(eval_time_elapsed))


if __name__ == "__main__":
  tf.get_logger().propagate = False
  init_flags()
  if FLAGS.enable_habana_backend:
    from habana_frameworks.tensorflow.multinode_helpers import comm_local_rank
    from habana_frameworks.tensorflow import load_habana_module
    from TensorFlow.common.horovod_helpers import hvd, hvd_init, hvd_size, hvd_rank, horovod_enabled
  else:
    from TensorFlow.common.horovod_helpers_gpu import hvd, hvd_init, hvd_size, hvd_rank, horovod_enabled, comm_local_rank

  print("*****************************************")
  print("Arguments passed to this program: run_pretraining.")
  for key in FLAGS.__flags.keys():
      print("{} = {}".format(key, getattr(FLAGS,key)))
  if FLAGS.horovod:
    hvd_init()
  if FLAGS.enable_habana_backend:
    load_habana_module()
  flags.mark_flag_as_required("input_files_dir")
  if FLAGS.do_eval:
    flags.mark_flag_as_required("eval_files_dir")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  if FLAGS.deterministic_run:
    tf.random.set_seed(1)
    modeling.dropout = lambda input_tensor, dropout_prob: input_tensor  # disable dropout

  if FLAGS.use_xla and FLAGS.manual_fp16:
    print('WARNING! Combining --use_xla with --manual_fp16 may prevent convergence.')
    print('         This warning message will be removed when the underlying')
    print('         issues have been fixed and you are running a TF version')
    print('         that has that fix.')
  tf.compat.v1.app.run()
