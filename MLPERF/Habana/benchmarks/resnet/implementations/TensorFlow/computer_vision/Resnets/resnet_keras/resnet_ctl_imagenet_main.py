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
# - loading habana module
# - added support for prefetching to HPU
# - added profiling callbacks support
# - changed include paths of modules
# - include mechanism for dumping tensors

# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company

"""Runs a ResNet model on the ImageNet dataset using custom training loops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import os
import time
import math

from TensorFlow.common.modeling import performance
from TensorFlow.common.training import controller
from TensorFlow.utils.flags import core as flags_core
from TensorFlow.utils.logs import logger
from TensorFlow.utils.misc import distribution_utils
from TensorFlow.utils.misc import keras_utils
from TensorFlow.utils.misc import model_helpers
from TensorFlow.computer_vision.common import imagenet_preprocessing
from TensorFlow.computer_vision.Resnets.utils.optimizers.keras import lars_util
from TensorFlow.computer_vision.Resnets.resnet_keras import common
from TensorFlow.computer_vision.Resnets.resnet_keras import resnet_runnable
from TensorFlow.computer_vision.Resnets.resnet_keras.common import get_global_batch_size
from habana_frameworks.tensorflow import load_habana_module
from TensorFlow.common.debug import dump_callback
from TensorFlow.common.tb_utils import write_hparams_v2
from TensorFlow.common.horovod_helpers import hvd, hvd_init, hvd_size, horovod_enabled, synapse_logger_init
from TensorFlow.computer_vision.Resnets.resnet_keras.mlp_log import get_mllog_mlloger


flags.DEFINE_boolean(name='use_tf_function', default=True,
                     help='Wrap the train and test step inside a '
                     'tf.function.')
flags.DEFINE_boolean(name='single_l2_loss_op', default=False,
                     help='Calculate L2_loss on concatenated weights, '
                     'instead of using Keras per-layer L2 loss.')
flags.DEFINE_boolean(name='cache_decoded_image',
                     default=False,
                     help='Whether or not to cache decoded images in the '
                     'input pipeline. If this flag and `cache` is enabled, '
                     'then TFExample protos will be parsed and then cached '
                     'which reduces the load on hosts.')
flags.DEFINE_boolean(name='dist_eval', default=True,
                     help='Partial eval in each rank and allreduce the partial result')
flags.DEFINE_boolean(name='enable_device_warmup',
                     default=False,
                     help='Whether or not to enable device warmup. This '
                     'includes training on dummy data and enabling graph/XLA '
                     'compilation before run_start.')
flags.DEFINE_integer(name='device_warmup_steps',
                     default=2,
                     help='The number of steps to apply for device warmup.')
flags.DEFINE_float('base_learning_rate', 0.1,
                   'Base learning rate. '
                   'This is the learning rate when using batch size 256; when using other '
                   'batch sizes, the learning rate will be scaled linearly.')
flags.DEFINE_boolean(name='profile', default=False,
                    help='Running RN50 with profiling')





def build_stats(runnable, time_callback):
    """Normalizes and returns dictionary of stats.

    Args:
      runnable: The module containing all the training and evaluation metrics.
      time_callback: Time tracking callback instance.

    Returns:
      Dictionary of normalized results.
    """
    stats = {}

    if not runnable.flags_obj.skip_eval:
        if runnable.test_loss:
            stats['eval_loss'] = runnable.test_loss.result().numpy()
        if runnable.test_accuracy:
            stats['eval_acc'] = runnable.eval_accuracy

        if runnable.train_loss:
            stats['train_loss'] = runnable.train_loss.result().numpy()
        if runnable.train_accuracy:
            stats['train_acc'] = runnable.train_accuracy.result().numpy()

    if time_callback:
        timestamp_log = time_callback.timestamp_log
        stats['step_timestamp_log'] = timestamp_log
        stats['train_finish_time'] = time_callback.train_finish_time
        if time_callback.epoch_runtime_log:
            stats['avg_exp_per_second'] = time_callback.average_examples_per_second

    return stats




def get_num_train_iterations(flags_obj):
    """Returns the number of training steps, train and test epochs."""
    global_batch_size = get_global_batch_size(flags_obj.batch_size)
    steps_per_epoch = math.ceil(imagenet_preprocessing.NUM_IMAGES['train'] / global_batch_size)
    train_epochs = flags_obj.train_epochs

    if train_epochs == 0 and flags_obj.train_steps > 0:
        steps_per_epoch = flags_obj.train_steps
        train_epochs = 1

    eval_batch_size = flags_obj.batch_size
    if flags_obj.dist_eval:
        eval_batch_size = global_batch_size
    eval_steps = (
        math.ceil(imagenet_preprocessing.NUM_IMAGES['validation'] / eval_batch_size))

    return steps_per_epoch, train_epochs, eval_steps


def _steps_to_run(steps_in_current_epoch, steps_per_epoch, steps_per_loop):
    """Calculates steps to run on device."""
    if steps_per_loop <= 0:
        raise ValueError('steps_per_loop should be positive integer.')
    if steps_per_loop == 1:
        return steps_per_loop
    return min(steps_per_loop, steps_per_epoch - steps_in_current_epoch)


def run(flags_obj):
    """Run ResNet ImageNet training and eval loop using custom training loops.

    Args:
      flags_obj: An object containing parsed flag values.

    Raises:
      ValueError: If fp16 is passed as it is not currently supported.

    Returns:
      Dictionary of training and eval stats.
    """
    tf.get_logger().propagate = False
    output_dir = None
    if "LOG_DIR" in os.environ:
        output_dir = os.environ["LOG_DIR"]
    mlperf_mlloger, mlperf_mllog = get_mllog_mlloger(output_dir)
    mlperf_mlloger.event(key=mlperf_mllog.constants.CACHE_CLEAR, value=True)
    mlperf_mlloger.start(key=mlperf_mllog.constants.INIT_START, value=None)
    mlperf_mlloger.event(key=mlperf_mllog.constants.SUBMISSION_BENCHMARK, value=mlperf_mllog.constants.RESNET)
    mlperf_mlloger.event(key=mlperf_mllog.constants.SUBMISSION_ORG, value='Habana')
    mlperf_mlloger.event(key=mlperf_mllog.constants.SUBMISSION_DIVISION, value='closed')
    mlperf_mlloger.event(key=mlperf_mllog.constants.SUBMISSION_PLATFORM, value='gaudi-{}'.format(flags_obj.num_gpus))
    mlperf_mlloger.event(key=mlperf_mllog.constants.SUBMISSION_STATUS, value='onprem')

    keras_utils.set_session_config(
        enable_eager=flags_obj.enable_eager,
        enable_xla=flags_obj.enable_xla)
    performance.set_mixed_precision_policy(flags_core.get_tf_dtype(flags_obj))

    # This only affects GPU.
    common.set_cudnn_batchnorm_mode()

    # TODO(anj-s): Set data_format without using Keras.
    data_format = flags_obj.data_format
    if data_format is None:
        data_format = ('channels_first'
                       if tf.test.is_built_with_cuda() else 'channels_last')
    tf.keras.backend.set_image_data_format(data_format)

    if horovod_enabled():
      model_dir = os.path.join(flags_obj.model_dir, "worker_" + str(hvd.rank()))
    else:
      model_dir = flags_obj.model_dir

    global_batch_size = get_global_batch_size(flags_obj.batch_size)

    strategy = distribution_utils.get_distribution_strategy(
        distribution_strategy=flags_obj.distribution_strategy,
        num_gpus=flags_obj.num_gpus,
        all_reduce_alg=flags_obj.all_reduce_alg,
        num_packs=flags_obj.num_packs,
        tpu_address=flags_obj.tpu)

    mlperf_mlloger.event(key=mlperf_mllog.constants.GLOBAL_BATCH_SIZE, value=global_batch_size)
    mlperf_mlloger.event(key=mlperf_mllog.constants.TRAIN_SAMPLES, value=imagenet_preprocessing.NUM_IMAGES['train'])
    mlperf_mlloger.event(key=mlperf_mllog.constants.EVAL_SAMPLES, value=imagenet_preprocessing.NUM_IMAGES['validation'])
    group_batch_norm = 1
    mlperf_mlloger.event(key=mlperf_mllog.constants.MODEL_BN_SPAN, value= flags_obj.batch_size * group_batch_norm)

    train_writer, eval_writer = None, None
    if flags_obj.enable_tensorboard:
        train_writer = tf.summary.create_file_writer(model_dir)
        eval_writer = tf.summary.create_file_writer(os.path.join(model_dir, 'eval'))
        hparams = flags_obj.flag_values_dict()
        write_hparams_v2(train_writer, hparams)

    per_epoch_steps, train_epochs, eval_steps = get_num_train_iterations(
        flags_obj)
    steps_per_loop = min(flags_obj.steps_per_loop, per_epoch_steps)
    train_steps = train_epochs * per_epoch_steps

    logging.info(
        'Training %d epochs, each epoch has %d steps, '
        'total steps: %d; Eval %d steps', train_epochs, per_epoch_steps,
        train_steps, eval_steps)

    time_callback = keras_utils.TimeHistory(
        global_batch_size,
        flags_obj.log_steps,
        summary_writer=train_writer,
        batch_size_per_node=flags_obj.batch_size)
    profiler_callback = None
    if flags_obj.profile_steps is not None:
        profiler_callback = keras_utils.get_profiler_callback(
            model_dir,
            flags_obj.profile_steps,
            flags_obj.enable_tensorboard,
            per_epoch_steps)
    with distribution_utils.get_strategy_scope(strategy):
        runnable = resnet_runnable.ResnetRunnable(flags_obj, time_callback,
                                                  train_steps,
                                                  per_epoch_steps,
                                                  profiler_callback,mlperf_mlloger,mlperf_mllog)

    eval_interval = flags_obj.epochs_between_evals * per_epoch_steps
    eval_offset = flags_obj.eval_offset_epochs * per_epoch_steps
    if eval_offset != 0:
        eval_offset -= eval_interval
    checkpoint_interval = (
        per_epoch_steps if flags_obj.enable_checkpoint_and_export else None)
    summary_interval = per_epoch_steps if flags_obj.enable_tensorboard else None

    checkpoint_manager = tf.train.CheckpointManager(
        runnable.checkpoint,
        directory=model_dir,
        max_to_keep=10,
        step_counter=runnable.global_step,
        checkpoint_interval=checkpoint_interval)

    device_warmup_steps = (
      flags_obj.device_warmup_steps if flags_obj.enable_device_warmup else 0)
    if flags_obj.enable_device_warmup:
      logging.info('Warmup for %d steps.', device_warmup_steps)

    train_steps=per_epoch_steps * train_epochs

    resnet_controller = controller.Controller(
        strategy,
        runnable.train,
        runnable.evaluate,
        runnable.warmup,
        global_step=runnable.global_step,
        steps_per_loop=steps_per_loop,
        train_steps=train_steps,
        checkpoint_manager=checkpoint_manager,
        summary_interval=summary_interval,
        eval_steps=eval_steps,
        eval_interval=eval_interval,
        eval_offset=eval_offset,
        device_warmup_steps=device_warmup_steps,
        train_summary_writer=train_writer,
        eval_summary_writer=eval_writer)

    if flags_obj.enable_device_warmup:
        resnet_controller.warmup()

    mlperf_mlloger.end(key=mlperf_mllog.constants.INIT_STOP)

    hvd.broadcast(0, 0)
    time_callback.on_train_begin()
    mlperf_mlloger.start(key=mlperf_mllog.constants.RUN_START)
    mlperf_mlloger.start(
        key=mlperf_mllog.constants.BLOCK_START, value=None,
        metadata={
            'first_epoch_num': 1,
            'epoch_count':
                (flags_obj.eval_offset_epochs if flags_obj.eval_offset_epochs > 0
                 else flags_obj.epochs_between_evals)
        })
    resnet_controller.train(evaluate=not flags_obj.skip_eval)
    if not flags_obj.skip_eval:
        eval_accuracy = resnet_controller.last_eval_output['test_accuracy']
        if eval_accuracy >= flags_obj.target_accuracy:
            mlperf_mlloger.end(key=mlperf_mllog.constants.RUN_STOP, value=None, metadata={'status': 'success'})
        else:
            mlperf_mlloger.end(key=mlperf_mllog.constants.RUN_STOP, value=None, metadata={'status': 'fail'})
    time_callback.on_train_end()


    stats = build_stats(runnable, time_callback)
    return stats


def main(_):
    common.initialize_preloading()
    if flags.FLAGS.use_horovod:
        hvd_init()
    else:
        synapse_logger_init()
    load_habana_module()

    with dump_callback():
        model_helpers.apply_clean(flags.FLAGS)
        with logger.benchmark_context(flags.FLAGS):
            stats =run (flags.FLAGS)
        logging.info('Run stats:\n%s', stats)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    common.define_keras_flags()
    common.define_habana_flags()
    lars_util.define_lars_flags()
    app.run(main)
