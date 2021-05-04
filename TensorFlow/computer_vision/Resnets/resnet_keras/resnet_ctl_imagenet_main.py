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
# - flag for setting tensorflow global seed
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
from TensorFlow.computer_vision.Resnets.resnet_keras.common import adjust_batch_size

from TensorFlow.common.library_loader import load_habana_module
from TensorFlow.common.debug import dump_callback
from TensorFlow.common.horovod_helpers import hvd, hvd_init, hvd_size, horovod_enabled, synapse_logger_init


flags.DEFINE_integer(name='global_seed', default=None, help='Global seed set using `tf.random.set_seed`')
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
flags.DEFINE_float('base_learning_rate', 0.1,
                   'Base learning rate. '
                   'This is the learning rate when using batch size 256; when using other '
                   'batch sizes, the learning rate will be scaled linearly.')


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
        stats['eval_loss'] = runnable.test_loss.result().numpy()
        stats['eval_acc'] = runnable.test_accuracy.result().numpy()

        stats['train_loss'] = runnable.train_loss.result().numpy()
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
    train_steps = (
        imagenet_preprocessing.NUM_IMAGES['train'] // adjust_batch_size(flags_obj.batch_size))
    train_epochs = flags_obj.train_epochs

    if flags_obj.train_steps:
        train_steps = min(flags_obj.train_steps, train_steps)
        train_epochs = 1

    eval_steps = (
        imagenet_preprocessing.NUM_IMAGES['validation'] // flags_obj.batch_size)

    return train_steps, train_epochs, eval_steps


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
      batch_size = adjust_batch_size(flags_obj.batch_size)
      model_dir = os.path.join(flags_obj.model_dir, "worker_" + str(hvd.rank()))
    else:
      batch_size = flags_obj.batch_size
      model_dir = flags_obj.model_dir


    strategy = distribution_utils.get_distribution_strategy(
        distribution_strategy=flags_obj.distribution_strategy,
        num_gpus=flags_obj.num_gpus,
        all_reduce_alg=flags_obj.all_reduce_alg,
        num_packs=flags_obj.num_packs,
        tpu_address=flags_obj.tpu)


    per_epoch_steps, train_epochs, eval_steps = get_num_train_iterations(
        flags_obj)
    steps_per_loop = min(flags_obj.steps_per_loop, per_epoch_steps)
    train_steps = train_epochs * per_epoch_steps

    logging.info(
        'Training %d epochs, each epoch has %d steps, '
        'total steps: %d; Eval %d steps', train_epochs, per_epoch_steps,
        train_steps, eval_steps)

    time_callback = keras_utils.TimeHistory(
        batch_size,
        flags_obj.log_steps,
        logdir=model_dir if flags_obj.enable_tensorboard else None)
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
                                                  profiler_callback)

    eval_interval = flags_obj.epochs_between_evals * per_epoch_steps
    checkpoint_interval = (
        per_epoch_steps if flags_obj.enable_checkpoint_and_export else None)
    summary_interval = per_epoch_steps if flags_obj.enable_tensorboard else None

    checkpoint_manager = tf.train.CheckpointManager(
        runnable.checkpoint,
        directory=model_dir,
        max_to_keep=10,
        step_counter=runnable.global_step,
        checkpoint_interval=checkpoint_interval)

    train_steps=per_epoch_steps * train_epochs

    resnet_controller = controller.Controller(
        strategy,
        runnable.train,
        runnable.evaluate,
        global_step=runnable.global_step,
        steps_per_loop=steps_per_loop,
        train_steps=train_steps,
        checkpoint_manager=checkpoint_manager,
        summary_interval=summary_interval,
        eval_steps=eval_steps,
        eval_interval=eval_interval)

    time_callback.on_train_begin()
    resnet_controller.train(evaluate=not flags_obj.skip_eval)
    if flags_obj.enable_checkpoint_and_export: 
        runnable.model.save("./saved_model/resnet50", save_format="tf", include_optimizer=False)
    time_callback.on_train_end()

    stats = build_stats(runnable, time_callback)
    return stats


def main(_):
    common.initialize_preloading()
    if flags.FLAGS.use_horovod:
        hvd_init()
    else:
        synapse_logger_init()
    log_info_devices = load_habana_module()
    logging.info('Devices:\n%s', log_info_devices)

    if flags.FLAGS.global_seed:
        tf.random.set_seed(flags.FLAGS.global_seed)

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
