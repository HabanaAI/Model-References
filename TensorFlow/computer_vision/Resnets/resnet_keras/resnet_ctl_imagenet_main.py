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
###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################
# List of changes:
# - loading habana module
# - added support for prefetching to HPU
# - added profiling callbacks support
# - changed include paths of modules
# - added logic for controlling habana bf16 conversion pass
# - include mechanism for dumping tensors
# - added logic for creating checkpoints in separate dir for each worker
# - conditionally set TF_DISABLE_MKL=1 and TF_ALLOW_CONTROL_EDGES_IN_HABANA_OPS=1

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
from local_flags import core as flags_core
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
from TensorFlow.common.multinode_helpers import comm_size, comm_rank
from TensorFlow.common.tb_utils import write_hparams_v2


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
        enable_xla=flags_obj.enable_xla,
        enable_scoped_allocator=flags_obj.enable_scoped_allocator)
    # Enable habana bf16 conversion pass only if native keras mixed precision is disabled
    if flags.FLAGS.dtype == 'bf16' and flags.FLAGS.use_native_keras_mixed_precision_policy == False:
        performance.set_mixed_precision_policy(tf.float32)
        os.environ['TF_BF16_CONVERSION'] = flags.FLAGS.bf16_config_path
    else:
        performance.set_mixed_precision_policy(flags_core.get_tf_dtype(flags_obj))

    os.environ.setdefault("TF_DISABLE_MKL", "1")
    os.environ.setdefault("TF_ALLOW_CONTROL_EDGES_IN_HABANA_OPS", "1")

    # This only affects GPU.
    common.set_cudnn_batchnorm_mode()

    # TODO(anj-s): Set data_format without using Keras.
    data_format = flags_obj.data_format
    if data_format is None:
        data_format = ('channels_first'
                       if tf.test.is_built_with_cuda() else 'channels_last')
    tf.keras.backend.set_image_data_format(data_format)
    batch_size = adjust_batch_size(flags_obj.batch_size)

    hls_addresses=str(os.environ.get("MULTI_HLS_IPS", "127.0.0.1")).split(",")
    TF_BASE_PORT = 2410
    mpi_rank = comm_rank()
    mpi_size = comm_size()

    if horovod_enabled():
      model_dir = os.path.join(flags_obj.model_dir, "worker_" + str(hvd.rank()))
    elif comm_rank() > 1:
      model_dir = os.path.join(flags_obj.model_dir, "worker_" + str(comm_rank()))
    else:
        model_dir = flags_obj.model_dir

    worker_hosts=""
    for address in hls_addresses:
        # worker_hosts: comma-separated list of worker ip:port pairs.
        worker_hosts  = worker_hosts + ",".join([address + ':' + str(TF_BASE_PORT + rank) for rank in range(mpi_size//len(hls_addresses))])
    task_index = mpi_rank

    # Configures cluster spec for distribution strategy.
    _ = distribution_utils.configure_cluster(worker_hosts, task_index)

    strategy = distribution_utils.get_distribution_strategy(
        distribution_strategy=flags_obj.distribution_strategy,
        num_gpus=flags_obj.num_gpus,
        all_reduce_alg=flags_obj.all_reduce_alg,
        num_packs=flags_obj.num_packs,
        tpu_address=flags_obj.tpu)

    train_writer, eval_writer = None, None
    if flags_obj.enable_tensorboard:
        train_writer = tf.summary.create_file_writer(model_dir)
        eval_writer = tf.summary.create_file_writer(os.path.join(model_dir, 'eval'))
        write_hparams_v2(train_writer, flags_obj.flag_values_dict())

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
                                                  profiler_callback)

    eval_interval = flags_obj.epochs_between_evals * per_epoch_steps
    checkpoint_interval = (
        per_epoch_steps if flags_obj.enable_checkpoint_and_export else None)
    summary_interval = flags_obj.log_steps if flags_obj.enable_tensorboard else None

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
        eval_interval=eval_interval,
        train_summary_writer=train_writer,
        eval_summary_writer=eval_writer)

    time_callback.on_train_begin()
    resnet_controller.train(evaluate=not flags_obj.skip_eval)
    time_callback.on_train_end()

    stats = build_stats(runnable, time_callback)
    return stats


def main(_):
    common.initialize_preloading()
    if flags.FLAGS.use_horovod and flags.FLAGS.distribution_strategy!="off":
        raise RuntimeError("Horovod and distribution strategy cannot be used together. Please select one of the scaleout methods.")
    if flags.FLAGS.distribution_strategy not in ["off", "hpu"]:
        raise RuntimeError("Currently HPU supports only HPUStrategy, please set --distribution_strategy=hpu or use horovod")
    if flags.FLAGS.use_horovod:
        if flags.FLAGS.horovod_hierarchical_allreduce:
            os.environ['HOROVOD_HIERARCHICAL_ALLREDUCE'] = "1"
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
