# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
###############################################################################
# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
###############################################################################
# Changes:
# - script migration to Tensorflow 2.x version
# - added HPU custom setup functions
# - added possibility to handle non-HPU runs
# - added possibility to log data from every horovod worker

import os

import dllogger as logger
import tensorflow as tf
from dllogger import StdOutBackend, Verbosity, JSONStreamBackend

from model.model_fn import unet_3d


def setup_horovod(params):
    params.hvd = None
    params.worker_id = 0
    params.num_workers = 1
    if params.use_horovod:
        if params.no_hpu:
            # Horovod on GPU
            import horovod.tensorflow as hvd
            hvd.init()
            os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())
        else:
            from TensorFlow.common.horovod_helpers import hvd, hvd_init
            hvd_init()
        params.worker_id = hvd.rank()
        params.num_workers = hvd.size()
        params.hvd = hvd
        if params.log_all_workers:
            params.log_dir = os.path.join(params.log_dir, f'worker_{params.worker_id}')
            params.model_dir = os.path.join(params.model_dir, f'worker_{params.worker_id}')

    return params


def set_flags(params):
    if params.tf_verbosity:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = params.tf_verbosity

    if params.no_hpu:
        os.environ['CUDA_CACHE_DISABLE'] = '1'
        os.environ['HOROVOD_GPU_ALLREDUCE'] = 'NCCL'
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '0'
        os.environ['TF_ADJUST_HUE_FUSED'] = '1'
        os.environ['TF_ADJUST_SATURATION_FUSED'] = '1'
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
        os.environ['TF_SYNC_ON_FINISH'] = '0'
    else:
        from habana_frameworks.tensorflow import load_habana_module
        load_habana_module()
        if params.dtype == 'bf16':
            os.environ['TF_BF16_CONVERSION'] = params.bf16_config_path


def prepare_model_dir(params):
    model_dir = os.path.join(params.model_dir, "model_checkpoint")
    model_dir = model_dir if ((params.worker_id == 0 or params.log_all_workers) and not params.benchmark) else None
    if model_dir is not None:
        os.makedirs(model_dir, exist_ok=True)
        if ('train' in params.exec_mode) and (not params.resume_training):
            os.system('rm -rf {}/*'.format(model_dir))

    return model_dir


def build_estimator(params, model_dir):
    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(), allow_soft_placement=True)

    if params.use_xla:
        config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1

    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = os.getenv('CUDA_VISIBLE_DEVICES', '0')

    if params.use_amp:
        config.graph_options.rewrite_options.auto_mixed_precision = 1

    checkpoint_steps = (params.max_steps // params.num_workers) if params.worker_id == 0 else None
    checkpoint_steps = checkpoint_steps if not params.benchmark else None
    run_config = tf.estimator.RunConfig(
        save_summary_steps=params.max_steps,
        session_config=config,
        save_checkpoints_steps=checkpoint_steps,
        keep_checkpoint_max=1)

    return tf.estimator.Estimator(
        model_fn=unet_3d,
        model_dir=model_dir,
        config=run_config,
        params=params)


def get_logger(params):
    backends = []
    if params.worker_id == 0 or params.log_all_workers:
        backends += [StdOutBackend(Verbosity.VERBOSE)]
        if params.log_dir:
            os.makedirs(params.log_dir, exist_ok=True)
            log_file = f"{params.log_dir}/log.json"
            backends += [JSONStreamBackend(Verbosity.VERBOSE, log_file)]
    logger.init(backends=backends)
    return logger
