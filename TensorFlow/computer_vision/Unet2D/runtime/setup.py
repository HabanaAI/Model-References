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
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################
# Changes:
# - horovod import changed with HPU horovod helpers
# - removed GPU specific flags
# - set TF_BF16_CONVERSION flag to default unet2d config for bfloat16 precision
# - changed GPU specific logic for configuration with HPU
# - fixed JSON logger output directory

import os
import multiprocessing

import numpy as np
import tensorflow as tf
import dllogger as logger
from dllogger import StdOutBackend, Verbosity, JSONStreamBackend

from TensorFlow.common.horovod_helpers import horovod_enabled, hvd_size, hvd_rank


def set_flags(params):
    if params.tf_verbosity:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = params.tf_verbosity

    if not params.no_hpu:
        from TensorFlow.common.library_loader import load_habana_module
        load_habana_module()
        if params.dtype == 'bf16':
            os.environ['TF_BF16_CONVERSION'] = params.bf16_config_path

    np.random.seed(params.seed)
    tf.random.set_seed(params.seed)

    if params.use_xla:
        tf.config.optimizer.set_jit(True)

    per_hpu_thread_count = 1
    num_hpus = hvd_size() if horovod_enabled() else 1
    cpu_count = multiprocessing.cpu_count()
    total_hpu_thread_count = per_hpu_thread_count * num_hpus

    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(cpu_count - total_hpu_thread_count)


def prepare_model_dir(params):
    worker_id = hvd_rank() if horovod_enabled() else 0
    if params.benchmark or (not params.log_all_workers and worker_id != 0):
        return None

    model_dir = os.path.join(params.model_dir, "model_checkpoint")
    if params.log_all_workers and horovod_enabled():
        model_dir = os.path.join(model_dir, f'worker_{worker_id}')

    os.makedirs(model_dir, exist_ok=True)
    if ('train' in params.exec_mode) and (not params.resume_training):
        os.system('rm -rf {}/*'.format(model_dir))
    return model_dir


def get_logger(params):
    backends = []
    worker_id = hvd_rank() if horovod_enabled() else 0
    if worker_id == 0:
        backends += [StdOutBackend(Verbosity.VERBOSE)]
        if params.log_dir:
            os.makedirs(params.log_dir, exist_ok=True)
            log_file = f"{params.log_dir}/log.json"
            backends += [JSONStreamBackend(Verbosity.VERBOSE, log_file)]
    logger.init(backends=backends)
    return logger
