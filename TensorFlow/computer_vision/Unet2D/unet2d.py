# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
# - renamed script from main.py to unet2d.py
# - included HPU horovod helpers
# - added tensorboard logging functionality

import os
from collections import namedtuple

import tensorflow as tf

from model.unet import Unet
from runtime.run import train, evaluate, predict
from runtime.setup import get_logger, set_flags, prepare_model_dir
from runtime.arguments import parse_args
from data_loading.data_loader import Dataset
from TensorFlow.common.debug import dump_callback
from TensorFlow.common.horovod_helpers import hvd_init, horovod_enabled, hvd_size, hvd_rank


def main():
    """
    Starting point of the application
    """
    params = parse_args(description="UNet-medical")
    if params.use_horovod:
        hvd_init()
    set_flags(params)

    model_dir = prepare_model_dir(params)
    params.model_dir = model_dir
    logger = get_logger(params)

    tb_logger = None
    if params.tensorboard_logging:
        log_dir = params.log_dir
        if horovod_enabled() and params.log_all_workers:
            log_dir = os.path.join(log_dir, f'worker_{hvd_rank()}')
        tb_logger = namedtuple('TBSummaryWriters', 'train_writer eval_writer')(
            tf.summary.create_file_writer(log_dir),
            tf.summary.create_file_writer(os.path.join(log_dir, 'eval')))

    model = Unet()

    dataset = Dataset(data_dir=params.data_dir,
                      batch_size=params.batch_size,
                      fold=params.fold,
                      augment=params.augment,
                      hpu_id=hvd_rank() if horovod_enabled() else 0,
                      num_hpus=hvd_size() if horovod_enabled() else 1,
                      seed=params.seed)

    if 'train' in params.exec_mode:
        with dump_callback(params.dump_config):
            train(params, model, dataset, logger, tb_logger)

    if 'evaluate' in params.exec_mode:
        evaluate(params, model, dataset, logger, tb_logger)

    if 'predict' in params.exec_mode:
        predict(params, model, dataset, logger)


if __name__ == '__main__':
    main()
