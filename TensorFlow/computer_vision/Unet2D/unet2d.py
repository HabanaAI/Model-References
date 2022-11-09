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
# Copyright (C) 2020-2022 Habana Labs, Ltd. an Intel Company
###############################################################################
# Changes:
# - renamed script from main.py to unet2d.py
# - wrapped horovod import in a try-catch block so that the user is not required to install this library
#   when the model is being run on a single card
# - added tensorboard logging functionality
# - added TimeToTrain callback for dumping evaluation timestamps

import os
from collections import namedtuple

import tensorflow as tf

from model.unet import Unet
from runtime.run import train, evaluate, predict
from runtime.setup import get_logger, set_flags, prepare_model_dir
from runtime.arguments import parse_args
from data_loading.data_loader import Dataset
from TensorFlow.common.debug import dump_callback
from TensorFlow.common.tb_utils import TimeToTrainKerasHook

try:
    import horovod.tensorflow as hvd
except ImportError:
    hvd = None


def main():
    """
    Starting point of the application
    """
    params = parse_args(description="UNet-medical")
    if params.use_horovod:
        if hvd is None:
            raise RuntimeError(
                "Problem encountered during Horovod import. Please make sure that habana-horovod package is installed.")
        hvd.init()
    set_flags(params)

    model_dir = prepare_model_dir(params)
    params.model_dir = model_dir
    logger = get_logger(params)

    tb_logger = None
    ttt_callback = None
    if params.tensorboard_logging:
        log_dir = params.log_dir
        if hvd is not None and hvd.is_initialized() and params.log_all_workers:
            log_dir = os.path.join(log_dir, f'worker_{hvd.rank()}')
        tb_logger = namedtuple('TBSummaryWriters', 'train_writer eval_writer')(
            tf.summary.create_file_writer(log_dir),
            tf.summary.create_file_writer(os.path.join(log_dir, 'eval')))
        ttt_callback = TimeToTrainKerasHook(os.path.join(log_dir, 'eval'))

    model = Unet()

    dataset = Dataset(data_dir=params.data_dir,
                      batch_size=params.batch_size,
                      fold=params.fold,
                      augment=params.augment,
                      hpu_id=hvd.rank() if hvd is not None and hvd.is_initialized() else 0,
                      num_hpus=hvd.size() if hvd is not None and hvd.is_initialized() else 1,
                      seed=params.seed,
                      gaudi_type=params.gaudi_type)

    if 'train' in params.exec_mode:
        with dump_callback(params.dump_config):
            train(params, model, dataset, logger, tb_logger, ttt_callback)

    if 'evaluate' in params.exec_mode:
        evaluate(params, model, dataset, logger, tb_logger, ttt_callback)

    if 'predict' in params.exec_mode:
        predict(params, model, dataset, logger)


if __name__ == '__main__':
    main()
