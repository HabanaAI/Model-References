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
# - default values for exec_mode, max_steps, log_dir, augment, use_xla, batch_size
# - added dtype, num_workers_per_hls, kubernetes_run, use_horovod, no-augment, no_xla,
#   seed, no_hpu, tensorboard_logging, log_all_workers, tf_verbosity, bf16_config_path options
# - included missing help for all available flags
# - parser has been wraper with Munch dict for more elegant parameter access


import os
import argparse
from pathlib import Path

from munch import Munch


def get_parser(description, distributed_launcher):

    parser = argparse.ArgumentParser(description=description)

    # Estimator flags
    parser.add_argument('--model_dir', required=True, type=str,
                        help="""Output directory for information related to the model""")
    parser.add_argument('--exec_mode', default="train_and_evaluate",
                        choices=['train', 'evaluate', 'train_and_evaluate', 'predict'], type=str,
                        help="""Execution mode of running the model""")

    # Training flags
    parser.add_argument('--benchmark', dest='benchmark', action='store_true', default=False,
                        help="""Collect performance metrics during training""")
    parser.add_argument('--max_steps', default=16000, type=int,
                        help="""Maximum number of steps used for training""")
    parser.add_argument('--learning_rate', default=0.0002, type=float,
                        help="""Learning rate coefficient for AdamOptimizer""")
    parser.add_argument('--log_every', default=100, type=int,
                        help="""Log data every n steps""")
    parser.add_argument('--log_dir', default="/tmp/unet3d_logs", type=str,
                        help="""Output directory for logs""")
    parser.add_argument('--loss', choices=['dice', 'ce', 'dice+ce'], default='dice+ce', type=str,
                        help="""Loss function to be used during training""")
    parser.add_argument('--warmup_steps', default=40, type=int,
                        help="""Number of warmup steps""")
    parser.add_argument('--normalization', choices=['instancenorm', 'batchnorm', 'groupnorm'],
                        default='instancenorm', type=str,
                        help="""Normalization block to be applied in the model""")
    parser.add_argument('--include_background', dest='include_background', action='store_true', default=False,
                        help="""Include background both in preditions and labels""")
    parser.add_argument('--resume_training', dest='resume_training', action='store_true', default=False,
                        help="""Resume training from a checkpoint""")
    parser.add_argument('--num_workers_per_hls', dest='num_workers_per_hls', type=int, default=1,
                        help="""Number of workers for single HLS""" if distributed_launcher else argparse.SUPPRESS)  # ignored by main.py
    parser.add_argument("--hls_type", default="HLS1", type=str,
                        help="Type of HLS" if distributed_launcher else argparse.SUPPRESS)  # ignored by main.py
    parser.add_argument("--kubernetes_run", default=False, type=bool,
                        help="Kubernetes run" if distributed_launcher else argparse.SUPPRESS)  # ignored by main.py
    parser.add_argument('--use_horovod', dest='use_horovod', action='store_true',
                        help="""Enable horovod usage""")

    # Augmentations
    parser.add_argument('--augment', dest='augment', action='store_true',
                        help="""Perform data augmentation during training""")
    parser.add_argument('--no-augment', dest='augment', action='store_false')
    parser.set_defaults(augment=True)

    # Dataset flags
    parser.add_argument('--data_dir', required=True, type=str,
                        help="""Input directory containing the dataset for training the model""")
    parser.add_argument('--batch_size', default=2, type=int,
                        help="""Size of each minibatch per device""")
    parser.add_argument('--fold', default=0, type=int,
                        help="""Chosen fold for cross-validation""")
    parser.add_argument('--num_folds', default=5, type=int,
                        help="""Number of folds in k-cross-validation of dataset""")

    # Tensorflow configuration flags
    parser.add_argument('--dtype', '-d', type=str, choices=['fp32', 'bf16'], default='bf16',
                        help='Data type for HPU: fp32 or bf16')
    parser.add_argument('--use_amp', '--amp', dest='use_amp', action='store_true', default=False,
                        help="""Train using TF-AMP for GPU/CPU""")
    parser.add_argument('--use_xla', '--xla', dest='use_xla', action='store_true',
                        help="""Train using XLA""")
    parser.add_argument('--no_xla', dest='use_xla', action='store_false')
    parser.set_defaults(use_xla=True)

    parser.add_argument('--seed', default=None, type=int,
                        help="""Random seed""")
    parser.add_argument('--no_hpu', dest='no_hpu', action='store_true',
                        help="""Do not load Habana modules. Train the model on CPU/GPU""")
    parser.add_argument('--tensorboard_logging', dest='tensorboard_logging', action='store_true',
                        help="""Enable tensorboard logging""")
    parser.add_argument('--log_all_workers', dest='log_all_workers', action='store_true',
                        help="""Enable logging data for every worker in a separate directory named `worker_N`""")
    parser.add_argument('--tf_verbosity', dest='tf_verbosity', type=int, choices=[0, 1, 2, 3],
                        help="""Logging level from Tensorflow.
                                0 = all messages are logged (default behavior)
                                1 = INFO messages are not printed
                                2 = INFO and WARNING messages are not printed
                                3 = INFO, WARNING, and ERROR messages are not printed""")
    DEFAULT_BF16_CONFIG_PATH = os.fspath(Path(os.path.realpath(
        __file__)).parents[3].joinpath("common/bf16_config/unet.json"))
    parser.add_argument('--bf16_config_path', metavar='</path/to/custom/bf16/config>', required=False, type=str, default=DEFAULT_BF16_CONFIG_PATH,
                        help=f'Path to custom mixed precision config to use given in JSON format. Defaults to {DEFAULT_BF16_CONFIG_PATH}')

    return parser


def parse_args(description="UNet-3D", distributed_launcher=False):
    flags = get_parser(description, distributed_launcher).parse_args()
    return Munch({
        'model_dir': flags.model_dir,
        'exec_mode': flags.exec_mode,
        'benchmark': flags.benchmark,
        'max_steps': flags.max_steps,
        'learning_rate': flags.learning_rate,
        'log_every': flags.log_every,
        'log_dir': flags.log_dir,
        'loss': flags.loss,
        'warmup_steps': flags.warmup_steps,
        'normalization': flags.normalization,
        'include_background': flags.include_background,
        'resume_training': flags.resume_training,
        'augment': flags.augment,
        'data_dir': flags.data_dir,
        'batch_size': flags.batch_size,
        'fold': flags.fold,
        'num_folds': flags.num_folds,
        'use_amp': flags.use_amp,
        'use_xla': flags.use_xla,
        'dtype': flags.dtype,
        'precision': flags.dtype,
        'num_workers_per_hls': flags.num_workers_per_hls,
        'hls_type': flags.hls_type,
        'kubernetes_run': flags.kubernetes_run,
        'use_horovod': flags.use_horovod,
        'seed': flags.seed,
        'no_hpu': flags.no_hpu,
        'tensorboard_logging': flags.tensorboard_logging,
        'log_all_workers': flags.log_all_workers,
        'tf_verbosity': flags.tf_verbosity,
        'bf16_config_path': flags.bf16_config_path,
    })
