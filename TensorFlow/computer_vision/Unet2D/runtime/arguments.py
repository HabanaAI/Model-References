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
# - default values for model_dir, log_dir, batch_size, max_steps, augment, xla
# - removed use_amp, use_trt flags
# - added dtype, hvd_workers, dump_config, no_hpu, synth_data, disable_ckpt_saving,
#   use_horovod, tensorboard_logging, bf16_config_path, tf_verbosity,
#   kubernetes_run options
# - SmartFormatter for textwrapping help message

"""Command line argument parsing"""

import os
import argparse
from pathlib import Path

from munch import Munch


class SmartFormatter(argparse.ArgumentDefaultsHelpFormatter):
    '''
         Custom Help Formatter used to split help text when '\n' was
         inserted in it.
    '''
    def _fill_text(self, text, width, indent):
        return ''.join(line for line in text.splitlines(keepends=True))

    def _split_lines(self, text, width):
        return [argparse.HelpFormatter._split_lines(self, t, width)[0] for t in text.splitlines()]


def get_parser(description, distributed_launcher):
    parser = argparse.ArgumentParser(description=description, formatter_class=SmartFormatter)

    parser.add_argument('--exec_mode',
                        type=str,
                        choices=['train', 'train_and_predict', 'predict', 'evaluate', 'train_and_evaluate'],
                        default='train_and_evaluate',
                        help="""Execution mode of running the model""")

    parser.add_argument('--model_dir',
                        type=str,
                        default='/tmp/unet2d',
                        help="""Output directory for information related to the model""")

    parser.add_argument('--data_dir',
                        type=str,
                        required=True,
                        help="""Input directory containing the dataset for training the model""")

    parser.add_argument('--log_dir',
                        type=str,
                        default="/tmp/unet2d",
                        help="""Output directory for training logs""")

    parser.add_argument('-b', '--batch_size',
                        type=int,
                        default=8,
                        help="""Size of each minibatch per HPU""")

    parser.add_argument('-d', '--dtype',
                        type=str,
                        default='bf16',
                        metavar='bf16/fp32',
                        choices=['fp32', 'bf16'],
                        help='Data type: fp32 or bf16')

    parser.add_argument('--fold',
                        type=int,
                        default=None,
                        help="""Chosen fold for cross-validation. Use None to disable cross-validation""")

    parser.add_argument('--max_steps',
                        type=int,
                        default=6400,
                        help="""Maximum number of steps (batches) used for training""")

    parser.add_argument('--log_every',
                        type=int,
                        default=100,
                        help="""Log data every n steps""")

    parser.add_argument('--evaluate_every',
                        type=int,
                        default=0,
                        help="""Evaluate every n steps""")

    parser.add_argument('--warmup_steps',
                        type=int,
                        default=200,
                        help="""Number of warmup steps""")

    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.0005,
                        help="""Weight decay coefficient""")

    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.0001,
                        help="""Learning rate coefficient for AdamOptimizer""")

    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help="""Random seed""")

    parser.add_argument('--dump_config',
                        type=str,
                        default=None,
                        help="""Directory for dumping debug traces""")

    parser.add_argument('--augment', dest='augment', action='store_true',
                        help="""Perform data augmentation during training""")
    parser.add_argument('--no-augment', dest='augment', action='store_false')
    parser.set_defaults(augment=True)

    parser.add_argument('--benchmark', dest='benchmark', action='store_true',
                        help="""Collect performance metrics during training""")
    parser.add_argument('--no-benchmark', dest='benchmark', action='store_false')

    parser.add_argument('--use_xla', '--xla', dest='use_xla', action='store_true',
                        help="""Train using XLA""")
    parser.add_argument('--no_xla', dest='use_xla', action='store_false')
    parser.set_defaults(use_xla=True)

    parser.add_argument('--resume_training', dest='resume_training', action='store_true',
                        help="""Resume training from a checkpoint""")

    parser.add_argument('--no_hpu', dest='no_hpu', action='store_true',
                        help="""Disables execution on HPU, train on CPU""")

    parser.add_argument('--synth_data', dest='synth_data', action='store_true',
                        help="""Use deterministic and synthetic data""")

    parser.add_argument('--disable_ckpt_saving', dest='disable_ckpt_saving', action='store_true',
                        help="""Disables saving checkpoints""")

    parser.add_argument('--hvd_workers', dest='hvd_workers', type=int, default=1,
                        help="""Number of Horovod workers for single HLS""" if distributed_launcher else argparse.SUPPRESS)  # ignored by unet2d.py

    parser.add_argument("--kubernetes_run", default=False, type=bool,
                        help="Kubernetes run" if distributed_launcher else argparse.SUPPRESS)  # ignored by unet2d.py

    parser.add_argument('--use_horovod', dest='use_horovod', action='store_true',
                        help="""Enable horovod usage""")

    parser.add_argument('--tensorboard_logging', dest='tensorboard_logging', action='store_true',
                        help="""Enable tensorboard logging""")

    parser.add_argument('--log_all_workers', dest='log_all_workers', action='store_true',
                        help="""Enable logging data for every horovod worker in a separate directory named `worker_N`""")

    DEFAULT_BF16_CONFIG_PATH = os.fspath(Path(os.path.realpath(__file__)).parents[3].joinpath("common/bf16_config/unet.json"))
    parser.add_argument('--bf16_config_path', metavar='</path/to/custom/bf16/config>', required=False, type=str, default=DEFAULT_BF16_CONFIG_PATH,
                        help="""Path to custom mixed precision config to use given in JSON format.""")

    parser.add_argument('--tf_verbosity', dest='tf_verbosity', type=int, choices=[0, 1, 2, 3],
                        help="""If set changes logging level from Tensorflow:
                                0 - all messages are logged (default behavior);
                                1 - INFO messages are not printed;
                                2 - INFO and WARNING messages are not printed;
                                3 - INFO, WARNING, and ERROR messages are not printed.""")
    return parser


def parse_args(description="UNet-medical", distributed_launcher=False):
    flags = get_parser(description, distributed_launcher).parse_args()
    return Munch({
        'exec_mode': flags.exec_mode,
        'model_dir': flags.model_dir,
        'data_dir': flags.data_dir,
        'log_dir': flags.log_dir,
        'batch_size': flags.batch_size,
        'dtype': flags.dtype,
        'fold': flags.fold,
        'max_steps': flags.max_steps,
        'log_every': flags.log_every,
        'evaluate_every': flags.evaluate_every,
        'warmup_steps': flags.warmup_steps,
        'weight_decay': flags.weight_decay,
        'learning_rate': flags.learning_rate,
        'seed': flags.seed,
        'dump_config': flags.dump_config,
        'augment': flags.augment,
        'benchmark': flags.benchmark,
        'use_xla': flags.use_xla,
        'resume_training': flags.resume_training,
        'no_hpu': flags.no_hpu,
        'synth_data': flags.synth_data,
        'disable_ckpt_saving': flags.disable_ckpt_saving,
        'hvd_workers': flags.hvd_workers,
        'kubernetes_run': flags.kubernetes_run,
        'use_horovod': flags.use_horovod,
        'tensorboard_logging': flags.tensorboard_logging,
        'log_all_workers': flags.log_all_workers,
        'bf16_config_path': flags.bf16_config_path,
        'tf_verbosity': flags.tf_verbosity,
    })
