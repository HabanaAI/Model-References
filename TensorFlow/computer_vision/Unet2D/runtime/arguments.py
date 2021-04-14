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
# - default values for model_dir, log_dir, batch_size, max_steps
# - removed use_amp, use_trt flags
# - added dtype, hvd_workers, dump_config, disable_hpu, synth_data, disable_ckpt_saving,
#   use_horovod, tensorboard_logging options

"""Command line argument parsing"""
import argparse
from munch import Munch

PARSER = argparse.ArgumentParser(description="UNet-medical")

PARSER.add_argument('--exec_mode',
                    choices=['train', 'train_and_predict', 'predict', 'evaluate', 'train_and_evaluate'],
                    type=str,
                    default='train_and_evaluate',
                    help="""Execution mode of running the model""")

PARSER.add_argument('--model_dir',
                    type=str,
                    default='/logs',
                    help="""Output directory for information related to the model""")

PARSER.add_argument('--data_dir',
                    type=str,
                    required=True,
                    help="""Input directory containing the dataset for training the model""")

PARSER.add_argument('--log_dir',
                    type=str,
                    default="logs",
                    help="""Output directory for training logs""")

PARSER.add_argument('--batch_size',
                    type=int,
                    default=8,
                    help="""Size of each minibatch per GPU""")

PARSER.add_argument('--dtype', '-d',
                    type=str,
                    default='bf16',
                    choices=['fp32', 'bf16'],
                    help='Data type: fp32 or bf16')

PARSER.add_argument('--fold',
                    type=int,
                    default=None,
                    help="""Chosen fold for cross-validation. Use None to disable cross-validation""")

PARSER.add_argument('--max_steps',
                    type=int,
                    default=6400,
                    help="""Maximum number of steps (batches) used for training""")

PARSER.add_argument('--log_every',
                    type=int,
                    default=100,
                    help="""Log data every n steps""")

PARSER.add_argument('--evaluate_every',
                    type=int,
                    default=0,
                    help="""Evaluate every n steps""")

PARSER.add_argument('--warmup_steps',
                    type=int,
                    default=200,
                    help="""Number of warmup steps""")

PARSER.add_argument('--weight_decay',
                    type=float,
                    default=0.0005,
                    help="""Weight decay coefficient""")

PARSER.add_argument('--learning_rate',
                    type=float,
                    default=0.0001,
                    help="""Learning rate coefficient for AdamOptimizer""")

PARSER.add_argument('--seed',
                    type=int,
                    default=0,
                    help="""Random seed""")

PARSER.add_argument('--hvd_workers',
                    type=int,
                    default=1,
                    help="""Number of Horovod workers, default 1 - Horovod disabled""")

PARSER.add_argument('--dump_config',
                    type=str,
                    default=None,
                    help="""Directory for dumping debug traces""")

PARSER.add_argument('--augment', dest='augment', action='store_true',
                    help="""Perform data augmentation during training""")
PARSER.add_argument('--no-augment', dest='augment', action='store_false')
PARSER.set_defaults(augment=False)

PARSER.add_argument('--benchmark', dest='benchmark', action='store_true',
                    help="""Collect performance metrics during training""")
PARSER.add_argument('--no-benchmark', dest='benchmark', action='store_false')

PARSER.add_argument('--use_xla', '--xla', dest='use_xla', action='store_true',
                    help="""Train using XLA""")

PARSER.add_argument('--resume_training', dest='resume_training', action='store_true',
                    help="""Resume training from a checkpoint""")

PARSER.add_argument('--disable_hpu', dest='disable_hpu', action='store_true',
                    help="""Disables execution on HPU""")

PARSER.add_argument('--synth_data', dest='synth_data', action='store_true',
                    help="""Use deterministic and synthetic data""")

PARSER.add_argument('--disable_ckpt_saving', dest='disable_ckpt_saving', action='store_true',
                    help="""Disables saving checkpoints""")

PARSER.add_argument('--use_horovod', dest='use_horovod', action='store_true',
                    help="""Enable horovod usage""")

PARSER.add_argument('--tensorboard_logging', dest='tensorboard_logging', action='store_true',
                    help="""Enable tensorboard logging""")

def parse_args(flags):
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
        'hvd_workers': flags.hvd_workers,
        'dump_config': flags.dump_config,
        'augment': flags.augment,
        'benchmark': flags.benchmark,
        'use_xla': flags.use_xla,
        'resume_training': flags.resume_training,
        'disable_hpu': flags.disable_hpu,
        'synth_data': flags.synth_data,
        'disable_ckpt_saving': flags.disable_ckpt_saving,
        'use_horovod': flags.use_horovod,
        'tensorboard_logging': flags.tensorboard_logging
    })
