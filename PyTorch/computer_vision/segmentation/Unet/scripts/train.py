# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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


import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from os.path import dirname
from subprocess import call

parser = ArgumentParser(ArgumentDefaultsHelpFormatter)
parser.add_argument("--task", type=str, default="01", help="Path to data")
parser.add_argument("--data", type=str, default="/data", help="Path to data directory")
parser.add_argument("--gpus", type=int, help="Number of GPUs")
parser.add_argument("--hpus", type=int, help="Number of HPUs")
parser.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4], help="Fold number")
parser.add_argument("--dim", type=int, required=True, choices=[2, 3], help="Dimension of UNet")
parser.add_argument("--save_ckpt", action="store_true", help="Enable saving checkpoint")
parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision")
parser.add_argument('--autocast', action="store_true", help='Enable autocast on HPU')
parser.add_argument("--tta", action="store_true", help="Enable test time augmentation")
parser.add_argument("--results", type=str, default="/results", help="Path to results directory")
parser.add_argument("--logname", type=str, default="log", help="Name of dlloger output")
parser.add_argument("--optimizer",
                    type=str,
                    default="adamw",
                    choices=["sgd", "radam", "adam", "adamw", "fusedadamw"],
                    help="Optimizer")
parser.add_argument('--habana_loader', action='store_true', help='Enable Habana Media Loader. Media loader is not supported on Gaudi(1)')

if __name__ == "__main__":
    args = parser.parse_args()
    path_to_main = os.path.join(dirname(dirname(os.path.realpath(__file__))), "main.py")
    cmd = f"python {path_to_main} --exec_mode train --task {args.task} --deep_supervision --data {args.data} "
    cmd += "--save_ckpt " if args.save_ckpt else ""
    cmd += "--habana_loader " if args.habana_loader else ""
    cmd += f"--results {args.results} "
    cmd += f"--logname {args.logname} "
    cmd += f"--dim {args.dim} "
    cmd += f"--batch_size {2 if args.dim == 3 else 64} "
    val_batch_size = 0
    if args.dim == 3:
        if args.hpus:
            val_batch_size = 2
        else:
            val_batch_size = 4
    else:
        val_batch_size = 64
    cmd += f"--val_batch_size {val_batch_size} "
    cmd += f"--fold {args.fold} "
    if args.gpus:
        cmd += f"--gpus {args.gpus} "
    if args.hpus:
        cmd += f"--hpus {args.hpus} "
    cmd += "--amp " if args.amp else ""
    cmd += "--autocast " if args.autocast else ""
    cmd += "--tta " if args.tta else ""
    cmd += f"--optimizer {args.optimizer}" if args.optimizer else ""
    call(cmd, shell=True)
