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
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################


import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from os.path import dirname
from subprocess import call

parser = ArgumentParser(ArgumentDefaultsHelpFormatter)
parser.add_argument("--mode", type=str, required=True, choices=["train", "predict"], help="Benchmarking mode")
parser.add_argument("--data", type=str, default="/data", help="Path to data directory")
parser.add_argument("--task", type=str, default="01", help="Task code")
parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs to use")
parser.add_argument("--hpus", type=int, default=0, help="Number of HPUs to use")
parser.add_argument("--dim", type=int, required=True, help="Dimension of UNet")
parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision")
parser.add_argument('--autocast', action="store_true", help='Enable autocast on HPU')
parser.add_argument("--train_batches", type=int, default=150, help="Number of batches for training")
parser.add_argument("--test_batches", type=int, default=150, help="Number of batches for inference")
parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations before collecting statistics")
parser.add_argument("--results", type=str, default="/results", help="Path to results directory")
parser.add_argument("--logname", type=str, default="perf.json", help="Name of dlloger output")
parser.add_argument("--profile", action="store_true", help="Enable dlprof profiling")
parser.add_argument("--optimizer",
                    type=str,
                    default="adam",
                    choices=["sgd", "radam", "adam", "adamw", "fusedadamw"],
                    help="Optimizer")
parser.add_argument('--habana_loader', action='store_true', help='Enable Habana Media Loader')
parser.add_argument("--inference_mode", type=str, default="graphs", choices=["lazy", "graphs"], help="inference mode to run")
parser.add_argument("--measurement_type", type=str, choices=["throughput", "latency"], default="throughput", help="Measurement mode for inference benchmark")

if __name__ == "__main__":
    args = parser.parse_args()
    path_to_main = os.path.join(dirname(dirname(os.path.realpath(__file__))), "main.py")
    cmd = ""
    cmd += f"python {path_to_main} --task {args.task} --benchmark --max_epochs 2 --min_epochs 1 --optimizer {args.optimizer} --data {args.data} "
    cmd += f"--results {args.results} "
    cmd += f"--logname {args.logname} "
    cmd += f"--exec_mode {args.mode} "
    cmd += f"--dim {args.dim} "
    cmd += "--habana_loader " if args.habana_loader else ""
    if args.gpus:
        cmd += f"--gpus {args.gpus} "
    if args.hpus:
        cmd += f"--hpus {args.hpus} "
    cmd += f"--train_batches {args.train_batches} "
    cmd += f"--test_batches {args.test_batches} "
    cmd += f"--warmup {args.warmup} "
    cmd += "--amp " if args.amp else ""
    cmd += "--autocast " if args.autocast else ""
    cmd += f"--inference_mode {args.inference_mode} "
    cmd += f"--measurement_type {args.measurement_type} "
    cmd += "--profile " if args.profile else ""
    if args.mode == "train":
        cmd += f"--batch_size {args.batch_size} "
    else:
        cmd += f"--val_batch_size {args.batch_size} "
    call(cmd, shell=True)
