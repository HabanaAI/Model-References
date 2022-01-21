#!/usr/bin/env python3

###############################################################################
# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
import subprocess
import sys

from central.training_run_config import TrainingRunHWConfig
from TensorFlow.common.common import setup_jemalloc
from TensorFlow.computer_vision.SSD_ResNet34.argparser import SSDArgParser


parser = SSDArgParser(is_demo=True)
args = parser.parse_args()
script_to_run = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "ssd.py"))

setup_jemalloc()     # libjemalloc for better allocations

if args.hvd_workers > 1:
    hw_config = TrainingRunHWConfig(
        scaleout=True,
        num_workers_per_hls=args.num_workers_per_hls,
        kubernetes_run=args.kubernetes_run,
        output_filename="demo_ssd"
    )
    cmd = hw_config.mpirun_cmd.split(" ") + \
        [sys.executable, str(script_to_run), "--use_horovod"]
else:
    cmd = [sys.executable, str(script_to_run)]
cmd += sys.argv[1:]
cmd_str = ' '.join(map(str, cmd))
print(f"Running: {cmd_str}", flush=True)
with subprocess.Popen(cmd_str, shell=True, executable='/bin/bash') as proc:
    proc.wait()
