###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
import sys

from runtime.arguments import parse_args
from central.training_run_config import TrainingRunHWConfig


description = "This script is a distributed launcher for unet2d.py " \
              "and accepts the same arguments as orginal unet2d.py script.\n" \
              "In case argument --hvd_workers > 1 is passed, " \
              "it runs 'unet2d.py [ARGS] --use_horovod' via mpirun with generated HCL config.\n"
params = parse_args(description, distributed_launcher=True)

script_to_run = str(os.path.abspath(os.path.join(os.path.dirname(__file__), "unet2d.py")))
command_to_run = [sys.executable, script_to_run]
# Prepare mpi command prefix for multinode run
if params.hvd_workers > 1:
    hw_config = TrainingRunHWConfig(
        scaleout=True,
        num_workers_per_hls=params.hvd_workers,
        kubernetes_run=params.kubernetes_run,
        output_filename="demo_unet2d"
    )
    mpirun_cmd = hw_config.mpirun_cmd.split(" ")
    command_to_run = mpirun_cmd + command_to_run + ["--use_horovod"]
command_to_run += sys.argv[1:]
command_str = ' '.join(command_to_run)

print(f"Running: {command_str}", flush=True)
os.system(command_str)
