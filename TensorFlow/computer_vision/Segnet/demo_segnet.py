#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
###############################################################################
import subprocess
import sys
from keras_segmentation.cli_interface import get_arg_parser

parser = get_arg_parser(True)
args = parser.parse_args()

cmd = [f"{sys.executable}", "-m", "keras_segmentation"]
if args.num_workers_per_hls > 1:
    assert args.distributed
    assert args.train_engine != 'cpu'
    if args.train_engine == 'hpu':
        from central.training_run_config import TrainingRunHWConfig
        hw_config = TrainingRunHWConfig(
            scaleout=True,
            num_workers_per_hls=args.num_workers_per_hls,
            hls_type=args.hls_type,
            kubernetes_run=args.kubernetes_run,
            output_filename="demo_segnet"
        )
        cmd = hw_config.mpirun_cmd.split(" ") + cmd
    elif args.train_engine == 'gpu':
        cmd = ["horovodrun", "-np", f"{args.num_workers_per_hls}"] + cmd
cmd += sys.argv[1:]
print(f"Running: {' '.join(map(str, cmd))}")
subprocess.run(cmd)
