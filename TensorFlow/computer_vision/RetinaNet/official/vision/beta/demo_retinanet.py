###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import argparse
import os
import subprocess

from central.training_run_config import TrainingRunHWConfig

def main():
    parser = argparse.ArgumentParser(add_help=False, usage=argparse.SUPPRESS)
    parser.add_argument("--num_workers_per_hls", default=1, type=int)
    parser.add_argument("--kubernetes_run", default=False, type=bool)
    args, unknown_args = parser.parse_known_args()
    script_to_run = str(os.path.abspath(os.path.join(os.path.dirname(__file__), "train.py")))
    if args.num_workers_per_hls > 1:
        hw_config = TrainingRunHWConfig(scaleout=True, num_workers_per_hls=args.num_workers_per_hls,
                                            kubernetes_run=args.kubernetes_run, output_filename="retinanet_log")
        cmd = list(hw_config.mpirun_cmd.split(" ")) + ["python3", script_to_run]
    else:
        cmd = ["python3", script_to_run]

    cmd.extend(unknown_args)
    cmd_str = ' '.join(map(str, cmd))
    print(f"Running: {cmd_str}", flush=True)
    with subprocess.Popen(cmd_str, shell=True, executable='/bin/bash') as proc:
        proc.wait()

if __name__ == "__main__":
    main()