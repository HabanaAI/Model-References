# ******************************************************************************
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
# ******************************************************************************
import os
import sys
import subprocess
import argparse

from central.training_run_config import TrainingRunHWConfig


if __name__ == "__main__":
    description = "demo_efficientdet.py is a distributed launcher for main.py \
                  It accepts the same arguments as main.py. \
                  it runs 'main.py [ARGS] --use_horovod' via mpirun with generated HCL config."
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', action='store_true', help=description)
    parser.add_argument('--use_horovod', metavar='<num_workers_per_hls>', required=False, type=int,
                        help='Use Horovod for training. num_workers_per_hls parameter is optional and defaults to 8')
    parser.add_argument("--kubernetes_run", default=False, type=bool, help="Kubernetes run")

    args, unknown = parser.parse_known_args()
    script_to_run = os.path.abspath(os.path.join(os.path.dirname(__file__), "main.py"))

    if args.help:
        parser.print_help()
        print("main scrpit flags: ")
        print("=" * 30)

    num_workers_per_hls = args.use_horovod if args.use_horovod is not None else 1
    use_horovod = args.use_horovod is not None
    hw_config = TrainingRunHWConfig(
        scaleout=use_horovod,
        num_workers_per_hls=num_workers_per_hls,
        kubernetes_run=args.kubernetes_run,
        output_filename="demo_efficientdet"
    )
    cmd = hw_config.mpirun_cmd.split(" ") if args.use_horovod else []
    cmd += [sys.executable, str(script_to_run)]
    cmd += sys.argv[1:]

    print(f"Running: {' '.join(map(str, cmd))}")
    subprocess.run(cmd)