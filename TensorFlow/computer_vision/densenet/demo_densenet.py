###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
import subprocess
import sys

from central.training_run_config import TrainingRunHWConfig
from utils.arguments import DenseNetArgumentParser


if __name__ == '__main__':
    script_to_run = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "train.py"))
    parser = DenseNetArgumentParser(
        description=(
            "demo_densenet.py is a distributed launcher for train.py. "
            "It accepts the same arguments as train.py. In case of num_workers_per_hls > 1, "
            "it runs 'train.py [ARGS]' via mpirun with generated HCL config."))
    # special arguments for multi-node/device training
    parser.add_argument('--num_workers_per_hls', type=int, default=1,
                        help="number of workers per HLS")
    parser.add_argument("--hls_type", default="HLS1", type=str,
                        help="type of HLS")
    parser.add_argument("--kubernetes_run", action='store_true',
                        help="whether it's kubernetes run")
    args = parser.parse_args()

    cmd = []
    if args.num_workers_per_hls > 1:
        hw_config = TrainingRunHWConfig(
            scaleout=True,
            num_workers_per_hls=args.num_workers_per_hls,
            hls_type=args.hls_type,
            kubernetes_run=args.kubernetes_run,
            output_filename="demo_densenet_log"
        )
        cmd += hw_config.mpirun_cmd.split(" ")

    cmd += [sys.executable, str(script_to_run)]
    cmd += sys.argv[1:]
    subprocess.run(cmd).check_returncode()
