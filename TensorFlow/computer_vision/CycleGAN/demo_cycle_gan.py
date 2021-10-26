###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
import subprocess
import sys

from central.training_run_config import TrainingRunHWConfig
from arguments import CycleGANArgParser


if __name__ == '__main__':
    script_to_run = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "cycle_gan.py"))
    parser = CycleGANArgParser(is_demo=True)
    args = parser.parse_args()
    if args.hvd_workers > 1:
        hw_config = TrainingRunHWConfig(
            scaleout=True,
            num_workers_per_hls=args.hvd_workers,
            hls_type=args.hls_type,
            kubernetes_run=args.kubernetes_run,
            output_filename="cycle_gan"
        )
        cmd = hw_config.mpirun_cmd.split(" ") + \
            [sys.executable, str(script_to_run), "--use_horovod"]
    else:
        cmd = [sys.executable, str(script_to_run)]
    cmd += sys.argv[1:]
    subprocess.run(cmd).check_returncode()
