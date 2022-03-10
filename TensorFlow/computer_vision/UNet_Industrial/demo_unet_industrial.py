###############################################################################
# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
import subprocess
import sys

from utils.cmdline_helper import parse_cmdline
from central.training_run_config import TrainingRunHWConfig

if __name__ == '__main__':
    script_to_run = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "main.py"))
    FLAGS = parse_cmdline()
    cmd = []
    if FLAGS.num_workers_per_hls > 1:
        hw_config = TrainingRunHWConfig(
            scaleout=True,
            num_workers_per_hls=FLAGS.num_workers_per_hls,
            hls_type=FLAGS.hls_type,
            kubernetes_run=FLAGS.kubernetes_run,
            output_filename="demo_unet_industrial_log"
        )
        cmd += hw_config.mpirun_cmd.split(" ")
    cmd += [sys.executable, str(script_to_run)]
    cmd += sys.argv[1:]
    print("cmd = ",cmd)
    subprocess.run(cmd).check_returncode()

