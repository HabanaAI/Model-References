###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
from pathlib import Path
import sys
import socket
import subprocess
from central.habana_model_runner_utils import get_canonical_path

def download_dataset_r(dataset_path):
    host_name = socket.gethostname()
    try:
        ds_path = get_canonical_path(dataset_path)
        if not os.path.isdir(ds_path):
            print(f"{host_name}: *** Downloading dataset...\n\n")
            os.makedirs(ds_path, exist_ok=True)
            download_script = Path(__file__).parent.joinpath("download_glue_data.py")
            sys.stdout.flush()
            sys.stderr.flush()
            with subprocess.Popen(f"{sys.executable} {str(download_script)} --data_dir {str(ds_path)} --tasks MRPC", shell=True, executable='/bin/bash') as proc:
                proc.wait()
    except Exception as exc:
        raise Exception(f"{host_name}: Error in {__file__} download_dataset_r({dataset_path})") from exc

if __name__ == "__main__":
    host_name = socket.gethostname()
    print(f"{host_name}: In {sys.argv[0]}")
    print(f"{host_name}: called with arguments: \"{sys.argv[1]}\"")
    dataset_path = sys.argv[1]
    print(f"{host_name}: MULTI_HLS_IPS = {os.environ.get('MULTI_HLS_IPS')}")
    download_dataset_r(dataset_path)
