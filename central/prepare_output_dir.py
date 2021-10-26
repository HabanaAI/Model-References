###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
import sys
import socket
#import shutil
from central.habana_model_runner_utils import get_canonical_path, get_canonical_path_str
from central.multi_node_utils import run_cmd_as_subprocess

def prepare_output_dir_r(output_dir):
    host_name = socket.gethostname()
    try:
        od_path = get_canonical_path(output_dir)
        if os.path.isdir(od_path):
            print(f"{host_name}: *** Cleaning existing {str(od_path)}...\n\n")
            #shutil.rmtree(od_path)
            cmd = f"rm -rf {get_canonical_path_str(od_path)}"
            run_cmd_as_subprocess(cmd)
        os.makedirs(od_path, mode=0o777, exist_ok=True)
    except Exception as exc:
        raise Exception(f"{host_name}: Error in {__file__} prepare_output_dir_r({output_dir})") from exc

if __name__ == "__main__":
    host_name = socket.gethostname()
    print(f"{host_name}: In {sys.argv[0]}")
    print(f"{host_name}: called with arguments: \"{sys.argv[1]}\"")
    output_dir = sys.argv[1]
    print(f"{host_name}: MULTI_HLS_IPS = {os.environ.get('MULTI_HLS_IPS')}")
    prepare_output_dir_r(output_dir)
