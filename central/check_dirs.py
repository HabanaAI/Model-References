###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
import sys
import socket
from central.habana_model_runner_utils import get_canonical_path

def check_dirs_r(largs):
    host_name = socket.gethostname()
    try:
        for fl in largs:
            print(f"Checking \'{fl}\'")
            if os.path.exists(fl) == False:
                raise Exception(f"{host_name}: Error: {str(fl)} file or directory missing. Please mount correctly")
    except Exception as exc:
        raise RuntimeError(f"{host_name}: Error in {__file__} check_dirs_r({largs})") from exc

if __name__ == "__main__":
    host_name = socket.gethostname()
    print(f"{host_name}: In {sys.argv[0]}")
    print(f"{host_name}: called with arguments: \"{str(sys.argv[1:])}\"")
    dir_list = sys.argv[1:]
    print(f"{host_name}: MULTI_HLS_IPS = {os.environ.get('MULTI_HLS_IPS')}")
    check_dirs_r(dir_list)
