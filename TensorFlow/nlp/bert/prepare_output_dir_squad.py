###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
from pathlib import Path
import sys
import socket
#import shutil
from central.multi_node_utils import run_cmd_as_subprocess

def prepare_output_dir_squad_r(output_dir, batch_size, max_seq_len):
    host_name = socket.gethostname()
    try:
        route0 = 0
        route1 = 0
        if os.path.isdir(output_dir):
            cfg_path = output_dir + ("/") + (f"last_config_{batch_size}_{max_seq_len}")
            if os.path.exists(cfg_path):
                route0 = 1
            else:
                route1 = 1
        else:
            os.makedirs(output_dir, exist_ok=True)

        if route0 == 1:
            print(f"{host_name}: *** Cleaning temp directory content in {output_dir}... (except *.tf_record files) \n\n")
            with os.scandir(output_dir) as it:
                for entry in it:
                    if entry.is_file():
                        if Path(entry.name).suffix != '.tf_record':
                            #os.remove(Path(entry.path))
                            cmd = f"rm -f {entry.path}"
                            run_cmd_as_subprocess(cmd)
                    elif entry.is_dir():
                        #shutil.rmtree(get_canonical_path(entry.path))
                        cmd = f"rm -rf {entry.path}"
                        run_cmd_as_subprocess(cmd)

        if route1 == 1:
            print(f"{host_name}: *** Cleaning temp directory content in {output_dir}... \n\n")
            # This throws an exception when remote hosts share the same file system paths
            #shutil.rmtree(od_path)
            cmd = f"rm -rf {output_dir}"
            run_cmd_as_subprocess(cmd)
            os.makedirs(output_dir, exist_ok=True)

        os.open(output_dir + ("/") + (f"last_config_{batch_size}_{max_seq_len}"), os.O_CREAT, mode=0o644)
    except Exception as exc:
        raise Exception(f"{host_name}: Error in {__file__} prepare_output_dir_squad_r({output_dir}, {batch_size}, {max_seq_len})") from exc

if __name__ == "__main__":
    host_name = socket.gethostname()
    print(f"{host_name}: In {sys.argv[0]}")
    print(f"{host_name}: called with arguments: \"{sys.argv[1]} {sys.argv[2]} {sys.argv[3]}\"")
    output_dir = sys.argv[1]
    batch_size = int(sys.argv[2])
    max_seq_len = int(sys.argv[3])
    print(f"{host_name}: MULTI_HLS_IPS = {os.environ.get('MULTI_HLS_IPS')}")
    prepare_output_dir_squad_r(output_dir, batch_size, max_seq_len)
