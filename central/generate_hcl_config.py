###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

"""Generate an HCL Config File

Generate hcl config file and set HCL_CONFIG_PATH
Globals:
  MULTI_HLS_IPS
Arguments:
  Path to save the file
  Number of devices per HLS
  HLS type, default HLS1
"""
import os
import sys
import socket
from central.habana_model_runner_utils import get_canonical_path, is_valid_multi_node_config, get_multi_node_config_nodes
from central.multi_node_utils import run_cmd_as_subprocess

def generate_hcl_config_r(file_path, devices_per_hls, hls_type="HLS1"):
    host_name = socket.gethostname()
    try:
        print(f"{host_name}: Generating HCL Config...")
        num_workers_total = devices_per_hls
        if is_valid_multi_node_config():
            num_nodes = get_multi_node_config_nodes()
            num_workers_total *= len(num_nodes)
        file_name = f"hcl_config_{num_workers_total}.json"
        os.makedirs(get_canonical_path(file_path), mode=0o777, exist_ok=True)
        hcl_config_path = get_canonical_path(file_path).joinpath(file_name)
        if os.path.exists(hcl_config_path):
            #os.remove(hcl_config_path)
            cmd = f"rm -f {str(hcl_config_path)}"
            run_cmd_as_subprocess(cmd)
        print(f"{host_name}: Path: {str(hcl_config_path)}")

        with open(hcl_config_path, 'a') as out_fid:
            config_str = "{\n\t\"HCL_PORT\": 5332,\n\t\"HCL_TYPE\": \"" + hls_type + "\",\n"
            if not is_valid_multi_node_config():
                config_str += "\t\"HCL_COUNT\": " + str(devices_per_hls) + "\n"
            else:
                config_str += "\t\"HCL_RANKS\": [\n"
                multi_hls_nodes = get_multi_node_config_nodes()
                for node_num, node in enumerate(multi_hls_nodes):
                    for i in range(devices_per_hls):
                        if i == devices_per_hls-1 and node_num == len(multi_hls_nodes)-1:
                            config_str += f"\t\t\"{node}\"\n"
                        else:
                            config_str += f"\t\t\"{node}\",\n"
                config_str += "\t]\n"
            config_str += "}\n"
            print(f"{host_name}: HCL Config: \n{config_str}")
            out_fid.write(config_str)

        os.environ['HCL_CONFIG_PATH'] = str(hcl_config_path)
        return hcl_config_path
    except Exception as exc:
        raise Exception(f"{host_name}: Error in {__file__} generate_hcl_config_r({file_path}, {devices_per_hls}, {hls_type})") from exc

if __name__ == "__main__":
    host_name = socket.gethostname()
    print(f"{host_name}: In {sys.argv[0]}")
    print(f"{host_name}: called with arguments: \"{sys.argv[1]} {sys.argv[2]} {sys.argv[3]}\"")
    file_path = sys.argv[1]
    devices_per_hls = int(sys.argv[2])
    hls_type = sys.argv[3]
    print(f"{host_name}: MULTI_HLS_IPS = {os.environ.get('MULTI_HLS_IPS')}")
    print(f"{host_name}: HCL_CONFIG_PATH = {os.environ.get('HCL_CONFIG_PATH')}")
    generate_hcl_config_r(file_path, devices_per_hls, hls_type)
