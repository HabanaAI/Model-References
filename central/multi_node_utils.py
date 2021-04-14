###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

"""Utilities for multi-card and scaleout training"""

import os
import sys
import subprocess
import socket
from central.habana_model_runner_utils import get_canonical_path, is_valid_multi_node_config, get_multi_node_config_nodes

def run_cmd_as_subprocess(cmd=str, use_devnull=False):
    print(cmd)
    sys.stdout.flush()
    sys.stderr.flush()
    if use_devnull:
        with subprocess.Popen(cmd, shell=True, executable='/bin/bash', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as proc:
            proc.wait()
    else:
        with subprocess.Popen(cmd, shell=True, executable='/bin/bash') as proc:
            proc.wait()

#--------------------------------------------------------------------------------
# For scaleout, depends on MULTI_HLS_IPS environment variable being set to contain
# ",' separated list of host IPs.
# If MULTI_HLS_IPS is not set, the command "cmd" will be run on the local host.
# Make sure that the command "cmd" being run on the remote host does not depend on
# any other environment variables being available on the remote host besides the ones
# in the "env_vars_for_mpi" list that this function will export to the remote IPs.
#--------------------------------------------------------------------------------

def run_per_ip(cmd, env_vars_for_mpi=None, use_devnull=False):
    if os.environ.get('OMPI_COMM_WORLD_SIZE') is not None:
        raise RuntimeError("Function run_per_ip is not meant to be run from within an OpenMPI context. It is intended to invoke mpirun by itelf.")

    if not is_valid_multi_node_config():
        print("************************* Single-HLS mode *************************")
        run_cmd_as_subprocess(cmd, use_devnull)
    else:
        if os.environ.get('DOCKER_SSHD_PORT'):
            portnum = os.environ.get('DOCKER_SSHD_PORT')
        else:
            portnum = 3022
        scmd = f"mpirun --allow-run-as-root --mca plm_rsh_args -p{portnum} --tag-output --merge-stderr-to-stdout --prefix /usr/lib/habanalabs/openmpi/ -H {os.environ.get('MULTI_HLS_IPS')} "
        if env_vars_for_mpi is not None:
            for env_var in env_vars_for_mpi:
                scmd += f"-x {env_var} "
        scmd += cmd
        print(f"{socket.gethostname()}: In MULTI NODE run_per_ip(): scmd = {scmd}")
        run_cmd_as_subprocess(scmd, use_devnull)


# Generate the MPI hostfile
def generate_mpi_hostfile(file_path, devices_per_hls=8):
    mpi_hostfile_path = ''
    if is_valid_multi_node_config():
        multi_hls_nodes = get_multi_node_config_nodes()
        print("Generating MPI hostfile...")
        file_name = "hostfile"
        os.makedirs(get_canonical_path(file_path), mode=0o777, exist_ok=True)
        mpi_hostfile_path = get_canonical_path(file_path).joinpath(file_name)
        if os.path.exists(mpi_hostfile_path):
            #os.remove(mpi_hostfile_path)
            cmd = f"rm -f {str(mpi_hostfile_path)}"
            run_cmd_as_subprocess(cmd)
        print(f"Path: {mpi_hostfile_path}")
        out_fid = open(mpi_hostfile_path, 'a')
        config_str = ''
        for node in multi_hls_nodes:
            config_str += f"{node} slots={devices_per_hls}\n"
        print(f"MPI hostfile: \n{config_str}")
        out_fid.write(config_str)
        out_fid.close()
    return mpi_hostfile_path

def print_file_contents(file_name):
    with open(file_name, 'r') as fl:
        for line in fl:
            print(line, end='')
