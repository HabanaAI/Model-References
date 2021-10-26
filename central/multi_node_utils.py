###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

"""Utilities for multi-card and scaleout training"""

import os
import socket
import subprocess
import sys
from functools import lru_cache

from central.habana_model_runner_utils import (get_canonical_path,
                                               get_multi_node_config_nodes,
                                               is_valid_multi_node_config)


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

# -------------------------------------------------------------------------------
# For scaleout, depends on MULTI_HLS_IPS environment variable being set to contain
# ",' separated list of host IPs.
# If MULTI_HLS_IPS is not set, the command "cmd" will be run on the local host.
# Make sure that the command "cmd" being run on the remote host does not depend on
# any other environment variables being available on the remote host besides the ones
# in the "env_vars_for_mpi" list that this function will export to the remote IPs.
# -------------------------------------------------------------------------------


def run_per_ip(cmd, env_vars_for_mpi=None, use_devnull=False, kubernetes_run=False):
    if kubernetes_run:
        print("************************* Kubernetes mode *************************")
        run_cmd_as_subprocess(cmd, use_devnull)
        return

    if os.environ.get('OMPI_COMM_WORLD_SIZE') is not None:
        raise RuntimeError(
            "Function run_per_ip is not meant to be run from within an OpenMPI context. It is intended to invoke mpirun by itelf.")

    if not is_valid_multi_node_config():
        print("************************* Single-HLS mode *************************")
        run_cmd_as_subprocess(cmd, use_devnull)
    else:
        if os.environ.get('DOCKER_SSHD_PORT'):
            portnum = os.environ.get('DOCKER_SSHD_PORT')
        else:
            portnum = 3022
        scmd = f"mpirun --allow-run-as-root --mca plm_rsh_args -p{portnum} --tag-output --merge-stderr-to-stdout --prefix {os.environ.get('MPI_ROOT')} -H {os.environ.get('MULTI_HLS_IPS')} "
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


def _is_relevant_env_var(env_var: str):
    """ Given an environment variable name, determines whether is it "relevant" for the child processes spawned by OpenMPI.
        OpenMPI passes the local environment only to the local child processes.
    """
    RELEVANT_ENV_VAR_INFIXES = [
        "PATH", "LD_",      # System-specific: PATH, PYTHONPATH, LD_LIBRARY_PATH, LD_PRELOAD
        "TF_",              # TensorFlow-specific, e.g.: TF_BF16_CONVERSION
        "TPC_",             # TPC-specific, e.g.: RUN_TPC_FUSER
        "GC_",              # GC-specific, e.g.: GC_KERNEL_PATH
        "HABANA", "HBN",    # Other Habana-specific, e.g.: HABANA_INITIAL_WORKSPACE_SIZE_MB
        "HOROVOD",          # Horovod-specific, e.g.: HOROVOD_LOG_LEVEL
        "SYN",              # Synapse-specific
        "HCL",              # HCL-specific, e.g.: HCL_CONFIG_PATH
        "HCCL", "NCCL",     # HCCL-specific: HCCL_SOCKET_IFNAME, HABANA_NCCL_COMM_API
        "LOG_LEVEL",        # Logger-specific, e.g.: LOG_LEVEL_HCL, LOG_LEVEL_SYN_API
    ]

    OTHER_RELEVANT_ENV_VARS = [
        "VIRTUAL_ENV",
        "ENABLE_CONSOLE",
    ]

    ENV_VARS_DEPRECATIONS = {
        "TF_ENABLE_BF16_CONVERSION": "Superceeded by TF_BF16_CONVERSION.",
        "HABANA_USE_PREALLOC_BUFFER_FOR_ALLREDUCE": "'same address' optimization for collective operations is no longer supported.",
        "HABANA_USE_STREAMS_FOR_HCL": "HCL streams are always used. Setting to 0 enforces blocking synchronization after collective operations (debug feature).",
    }

    if env_var in ENV_VARS_DEPRECATIONS:
        print(
            f"warninig: Environment variable '{env_var}' is deprecated: {ENV_VARS_DEPRECATIONS[env_var]}")

    if env_var in OTHER_RELEVANT_ENV_VARS:
        return True
    for infix in RELEVANT_ENV_VAR_INFIXES:
        if infix in env_var:
            return True
    return False


@lru_cache()
def get_relevant_env_vars():
    """ Retrieves the list of those environment variables, which should be passed to the child processes spawned by OpenMPI.
    """
    return [env_var for env_var in os.environ if _is_relevant_env_var(env_var)]
