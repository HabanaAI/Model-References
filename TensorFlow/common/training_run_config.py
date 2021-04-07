###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
###############################################################################

"""Encapsulates the hardware configuration for scaleout training

This class encapsulates all the single-card and multi-card/multi-node training
run hardware constraints for Horovod.
"""
import os
from pathlib import Path
import subprocess
from TensorFlow.common.habana_model_runner_utils import HabanaEnvVariables, get_canonical_path, get_canonical_path_str, is_valid_multi_node_config, get_multi_node_config_nodes
from TensorFlow.common.multi_node_utils import run_cmd_as_subprocess, run_per_ip, generate_mpi_hostfile, print_file_contents
import TensorFlow.common.generate_hcl_config as generate_hcl_config

class TrainingRunHWConfig():
    def __init__(self, args):
        self.args = args
        self.use_horovod = False
        self.num_workers_per_hls = 1
        self.scaleout = False
        self.hls_ips = ''
        self.mpirun_cmd = ''
        if self.args.use_horovod is not None:
            self.use_horovod = True
            self.num_workers_per_hls = self.args.use_horovod
            self.scaleout = True

        self.num_workers_total = self.num_workers_per_hls

        print(f"use_horovod = {self.use_horovod}, num_workers_per_hls = {self.num_workers_per_hls}")

        self.run_config_env_variables = {}

        os.makedirs(get_canonical_path("$HOME/tmp/"), mode=0o777, exist_ok=True)
        if self.use_horovod:
            self.create_multi_worker_setup()
        else:
            self.create_single_worker_setup()

    def get_env_vars(self):
        return self.run_config_env_variables

    # This handles single-card run configuration
    def create_single_worker_setup(self):
        assert self.use_horovod == False, "Horovod is NOT set for single-worker run configuration"
        self.run_config_env_variables['NUM_WORKERS_PER_HLS'] = '1'

        print(f"{self.__class__.__name__} create_single_worker_setup(): self.mpirun_cmd = {self.mpirun_cmd}")
        print(f"{self.__class__.__name__} create_single_worker_setup(): MPIRUN_CMD = {os.environ.get('MPIRUN_CMD')}")

        if os.environ.get('MULTI_HLS_IPS'):
            print(f"Warning: In non-Horovod scenario, variable MULTI_HLS_IPS==\'{os.environ.get('MULTI_HLS_IPS')}\' has no effect.")

    # This handles Single-HLS run configuration
    def create_single_hls_setup(self, tmp_dir):
        #
        # Single-HLS Mode
        #
        print(f"self.num_workers_total = {self.num_workers_total}")
        hcl_config_path = generate_hcl_config.generate_hcl_config_r(str(tmp_dir), self.num_workers_per_hls)
        print(f"---------- Single-HLS ({self.num_workers_per_hls}-cards): HCL_CONFIG_PATH = {str(os.environ.get('HCL_CONFIG_PATH'))}")
        self.mpirun_cmd += f" -np {self.num_workers_per_hls}"
        return hcl_config_path

    # This handles Multi-HLS run configuration, driven by the MULTI_HLS_IPS environment variable
    def create_multi_hls_setup(self, tmp_dir):
        #
        # Multi-HLS Mode
        #
        if os.environ.get('MPI_TPC_INCLUDE'):
            mpi_tpc_include = os.environ.get('MPI_TPC_INCLUDE')
        else:
            mpi_tpc_include = "enp3s0"
        print(f"mpi_tpc_include = {mpi_tpc_include}")

        gen_hcl_path = Path(__file__).parent.joinpath('generate_hcl_config.py')
        # Create HCL config on each remote IP.
        run_per_ip(f"python3 {str(gen_hcl_path)} {str(tmp_dir)} {self.num_workers_per_hls} \"HLS1\"", ['MULTI_HLS_IPS', 'PYTHONPATH'], False)

        # Set HCL_CONFIG_PATH in this script, so it can be propagated in self.mpirun_cmd to remote IPs.
        hcl_config_path = generate_hcl_config.generate_hcl_config_r(str(tmp_dir), self.num_workers_per_hls)

        multi_hls_nodes = get_multi_node_config_nodes()
        self.num_workers_total = len(multi_hls_nodes) * self.num_workers_per_hls
        print(f"self.num_workers_total = {self.num_workers_total}")
        print(f"++++++++++ Multi-HLS ({self.num_workers_total}-cards): HCL_CONFIG_PATH = {str(os.environ.get('HCL_CONFIG_PATH'))}")

        mpi_hostfile_path = generate_mpi_hostfile(str(tmp_dir))
        assert mpi_hostfile_path != '', "Don\'t have a valid mpi_hostfile_path for MULTI_HLS_IPS scenario"
        print(f"mpi_hostfile_path = {mpi_hostfile_path} ->")
        print_file_contents(mpi_hostfile_path)

        self.mpirun_cmd += f" -np {self.num_workers_total}"
        if os.environ.get('DOCKER_SSHD_PORT'):
            portnum = os.environ.get('DOCKER_SSHD_PORT')
        else:
            portnum = 3022
        self.mpirun_cmd += f" --mca plm_rsh_args -p{portnum}"
        self.mpirun_cmd += f" --mca btl_tcp_if_include {mpi_tpc_include}"
        self.mpirun_cmd += f" -hostfile {mpi_hostfile_path}"
        # in case you deployed a docker image
        self.mpirun_cmd += " --prefix /usr/lib/habanalabs/openmpi/"
        # in case you invoked build_horovod manually
        #self.mpirun_cmd += " --prefix $HOME/.openmpi/"
        self.mpirun_cmd += " -x HCL_CONFIG_PATH"

        self.mpirun_cmd += " -x HABANA_USE_PREALLOC_BUFFER_FOR_ALLREDUCE"
        self.mpirun_cmd += " -x TF_ENABLE_BF16_CONVERSION"
        self.mpirun_cmd += " -x TF_ALLOW_CONTROL_EDGES_IN_HABANA_OPS"
        self.mpirun_cmd += " -x HBN_TF_REGISTER_DATASETOPS"
        self.mpirun_cmd += " -x HABANA_USE_STREAMS_FOR_HCL"
        self.mpirun_cmd += " -x TF_PRELIMINARY_CLUSTER_SIZE"
        self.mpirun_cmd += " -x HABANA_INITIAL_WORKSPACE_SIZE_MB"
        self.mpirun_cmd += " -x RUN_TPC_FUSER"
        self.mpirun_cmd += " -x TF_DISABLE_SCOPED_ALLOCATOR"

        self.mpirun_cmd += " -x LD_PRELOAD"
        self.mpirun_cmd += " -x TF_MODULES_RELEASE_BUILD"
        self.mpirun_cmd += " -x PYTHONPATH"
        self.mpirun_cmd += " -x GC_KERNEL_PATH"
        self.mpirun_cmd += " -x HABANA_LOGS"
        self.mpirun_cmd += " -x VIRTUAL_ENV"
        self.mpirun_cmd += " -x PATH"
        self.mpirun_cmd += " -x LD_LIBRARY_PATH"
        return hcl_config_path

    # This handles Single-HLS and Multi-HLS scaleout run configurations
    def create_multi_worker_setup(self):
        assert self.use_horovod and self.num_workers_per_hls > 1, "Horovod run requires at least 2 workers"
        self.run_config_env_variables['NUM_WORKERS_PER_HLS'] = f"{self.num_workers_per_hls}"
        tmp_dir = get_canonical_path("$HOME/tmp/")
        run_per_ip(f"mkdir -p {str(tmp_dir)}", ['MULTI_HLS_IPS', 'PYTHONPATH'], False)
        print(f"MULTI_HLS_IPS={os.environ.get('MULTI_HLS_IPS')}")

        # OpenMPI process bind resource type.
        mpi_map_by = "socket"

        # Get lscpu
        cmd = 'lscpu | grep \"CPU(s):\"'
        lscpu_output = []
        with subprocess.Popen(cmd, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as proc:
            lscpu_output = proc.stdout.read()
        # Determine the optimal value of resources per process of OpenMPI binding based on local lscpu.
        if mpi_map_by == "socket":
            mpi_map_by_pe = int(lscpu_output.split()[1])//self.num_workers_per_hls//2
        elif mpi_map_by == "slot":
            mpi_map_by_pe= int(lscpu_output.split()[1])//self.num_workers_per_hls
        else:
            raise Exception("mpi_map_by must be either 'socket' or 'slot'.")

        print(f"mpi_map_by_pe = {mpi_map_by_pe}")

        output_file_name = str(tmp_dir.joinpath("demo_bert_log/"))
        self.mpirun_cmd =  "mpirun"
        self.mpirun_cmd += " --allow-run-as-root"
        self.mpirun_cmd += f" --tag-output --merge-stderr-to-stdout --output-filename {output_file_name}"

        if mpi_map_by_pe > 0:
            self.mpirun_cmd += f" --bind-to core --map-by {mpi_map_by}:PE={mpi_map_by_pe}"

        hcl_config_path = ''

        if is_valid_multi_node_config():
            hcl_config_path = self.create_multi_hls_setup(tmp_dir)
        else:
            hcl_config_path = self.create_single_hls_setup(tmp_dir)

        print(f"HCL_CONFIG_PATH = {str(os.environ.get('HCL_CONFIG_PATH'))}")
        print(f"hcl_config_path = {hcl_config_path} ->")
        print_file_contents(hcl_config_path)

        os.environ['MPIRUN_CMD'] = self.mpirun_cmd
        print(f"{self.__class__.__name__} create_multi_worker_setup(): self.mpirun_cmd = {self.mpirun_cmd}")
        print(f"{self.__class__.__name__} create_multi_worker_setup(): MPIRUN_CMD = {os.environ.get('MPIRUN_CMD')}")
