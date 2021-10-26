###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

"""Encapsulates the hardware configuration for scaleout training

This class encapsulates all the single-card and multi-card/multi-node training
run hardware constraints for scaleout.
"""
import os
import sys
import shlex
import socket
import subprocess
from pathlib import Path

import central.generate_hcl_config as generate_hcl_config
from central.habana_model_runner_utils import (HabanaEnvVariables,
                                               get_canonical_path,
                                               get_canonical_path_str,
                                               get_multi_node_config_nodes,
                                               is_valid_multi_node_config)
from central.multi_node_utils import (generate_mpi_hostfile,
                                      get_relevant_env_vars,
                                      print_file_contents,
                                      run_cmd_as_subprocess, run_per_ip)


class TrainingRunHWConfig():
    def __init__(self, scaleout=False, num_workers_per_hls=1, hls_type="HLS1", kubernetes_run=False, output_filename="training_run_log"):
        self.scaleout = scaleout
        self.num_workers_per_hls = num_workers_per_hls
        self.hls_type = hls_type
        self.kubernetes_run = kubernetes_run
        self.output_filename = output_filename
        self.hls_ips = ''
        self.mpirun_cmd = ''

        self.num_workers_total = self.num_workers_per_hls

        print(f"scaleout = {self.scaleout}, num_workers_per_hls = {self.num_workers_per_hls}, hls_type = {self.hls_type}, kubernetes_run={self.kubernetes_run}")

        self.run_config_env_variables = {}

        os.makedirs(get_canonical_path("$HOME/tmp/"),
                    mode=0o777, exist_ok=True)
        if self.scaleout:
            self.create_multi_worker_setup()
        else:
            self.create_single_worker_setup()

    def get_env_vars(self):
        return self.run_config_env_variables

    # This handles single-card run configuration
    def create_single_worker_setup(self):
        assert self.scaleout == False, "Scaleout is set for single-worker run configuration"
        if self.kubernetes_run:
            return

        print(f"{self.__class__.__name__} create_single_worker_setup(): self.mpirun_cmd = {self.mpirun_cmd}")

        if os.environ.get('MULTI_HLS_IPS'):
            print(
                f"Warning: In non-scaleout scenario, variable MULTI_HLS_IPS==\'{os.environ.get('MULTI_HLS_IPS')}\' has no effect.")

    # This handles Single-HLS and Multi-HLS scaleout run configurations
    def create_multi_worker_setup(self):
        if not self.kubernetes_run:
            assert self.scaleout and self.num_workers_per_hls > 1, "Scaleout run requires at least 2 workers"
        tmp_dir = get_canonical_path("$HOME/tmp/")
        run_per_ip(f"mkdir -p {str(tmp_dir)}",
                   ['MULTI_HLS_IPS', 'PYTHONPATH'], False, self.kubernetes_run)
        hcl_config_path = ''

        if self.kubernetes_run:
            hcl_config_path = get_canonical_path(
                os.environ.get('HCL_CONFIG_PATH'))
            print(
                f"HCL_CONFIG_PATH = {str(os.environ.get('HCL_CONFIG_PATH'))}")
            print(f"hcl_config_path = {hcl_config_path} ->")
            print_file_contents(hcl_config_path)
            return

        print(f"MULTI_HLS_IPS={os.environ.get('MULTI_HLS_IPS')}")

        output_file_name = str(tmp_dir.joinpath(self.output_filename))
        self.mpirun_cmd = self.create_mpi_cmdline(output_file_name)

        if is_valid_multi_node_config():
            hcl_config_path = self.create_multi_hls_setup(tmp_dir)
        else:
            hcl_config_path = self.create_single_hls_setup(tmp_dir)

        print(f"HCL_CONFIG_PATH = {str(os.environ.get('HCL_CONFIG_PATH'))}")
        print(f"hcl_config_path = {hcl_config_path} ->")
        print_file_contents(hcl_config_path)

        print(f"{self.__class__.__name__} create_multi_worker_setup(): self.mpirun_cmd = {self.mpirun_cmd}")

    def create_mpi_cmdline(self, output_file_name):
        # OpenMPI process bind resource type.
        mpi_map_by = "socket"

        # Get lscpu
        cmd = 'lscpu | grep \"CPU(s):\"'
        lscpu_output = []
        with subprocess.Popen(cmd, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env={"LD_PRELOAD": ""}) as proc:
            lscpu_output = proc.stdout.read()
        # Determine the optimal value of resources per process of OpenMPI binding based on local lscpu.
        if mpi_map_by == "socket":
            mpi_map_by_pe = int(lscpu_output.split()[
                                1])//self.num_workers_per_hls//2
        elif mpi_map_by == "slot":
            mpi_map_by_pe = int(lscpu_output.split()[
                                1])//self.num_workers_per_hls
        else:
            raise Exception("mpi_map_by must be either 'socket' or 'slot'.")

        print(f"mpi_map_by_pe = {mpi_map_by_pe}")
        mpi_cmd = "mpirun"
        mpi_cmd += " --allow-run-as-root"
        mpi_cmd += f" --tag-output --merge-stderr-to-stdout --output-filename {output_file_name}"

        if mpi_map_by_pe > 0:
            mpi_cmd += f" --bind-to core --map-by {mpi_map_by}:PE={mpi_map_by_pe}"
        return mpi_cmd

    # This handles Single-HLS run configuration
    def create_single_hls_setup(self, tmp_dir):
        #
        # Single-HLS Mode
        #
        print(f"self.num_workers_total = {self.num_workers_total}")
        hcl_config_path = generate_hcl_config.generate_hcl_config_r(
            str(tmp_dir), self.num_workers_per_hls, self.hls_type)
        print(
            f"---------- Single-HLS ({self.num_workers_per_hls}-cards): HCL_CONFIG_PATH = {str(os.environ.get('HCL_CONFIG_PATH'))}")
        self.mpirun_cmd += f" -np {self.num_workers_per_hls}"
        return hcl_config_path

    def deduce_ip_addr(self):
        """ Deduces the IP address of the host running this process used for connecting to the other machines.
        """
        # The first method: deduce using a default network interface.
        try:
            dummy_ip_endpoint = ("4.3.2.1", 80)
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(dummy_ip_endpoint)
            return s.getsockname()[0]
        except Exception:
            pass

        # The second method: Call 'hostname -I' and take the first IP address (unreliable in all cases).
        return subprocess.check_output(["hostname", "-I"], encoding="ascii").split(" ")[0].strip()

    # This handles Multi-HLS run configuration, driven by the MULTI_HLS_IPS environment variable
    def create_multi_hls_setup(self, tmp_dir):
        #
        # Multi-HLS Mode
        #
        if 'MPI_TCP_INCLUDE' in os.environ:
            mpi_tcp_include = os.environ['MPI_TCP_INCLUDE']
            print(
                f"Setting mpi_tcp_include to '{mpi_tcp_include}' (provided with MPI_TCP_INCLUDE env var)")
        else:
            mpi_tcp_include = self.deduce_ip_addr() + "/24"
            print(
                f"Setting mpi_tcp_include to '{mpi_tcp_include}' (deduced automatically)")

        gen_hcl_path = Path(__file__).parent.joinpath('generate_hcl_config.py')
        # Create HCL config on each remote IP.
        run_per_ip(f"{sys.executable} {str(gen_hcl_path)} {str(tmp_dir)} {self.num_workers_per_hls} {self.hls_type}", [
                   'MULTI_HLS_IPS', 'PYTHONPATH', 'HOROVOD_HIERARCHICAL_ALLREDUCE'], False)

        # Set HCL_CONFIG_PATH in this script, so it can be propagated in self.mpirun_cmd to remote IPs.
        hcl_config_path = generate_hcl_config.generate_hcl_config_r(
            str(tmp_dir), self.num_workers_per_hls, self.hls_type)

        multi_hls_nodes = get_multi_node_config_nodes()
        self.num_workers_total = len(
            multi_hls_nodes) * self.num_workers_per_hls
        print(f"self.num_workers_total = {self.num_workers_total}")
        print(
            f"++++++++++ Multi-HLS ({self.num_workers_total}-cards): HCL_CONFIG_PATH = {str(os.environ.get('HCL_CONFIG_PATH'))}")

        mpi_hostfile_path = generate_mpi_hostfile(
            str(tmp_dir), self.num_workers_per_hls)
        assert mpi_hostfile_path != '', "Don\'t have a valid mpi_hostfile_path for MULTI_HLS_IPS scenario"
        print(f"mpi_hostfile_path = {mpi_hostfile_path} ->")
        print_file_contents(mpi_hostfile_path)

        self.mpirun_cmd += f" -np {self.num_workers_total}"
        if os.environ.get('DOCKER_SSHD_PORT'):
            portnum = os.environ.get('DOCKER_SSHD_PORT')
        else:
            portnum = 3022
        self.mpirun_cmd += f" --mca plm_rsh_args -p{portnum}"
        self.mpirun_cmd += f" --mca btl_tcp_if_include {mpi_tcp_include}"
        self.mpirun_cmd += f" -hostfile {mpi_hostfile_path}"
        self.mpirun_cmd += " --prefix $MPI_ROOT"

        for env_var in get_relevant_env_vars():
            self.mpirun_cmd += f" -x {env_var}={shlex.quote(os.environ[env_var])}"
            # Note that =value above in not necessary, but provides a vital information when presented this way in the log file.

        return hcl_config_path
