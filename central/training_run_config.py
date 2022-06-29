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
import subprocess
from pathlib import Path

from central.multi_node_utils import (generate_mpi_hostfile,
                                      get_mpi_tcp_include,
                                      get_relevant_env_vars,
                                      print_file_contents, run_per_ip,
                                      is_valid_multi_node_config,
                                      get_multi_node_config_nodes)

ALLOWED_MPI_MAP_BY = ["socket", "slot", "none", ""]
DEFAULT_MPI_MAP_BY = ALLOWED_MPI_MAP_BY[0]


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

        os.makedirs(os.path.expandvars(os.path.expanduser("$HOME/tmp/")),
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
        tmp_dir = Path(os.path.expandvars(os.path.expanduser("$HOME/tmp/")))
        run_per_ip(f"mkdir -p {str(tmp_dir)}",
                   ['MULTI_HLS_IPS', 'PYTHONPATH'], False, self.kubernetes_run)

        print(f"MULTI_HLS_IPS={os.environ.get('MULTI_HLS_IPS')}")

        output_file_name = str(tmp_dir.joinpath(self.output_filename))
        self.mpirun_cmd = self.create_mpi_cmdline(output_file_name)

        if is_valid_multi_node_config():
            self.create_multi_hls_setup(tmp_dir)
        else:
            self.create_single_hls_setup(tmp_dir)

        print(f"{self.__class__.__name__} create_multi_worker_setup(): self.mpirun_cmd = {self.mpirun_cmd}")

    def create_mpi_cmdline(self, output_file_name):
        # OpenMPI process bind resource type.
        mpi_map_by = os.environ.get("MPI_MAP_BY", DEFAULT_MPI_MAP_BY).lower()
        assert mpi_map_by in ALLOWED_MPI_MAP_BY, f"MPI_MAP_BY must be one of {ALLOWED_MPI_MAP_BY}, but is: {mpi_map_by}"

        mpi_cmd = "mpirun"
        mpi_cmd += " --allow-run-as-root"
        mpi_cmd += f" --tag-output --merge-stderr-to-stdout --output-filename {output_file_name}"

        if mpi_map_by not in ["none", ""]:
            # Get lscpu
            cmd = 'lscpu | grep \"CPU(s):\"'
            lscpu_output = []
            with subprocess.Popen(cmd, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env={"LD_PRELOAD": ""}) as proc:
                lscpu_output = proc.stdout.read()
            # Determine the optimal value of resources per process of OpenMPI binding based on local lscpu.
            if mpi_map_by == "socket":
                mpi_map_by_pe = int(lscpu_output.split()[
                                    1])//self.num_workers_per_hls//2
            else:
                assert mpi_map_by == "slot"
                mpi_map_by_pe = int(lscpu_output.split()[
                                    1])//self.num_workers_per_hls

            print(f"mpi_map_by_pe = {mpi_map_by_pe}")

            if mpi_map_by_pe > 0:
                mpi_cmd += f" --bind-to core --map-by {mpi_map_by}:PE={mpi_map_by_pe}"

        return mpi_cmd

    # This handles Single-HLS run configuration
    def create_single_hls_setup(self, tmp_dir):
        #
        # Single-HLS Mode
        #
        print(f"self.num_workers_total = {self.num_workers_total}")
        print(
            f"---------- Single-HLS ({self.num_workers_per_hls}-cards)")
        self.mpirun_cmd += f" -np {self.num_workers_per_hls}"

    # This handles Multi-HLS run configuration, driven by the MULTI_HLS_IPS environment variable

    def create_multi_hls_setup(self, tmp_dir):
        #
        # Multi-HLS Mode
        #

        multi_hls_nodes = get_multi_node_config_nodes()
        self.num_workers_total = len(
            multi_hls_nodes) * self.num_workers_per_hls
        print(f"self.num_workers_total = {self.num_workers_total}")
        print(
            f"++++++++++ Multi-HLS ({self.num_workers_total}-cards)")

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
        self.mpirun_cmd += f" --mca btl_tcp_if_include {get_mpi_tcp_include()}"
        self.mpirun_cmd += f" -hostfile {mpi_hostfile_path}"
        self.mpirun_cmd += " --prefix $MPI_ROOT"

        for env_var in get_relevant_env_vars():
            self.mpirun_cmd += f" -x {env_var}={shlex.quote(os.environ[env_var])}"
            # Note that =value above in not necessary, but provides a vital information when presented this way in the log file.

