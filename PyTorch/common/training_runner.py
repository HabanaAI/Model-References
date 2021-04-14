###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
import subprocess
import sys
from central.habana_model_runner_utils import HabanaEnvVariables, print_env_info
import central.generate_hcl_config as generate_hcl_config


class TrainingRunner(object):
    """
    Set up training HW configurations and run training commands
    """

    def __init__(self, command_list=[], model_env_vars={}, world_size=1, dist=False, use_mpi=False, mpi_run=False):
        self.__commands = command_list
        self.__model_env_vars = model_env_vars
        self.__world_size = world_size
        self.__use_mpi = use_mpi
        self.__dist = dist
        self.__mpi_run = mpi_run
        self.__config_env_vars = {}
        self.__interpreter = "python3 "

        print(f"dist training = {self.__dist}, world_size = {self.__world_size}")

        if self.__world_size > 1:
            if self.__use_mpi:
                self.create_multi_hls_setup()
            elif mpi_run:
                self.create_single_hls_setup_mpirun()
            else:
                self.create_single_hls_setup()
        else:
            self.create_single_card_setup()

    def setup_config_env(self):
        print(f"self.world_size = {self.__world_size}")

        # HCL_CONFIG_PATH env var is set in generate_hcl_config_r()
        generate_hcl_config.generate_hcl_config_r('/tmp', self.__world_size)
        print(
            f"Single-HLS ({self.__world_size}): HCL_CONFIG_PATH = {str(os.environ.get('HCL_CONFIG_PATH'))}")
        # set up BACKEND env var and WORLD_SIZE env var
        self.__config_env_vars['BACKEND'] = 'hcl'
        self.__config_env_vars['WORLD_SIZE'] = self.__world_size
        self.__config_env_vars['MAX_WAIT_ATTEMPTS'] = '30'

    def setup_config_env_mpirun(self):
        cmd1 = "lscpu 2>/dev/null | awk '/Socket\(s\)/  { print $2 }'"
        cmd2 = "lscpu 2>/dev/null | awk '/Core\(s\) per socket/  { print $4 }'"
        with subprocess.Popen(cmd1, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as proc:
            lscpu_output1 = proc.stdout.read()
        with subprocess.Popen(cmd2, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as proc:
            lscpu_output2 = proc.stdout.read()
        sockets= int(lscpu_output1)
        corespsocket= int(lscpu_output2)
        peval = (sockets * corespsocket) // self.__world_size
        if peval:
            map_cmd = f"--map-by slot:PE={peval}"
        return map_cmd

    def create_single_card_setup(self):
        self.setup_config_env()
        self.__interpreter = "python3 "

    def create_single_hls_setup_mpirun(self):
        """ set up single hls multi-cards configurations for mpirun"""
        self.setup_config_env()
        mpi_cmd = self.setup_config_env_mpirun()
        self.__interpreter = f"mpirun -n {self.__world_size} --bind-to core {mpi_cmd} --rank-by core --report-bindings --allow-run-as-root python3 "

    def create_single_hls_setup(self):
        """ set up single hls multi-cards configurations"""

        self.setup_config_env()
        self.__interpreter = f"python3 -um torch.distributed.launch --nproc_per_node={self.__world_size} "

    def create_multi_hls_setup(self, tmp_dir):
        # placeholder for multi-hls setup
        pass

    def set_model_env_vars(self, model_env_vars):
        self.__model_env_vars = model_env_vars

    def run(self):
        try:
            print('HW config env vars: ', self.__config_env_vars)
            print('Model specific env vars: ', self.__model_env_vars)
            with HabanaEnvVariables(env_vars_to_set=self.__config_env_vars), \
                    HabanaEnvVariables(env_vars_to_set=self.__model_env_vars):
                for command in self.__commands:
                    command = self.__interpreter + command
                    print_env_info(command, self.__config_env_vars)
                    print_env_info(command, self.__model_env_vars)
                    print(f"{self.__class__.__name__} run(): command = {command}")
                    sys.stdout.flush()
                    sys.stderr.flush()
                    with subprocess.Popen(command, shell=True, executable='/bin/bash') as proc:
                        proc.wait()
                    sys.stdout.flush()
                    sys.stderr.flush()
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} run()") from exc
