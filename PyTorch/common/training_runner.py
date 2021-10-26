###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
import subprocess
import sys
from pathlib import Path
from central.habana_model_runner_utils import HabanaEnvVariables, print_env_info
import central.generate_hcl_config as generate_hcl_config
from central.multi_node_utils import run_cmd_as_subprocess
from central.multi_node_utils import run_per_ip
from central.habana_model_runner_utils import is_valid_multi_node_config
import socket


class TrainingRunner(object):
    """
    Set up training HW configurations and run training commands
    """

    def __init__(self, command_list=[], model_env_vars={}, world_size=1,
                 dist=False, use_mpi=False, mpi_run=False,
                 use_env=False, map_by='socket', hls_type="HLS1",
                 multi_hls=False):
        self.__commands = command_list
        self.__model_env_vars = model_env_vars
        self.__world_size = world_size
        self.__map_by = map_by
        self.__multi_hls = multi_hls
        self.__dist = dist
        self.__mpi_run = mpi_run
        self.__config_env_vars = {}
        self.__use_env = use_env
        self.__interpreter = f"{sys.executable} "
        # Change the hls_type to actual type
        # Based on the machine type passed in constructor
        self.___hls_type = hls_type

        print(f"dist training = {self.__dist}, world_size = {self.__world_size}")

        if self.__world_size > 1:
            if self.__multi_hls:
                self.create_multi_hls_setup()
            elif mpi_run:
                self.create_single_hls_setup_mpirun()
            else:
                self.create_single_hls_setup()
        else:
            self.create_single_card_setup()


    def setup_config_env(self):
        print(f"self.world_size = {self.__world_size}")

        tmp_dir = '/tmp'
        __worker_per_node = self.__world_size
        if self.__multi_hls:
            __cnt = len(os.getenv("MULTI_HLS_IPS").split(','))
            gen_hcl_path = Path(__file__).parent.parent.parent.joinpath('central/generate_hcl_config.py')
            # Create HCL config on each remote IP.
            if os.getenv("HCL_CONFIG_PATH"):
                del os.environ['HCL_CONFIG_PATH']
            __worker_per_node = self.__world_size // __cnt
            run_per_ip((f"{sys.executable} {str(gen_hcl_path)} {tmp_dir} "
                             f"{__worker_per_node} {self.___hls_type}"),
                            ['MULTI_HLS_IPS', 'PYTHONPATH'], False)

        # HCL_CONFIG_PATH env var is set in generate_hcl_config_r()
        generate_hcl_config.generate_hcl_config_r(f'{tmp_dir}', __worker_per_node,
                                                  hls_type=self.___hls_type)
        print(
            f"Single-HLS ({self.__world_size}): HCL_CONFIG_PATH = {str(os.environ.get('HCL_CONFIG_PATH'))}")

    def get_peval(self):
        """ get_peval """
        cmd1 = "lscpu 2>/dev/null | awk '/Socket\(s\)/  { print $2 }'"
        cmd2 = "lscpu 2>/dev/null | awk '/Core\(s\) per socket/  { print $4 }'"
        with subprocess.Popen(cmd1, shell=True, executable='/bin/bash', stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT) as proc:
            lscpu_output1 = proc.stdout.read()
        with subprocess.Popen(cmd2, shell=True, executable='/bin/bash', stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT) as proc:
            lscpu_output2 = proc.stdout.read()
        sockets = int(lscpu_output1)
        corespsocket = int(lscpu_output2)
        if corespsocket == 1:  # running inside VM?
            peval = 1
            print(f"Warning !! cores per socket is {corespsocket}")
            print((f"Warning !! running with default PE value {peval},"
                  "it can impact performance"))
            return peval, sockets, corespsocket
        else:
            peval = (sockets * corespsocket) // self.__world_size
            return peval, sockets, corespsocket

    def setup_config_env_mpirun(self):
        peval, _, _ = self.get_peval()
        if peval:
            map_cmd = f"--map-by {self.__map_by}:PE={peval}"
        return map_cmd

    def create_single_card_setup(self):
        self.setup_config_env()
        self.__interpreter = f"{sys.executable} "

    def create_single_hls_setup_mpirun(self):
        """ set up single hls multi-cards configurations for mpirun"""
        self.setup_config_env()
        mpi_cmd = self.setup_config_env_mpirun()
        self.__interpreter = f"mpirun -n {self.__world_size} --bind-to core {mpi_cmd} --rank-by core --report-bindings --allow-run-as-root {sys.executable} "

    def create_single_hls_setup(self):
        """ set up single hls multi-cards configurations"""

        use_env_param = "--use_env" if self.__use_env else ""

        self.setup_config_env()
        self.__interpreter = f"{sys.executable} -um torch.distributed.launch --nproc_per_node={self.__world_size} {use_env_param} "

    def create_multi_hls_setup(self):
        """ set up multi hls configurations for mpirun"""
        self.setup_config_env()
        envlist = [
            "HCL_CONFIG_PATH",
            "MAX_WAIT_ATTEMPTS",
            "PT_USE_HCL_SYNC",
            "LOG_LEVEL_ALL",
            "LOG_LEVEL_SYN_API",
            "LD_LIBRARY_PATH",
            "PYTORCH_MODULES_ROOT_PATH",
            "BUILD_ROOT_LATEST",
            "PYTHONPATH",
            "HABANA_LOGS",
            "GC_KERNEL_PATH"
        ]
        assert os.getenv("MULTI_HLS_IPS"), "environment variable MULTI_HLS_IPS is not set"
        __hls_list = str(os.getenv("MULTI_HLS_IPS", "")).split(',')
        __num_hls = len(__hls_list)
        __world_size = self.__world_size
        __sshport = os.getenv('DOCKER_SSHD_PORT', 3022)
        __master_port = os.getenv('MASTER_PORT', 12345)
        __master_addr = os.getenv('MASTER_ADDR', __hls_list[0])
        __per_node_processes = int(__world_size/len(__hls_list))
        __pe_val, __sockets, __corespsocket = self.get_peval()
        __pe_val = __num_hls * __pe_val
        __cores_per_node = __per_node_processes * __pe_val
        __process_per_socket = __per_node_processes // __sockets
        envset_cmds = []
        for __env in envlist:
            __val = os.getenv(__env, None)
            if __val:
                __arg = f"-x {__env}=\"{__val}\""
                envset_cmds.append(__arg)
        envset_cmd = " ".join(envset_cmds)
        hls_nodes = []
        for hls in __hls_list:
            hls_node = hls.split('-')[0]
            hls_nodes.append(f"{hls_node}:{__cores_per_node}")
        hls_info = ",".join(hls_nodes)
        __master_addr = __hls_list[0]
        network = __master_addr.split('.')
        network[-1] = '0'
        network_id = '.'.join(network) + "/24"
        cmd = "mpirun --allow-run-as-root "
        cmd += f" {envset_cmd} "
        cmd += f"--prefix {os.getenv('MPI_ROOT', '/usr/lib/habanalabs/openmpi')} "
        cmd += f"--mca btl_tcp_if_include {network_id} "
        cmd += f"-x MASTER_ADDR={__master_addr} "
        cmd += f"-x MASTER_PORT={__master_port} "
        cmd += f"--mca plm_rsh_args \"-p {__sshport}\" --bind-to core "
        cmd += f"-H {hls_info} -n {__world_size} "
        if __process_per_socket > 0:
            cmd += f"--map-by ppr:{__process_per_socket}:socket:PE={__pe_val} "
        else:
            cmd += f"--map-by ppr:{__per_node_processes}:node:PE={__pe_val} "
        cmd += f"--rank-by core --report-bindings {sys.executable} "
        self.__interpreter = cmd

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
