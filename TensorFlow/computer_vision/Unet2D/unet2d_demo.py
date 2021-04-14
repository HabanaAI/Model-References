###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
import pathlib
import subprocess
import sys

from runtime.arguments import PARSER, parse_args

def get_mpi_pe(num_workers):
    lscpu = subprocess.check_output("lscpu", shell=True).strip().decode().split('\n')
    for line in lscpu:
        if line.startswith("CPU(s):"):
            cpus = int(line.split()[1])
            return cpus//num_workers//2

def create_mpi_cmd(num_workers):
    mpi_cmd_parts = []
    mpi_cmd_parts.append("mpirun")
    mpi_cmd_parts.append("--allow-run-as-root")
    mpi_pe = get_mpi_pe(num_workers)
    mpi_cmd_parts.append("--bind-to core --map-by socket:PE={}".format(mpi_pe))
    mpi_cmd_parts.append("--np {}".format(num_workers))

    return ' '.join(mpi_cmd_parts)

def find_common_path():
    current_path = pathlib.Path().absolute()
    root_dir = pathlib.Path()
    for p in current_path.parents:
        for d in p.iterdir():
            if d.match('TensorFlow'):
                root_dir = p
                break
    return root_dir.joinpath('TensorFlow', 'common').resolve()

def generate_hcl_config(num_workers):
    os.environ['TF_DISABLE_SCOPED_ALLOCATOR'] = 'true'
    os.environ['HABANA_USE_STREAMS_FOR_HCL'] = 'true'
    os.environ['HABANA_USE_PREALLOC_BUFFER_FOR_ALLREDUCE'] = 'false'
    common_path = find_common_path()
    common_sh = common_path.joinpath('common.sh')
    script_path = os.path.dirname(os.path.realpath(__file__))
    subprocess.Popen(['bash', '-c', 'source {}; generate_hcl_config {} {}'.format(common_sh, script_path, num_workers)])
    config_file_name = "config_{}.json".format(num_workers)
    os.environ['HCL_CONFIG_PATH'] = "{}/{}".format(script_path, config_file_name)


if __name__ == '__main__':
    params = parse_args(PARSER.parse_args())
    command_to_run = ["python3 runtime/run.py"]
    # Prepare mpi command prefix for multinode run
    num_workers = params.hvd_workers
    if num_workers > 1:
        generate_hcl_config(num_workers)
        mpi_command = create_mpi_cmd(num_workers)
        command_to_run.insert(0, mpi_command)
        command_to_run.append("--use_horovod")
    command_to_run += sys.argv[1:]

    command_str = " ".join(command_to_run)

    os.environ['PYTHONPATH'] = f"{os.environ['PYTHONPATH']}:{os.path.dirname(os.path.realpath(__file__))}"
    os.system(command_str)
