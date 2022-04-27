# Copyright (c) 2021, Habana Labs Ltd.  All rights reserved.

import sys
import re
import os
import io
import socket
import argparse
model_garden_path = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../.."))
os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + ":" +  model_garden_path
sys.path.append(model_garden_path)
from central.multi_node_utils import is_valid_multi_node_config, get_multi_node_config_nodes
from PyTorch.common.training_runner import TrainingRunner

class Argparser(argparse.ArgumentParser):
  def __init__(self):
    super(Argparser, self).__init__()
    self.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='batch size (default: 256)')
    self.add_argument('--use_lazy_mode', action='store_true',
                    help='whether to enable Lazy mode, if it is not set, your code will run in Eager mode')
    self.add_argument('--data-type', metavar='<data type>', default='fp32', choices=['fp32', 'bf16'],
                      help='Data Type. fp32, bf16. Default: fp32')
    self.add_argument('--no-cuda', action='store_true', help='disables CUDA training')
    self.add_argument("--hmp", action="store_true", help="Enable HMP")
    self.add_argument('--hmp-bf16', default='ops_bf16_mnist.txt', help='path to bf16 ops list in hmp O1 mode')
    self.add_argument('--hmp-fp32', default='ops_fp32_mnist.txt', help='path to fp32 ops list in hmp O1 mode')
    self.add_argument('--hmp-opt-level', default='O1', help='choose optimization level for hmp')
    self.add_argument('--hmp-verbose', action='store_true', help='enable verbose mode for hmp')
    self.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
    self.add_argument('--world_size', type=int, default=1, help='Training device size')
    self.add_argument('--distributed', action='store_true', help='Distribute training')
    self.add_argument('--hpu', action='store_true', help='Gaudi training')
    self.add_argument('--process-per-node', default=0, type=int, metavar='N',
                      help='Number of process per node')
    self.add_argument('--lr', default=1.0, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
    self.add_argument('--gamma', default=0.7, type=float, help='decrease lr by a factor of lr-gamma')
    self.add_argument('--data_type', type=str, choices=["bf16", "fp32"], default='bf16',help="Specify data type to be either bf16 or fp32.")


def build_command(args, path, train_script):
    """ Constructing the training command """
    init_command = f"{path + '/' + str(train_script)}"
    command = (
        f"{init_command}"
        f" --batch-size={args.batch_size}"
        f" --epochs={args.epochs}"
        f" --lr={args.lr}"
        f" --gamma={args.gamma}"
    )
    if args.hpu:
        command += f" --hpu"
    if args.use_lazy_mode:
        command += f" --use_lazy_mode"
    if args.data_type == 'bf16':
        command += f" --hmp"
        command += f" --hmp-bf16={args.hmp_bf16}"
        command += f" --hmp-fp32={args.hmp_fp32}"


    if is_valid_multi_node_config() and args.world_size > 0:
        if args.process_per_node == 0:
            nodes = get_multi_node_config_nodes()
            args.process_per_node = args.world_size // len(nodes)
        command += f" --process-per-node={args.process_per_node}"
        args.multi_hls = True
    else:
        args.multi_hls = False

    return command

#Check if distributed training is true
def check_world_size(args):
    world_size = args.world_size
    if args.distributed and world_size == 1:
        args.world_size = 8
    return args.world_size

#Setup the environment variable
def set_env(args, cur_path):
    env_vars = {}
    return env_vars

def main():
    args = Argparser().parse_args()
    print(args)
    # Get the path for the train file.
    path = os.path.abspath(os.path.dirname(__file__))
    train_script = "mnist.py"
    # Check if the world_size > 1 for distributed training in hls
    check_world_size(args)
    # Set environment variables
    env_vars = set_env(args, path)
    # Build the command line
    command = build_command(args, path, train_script)

    if args.world_size == 1:
        training_runner = TrainingRunner([command],env_vars,args.world_size,False, map_by='slot')
    else:
        mpi_runner = True
        training_runner = TrainingRunner([command],env_vars,args.world_size,args.distributed,False,mpi_runner,
                                         map_by='slot', multi_hls=args.multi_hls)

    training_runner.run()

if __name__=="__main__":
  main()
