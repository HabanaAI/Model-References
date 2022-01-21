# Copyright (c) 2021, Habana Labs Ltd.  All rights reserved.

import sys
import os
import argparse
model_garden_path = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../../"))
os.environ['PYTHONPATH'] = os.getenv('PYTHONPATH', default='') + ":" +  model_garden_path
sys.path.append(model_garden_path)
from central.multi_node_utils import is_valid_multi_node_config, get_multi_node_config_nodes
from PyTorch.common.training_runner import TrainingRunner

class Argparser(argparse.ArgumentParser):
  def __init__(self):
    super(Argparser, self).__init__()
    self.add_argument('--amp', dest='is_amp', action='store_true', help='enable GPU amp mode')
    self.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='batch size (default: 256)')
    self.add_argument('--data-path', default='/root/software/data/pytorch/imagenet/ILSVRC2012/', metavar='DIR', help='Path to the training dataset')
    self.add_argument('--data-type', metavar='<data type>', default='fp32', choices=['fp32', 'bf16'],
                      help='Data Type. fp32, bf16. Default: fp32')
    self.add_argument('--dataset-type', help='Dataset type',
                    type=str, choices=['Imagenet'],
                    default='Imagenet')
    self.add_argument('--device', help='cpu,hpu,gpu', type=str, choices=['cpu', 'hpu', 'gpu'], default='hpu')
    self.add_argument('--distributed', action='store_true', help='Distributed training')
    self.add_argument('--dl-worker-type', default='MP', type=lambda x: x.upper(),
                    choices = ["MP", "HABANA"], help='select multiprocessing or habana accelerated')
    self.add_argument('--enable-lazy', action='store_true',
                    help='whether to enable Lazy mode, if it is not set, your code will run in Eager mode')
    self.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run.')
    self.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
    self.add_argument('--hmp-bf16', default='ops_bf16_googlenet.txt', help='path to bf16 ops list in hmp O1 mode')
    self.add_argument('--hmp-fp32', default='ops_fp32_googlenet.txt', help='path to fp32 ops list in hmp O1 mode')
    self.add_argument('--hmp-opt-level', default='O1', help='choose optimization level for hmp')
    self.add_argument('--hmp-verbose', action='store_true', help='enable verbose mode for hmp')
    self.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
    self.add_argument('--lr-step-size', default=30, type=int, help='decrease lr every step-size epochs')
    self.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    self.add_argument('--mode', metavar='<mode>', type=str, default='eager', choices=['lazy', 'eager'],
                    help='Different modes available. lazy, eager. Default: eager')
    self.add_argument('--model', default='googlenet', help='model')
    self.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    self.add_argument('--no-aux-logits', action='store_true', help='disable aux logits in GoogleNet')
    self.add_argument('--num-train-steps', type=int, default=sys.maxsize,
                    help='Number of training steps to run')
    self.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
    self.add_argument('-p', '--print-interval', default=1, type=int,
                    metavar='N', help='print frequency (default: 1)')
    self.add_argument('--process-per-node', default=8, type=int, metavar='N',
                      help='Number of process per node')
    self.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: None)')
    self.add_argument('--save-checkpoint', default=10, const=5, type=int, nargs='?', help='Save checkpoint after every <N> epochs')
    self.add_argument('--seed', type=int, default=None, help='seed for initializing training')
    self.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    self.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
    self.add_argument('--world-size', type=int, default=1, help='Training device size')
    self.add_argument('-j', '--workers', metavar='<workers>', type=int, default=10, help='number of dataloader workers')

def build_command(args, path, train_script):
    """ Constructing the training command """
    init_command = f"{path + '/' + str(train_script)}"
    command = (
        f"{init_command}"
        f" --batch-size={args.batch_size}"
        f" --data-path={args.data_path}"
        f" --dataset-type={args.dataset_type}"
        f" --device={args.device}"
        f" --dl-worker-type={args.dl_worker_type}"
        f" --epochs={args.epochs}"
        f" --learning-rate={args.lr}"
        f" --lr-step-size={args.lr_step_size}"
        f" --lr-gamma={args.lr_gamma}"
        f" --momentum={args.momentum}"
        f" --print-interval={args.print_interval}"
        f" --save-checkpoint={args.save_checkpoint}"
        f" --start-epoch={args.start_epoch}"
        f" --weight-decay={args.weight_decay}"
        f" --workers={args.workers}"
    )

    if args.is_amp:
        command += " --amp"
    if args.data_type == 'bf16':
        command += " --hmp --hmp-bf16 " + path + "/" + args.hmp_bf16 + ' --hmp-fp32 ' + path + "/" + args.hmp_fp32
        command += f" --hmp-opt-level={args.hmp_opt_level}"
        if args.hmp_verbose:
            command += " --hmp-verbose"
    if args.evaluate:
        command += " --evaluate"
    if args.mode == "lazy":
        command += " --enable-lazy"
    command += f" --num-train-steps={args.num_train_steps}"
    if args.no_aux_logits:
        command += " --no-aux-logits"
    if args.pretrained:
        command += " --pretrained"
    if args.resume is not None:
        command += " --resume=" + args.resume
    if args.seed is not None:
        command += f" --seed={args.seed}"
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
    # print (args)
    # Get the path for the train file.
    path = os.path.abspath(os.path.dirname(__file__))
    train_script = "main.py"
    # Check if the world_size > 1 for distributed training in hls
    check_world_size(args)
    # Set environment variables
    env_vars = set_env(args, path)
    # Build the command line
    command = build_command(args, path, train_script)

    if args.world_size == 1:
        training_runner = TrainingRunner([command], env_vars, args.world_size, False, map_by='slot')
    else:
        mpi_runner = True
        training_runner = TrainingRunner([command], env_vars, args.world_size, args.distributed, False, mpi_runner,
                                         map_by='slot', multi_hls=args.multi_hls)

    training_runner.run()

if __name__=="__main__":
  main()
