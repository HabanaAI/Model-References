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

class SwinTransformerArgparser(argparse.ArgumentParser):
  """docstring for SwinTransformerArgparser"""
  def __init__(self):
    super(SwinTransformerArgparser, self).__init__()
    self.add_argument('-p', '--data-path', metavar='<data_path>', default="/root/software/data/pytorch/imagenet/ILSVRC2012/",
                      help='Path to the training dataset')
    self.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    self.add_argument('--batch-size', type=int, default=128, help="batch size for single HPU")
    self.add_argument('--mode', required=True, metavar='<mode>', type=str, default='lazy', choices=[
                      'lazy', 'eager'], help='Different modes avaialble. Possible values: lazy, eager. Default: lazy')
    self.add_argument('--data-type', required=True, metavar='<data type>', default='fp32', choices=['fp32', 'bf16'],
                      help='Data Type. Possible values: fp32, bf16. Default: fp32')
    self.add_argument('-w', '--world-size', metavar='<world_size>', type=int, default=8, help='Training device size')
    self.add_argument('-e', '--epochs', required=False, metavar='<epochs>', type=int, help='Number of epochs. Default: 1')
    self.add_argument('--train-steps', required=False, metavar='<num_train_steps>', type=int, default=None,
                      help='Number of training steps.')
    self.add_argument('--test-steps', required=False, metavar='<num_eval_steps>', type=int, default=None,
                      help='Number of evaluation steps.')
    self.add_argument('--output', required=False, type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    self.add_argument('--tag', required=False, help='tag of experiment',)
    self.add_argument('--resume', required=False, metavar='<checkpt_path>', default='', help='resume from checkpoint')
    self.add_argument('--process-per-node', default=0, type=int, metavar='N',
                      help='Number of process per node')
    self.add_argument('--dist', action='store_true', help='Distribute training')

def build_command(args, path, train_script):
    """ Constructing the training command """

    init_command = f"{path + '/' + str(train_script)}"
    command = (
        f"{init_command}"
        f" --data-path {args.data_path}"
        f" --batch-size {args.batch_size}"
        f" --mode {args.mode}"
        f" --cfg {os.path.join(path, args.cfg)}"
    )

    if args.data_type == 'bf16':
        command += " --hmp --hmp-bf16 " + path + '/ops_bf16_swin_transformer.txt' + ' --hmp-fp32 ' + path + '/ops_fp32_swin_transformer.txt'
    if args.test_steps is not None:
        command += f" --test-steps {args.test_steps}"
    if args.train_steps is not None:
        command += f" --train-steps {args.train_steps}"
    if args.tag:
        command += f" --tag {args.tag}"
    if args.resume:
        command += f" --resume {args.resume}"
    if args.epochs is not None:
        command += f" --epochs {args.epochs}"
    if args.output:
        command += f" --output {args.output}"

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
    if args.dist and world_size == 1:
        args.world_size = 8
    return args.world_size

#Setup the environment variable
def set_env(args, cur_path):
  env_vars = {}

  return env_vars

def main():

    args = SwinTransformerArgparser().parse_args()
    print (args)
    swin_transformer_path = os.path.abspath(os.path.dirname(__file__))
    train_script = "main.py"
    # Check if the world_size > 1 for distributed training in hls
    check_world_size(args)
    # Set environment variables
    env_vars = set_env(args, swin_transformer_path)
    # Build the command line
    command = build_command(args, swin_transformer_path, train_script)

    if args.world_size == 1:
        training_runner = TrainingRunner([command],env_vars,args.world_size,False, map_by='slot')
    else:
        mpi_runner = True
        training_runner = TrainingRunner([command],env_vars,args.world_size,args.dist,False,mpi_runner,
                                         map_by='slot', multi_hls=args.multi_hls)

    ret_code = training_runner.run()
    return ret_code

if __name__=="__main__":
    ret = main()
    sys.exit(ret)
