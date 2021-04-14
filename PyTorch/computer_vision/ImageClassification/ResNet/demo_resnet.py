import sys
import re
import os
import io
import socket
import argparse
from central.habana_model_runner_utils import get_canonical_path_str
from PyTorch.common.training_runner import TrainingRunner

class ResNet50Argparser(argparse.ArgumentParser):
  """docstring for ResNet50Argparser"""
  def __init__(self):
    super(ResNet50Argparser, self).__init__()
    self.add_argument('-p', '--data-path', metavar='<data_path>', default="/root/software/data/pytorch/imagenet/ILSVRC2012/",
                      help='Path to the training dataset')
    self.add_argument('-m', '--model', required=False, metavar='<model>', type=str, default="resnet50",
                      help='The model variant. Default: resnet50')
    self.add_argument('-d', '--device', required=False, metavar='<device>', type=str, default="habana",
                      help='The device for training. Default: habana')
    self.add_argument('-b', '--batch-size', required=False, metavar='<batch_size>', type=int, default=128,
                      help='Batch size. Default: 128')
    self.add_argument('--mode', required=False, metavar='<mode>', type=str, default='lazy', choices=[
                      'lazy', 'eager'], help='Different modes avaialble. Possible values: lazy, eager. Default: lazy')
    self.add_argument('--data-type', required=False, metavar='<data type>', default='fp32', choices=['fp32', 'bf16'],
                      help='Data Type. Possible values: fp32, bf16. Default: fp32')
    self.add_argument('-e', '--epochs', required=False, metavar='<epochs>', type=int, default=1,
                      help='Number of epochs. Default: 1')
    self.add_argument('-j', '--workers', metavar='<workers>', type=int, default=0, help='Training device size')
    self.add_argument('--print-freq', required=False, metavar='<print_freq>', type=int, default=1,
                      help='Frequency of printing. Default: 1')
    self.add_argument('-w', '--world-size', metavar='<world_size>', type=int, default=8, help='Training device size')
    self.add_argument('--num-train-steps', required=False, metavar='<num_train_steps>', type=int, default=None,
                      help='Number of training steps. Default: 100 ')
    self.add_argument('--num-eval-steps', required=False, metavar='<num_eval_steps>', type=int, default=None,
                      help='Number of evaluation steps. Default: 30')
    self.add_argument('--custom-lr-values', required=False, metavar='<custom_lr_values>', default=None,
                      help='custom lr values list')
    self.add_argument('--custom-lr-milestones', required=False, metavar='<custom_lr_milestones>', default=None,
                      help='custom lr milestones list')
    self.add_argument('--dist', action='store_true', help='Distribute training')
    self.add_argument('--output-dir', default='.', metavar='<output_dir>', help='path where to save')

def build_command(args, path, train_script):
    """ Constructing the training command """

    init_command = f"{path + '/' + str(train_script)}"
    command = (
        f"{init_command}"
        f" --data-path={args.data_path}"
        f" --model={args.model}"
        f" --device={args.device}"
        f" --batch-size={args.batch_size}"
        f" --epochs={args.epochs}"
        f" --workers={args.workers}"
        f" --print-freq={args.print_freq}"
        f" --num-train-steps={args.num_train_steps}"
        f" --num-eval-steps={args.num_eval_steps}"
        f" --output-dir={args.output_dir}"
    )

    if args.mode == "lazy":
        command += " --run-lazy-mode"
    if args.data_type == 'bf16':
        command += " --hmp --hmp-bf16 " + path + '/ops_bf16_Resnet.txt' + ' --hmp-fp32 ' + path + '/ops_fp32_Resnet.txt'

    #Check each value in list args.custom_lr_values and args.custom_lr_milestones are valid
    lr_val_flag = 0
    lr_mil_flag = 0
    perform_multi_val_flag = 0
    perform_multi_mil_flag = 0

    if args.custom_lr_values is not None:
        perform_multi_val_flag = 1

    if args.custom_lr_milestones is not None:
        perform_multi_mil_flag = 1

    if perform_multi_val_flag ==1:
        for lr_val in args.custom_lr_values.split(","):
            if (bool(re.match(r'^([0-9]+)?[.][0-9]+$', lr_val))):
                lr_val_flag =1
            else:
                exit ("Invalid --custom-lr-values values")
        space_lr_val_list = args.custom_lr_values.replace(","," ")

    if perform_multi_mil_flag ==1:
        for lr_mil in args.custom_lr_milestones.split(","):
            if (bool(re.match(r'^[0-9]+$', lr_mil))):
                lr_mil_flag =1
            else:
                exit ("Invalid --custom-lr-milestones values")
        space_lr_mil_list = args.custom_lr_milestones.replace(","," ")

    if (lr_val_flag == 1) and (lr_mil_flag == 1):
        command += ( f" --custom-lr-values {space_lr_val_list}"
                     f" --custom-lr-milestones {space_lr_mil_list}" )
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
  env_vars['PT_HABANA_LOG_MOD_MASK'] = 'FFFF'
  env_vars['PT_HABANA_LOG_TYPE_MASK'] = '1'
  env_vars['RUN_TPC_FUSER'] = '0'
  env_vars['HCL_CPU_AFFINITY'] = '1'
  env_vars['PT_ENABLE_SYNC_OUTPUT_HOST'] = 'false'

  return env_vars

def main():

    args = ResNet50Argparser().parse_args()
    print (args)
    # Get the path for the resnet50 train file.
    resnet50_path = get_canonical_path_str("./")
    train_script = "train.py"
    # Check if the world_size > 1 for distributed training in hls
    check_world_size(args)
    # Set environment variables
    env_vars = set_env(args, resnet50_path)
    # Build the command line
    command = build_command(args, resnet50_path, train_script)

    if args.world_size == 1:
        training_runner = TrainingRunner([command],env_vars,args.world_size,False)
    else:
        mpi_runner = True
        training_runner = TrainingRunner([command],env_vars,args.world_size,args.dist,False,mpi_runner)

    training_runner.run()

if __name__=="__main__":
  main()
