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

class MobileNetArgparser(argparse.ArgumentParser):
  """docstring for MobileNetArgparser"""
  def __init__(self):
    super(MobileNetArgparser, self).__init__()
    self.add_argument('-p', '--data-path', metavar='<data_path>', default="/root/software/data/pytorch/imagenet/ILSVRC2012/",
                      help='Path to the training dataset')
    self.add_argument('-m', '--model', required=False, metavar='<model>', type=str, default="mobilenet_v2",
                      help='The model variant. Default: mobilenet_v2')
    self.add_argument('-d', '--device', required=False, metavar='<device>', type=str, default="hpu",
                      help='The device for training. Default: hpu')
    self.add_argument('-b', '--batch-size', required=False, metavar='<batch_size>', type=int, default=128,
                      help='Batch size. Default: 128')
    self.add_argument('--mode', required=False, metavar='<mode>', type=str, default='lazy', choices=[
                      'lazy', 'eager'], help='Different modes avaialble. Possible values: lazy, eager. Default: lazy')
    self.add_argument('--data-type', required=False, metavar='<data type>', default='fp32', choices=['fp32', 'bf16'],
                      help='Data Type. Possible values: fp32, bf16. Default: fp32')
    self.add_argument('-e', '--epochs', required=False, metavar='<epochs>', type=int, default=1,
                      help='Number of epochs. Default: 1')
    self.add_argument('--dl-worker-type', default='MP', type=lambda x: x.upper(),
                      choices = ["MP", "HABANA"], help='select multiprocessing or habana accelerated')
    self.add_argument('--dl-time-exclude', default=True, type=lambda x: x.lower() == 'true', help='Set to False to include data load time')
    self.add_argument('-j', '--workers', metavar='<workers>', type=int, default=10, help='number of dataloader workers')
    self.add_argument('--print-freq', required=False, metavar='<print_freq>', type=int, default=1,
                      help='Frequency of printing. Default: 1')
    self.add_argument('-w', '--world-size', metavar='<world_size>', type=int, default=8, help='Training device size')
    self.add_argument('--num-train-steps', required=False, metavar='<num_train_steps>', type=int, default=None,
                      help='Number of training steps. Default: the maximum train steps')
    self.add_argument('--num-eval-steps', required=False, metavar='<num_eval_steps>', type=int, default=None,
                      help='Number of evaluation steps. Default: the maximum eval steps')
    self.add_argument('--custom-lr-values', required=False, metavar='<custom_lr_values>', default=None,
                      help='custom lr values list')
    self.add_argument('--custom-lr-milestones', required=False, metavar='<custom_lr_milestones>', default=None,
                      help='custom lr milestones list')
    self.add_argument('--dist', action='store_true', help='Distribute training')
    self.add_argument('--output-dir', default='.', metavar='<output_dir>', help='path where to save')
    self.add_argument('--channels-last', default='True', type=lambda x: x.lower() == 'true',
                      help='Use channel last ordering')
    self.add_argument('--deterministic', action="store_true", help='make data loading deterministic')
    self.add_argument('--seed', type=int, default=123, help='random seed for data')
    self.add_argument('--resume', required=False, metavar='<checkpt_path>', default='',
                      help='resume from checkpoint')
    self.add_argument('--process-per-node', default=0, type=int, metavar='N',
                      help='Number of process per node')
    self.add_argument('--save-checkpoint', action="store_true",
                        help='Whether or not to save model/checkpont; True: to save, False to avoid saving')
    self.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    self.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    self.add_argument('--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    self.add_argument('--lr-step-size', default=30, type=int, help='decrease lr every step-size epochs')
    self.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')

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
        f" --dl-worker-type={args.dl_worker_type}"
        f" --dl-time-exclude={args.dl_time_exclude}"
        f" --print-freq={args.print_freq}"
        f" --output-dir={args.output_dir}"
        f" --channels-last={str(args.channels_last)}"
        f" --seed={str(args.seed)}"
        f" --lr={str(args.lr)}"
        f" --momentum={str(args.momentum)}"
        f" --wd={str(args.wd)}"
        f" --lr-step-size={str(args.lr_step_size)}"
        f" --lr-gamma={str(args.lr_gamma)}"
    )

    if args.mode == "eager":
        command += " --run-lazy-mode False"
    if args.data_type == 'bf16':
        command += " --hmp --hmp-bf16 " + path + '/ops_bf16_Mobilenet.txt' + ' --hmp-fp32 ' + path + '/ops_fp32_Mobilenet.txt'
    if args.num_train_steps is not None:
        command += f" --num-train-steps={args.num_train_steps}"
    if args.num_eval_steps is not None:
        command += f" --num-eval-steps={args.num_eval_steps}"
    if args.save_checkpoint:
        command += f" --save-checkpoint"

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

    if args.deterministic:
        command += " --deterministic "

    if args.resume != '':
        command += " --resume=" + args.resume

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

    args = MobileNetArgparser().parse_args()
    print (args)
    # Get the path for the mobilenet train file.
    script_dir = os.path.dirname(__file__)
    mobilenet_path = os.path.abspath(os.path.dirname(__file__))
    train_script = "train.py"
    # Check if the world_size > 1 for distributed training in hls
    check_world_size(args)
    # Set environment variables
    env_vars = set_env(args, mobilenet_path)
    # Build the command line
    command = build_command(args, mobilenet_path, train_script)

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
