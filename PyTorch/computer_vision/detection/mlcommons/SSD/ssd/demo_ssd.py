import sys
import re
import os
import random
import argparse
model_garden_path = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../../../../"))
os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + ":" +  model_garden_path
sys.path.append(model_garden_path)
from PyTorch.common.training_runner import TrainingRunner

class SSDPArgparser(argparse.ArgumentParser):
  """docstring for SSDPArgparser"""
  def __init__(self):
    super(SSDPArgparser, self).__init__()
    self.add_argument('-d', '--data',  metavar='<data_path>', type=str, default="/root/software/data/coco2017",
                      help="Path to the training dataset")
    self.add_argument('--pretrained-backbone',  metavar='<model>', type=str, default=None,
                      help="path to pretrained backbone weights file, "
                           "'default is to get it fro   m online torchvision repository'")
    self.add_argument('-e', '--epochs',  metavar='<epochs>', type=int, default=1,
                      help="number of epochs for training")
    self.add_argument('-b', '--batch-size',  metavar='<batch_size>', type=int, default=32,
                      help="number of examples for each training iteration")
    self.add_argument('--data-type',  metavar='<data_type>', type=str, default='float32',
                      help="precision to be used, supported values are float32, bfloat16")
    self.add_argument('--val-batch-size',  metavar='<val_batch_size>', type=int, default=None,
                      help="number of examples for each validation iteration (defaults to --batch-size)")
    self.add_argument('--device',  metavar='<device>', type=str, default='hpu',
                      help="device to be used")
    self.add_argument('--mode',  metavar='<exec_mode>', type=str, default='eager',
                      help="Execution mode")
    self.add_argument('--hpu-channels-last',  action='store_true',
                      help="convert images to channels last for HPUs")
    self.add_argument('--hmp-opt-level', type=str, default='01',
                      help="choose optimization level for hmp")
    self.add_argument('--hmp-verbose',  action='store_true',
                      help="enable verbose mode for hmp")
    self.add_argument('--seed',  metavar='<seed>', type=int, default=random.SystemRandom().randint(0, 2**32 - 1),
                      help="manually set random seed for torch")
    self.add_argument('--deterministic',  action='store_true', default=False,
                      help="force deterministic training")
    self.add_argument('--lowp',  metavar='<lowp>', type=str,
                      help="cast to BF16 if required")
    self.add_argument('--start-iteration',  metavar='<start_iteration>', type=int, default=0,
                      help="iteration to start from")
    self.add_argument('--end-iteration',  metavar='<end_iteration>', type=int, default=None,
                      help="iteration to end upon")
    self.add_argument('--checkpoint',  metavar='<checkpoint>', type=str, default=None,
                      help="path to model checkpoint")
    self.add_argument('--nosave-checkpoint',  action='store_true', default=False,
                      help="save model checkpoints")
    self.add_argument('--val-interval',  metavar='<evaluatio_interval>', type=int, default=5,
                      help="epoch interval for validation in addition to --val-epochs.")
    self.add_argument('--val-epochs',  metavar='<evaluation_epochs>', nargs='*', default=[],
                      help="epochs at which to evaluate in addition to --val-interval")
    self.add_argument('--batch-splits',  metavar='<batch_splits>', type=int, default=1,
                      help="Split batch to N steps (gradient accumulation)")
    self.add_argument('--learning-rate',  metavar='<learning_rate>', type=float,
                      help="base learning rate.")
    self.add_argument('--log-interval',  metavar='<logging_interval>', type=int, default=100,
                      help="Logging mini-batch interval")
    self.add_argument('--num-workers',  metavar='<number_of_workers>', type=int,
                      help="Number of workers for dataloader.")
    self.add_argument('--world_size',  metavar='<world_size>', type=int, default=1,
                      help="Number of nodes/devices to be used")
    self.add_argument('--distributed', action='store_false', default=False,
                      help="Enable distributed mode for training")
    self.add_argument('--warmup', metavar='<warmup factor>', type=float, default=None,
                      help="Warmup factor for adjusting the learning rate")

def build_command(args, path, train_script):
    """ Constructing the training command """

    init_command = f"{path + '/' + str(train_script)}"
    command = (
        f"{init_command}"
        f" --data {args.data}"
        f" --epochs {args.epochs}"
        f" --batch-size {args.batch_size}"
        f" --log-interval {args.log_interval}"
        f" --val-interval {args.val_interval}"
    )

    if args.mode == "lazy":
        command += " --hpu-lazy-mode"
    if args.device == "hpu":
        command += " --use-hpu"
    if args.data_type == 'bfloat16':
        command += " --hmp --hmp-bf16 " + path + '/ops_bf16_ssdrn34.txt' + ' --hmp-fp32 ' + path + '/ops_fp32_ssdrn34.txt'
    if args.start_iteration:
        command += f" --iteration {args.start_iteration}"
    if args.end_iteration:
        command += f" --end-iteration {args.end_iteration}"
    if args.checkpoint:
        command += f" --checkpoint {args.checkpoint}"
    if args.nosave_checkpoint:
        command += f" --no-save"
    if args.learning_rate:
        command += f" --lr {args.learning_rate}"
    if args.num_workers:
        command += f" --num-workers {args.num_workers}"
    if args.warmup:
        command += f" --warmup {args.warmup}"

    print(f"PYDEMO COMMAND : {command}")
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

    args = SSDPArgparser().parse_args()
    print (args)
    # Get the path for the resnet50 train file.
    ssd_path = os.path.abspath(os.path.dirname(__file__))
    train_script = "train.py"
    # Check if the world_size > 1 for distributed training in hls
    check_world_size(args)
    # Set environment variables
    env_vars = set_env(args, ssd_path)
    # Build the command line
    command = build_command(args, ssd_path, train_script)

    if args.world_size == 1:
        training_runner = TrainingRunner(command_list=[command],
                                         model_env_vars=env_vars,
                                         world_size=args.world_size,
                                         use_mpi=False)
    else:
        mpi_runner = True
        training_runner = TrainingRunner(command_list=[command],
                                         model_env_vars=env_vars,
                                         world_size=args.world_size,
                                         dist=args.distributed,
                                         use_mpi=False,
                                         mpi_run=mpi_runner)

    ret_code = training_runner.run()
    return ret_code

if __name__=="__main__":
    ret = main()
    sys.exit(ret)
