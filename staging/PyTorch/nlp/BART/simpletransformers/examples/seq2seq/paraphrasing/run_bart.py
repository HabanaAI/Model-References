import sys
import os
import argparse

model_garden_path = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../../../../../.."))
os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + ":" +  model_garden_path
sys.path.append(model_garden_path)
from central.multi_node_utils import (is_valid_multi_node_config, get_multi_node_config_nodes)
from PyTorch.common.training_runner import TrainingRunner

class BARTArgparser(argparse.ArgumentParser):
  """docstring for BARTArgparser"""
  def __init__(self):
    super(BARTArgparser, self).__init__()
    self.add_argument(
        "--use_habana",
        action="store_true",
        help="Whether not to use Habana device when available"
    )
    self.add_argument(
        "--lazy_mode",
        action="store_true",
        help="Enable lazy mode or not",
    )
    self.add_argument(
        "--output_dir",
        default='/tmp/bart',
        type=str,
        help="Output dir",
    )
    self.add_argument(
        "--no_cuda",
        action="store_true",
        help="Whether not to use CUDA when available"
    )
    self.add_argument(
        "--use_fused_adam",
        action="store_true",
        help="Whether to use fused adamw on habana device"
    )
    self.add_argument(
       "--use_fused_clip_norm",
       action="store_true",
       help="Whether to use fused clip norm on habana device"
    )
    self.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus"
    )
    self.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for initialization"
    )
    self.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="maximum input sequence length"
    )
    self.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="batch size for training"
    )
    self.add_argument(
       "--fp16",
       action="store_true",
       help="Whether to use fp16"
    )
    self.add_argument(
       "--bf16",
       action="store_true",
       help="Whether to use bf16"
    )
    self.add_argument(
        "--hmp_bf16",
        default="ops_bf16_bart.txt",
        help="path to bf16 ops list in hmp O1 mode"
    )
    self.add_argument(
        "--hmp_fp32",
        default="ops_fp32_bart.txt",
        help="path to fp32 ops list in hmp O1 mode"
    )
    self.add_argument(
        "--hmp_opt_level",
        default="O1",
        help="choose optimization level for hmp"
    )
    self.add_argument(
        "--hmp_verbose",
        action="store_true",
        help="enable verbose mode for hmp"
    )
    self.add_argument(
       "--debug",
       action="store_true",
       help="Whether in debug mode"
    )
    self.add_argument(
        "--save_steps",
        type=int,
        default=-1,
        help="number of steps to save the model"
    )
    self.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="number of maximum training steps"
    )
    self.add_argument(
       "--save_optimizer_and_scheduler",
       action="store_true",
       help="Whether save optimizer and scheduler"
    )
    self.add_argument(
        "--eval_batch_size",
        type=int,
        default=64,
        help="batch size for evaluation"
    )
    self.add_argument(
       "--evaluate_during_training",
       action="store_true",
       help="Whether evaluate during training"
    )
    self.add_argument(
        "--evaluate_during_training_steps",
        type=int,
        default=-1,
        help="evaluate every training steps"
    )
    self.add_argument(
       "--evaluate_each_epoch",
       action="store_true",
       help="Whether evaluate after each epoch"
    )
    self.add_argument(
       "--evaluate_generated_text",
       action="store_true",
       help="Whether evaluate the generated text"
    )
    self.add_argument(
       "--save_model_every_epoch",
       action="store_true",
       help="Whether save the model after each epoch"
    )
    self.add_argument(
       "--save_eval_checkpoints",
       action="store_true",
       help="Whether save the checkpoint after evaluation"
    )
    self.add_argument(
       "--save_best_model",
       action="store_true",
       help="Whether save the best model"
    )
    self.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="number of logging steps"
    )
    self.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="number of epochs for training"
    )
    self.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="number of return sequences during beam sampling"
    )
    self.add_argument(
       "--predict",
       action="store_true",
       help="Whether generate text given input"
    )
#################### distributed training ######################
    self.add_argument(
        '--dl_worker_type',
        default='MP',
        type=lambda x: x.upper(),
        choices = ["MT", "MP"],
        help='select multithreading or multiprocessing'
    )
    self.add_argument(
        '--world_size',
        default=1,
        type=int,
        metavar='N',
        help='number of total workers (default: 1)'
    )
    self.add_argument(
        '--process_per_node',
        default=8,
        type=int,
        metavar='N',
        help='Number of process per node'
    )
    self.add_argument(
        '--distributed',
        action='store_true',
        help='whether to enable distributed mode and run on multiple devices'
    )
    self.add_argument(
        '--dist_url',
        default='env://',
        help='url used to set up distributed training'
    )
    self.add_argument(
        "--data_dir",
        default="",
        type=str,
        help="The input data dir. If no data dir, will run with ./data under local directory.",
    )

def build_command(args, path, train_script):
    """ Constructing the training command """
    init_command = f"{path + '/' + str(train_script)}"
    command = (
        f"{init_command}"
    )
    if args.use_habana:
        command += f" --use_habana"
    if args.lazy_mode:
        command += f" --lazy_mode"
    if args.no_cuda:
        command += f" --no_cuda"
    if args.use_fused_adam:
        command += f" --use_fused_adam"
    if args.use_fused_clip_norm:
        command += f" --use_fused_clip_norm"
    command += f" --max_seq_length={args.max_seq_length}"
    if args.fp16:
        command += f" --fp16"
    if args.bf16:
        command += f" --bf16"
        command += f" --hmp_bf16={args.hmp_bf16}"
        command += f" --hmp_fp32={args.hmp_fp32}"
    if args.distributed:
        command += f" --distributed"
    command += f" --train_batch_size={args.train_batch_size}"
    command += f" --num_train_epochs={args.num_train_epochs}"
    command += f" --logging_steps={args.logging_steps}"
    if args.max_steps != -1:
        command += f" --max_steps={args.max_steps}"
    if args.save_best_model:
        command += f" --save_best_model"
    if args.save_model_every_epoch:
        command += f" --save_model_every_epoch"
    if args.save_eval_checkpoints:
        command += f" --save_eval_checkpoints"
    if args.save_optimizer_and_scheduler or args.save_steps != -1:
        command += f" --save_optimizer_and_scheduler"
    if args.save_steps != -1:
        command += f" --save_steps={args.save_steps}"
    command += f" --output_dir={args.output_dir}"
    if args.predict:
        command += f" --predict"
    if args.evaluate_during_training:
        command += f" --evaluate_during_training"
        command += f" --evaluate_during_training_steps={args.evaluate_during_training_steps}"
    if args.evaluate_each_epoch:
        command += f" --evaluate_each_epoch"
    if args.evaluate_generated_text:
        command += f" --evaluate_generated_text"
    if args.debug:
        command += f" --debug"

    if args.use_habana and is_valid_multi_node_config() and args.world_size > 0:
        if args.process_per_node == 0:
            nodes = get_multi_node_config_nodes()
            args.process_per_node = args.world_size // len(nodes)
        command += f" --process-per-node={args.process_per_node}"
        args.multi_hls = True
    else:
        args.multi_hls = False
    command += f" --data_dir={args.data_dir}"
    print('################# multi_hls #############', args.multi_hls)
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
    args = BARTArgparser().parse_args()
    print (args)

    script_dir = os.path.dirname(__file__)
    bart_path = str(script_dir)
    train_script = "train.py"

    # Check if the world_size > 1 for distributed training in hls
    check_world_size(args)

    # Set environment variables
    env_vars = set_env(args, bart_path)

    # Build the command line
    command = build_command(args, bart_path, train_script)

    if args.world_size == 1:
        training_runner = TrainingRunner([command],env_vars,args.world_size,False, map_by='slot')
    else:
        mpi_runner = True
        training_runner = TrainingRunner([command],env_vars,args.world_size,args.distributed,False,mpi_runner,
                                         map_by='slot', multi_hls=args.multi_hls)

    ret_code = training_runner.run()
    return ret_code


if __name__=="__main__":
    ret = main()
    sys.exit(ret)
