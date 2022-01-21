import sys
import re
import os
import io
import socket
import argparse

model_ref_path = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../.."))
os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + ":" +  model_ref_path
sys.path.append(model_ref_path)
from PyTorch.common.training_runner import TrainingRunner

class UnetArgparser(argparse.ArgumentParser):
  """docstring for UnetArgparser"""
  def __init__(self):
    super(UnetArgparser, self).__init__()
    self.add_argument('--exec_mode', metavar='<exec_mode>', default='train', choices=['train', 'evaluate', 'predict'],
                    help='Execution mode to run the model')
    self.add_argument('-p', '--data', metavar='<data>', default="/root/software/data/pytorch/unet/01_2d/",
                      help='Path to the training dataset')
    self.add_argument('--task', required=False, metavar='<task>', type=int, default=None,
                      help='Task number. MSD uses numbers 01-10. Default: None')
    self.add_argument('--gpus', required=False, metavar='<gpus>', type=int, default=0,
                      help='Number of gpus. Default: 0')
    self.add_argument('--hpus', required=False, metavar='<hpus>', type=int, default=0,
                      help='Number of hpus. Default: 0')
    self.add_argument('--learning_rate', required=False, metavar='<learning_rate>', type=float, default=0.001,
                    help='Learning rate. Default: 0.0001')
    self.add_argument('--tta', action="store_true", help='Enable test time augmentation')
    self.add_argument('--deep_supervision', action="store_true", help='Enables deep supervision')
    self.add_argument('--save_ckpt', action="store_true", help='Enable saving checkpoint')
    self.add_argument('--seed', type=int, default=1, help='random seed for data')
    self.add_argument('--fold', required=False, metavar='<fold>', type=int, default=0,
                      help='Fold number . Default: 0')
    self.add_argument('--batch_size', required=False, metavar='<batch_size>', type=int, default=2,
                      help='Batch size . Default: 2')
    self.add_argument('--val_batch_size', required=False, metavar='<val_batch_size>', type=int, default=4,
                      help='Validation batch size . Default: 4')
    self.add_argument('--factor', required=False, metavar='<factor>', type=float, default=0.3,
                    help='Schedule factor. Default: 0.3')
    self.add_argument('--num_workers', required=False, metavar='<num_workers>', type=int, default=8,
                      help='Number of subprocesses to use for data loading. Default: 8')
    self.add_argument( '--min_epochs', required=False, metavar='<min_epochs>', type=int, default=30,
                      help='Force training for at least these many epochs. Default: 30')
    self.add_argument( '--max_epochs', required=False, metavar='<max_epochs>', type=int, default=10000,
                      help='Stop training after this number of epochs. Default: 10000')
    self.add_argument('--channels_last', default='True', type=lambda x: x.lower() == 'true',
                      help='Use channel last ordering')
    self.add_argument('--scheduler', required=False, metavar='<scheduler>', type=str, default='none', choices=['none','multistep',
                      'cosine','plateau'], help='Learning rate scheduler. Possible values: multistep, cosine, plateau.Default: none')
    self.add_argument('--optimizer', required=False, metavar='<optimizer>', type=str, default='adamw', choices=['sgd',
                      'radam', 'adam', 'adamw', 'fusedadamw'], help='Different optimizers modes. Possible values: sgd, adam, radam, adamw, fusedadamw. Default: adamw')
    self.add_argument('--train_batches', required=False, metavar='<train_batches>', type=int, default=0,
                      help='Limit number of batches for training. Default: 0')
    self.add_argument('--test_batches', required=False, metavar='<test_batches>', type=int, default=0,
                      help='Limit number of batches for inference. Default: 0')
    self.add_argument('--mode', required=False, metavar='<mode>', type=str, default='lazy', choices=[
                      'lazy', 'eager'], help='Different modes avaialble. Possible values: lazy, eager. Default: lazy')
    self.add_argument('--data_type', required=False, metavar='<data type>', default='fp32', choices=['fp32', 'bf16'],
                      help='Data Type. Possible values: fp32, bf16. Default: fp32')
    self.add_argument('--dist', action='store_true', help='Distribute training')
    self.add_argument("--dim", type=int, choices=[2, 3], default=2, help="UNet dimension")
    self.add_argument("--norm", type=str, choices=["instance", "batch", "group"], default="instance", help="Normalization layer")
    self.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint")
    self.add_argument('--benchmark', action='store_true', help='Enable benchmark')

def build_command(args, path, train_script):
    """ Constructing training command """
    output_dir = f"/tmp/Unet/fold_{args.fold}"
    CHECK_DIR = os.path.isdir(output_dir)
    if not CHECK_DIR:
        os.makedirs(output_dir)
    init_command = f"{path + '/' + str(train_script)}"
    command = (
        f"{init_command}"
        f" --exec_mode={args.exec_mode}"
        f" --data={args.data}"
        f" --gpus={args.gpus}"
        f" --hpus={args.hpus}"
        f" --learning_rate={args.learning_rate}"
        f" --seed={str(args.seed)}"
        f" --fold={str(args.fold)}"
        f" --batch_size={args.batch_size}"
        f" --val_batch_size={args.val_batch_size}"
        f" --factor={args.factor}"
        f" --num_workers={str(args.num_workers)}"
        f" --min_epochs={args.min_epochs}"
        f" --max_epochs={args.max_epochs}"
        f" --channels_last={args.channels_last}"
        f" --scheduler={args.scheduler}"
        f" --optimizer={args.optimizer}"
        f" --train_batches={args.train_batches}"
        f" --test_batches={args.test_batches}"
        f" --norm={args.norm}"
        )

    if args.data_type == 'bf16':
        config_path = os.path.join(path, 'config')
        command += " --hmp --hmp-bf16 " + config_path + '/ops_bf16_unet.txt' + ' --hmp-fp32 ' + config_path + '/ops_fp32_unet.txt'

    if args.deep_supervision:
        command += " --deep_supervision "

    if args.task:
        command += " --task=" + str(args.task)

    if args.tta:
        command += " --tta "

    if args.save_ckpt:
        command += " --save-ckpt "

    if args.mode == "lazy":
        command += " --run_lazy_mode "

    if args.dim:
        command += " --dim=" + str(args.dim)

    if args.ckpt_path:
        command += " --ckpt_path=" + args.ckpt_path
        command += " --resume_training"

    if args.benchmark:
        command += " --benchmark "

    command += " --results=" + output_dir
    return command

#Setup the environment variable
def set_env(args, cur_path):
  env_vars = {}

  return env_vars

def main():
    args = UnetArgparser().parse_args()
    # Get the path for the unet train file.
    unet_path = os.path.abspath(os.path.dirname(__file__))
    train_script = "main.py"
    # Check if the world_size > 1 for distributed training in hls
    world_size = args.hpus
    # Set environment variables
    env_vars = set_env(args, unet_path)
    # Set if topology uses pytorch lightning framework
    use_pt_lightning = True
    # Build the command line
    command = build_command(args, unet_path, train_script)

    training_runner = TrainingRunner([command], env_vars, world_size, False, map_by='slot',
                                        use_pt_lightning=use_pt_lightning)
    ret_code = training_runner.run()
    return ret_code

if __name__=="__main__":
  ret = main()
  sys.exit(ret)
