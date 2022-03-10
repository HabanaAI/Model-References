###############################################################################
# Copyright (C) 2021-2022 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
import io
import sys
import argparse
from pathlib import Path

from TensorFlow.common.common import setup_jemalloc
from central.multi_node_utils import run_per_ip, run_cmd_as_subprocess, is_valid_multi_node_config
import distilbert_squad_main
import download.download_pretrained_model as download_pretrained_model

BERT_ARGPARSER_HELP = ''
DEFAULT_BF16_CONFIG_PATH = os.fspath(Path(os.path.realpath(__file__)).parent.joinpath("distilbert_bf16.json"))


class Bert(object):
  def __init__(self):
    super(Bert, self).__init__()
    self.pretrained_url = "https://storage.googleapis.com/bert_models/2020_02_20/"
    self.pretrained_model = ''
    self.pretrained_distilbert = ''

  def download_pretrained_model(self, horovod_run):
    try:
      download_pretrained_model_path = Path(__file__).parent.joinpath('download').joinpath('download_pretrained_model.py')
      if horovod_run and is_valid_multi_node_config():
        run_per_ip(f"{sys.executable} {str(download_pretrained_model_path)} {self.pretrained_url} {self.pretrained_model} {self.pretrained_distilbert} False", [
                   'MULTI_HLS_IPS', 'PYTHONPATH'], False)
      else:
        run_cmd_as_subprocess(f"{sys.executable} {str(download_pretrained_model_path)} {self.pretrained_url} {self.pretrained_model} {self.pretrained_distilbert} False")
    except Exception as exc:
      raise RuntimeError(f"Error in {self.__class__.__name__} download_pretrained_model()") from exc

  def finetuning(self, args):
    if args.use_horovod is not None:
      hw_config_use_horovod = True
      hw_config_num_workers_per_hls = args.use_horovod
    else:
      hw_config_use_horovod = False
      hw_config_num_workers_per_hls = 1

    if args.checkpoint_folder is not None:
        self.pretrained_model = args.checkpoint_folder
    if args.test_set is None:
      print(BERT_ARGPARSER_HELP)
    if args.test_set == "squad":
      if args.epochs:
        e = args.epochs
      else:
        e = 2
      if args.batch_size:
        b = args.batch_size
      else:
        if args.data_type == "bf16":
          b = 32
        else:
          b = 32
      if args.max_seq_length:
        s = args.max_seq_length
      else:
        s = 384
      print(
          f"running SQUAD on BERT {args.model_variant} {self.pretrained_model} epochs: {e} BF16: {os.environ.get('TF_BF16_CONVERSION')} BS: {b} T: {s}")
      squad_runner = distilbert_squad_main.DistilBertFinetuningSQUAD(hw_config_use_horovod, hw_config_num_workers_per_hls, "HLS1", args.kubernetes_run,
                                                         args, e, b, s, self.pretrained_model, args.enable_scoped_allocator)
      squad_runner.run()
    else:
      raise Exception("Incorrect test set passed to -t option")


class BertBase(Bert):
  """docstring for BertBase"""

  def __init__(self):
    super(BertBase, self).__init__()
    self.pretrained_model = "uncased_L-12_H-768_A-12"
    self.pretrained_distilbert = "distilbert-base-uncased"


class BertArgparser(argparse.ArgumentParser):
  """docstring for BertArgparser"""

  def __init__(self):
    super(BertArgparser, self).__init__()
    self.add_argument('-c', '--command', required=True, metavar='<command>',
                      choices=['finetuning'], help='Command, possible values: finetuning')
    self.add_argument('-d', '--data_type', required=True, metavar='<data_type>',
                      choices=['fp32', 'bf16'], help='Data type, possible values: fp32, bf16')
    self.add_argument('-m', '--model_variant', required=True, metavar='<model_variant>', choices=[
                      'base'], help='Model variant, possible values: base')
    self.add_argument('--bert-config-dir', required=False, metavar='<bert_config_dir>', type=str,
                      help="Path to directory containing bert config files needed for chosen training type. If not specified the zip file will be downloaded.")
    self.add_argument('--model', required=False, choices=['distilbert'], help='The model name is distilbert')
    self.add_argument('-o', '--output_dir', metavar='<dir>', help='Output directory. Default is /tmp/distilbert.', default="/tmp/distilbert")
    self.add_argument('-v', '--use_horovod', metavar='<num_horovod_workers_per_hls>', type=int, required=False,
                      help='Use Horovod for multi-card distributed training with specified num_horovod_workers_per_hls.')
    self.add_argument('-t', '--test_set', metavar='<test_set>', default='squad', choices=[
                      'squad'], help='Benchmark dataset, possible finetuning values: squad. Default: squad.')
    self.add_argument('-e', '--epochs', metavar='<val>',
                      help='Number of epochs. If not set, defaults to 2.0 for squad.')
    self.add_argument('-s', '--max_seq_length', metavar='<val>', help='Number of tokens in each sequence. If not set, defaults to 384 for squad.')
    self.add_argument('-fpo', '--fast_perf_only', metavar='<for_perf_measurements>', required=False, type=int,
                      default=0, help='Defaults to 0. Set to 1 to run smaller global batch size for perf measurement.')
    self.add_argument('--save_ckpt_steps', metavar='<val1>', type=int, default=100,
                      help='How often to save the model checkpoint. Default: 100.')
    self.add_argument('-b', '--batch_size', metavar='<batch_size>', help='Batch size. Defaults for bf16/fp32: 32/32 for squad.')
    self.add_argument('--learning_rate', metavar='<learning_rate>', required=False,
                      type=float, default=5e-5, help='Learning rate. Default: 5e-5.')
    self.add_argument('--dataset_path', metavar='<dataset_path>', required=False,
                      type=str, default='./dataset/', help='Path to training dataset. Default: ./dataset/')
    self.add_argument('--checkpoint_folder', metavar='<checkpoint_folder>', required=False,
                      type=str, default=None, help='Init checkpoint folder for use in finetuning')
    self.add_argument('--bf16_config_path', metavar='</path/to/custom/bf16/config>', required=False, type=str, default=DEFAULT_BF16_CONFIG_PATH,
                      help=f'Path to custom mixed precision config to use, given in JSON format. Defaults to {DEFAULT_BF16_CONFIG_PATH}. Applicable only if --data_type = bf16.')
    self.add_argument('--kubernetes_run', metavar='<kubernetes_run>', required=False,
                      type=bool, default=False, help='Set to True for running training on a Kubernetes cluster. Default: False.')
    self.add_argument('--enable_scoped_allocator', metavar='<enable_scoped_allocator>', required=False,
                      type=bool, default=False, help="Set to True to enable scoped allocator optimization. Default: False.")
    self.add_argument('--horovod_hierarchical_allreduce', metavar='<horovod_hierarchical_allreduce>', required=False,
                      type=bool, default=False, help="Enables hierarchical allreduce in Horovod. Default: False. Set this option to True to run multi-HLS scale-out training over host NICs. This will cause the environment variable `HOROVOD_HIERARCHICAL_ALLREDUCE` to be set to `1`.")
    self.add_argument('--deterministic_run', metavar='<deterministic_run>', required=False,
                      type=bool, default=False, help="If set run will be deterministic (set random seed, read dataset in single thread, disable dropout). Default: False.")
    self.add_argument('--deterministic_seed', metavar='<deterministic_seed>', required=False,
                      type=int, default=None, help="Seed vaule to be used in deterministic mode for all pseudorandom sequences.")


def prevent_mpirun_execution():
  if os.environ.get('OMPI_COMM_WORLD_SIZE'):
    raise RuntimeError(
        "This script is not meant to be run from within an OpenMPI context (using mpirun). Use -v parameter to enable the multi-node mode.")


def check_data_type_and_tf_bf16_conversion(args):
  # The bf16 data-type is run as a mixed-precision fp32+bf16 training using the 'bert' config
  if args.data_type == 'bf16':
    os.environ['TF_BF16_CONVERSION'] = args.bf16_config_path
  elif args.data_type != 'fp32':
    raise Exception("data_type can only be \'bf16\' or \'fp32\'")


def check_and_log_synapse_env_vars():
  run_tpc_fuser_env = os.environ.get('RUN_TPC_FUSER')
  if run_tpc_fuser_env is not None:
    print(f"RUN_TPC_FUSER={run_tpc_fuser_env}")
  habana_synapse_logger_env = os.environ.get('HABANA_SYNAPSE_LOGGER')
  if habana_synapse_logger_env is not None:
    print(f"HABANA_SYNAPSE_LOGGER={habana_synapse_logger_env}")
  log_level_all_env = os.environ.get('LOG_LEVEL_ALL')
  if log_level_all_env is not None:
    print(f"LOG_LEVEL_ALL={log_level_all_env}")


def get_model(args):
  try:
    the_map = {
        'base': BertBase
    }
    return the_map[args.model_variant]()
  except Exception as exc:
    print("Error: Incorrect model passed to -m option")
    raise RuntimeError("Error in get_model") from exc


def main():
  try:
    args = BertArgparser().parse_args()

    if not args.kubernetes_run:
      prevent_mpirun_execution()

    if args.horovod_hierarchical_allreduce:
      os.environ['HOROVOD_HIERARCHICAL_ALLREDUCE'] = "1"

    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    BertArgparser().print_help()
    BERT_ARGPARSER_HELP = new_stdout.getvalue()
    sys.stdout = old_stdout

    print(f"{__file__}: PYTHONPATH = {os.environ.get('PYTHONPATH')}")
    setup_jemalloc()
    check_data_type_and_tf_bf16_conversion(args)
    check_and_log_synapse_env_vars()
    model = get_model(args)
    if args.bert_config_dir is not None:
      model.pretrained_model = args.bert_config_dir
    # This downloads the model on all remote IPs if env variable MULTI_HLS_IPS is set
    if args.checkpoint_folder is not None:
      print(f"Using custom model folder: {args.checkpoint_folder}")
    elif args.use_horovod is not None and not args.kubernetes_run:
      model.download_pretrained_model(True)
    else:
      model.download_pretrained_model(False)
    the_map = {
        "finetuning": model.finetuning
    }
    the_map[args.command](args)
  except Exception as exc:
    raise RuntimeError("Error in demo_distilbert.py") from exc


if "__main__" == __name__:
  main()
