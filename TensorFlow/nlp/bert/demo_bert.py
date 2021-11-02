###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import sys
import os
from pathlib import Path
import io
import socket
import argparse
import bert_mrpc_main
import bert_squad_main
import bert_pretraining_bookswiki_main
import bert_pretraining_overfit_main
from central.habana_model_runner_utils import get_canonical_path, get_canonical_path_str, is_valid_multi_node_config, get_multi_node_config_nodes
from central.multi_node_utils import run_per_ip
import TensorFlow.nlp.bert.download.download_pretrained_model as download_pretrained_model
from TensorFlow.common.common import setup_jemalloc

BERT_ARGPARSER_HELP = ''
DEFAULT_BF16_CONFIG_PATH = os.fspath(Path(os.path.realpath(__file__)).parents[2].joinpath("common/bf16_config/bert.json"))


class Bert(object):
  def __init__(self):
    super(Bert, self).__init__()
    self.pretrained_url = "https://storage.googleapis.com/bert_models/2020_02_20/"
    self.pretrained_model = ''

  def download_pretrained_model(self, horovod_run):
    try:
      if horovod_run and is_valid_multi_node_config():
        download_pretrained_model_path = Path(__file__).parent.joinpath(
            'download').joinpath('download_pretrained_model.py')
        run_per_ip(f"{sys.executable} {str(download_pretrained_model_path)} {self.pretrained_url} {self.pretrained_model} False", [
                   'MULTI_HLS_IPS', 'PYTHONPATH'], False)
      else:
        download_pretrained_model.download_pretrained_model_r(self.pretrained_url, self.pretrained_model, False)
    except Exception as exc:
      raise RuntimeError(f"Error in {self.__class__.__name__} download_pretrained_model()") from exc

  def pretraining(self, args):
    if args.use_horovod is not None:
      hw_config_use_horovod = True
      hw_config_num_workers_per_hls = args.use_horovod
    else:
      hw_config_use_horovod = False
      hw_config_num_workers_per_hls = 1

    if args.test_set is None:
      print(BERT_ARGPARSER_HELP)
    if args.test_set == "bookswiki":
      if args.epochs:
        e = args.epochs
      else:
        e = 40
      if args.iters:
        if len(args.iters.split(',')) > 1:
          p1 = args.iters.split(',')[0]
          p2 = args.iters.split(',')[1]
        else:
          p1 = args.iters
          p2 = 782
      else:
        p1 = 7038
        p2 = 782
      if args.global_batch_size:
        if len(args.iters.split(',')) > 1:
          gbs1 = args.iters.split(',')[0]
          gbs2 = args.iters.split(',')[1]
        else:
          gbs1 =  args.global_batch_size
          gbs2 = 32768
      else:
        gbs1 = 65536
        gbs2 = 32768
      if args.warmup:
        if len(args.warmup.split(',')) > 1:
          w1 = args.warmup.split(',')[0]
          w2 = args.warmup.split(',')[1]
        else:
          w1 = args.warmup
          w2 = 200
      else:
        w1 = 2000
        w2 = 200
      if args.batch_size:
        if len(args.batch_size.split(',')) > 1:
          b1 = args.batch_size.split(',')[0]
          b2 = args.batch_size.split(',')[1]
        else:
          if args.data_type == "bf16":
            b1 = args.batch_size
            b2 = 8
          elif args.data_type == "fp32":
            b1 = args.batch_size
            b2 = 8
      else:
        if args.data_type == "bf16":
          b1 = 64
          b2 = 8
        elif args.data_type == "fp32":
          b1 = 32
          b2 = 8
      if args.max_seq_length:
        if len(args.max_seq_length.split(',')) > 1:
          s1 = args.max_seq_length.split(',')[0]
          s2 = args.max_seq_length.split(',')[1]
        else:
          s1 = args.max_seq_length
          s2 = 512
      else:
        s1 = 128
        s2 = 512
      print(f"running Books & Wiki on BERT {args.model_variant} {self.pretrained_model} epochs: {e}; p1: {p1} steps, {w1} warmup, {b1} batch, {gbs1} global batch, {s1} max_seq_length; "
            f"p2: {p2} steps, {w2} warmup, {b2} batch, {gbs2} global batch, {s2} max_seq_length; BF16: {os.environ.get('TF_BF16_CONVERSION')}")
      bookswiki_runner = bert_pretraining_bookswiki_main.BertPretrainingBooksWiki(
          hw_config_use_horovod, hw_config_num_workers_per_hls, "HLS1", args.kubernetes_run,
          args, e, p1, w1, b1, gbs1, s1, p2, w2, b2, gbs2, s2, self.pretrained_model, args.enable_scoped_allocator)
      bookswiki_runner.run()
    elif args.test_set == "overfit":
      if args.epochs:
        e = args.epochs
      else:
        e = 1
      if args.iters:
        p1 = args.iters
      else:
        p1 = 200
      if args.warmup:
        w1 = args.warmup
      else:
        w1 = 10
      if args.batch_size:
        b1 = args.batch_size
      else:
        b1 = 32
      if args.max_seq_length:
        s1 = args.max_seq_length
      else:
        s1 = 128
      print(f"running overfit on BERT {args.model_variant} {self.pretrained_model} epochs: {e}; {p1} steps, {w1} warmup, {b1} batch, {s1} max_seq_length; BF16: {os.environ.get('TF_BF16_CONVERSION')}")
      overfit_runner = bert_pretraining_overfit_main.BertPretrainingOverfit(
          hw_config_use_horovod, hw_config_num_workers_per_hls, "HLS1", args.kubernetes_run,
          args, e, p1, w1, b1, s1, self.pretrained_model, args.enable_scoped_allocator)
      overfit_runner.run()

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
    if args.test_set == "mrpc":
      if args.epochs:
        e = args.epochs
      else:
        e = 3
      if args.batch_size:
        b = args.batch_size
      else:
        if args.data_type == "bf16":
          b = 64
        else:
          b = 32
      if args.max_seq_length:
        s = args.max_seq_length
      else:
        s = 128
      print(
          f"running MRPC on BERT {args.model_variant} {self.pretrained_model} epochs: {e} BF16: {os.environ.get('TF_BF16_CONVERSION')} BS: {b} T: {s}")
      mrpc_runner = bert_mrpc_main.BertFinetuningMRPC(hw_config_use_horovod, hw_config_num_workers_per_hls, "HLS1", args.kubernetes_run,
                                                      args, e, b, s, self.pretrained_model, args.enable_scoped_allocator)
      mrpc_runner.run()
    elif args.test_set == "squad":
      if args.epochs:
        e = args.epochs
      else:
        e = 2
      if args.batch_size:
        b = args.batch_size
      else:
        if args.data_type == "bf16":
          b = 24
        else:
          b = 10
      if args.max_seq_length:
        s = args.max_seq_length
      else:
        s = 384
      print(
          f"running SQUAD on BERT {args.model_variant} {self.pretrained_model} epochs: {e} BF16: {os.environ.get('TF_BF16_CONVERSION')} BS: {b} T: {s}")
      squad_runner = bert_squad_main.BertFinetuningSQUAD(hw_config_use_horovod, hw_config_num_workers_per_hls, "HLS1", args.kubernetes_run,
                                                         args, e, b, s, self.pretrained_model, args.enable_scoped_allocator)
      squad_runner.run()
    else:
      raise Exception("Incorrect test set passed to -t option")


class BertMedium(Bert):
  """docstring for BertMedium"""

  def __init__(self):
    super(BertMedium, self).__init__()
    self.pretrained_model = "uncased_L-8_H-512_A-8"


class BertSmall(Bert):
  """docstring for BertSmall"""

  def __init__(self):
    super(BertSmall, self).__init__()
    self.pretrained_model = "uncased_L-4_H-512_A-8"


class BertMini(Bert):
  """docstring for BertMini"""

  def __init__(self):
    super(BertMini, self).__init__()
    self.pretrained_model = "uncased_L-4_H-256_A-4"


class BertTiny(Bert):
  """docstring for BertTiny"""

  def __init__(self):
    super(BertTiny, self).__init__()
    self.pretrained_model = "uncased_L-2_H-128_A-2"


class BertLarge(Bert):
  """docstring for BertLarge"""

  def __init__(self):
    super(BertLarge, self).__init__()
    self.pretrained_url = "https://storage.googleapis.com/bert_models/2019_05_30/"
    self.pretrained_model = "wwm_uncased_L-24_H-1024_A-16"

  def download_pretrained_model(self, horovod_run):
    try:
      if horovod_run and is_valid_multi_node_config():
        download_pretrained_model_path = Path(__file__).parent.joinpath(
            'download').joinpath('download_pretrained_model.py')
        run_per_ip(f"{sys.executable} {str(download_pretrained_model_path)} {self.pretrained_url} {self.pretrained_model} True", [
                   'MULTI_HLS_IPS', 'PYTHONPATH'], False)
      else:
        download_pretrained_model.download_pretrained_model_r(self.pretrained_url, self.pretrained_model, True)
    except Exception as exc:
      raise RuntimeError(f"Error in {self.__class__.__name__} download_pretrained_model()") from exc


class BertBase(Bert):
  """docstring for BertBase"""

  def __init__(self):
    super(BertBase, self).__init__()
    self.pretrained_model = "uncased_L-12_H-768_A-12"


class BertArgparser(argparse.ArgumentParser):
  """docstring for BertArgparser"""

  def __init__(self):
    super(BertArgparser, self).__init__()
    self.add_argument('-c', '--command', required=True, metavar='<command>',
                      choices=['pretraining', 'finetuning'], help='Command, possible values: pretraining, finetuning')
    self.add_argument('-d', '--data_type', required=True, metavar='<data_type>',
                      choices=['fp32', 'bf16'], help='Data type, possible values: fp32, bf16')
    self.add_argument('-m', '--model_variant', required=True, metavar='<model_variant>', choices=[
                      'tiny', 'mini', 'small', 'medium', 'base', 'large'], help='Model variant, possible values: tiny, mini, small, medium, base, large')
    self.add_argument('--bert-config-dir', required=False, metavar='<bert_config_dir>', type=str,
                      help="Path to directory containing bert config files needed for chosen training type. If not specified the zip file will be downloaded.")
    self.add_argument('--model', required=False, choices=['bert'], help='The model name is bert')
    self.add_argument('-o', '--output_dir', metavar='<dir>', help='Output directory. Default is /tmp/bert.', default="/tmp/bert")
    self.add_argument('-v', '--use_horovod', metavar='<num_horovod_workers_per_hls>', type=int, required=False,
                      help='Use Horovod for multi-card distributed training with specified num_horovod_workers_per_hls.')
    self.add_argument('-t', '--test_set', metavar='<test_set>', default='bookswiki', choices=[
                      'bookswiki', 'overfit', 'mrpc', 'squad'], help='Benchmark dataset, possible finetuning values: mrpc, squad; possible pretraining values: bookswiki, overfit. Default: bookswiki.')
    self.add_argument('-e', '--epochs', metavar='<val>',
                      help='Number of epochs. If not set, defaults to 3.0 for mrpc, 2.0 for squad and 40.0 for bookswiki.')
    self.add_argument('-s', '--max_seq_length', metavar='<val>', help='Number of tokens in each sequence. If not set, defaults to 128 for mrpc; 384 for squad; 128,512 for bookswiki; 128 for overfit.')
    self.add_argument(
        '-i', '--iters', metavar='<val1[,val2]>', help='Number of steps for each phase of pretraining. Default: 7038,782 for bookswiki and 200 for overfit.')
    self.add_argument('-w', '--warmup', metavar='<val1[,val2]>',
                      help='Number of warmup steps for each phase of pretraining. Default: 2000,200 for bookswiki and 10 for overfit.')
    self.add_argument('-fpo', '--fast_perf_only', metavar='<for_perf_measurements>', required=False, type=int,
                      default=0, help='Defaults to 0. Set to 1 to run smaller global batch size for perf measurement.')
    self.add_argument('--save_ckpt_steps', metavar='<val1>', type=int, default=100,
                      help='How often to save the model checkpoint. Default: 100.')
    self.add_argument('--no_steps_accumulation', metavar='<for_bookswiki_pretraining>', required=False, type=int,
                      default=0, help='Defaults to 0. Set to 1 for no steps accumulation during BooksWiki pretraining.')
    self.add_argument('-b', '--batch_size', metavar='<batch_size>', help='Batch size. Defaults for bf16/fp32: 64/32 for mrpc; 24/10 for squad; 64,8/32,8 for bookswiki; 32 for overfit.')
    self.add_argument('--global_batch_size', metavar='<val1[,val2]>', help='Global batch size for each phase of pretraining. Default: 65536, 32768 for bookswiki.')
    self.add_argument('--learning_rate', metavar='<learning_rate>', required=False,
                      type=float, default=2e-5, help='Learning rate. Default: 2e-5.')
    self.add_argument('--dataset_path', metavar='<dataset_path>', required=False,
                      type=str, default='./dataset/', help='Path to training dataset. Default: ./dataset/')
    self.add_argument('--init_checkpoint_path', metavar='<overfit_init_checkpoint_path>', required=False,
                      type=str, default='', help='Init checkpoint path for use in overfit pretraining')
    self.add_argument('--checkpoint_folder', metavar='<checkpoint_folder>', required=False,
                      type=str, default=None, help='Init checkpoint folder for use in finetuning')
    self.add_argument('--bf16_config_path', metavar='</path/to/custom/bf16/config>', required=False, type=str, default=DEFAULT_BF16_CONFIG_PATH,
                      help=f'Path to custom mixed precision config to use, given in JSON format. Defaults to {DEFAULT_BF16_CONFIG_PATH}. Applicable only if --data_type = bf16.')
    self.add_argument('--kubernetes_run', metavar='<kubernetes_run>', required=False,
                      type=bool, default=False, help='Set to True for running training on a Kubernetes cluster. Default: False.')
    self.add_argument('--run_phase2', metavar='<run_phase2>', required=False,
                      type=bool, default=False, help="Set to True for running Phase 2 of multi-card pretraining on a Kubernetes cluster, after Phase 1 has successfully completed. Default: False.")
    self.add_argument('--enable_scoped_allocator', metavar='<enable_scoped_allocator>', required=False,
                      type=bool, default=False, help="Set to True to enable scoped allocator optimization. Default: False.")
    self.add_argument('--horovod_hierarchical_allreduce', metavar='<horovod_hierarchical_allreduce>', required=False,
                      type=bool, default=False, help="Enables hierarchical allreduce in Horovod. Default: False. Set this option to True to run multi-HLS scale-out training over host NICs. This will cause the environment variable `HOROVOD_HIERARCHICAL_ALLREDUCE` to be set to `1`.")


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
        'base': BertBase,
        'large': BertLarge,
        'tiny': BertTiny,
        'mini': BertMini,
        'small': BertSmall,
        'medium': BertMedium
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

    if os.environ.get('PYTHONPATH'):
      os.environ['PYTHONPATH'] = get_canonical_path_str("./") + ":" + \
          get_canonical_path_str("../../common") + ":" + \
          get_canonical_path_str("../../") + ":" + \
          get_canonical_path_str("../../../central/") + ":" + os.environ.get('PYTHONPATH')
    else:
      os.environ['PYTHONPATH'] = get_canonical_path_str("./") + ":" + \
          get_canonical_path_str("../../common") + ":" + \
          get_canonical_path_str("../../") + ":" + \
          get_canonical_path_str("../../../central/")

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
        "pretraining": model.pretraining,
        "finetuning": model.finetuning
    }
    the_map[args.command](args)
  except Exception as exc:
    raise RuntimeError("Error in demo_bert.py") from exc


if "__main__" == __name__:
  main()
