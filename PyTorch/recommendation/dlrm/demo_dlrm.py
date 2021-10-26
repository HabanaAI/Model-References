import argparse
import subprocess
import os
import sys
import io
sys.path.append(os.path.realpath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../../..")))
from central.habana_model_runner_utils import get_canonical_path, get_canonical_path_str
from PyTorch.common.training_runner import TrainingRunner

class DlrmArgparser(argparse.ArgumentParser):
  """docstring for DlrmArgparser"""
  def __init__(self):
    super(DlrmArgparser, self).__init__()
    self.add_argument('--mode', required=True, metavar='<mode>', default='lazy', choices=['lazy','custom','cpu'], help='Mode, possible values= lazy, custom, cpu')
    self.add_argument('--arch-sparse-feature-size', metavar='<arch_sparse_feature_size>', type=int, default=2)
    self.add_argument('--arch-embedding-size', metavar='<arch_embedding_size>', type=str, default='4-3-2')
    self.add_argument('--arch-mlp-bot', metavar='<arch_mlp_bot>', type=str, default='4-3-2')
    self.add_argument('--arch-mlp-top', metavar='<arch_mlp_top>', type=str, default='4-2-1')
    self.add_argument('--arch-interaction-op', metavar='<arch_interaction_op>', type=str, default='dot')
    self.add_argument('--arch-interaction-itself', action='store_true')
    self.add_argument('--activation-function', metavar='<activation_function>', type=str, default='relu')
    self.add_argument('--loss-function', metavar='<loss_function>', type=str, default='bce', choices=['mse','bce'])
    self.add_argument('--loss-threshold', metavar='<loss_threshold>', type=float, default=0.0)
    self.add_argument('--round-targets', action='store_true')
    self.add_argument('--data-size', metavar='<data_size>', type=int, default=8)
    self.add_argument('--num-batches', metavar='<num_batches>', type=int, default=0)
    self.add_argument('--data-generation', metavar='<data_generation>', type=str, default='random', choices=['random','synthetic','dataset'])
    self.add_argument('--data-trace-file', metavar='<data_trace_file>', type=str, default='./input/dist_emb_j.log')
    self.add_argument('--data-set', metavar='<data_set>', type=str, default='kaggle', choices=['kaggle','terabyte'])
    self.add_argument('--raw-data-file', metavar='<raw_data_file>', type=str, default='')
    self.add_argument('--processed-data-file', metavar='<processed_data_file>', type=str, default='')
    self.add_argument('--data-randomize', metavar='<data_randomize>', type=str, default='total', choices=['total','day','none'])
    self.add_argument('--data-trace-enable-padding', action='store_true')
    self.add_argument('--max-ind-range', metavar='<max_ind_range>', type=int, default=-1)
    self.add_argument('--data-sub-sample-rate', metavar='<data_sub_sample_rate>', type=float, default=0.0, help='Float value ranging in [0,1]')
    self.add_argument('--num-indices-per-lookup', metavar='<num_indices_per_lookup>', type=int, default=10)
    self.add_argument('--num-indices-per-lookup-fixed', action='store_true')
    self.add_argument('--memory-map', action='store_true')
    self.add_argument('-b', '--mini-batch-size', metavar='<mini_batch_size>', type=int, default=1)
    self.add_argument('--nepochs', metavar='<nepochs>', type=int, default=1)
    self.add_argument('--learning-rate', metavar='<learning_rate>', type=float, default=0.01)
    self.add_argument('--print-precision', metavar='<print_precision>', type=int, default=5)
    self.add_argument('--numpy-rand-seed', metavar='<numpy_rand_seed>', type=int, default=123)
    self.add_argument('--inference-only', action='store_true')
    self.add_argument('--print-freq', metavar='<print_freq>', type=int, default=1)
    self.add_argument('--test-freq', metavar='<test_freq>', type=int, default=-1)
    self.add_argument('--print-time', action='store_true')
    self.add_argument('--mlperf-logging', action='store_true', default=False)
    self.add_argument('--measure-perf', action='store_true')
    self.add_argument('-d', '--data_type', metavar='<data_type>', default='fp32', choices=['fp32','bf16'])
    self.add_argument('--optimizer', metavar='<optimizer>', type=str, default='sgd', choices=['sgd','adagrad'])
    self.add_argument('-w', '--world_size', metavar='<world_size>', type=int, default=1, choices=range(1,9))
    self.add_argument('--print-all-ranks', action='store_true')

def set_env(args, cur_path):
  env_vars = {}

  return env_vars

def build_command(args, path, script_name):
  init_command = f"{path + '/' + str(script_name)}"
  command = (
    f"{init_command}"
    f" --arch-interaction-op={args.arch_interaction_op}"
    f" --arch-sparse-feature-size={args.arch_sparse_feature_size}"
    f" --arch-mlp-bot={args.arch_mlp_bot} --arch-mlp-top={args.arch_mlp_top}"
    f" --arch-embedding-size={args.arch_embedding_size}"
    f" --num-indices-per-lookup={args.num_indices_per_lookup}"
    f" --mini-batch-size={args.mini_batch_size} --learning-rate={args.learning_rate}"
    f" --num-batches={args.num_batches} --numpy-rand-seed={args.numpy_rand_seed}"
    f" --print-precision={args.print_precision} --data-size={args.data_size}"
    f" --activation-function={args.activation_function}"
    f" --loss-function={args.loss_function} --loss-threshold={args.loss_threshold}"
    f" --data-generation={args.data_generation} --data-trace-file={args.data_trace_file}"
    f" --data-set={args.data_set} --nepochs={args.nepochs}"
    f" --processed-data-file={args.processed_data_file} --data-randomize={args.data_randomize}"
    f" --max-ind-range={args.max_ind_range} --data-sub-sample-rate={args.data_sub_sample_rate}"
    f" --print-freq={args.print_freq} --optimizer={args.optimizer}"
    f" --raw-data-file={args.raw_data_file}"
  )

  if args.mode == 'lazy':
    command += " --run-lazy-mode"
  if args.print_time:
    command += " --print-time"
  if args.inference_only:
    command += " --inference-only"
  if args.memory_map:
    command += " --memory-map"
  if args.arch_interaction_itself:
    command += " --arch-interaction-itself"
  if args.mlperf_logging:
    command += " --mlperf-logging"
  if args.measure_perf:
    command += " --measure-perf"
  if args.data_type == 'bf16':
    command += " --hmp --hmp-bf16 " + path + '/ops_bf16_dlrm.txt' + ' --hmp-fp32 ' + path + '/ops_fp32_dlrm.txt'
  if args.data_trace_enable_padding:
    command += " --data-trace-enable-padding"
  if args.num_indices_per_lookup_fixed:
    command += " --num-indices-per-lookup-fixed"
  if args.round_targets:
    command += " --round-targets"
  if args.print_all_ranks:
    command += " --print-all-ranks"

  print(command)
  return command

def main():
  args = DlrmArgparser().parse_args()

  script_dir = os.path.dirname(__file__)
  dlrm_path = get_canonical_path_str(script_dir)
  if args.mode == 'lazy' or args.mode == 'custom':
    script_name = "dlrm_s_pytorch_hpu_custom.py"
  elif args.mode == 'cpu':
    script_name = "dlrm_s_pytorch.py"

  env_vars = set_env(args, dlrm_path)
  command = build_command(args, dlrm_path, script_name)

  training_runner = TrainingRunner([command],env_vars,args.world_size,False, use_env=True)
  training_runner.run()

if __name__=="__main__":
  main()
