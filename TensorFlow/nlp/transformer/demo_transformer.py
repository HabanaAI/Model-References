#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import argparse
import glob
import os
import sys
from pathlib import Path
import re
import subprocess

from central.generate_hcl_config import generate_hcl_config_r
from TensorFlow.common.mpi_common import create_mpi_cmd
from TensorFlow.common.tb_utils import TBSummary


problem = 'translate_ende_wmt32k_packed'
default_dataset_dir = '/software/data/tf/data/wmt32k_packed'
default_train_dir = f'{default_dataset_dir}/train'
default_val_dir = f'{default_dataset_dir}/val'

default_output_dir = './t2t_train/{problem}/transformer_{model}/bs{batch_size}'
default_decode_from_file = f'{default_val_dir}/wmt14.src.tok'
default_decode_to_file = './wmt14.tgt.tok'

default_bf16_config_path = os.path.normpath(
    os.path.join(os.path.realpath(__file__), '..', '..', '..',
                 'common', 'bf16_config', 'transformer.json'))


def get_args():
    parser = argparse.ArgumentParser(
        description='Runs Transformer demo.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dtype', '-d', choices=['fp32', 'bf16'],
                        default='bf16', help='Data type used for training. Note: Inference always runs on fp32')
    parser.add_argument('--batch_size', '-bs', type=int,
                        default=4096, help='Batch size (in number of tokens).')
    parser.add_argument('--max_length', '-ml', type=int,
                        default=256, help='Maximum length of sequence used during training.')
    parser.add_argument('--model', '-m', choices=['tiny', 'base', 'big'],
                        default='big', help='Model variant.')
    parser.add_argument('--data_dir', '-dd', type=str,
                        default=default_train_dir, help='Path to dataset.')
    parser.add_argument('--no_hpu', '-c', action='store_true',
                        default=False, help='Set this flag to run on CPU/GPU.')
    parser.add_argument('--log_steps', '-ls', type=int,
                        default=1, help='Steps between logging.')
    parser.add_argument('--summary_steps', '-ss', type=int, default=1,
                        help='How often to save summaries to TensorBoard.')
    parser.add_argument('--train_steps', '-ts', type=int, default=300000,
                        help='Number of training steps in total.')
    parser.add_argument('--eval_freq', '-ef', type=int, default=1000,
                        help='Number of training steps between evaluations.')
    parser.add_argument('--schedule', '-s', choices=[
                        'train', 'continuous_train_and_eval', 'calc_bleu'],
                        default='train', help='Schedule to execute.')
    parser.add_argument('--output_dir', '-od', type=str,
                        help='Output directory.', default=argparse.SUPPRESS)
    parser.add_argument('--optionally_use_dist_strat', default=False, action='store_true',
                        help='Whether to use distributed strategy.')
    parser.add_argument('--num_gpus', default=1, type=int,
                        help='Number of GPUs available for training.')
    parser.add_argument('--random_seed', default=None, type=int,
                        help='Random seed to use during training. Setting this enables deterministic dataloader.')
    parser.add_argument('--keep_checkpoint_max', default=0, type=int,
                        help='The maximum number of recent checkpoint files to keep.')
    parser.add_argument('--decode_from_file', '-dff', type=str,
                        help='File path to use when reading inference data.', default=default_decode_from_file)
    parser.add_argument('--decode_to_file', '-dtf', type=str,
                        help='File path to use when saving inference results.', default=default_decode_to_file)
    parser.add_argument('--hvd_workers', default=0, type=int,
                        help='Number of horovod workers. When 0 horovod is disabled.')
    parser.add_argument('--learning_rate_constant', '-lrc', default=2.0, type=float,
                        help='Learning rate constant.')
    parser.add_argument('--bf16_config_path', metavar='<path/to/custom/config.json>',
                        default=default_bf16_config_path, type=str,
                        help='Path to custom mixed precision config (in JSON format).')
    parser.add_argument('--checkpoint_path', '-cp', default=None, type=str,
                        help='Checkpoint path used for inference without any suffixes. For example ./tiny_bs4096/model.ckpt-25000')
    parser.add_argument('--deterministic_dataset', action='store_true', default=False,
                        help='Enable deterministic dataloader.')
    parser.add_argument('--no_dropout', action='store_true', default=False,
                        help='Disable dropout during training (useful for tests).')
    parser.add_argument('--no_checkpoints', action='store_true', default=False,
                        help='Disable saving checkpoints.')
    parser.add_argument('--dump_config', default=None, type=str,
                        help='Side-by-side config file. Internal, do not use.')

    args = parser.parse_args()
    return args

def get_flag_str(name, value):
    if value is not None:
        return f'--{name}={value}'
    return ''

def get_output_dir(args):
    if not hasattr(args, 'output_dir'):
        return default_output_dir.format(problem=problem, model=args.model, batch_size=args.batch_size)
    return args.output_dir

def get_checkpoint_dir(args):
    # Here we need to distinguish between training with and without horovod.
    # If training used horovod then checkpoint will be saved into "worker_0"
    # subdirectory.
    checkpoint_dir = Path(get_output_dir(args)) / 'worker_0'
    if not checkpoint_dir.exists():
        checkpoint_dir = checkpoint_dir.parent
    checkpoint_dir = str(checkpoint_dir)
    return checkpoint_dir

def get_training_cmd(script_dir, args):
    launcher = os.path.join(script_dir, 'trainer.py')

    mpi_prefix = []
    if args.hvd_workers > 1:
        generate_hcl_config_r(file_path=".", devices_per_hls=args.hvd_workers)
        mpi_prefix = create_mpi_cmd(num_workers=args.hvd_workers, tag_output=True)

    dropout_hparams = ''
    if args.no_dropout:
        dropout_hparams = (f',layer_prepostprocess_dropout=0.0'
                           f',attention_dropout=0.0'
                           f',relu_dropout=0.0')

    deterministic_dataset = args.deterministic_dataset or (args.random_seed is not None)

    params = [f'{sys.executable} {launcher}',
              get_flag_str('data_dir', args.data_dir),
              get_flag_str('problem', problem),
              get_flag_str('model', 'transformer'),
              get_flag_str('hparams_set', f'transformer_{args.model}'),
              get_flag_str(
                  'hparams',
                  f'batch_size={args.batch_size},max_length={args.max_length},'
                  f'learning_rate_constant={args.learning_rate_constant},no_data_parallelism=False'
                  f'{dropout_hparams}'),
              get_flag_str('output_dir', get_output_dir(args)),
              get_flag_str('worker_gpu', args.num_gpus),
              get_flag_str('optionally_use_dist_strat', args.optionally_use_dist_strat),
              get_flag_str('keep_checkpoint_max', args.keep_checkpoint_max),
              get_flag_str('local_eval_frequency', args.eval_freq),
              get_flag_str('train_steps', args.train_steps),
              get_flag_str('log_step_count_steps', args.log_steps),
              get_flag_str('schedule', args.schedule),
              get_flag_str('save_summary_steps', args.summary_steps),
              get_flag_str('deterministic_dataset', deterministic_dataset),
              get_flag_str('random_seed', args.random_seed),
              get_flag_str('use_horovod', args.hvd_workers > 1),
              get_flag_str('use_hpu', not args.no_hpu),
              get_flag_str('use_bf16', args.dtype == 'bf16'),
              get_flag_str('bf16_config_path', args.bf16_config_path),
              get_flag_str('no_checkpoints', args.no_checkpoints),
              get_flag_str('dump_config', args.dump_config),
              ]
    return ' '.join(mpi_prefix + params)

def get_inference_cmd(script_dir, args):
    launcher = os.path.join(script_dir, 'decoder.py')

    assert args.checkpoint_path is not None, (
        'Missing checkpoint_path argument!')
    output_dir = os.path.join(get_checkpoint_dir(args), 'decode')

    params = [f'{sys.executable} {launcher}',
              get_flag_str('data_dir', args.data_dir),
              get_flag_str('problem', problem),
              get_flag_str('model', 'transformer'),
              get_flag_str('hparams_set', f'transformer_{args.model}'),
              get_flag_str('output_dir', output_dir),
              get_flag_str('checkpoint_path', args.checkpoint_path),
              get_flag_str('decode_from_file', args.decode_from_file),
              get_flag_str('decode_to_file', args.decode_to_file),
              get_flag_str('use_hpu', not args.no_hpu),
              get_flag_str('use_bf16', args.dtype == 'bf16'),
              get_flag_str('bf16_config_path', args.bf16_config_path),
              ]
    return ' '.join(params)


def get_bleu_cmd(script_dir, args, log_dir=None):
    launcher = os.path.join(script_dir, 'compute_bleu.py')

    params = [f'{sys.executable} {launcher}',
              get_flag_str('decoded_file', args.decode_to_file),
              get_flag_str('log_dir', log_dir),
              ]
    return ' '.join(params)


def calc_bleu(script_dir, args):
    best_bleu = 0.
    log_dir = os.path.join(get_output_dir(args), 'eval')
    with TBSummary(log_dir) as summary:
        if args.checkpoint_path is not None:
            checkpoint_dir = args.checkpoint_path + '.meta'
        else:
            checkpoint_dir = os.path.join(get_checkpoint_dir(args), 'model.ckpt-*.meta')

        all_checkpoints = sorted(glob.glob(checkpoint_dir), key=lambda x: (len(x), x))
        if len(all_checkpoints) == 0:
            print("No checkpoints found!", flush=True)

        for ckpt in all_checkpoints:
            ckpt = str(Path(ckpt).with_suffix(''))
            step = int(re.findall('model.ckpt-(\d+)', ckpt)[0])
            print(f'Processing checkpoint: "{ckpt}"', flush=True)

            args.checkpoint_path = ckpt

            inference_cmd = get_inference_cmd(script_dir, args)
            print(f'Running inference: "{inference_cmd}"', flush=True)
            subprocess.run(inference_cmd.split(), check=True)

            bleu_cmd = get_bleu_cmd(script_dir, args, log_dir=None)
            print(f'Computing bleu: "{bleu_cmd}"', flush=True)
            bleu = subprocess.run(bleu_cmd.split(), check=True, stdout=subprocess.PIPE)

            try:
                bleu = float(bleu.stdout.decode().split()[-1])
                best_bleu = max(best_bleu, bleu)
                summary.add_scalar('BLEU', bleu, step)
                print('BLEU:', bleu, flush=True)
            except ValueError:
                print(f'Error when calculating BLEU score for step {step}!', flush=True)

        summary.add_scalar('accuracy', best_bleu, 0)


if __name__ == "__main__":
    args = get_args()
    script_dir = os.path.dirname(__file__)
    if not script_dir:
        script_dir = '.'

    if args.no_hpu:
        print('Running on CPU/GPU')
    else:
        print('Running on HPU')

    if 'train' in args.schedule:
        train_cmd = get_training_cmd(script_dir, args)
        print('Running:\n{}'.format(train_cmd), flush=True)
        os.system(train_cmd)

    if 'calc_bleu' in args.schedule:
        calc_bleu(script_dir, args)
