# coding=utf-8
# Copyright (c) 2021, Habana Labs Ltd.  All rights reserved.
# Copyright (c) 2019 - 2022 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
# ==================
import csv
import json
import math
import multiprocessing
import os
import random
import re
import sys
import threading
import time
import warnings
from collections import OrderedDict, defaultdict

import h5py
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import modeling
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from schedulers import LinearWarmUpScheduler, PolyWarmUpScheduler
from tokenization import BertTokenizer
from utils import format_step, get_rank, get_world_size, is_main_process, convert_weight_name

try:
    import amp_C
    import apex_C
    from apex import amp
    from apex.optimizers import FusedLAMB
    from apex.parallel import DistributedDataParallel as DDP
    from apex.parallel.distributed import flat_dist_call
except ImportError:
    if torch.cuda.is_available():
        raise ImportError("Please install apex from "
                          "https://www.github.com/nvidia/apex")
    else:
        from torch.nn.parallel import DistributedDataParallel as DDP

from concurrent.futures import ProcessPoolExecutor

import dllogger

from lamb import NVLAMB

from synapse_profiler_adapter import SynapseProfilerAdatper

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

avg_seq_per_pack = 1.0

# Track whether a SIGTERM (cluster time up) has been handled
timeout_sent = False

import signal

# gradient_buffer is created and used when: --distribution_strategy gradientbuffer
gradient_buffer = None

# handle SIGTERM sent from the scheduler and mark so we
# can gracefully save & exit
def signal_handler(sig, frame):
    global timeout_sent
    timeout_sent = True

signal.signal(signal.SIGTERM, signal_handler)

def get_mllog_mlloger(args):
    attr = 'mllog_mlloger'

    if not hasattr(get_mllog_mlloger, attr):
        from mlperf_logging import mllog
        mllogger = mllog.get_mllogger()

        if args.output_dir is not None:
            log_dir = args.output_dir
        else:
            log_dir = './log'

        if torch.distributed.is_initialized():
            workername =  str(torch.distributed.get_rank())
        else:
            workername = "0"

        filenames = os.path.normpath(log_dir) + "/result_rank_" + workername + ".txt"
        mllog.config(filename=filenames)

        mllog.config(
                default_namespace = "worker"+workername,
                default_stack_offset = 1,
                default_clear_line = False,
                root_dir = os.path.normpath(
                os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")))
        setattr(get_mllog_mlloger, attr, (mllogger, mllog))

        return getattr(get_mllog_mlloger, attr)


def create_dataset(dataset, args, worker_init, input_file=None):
    num_workers = 0 if args.use_habana else 4
    train_data = dataset(max_pred_length=args.max_predictions_per_seq, max_seq_length=args.max_seq_length, max_packed_sequences=args.max_packed_sequences, masked_lm_positions_size=args.masked_lm_positions_size, enable_packed_data_mode=args.enable_packed_data_mode, input_file=input_file, batch_size=args.train_batch_size * args.n_pu)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=args.train_batch_size * args.n_pu,
                                  num_workers=num_workers, worker_init_fn=worker_init,
                                  drop_last=True, pin_memory=True, persistent_workers=(num_workers > 0))
    return train_dataloader

def create_eval_dataset(args, worker_init_fn):
    eval_data = []
    for eval_file in sorted(os.listdir(args.eval_dir)):
        eval_file_path = os.path.join(args.eval_dir, eval_file)
        if os.path.isfile(eval_file_path) and 'part' in eval_file_path:
            eval_data.extend(pretraining_dataset(max_pred_length=args.max_predictions_per_seq, max_seq_length=args.max_seq_length, max_packed_sequences=args.max_packed_sequences, masked_lm_positions_size=args.masked_lm_positions_size, input_file=eval_file_path, batch_size=args.train_batch_size * args.n_pu))
            if len(eval_data) > args.num_eval_examples:
                eval_data = eval_data[:args.num_eval_examples]
                break
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank() % args.num_eval_workers
        chunk_size = args.num_eval_examples // args.num_eval_workers
        remainder = args.num_eval_examples % args.num_eval_workers
        if rank<remainder:
            eval_data = eval_data[(chunk_size+1)*rank : (chunk_size+1)*(rank+1)]
        else:
            eval_data = eval_data[chunk_size*rank+remainder : chunk_size*(rank+1)+remainder]
    else:
        chunk_size = args.num_eval_examples

    num_workers=0 if min(chunk_size, args.eval_batch_size)<=10 else 4
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                         num_workers=num_workers, worker_init_fn=worker_init_fn, pin_memory=True, persistent_workers=(num_workers > 0))

    return eval_dataloader

class warmup_dataset(Dataset):
    @staticmethod
    def name():
        return "  Warmup"

    def __init__(self, max_pred_length, max_seq_length, max_packed_sequences, masked_lm_positions_size, enable_packed_data_mode:bool=False, input_file=None, batch_size=0):
        self.batch_size = batch_size
        self.max_pred_length = max_pred_length
        self.masked_lm_positions_size = masked_lm_positions_size
        self.max_seq_length = max_seq_length
        self.max_packed_sequences = max_packed_sequences
        self.enable_packed_data_mode = enable_packed_data_mode

    def __len__(self):
        # NOTE: We only need 2 iterations for warmup
        return 2 * self.batch_size

    def __getitem__(self, index):
        sequences = self.max_packed_sequences if self.enable_packed_data_mode else 1
        segment_size = self.max_seq_length

        next_sentence_positions = [0]

        if sequences > 1:
            next_sentence_positions.extend( random.sample(range(1, segment_size - 1), sequences - 1) )
            next_sentence_positions.sort()

        input_mask = []
        positions = []

        input_masks = next_sentence_positions[1:]
        input_masks.append(segment_size)
        last_size=0
        for segment, size in enumerate(input_masks):
            input_mask.extend([segment + 1] * (size - last_size))
            positions.extend(range(size - last_size))
            last_size = size

        segment_ids = []

        current_segment = 0
        current_segment_val = 0
        while current_segment < segment_size:
            new_segment = random.randint(10, 300)
            if new_segment + current_segment > segment_size:
                new_segment = segment_size - current_segment

            segment_ids.extend([current_segment_val] * new_segment)
            current_segment_val = (current_segment_val + 1) % 2
            current_segment += new_segment

        input_ids = [random.randint(100, 20000) for _ in range(segment_size)]

        next_sentence_labels = [random.randint(-1, 1) for _ in range(sequences)] if sequences > 1 else [0]

        masked_lm_positions = random.sample(range(1, segment_size - 1), self.masked_lm_positions_size)
        masked_lm_positions.sort()

        masked_lm_ids = [random.randint(1000, 20000) for _ in range(self.masked_lm_positions_size)]

        input_ids = torch.tensor(input_ids)
        segment_ids = torch.tensor(segment_ids)
        input_mask = torch.tensor(input_mask)
        positions = torch.tensor(positions)
        masked_lm_positions = torch.tensor(masked_lm_positions)
        masked_lm_ids = torch.tensor(masked_lm_ids)
        next_sentence_positions = torch.tensor(next_sentence_positions)
        next_sentence_labels = torch.tensor(next_sentence_labels)

        if self.enable_packed_data_mode:
            return [input_ids, segment_ids, input_mask, positions, masked_lm_positions, masked_lm_ids, next_sentence_positions, next_sentence_labels]
        else:
            return [input_ids, segment_ids, input_mask, masked_lm_positions, masked_lm_ids, next_sentence_labels]

class pretraining_dataset(Dataset):
    @staticmethod
    def name():
        return "Training"

    def __init__(self, max_pred_length, max_seq_length, max_packed_sequences, masked_lm_positions_size, enable_packed_data_mode:bool=False, input_file=None, batch_size=0):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        self.max_seq_length = max_seq_length
        self.masked_lm_positions_size = masked_lm_positions_size
        self.max_packed_sequences = max_packed_sequences
        f = h5py.File(input_file, "r")
        if enable_packed_data_mode:
            keys = ['input_ids', 'input_mask', 'segment_ids', 'positions',
                    'masked_lm_positions', 'masked_lm_ids',
                    'next_sentence_positions', 'next_sentence_labels', 'next_sentence_weights']
        else:
            keys = ['input_ids', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids', 'next_sentence_labels']
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()
        self.enable_packed_data_mode = enable_packed_data_mode

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):
        if self.enable_packed_data_mode:
            [input_ids, input_mask, segment_ids, positions,
             masked_lm_positions, masked_lm_ids,
             next_sentence_positions, next_sentence_labels, next_sentence_weights] = [torch.from_numpy(input[index].astype(np.int64)) for input in self.inputs]

            next_sentence_labels = (next_sentence_weights == 1) * next_sentence_labels + (next_sentence_weights == 0) * -1

            assert next_sentence_labels.shape[0] == self.max_packed_sequences, f"Expected sentence label tensor of size {self.max_packed_sequences}, got {next_sentence_labels.shape[0]}"
            assert masked_lm_positions.shape[0] == self.masked_lm_positions_size, f"Expected masked lm positions of size {self.masked_lm_positions_size}, got {masked_lm_positions.shape[0]}"

            return [input_ids, segment_ids, input_mask, positions, masked_lm_positions, masked_lm_ids, next_sentence_positions, next_sentence_labels]
        else:
            input_ids = np.zeros((self.max_seq_length)).astype(np.int64)
            input_mask= np.zeros((self.max_seq_length)).astype(np.int64)
            segment_ids=np.zeros((self.max_seq_length)).astype(np.int64)
            [_input_ids, _segment_ids, _masked_lm_positions, _masked_lm_ids, _next_sentence_labels] = [
                input[index].astype(np.int64) if indice < 4 else
                np.asarray(input[index].astype(np.int64)) for indice, input in enumerate(self.inputs)]

            input_mask_len = _input_ids.shape[-1]
            input_ids[:input_mask_len] = _input_ids
            input_mask[:input_mask_len] = np.ones((1,input_mask_len)).astype(np.int64)
            segment_ids[:input_mask_len] = _segment_ids

            pad_size = self.max_pred_length - _masked_lm_ids.shape[0]

            if pad_size > 0:
                masked_lm_ids = np.concatenate([_masked_lm_ids, np.zeros(pad_size)], axis = 0).astype(np.int64)
                masked_lm_positions = np.concatenate([_masked_lm_positions, np.zeros(pad_size)], axis = 0).astype(np.int64)
            else:
                masked_lm_ids = _masked_lm_ids.astype(np.int64)
                masked_lm_positions = _masked_lm_positions.astype(np.int64)

            next_sentence_labels = _next_sentence_labels
            return [torch.from_numpy(input_ids), torch.from_numpy(segment_ids),
                    torch.from_numpy(input_mask), torch.from_numpy(masked_lm_positions), torch.from_numpy(masked_lm_ids), torch.from_numpy(next_sentence_labels)]

class BertPretrainingCriterion(torch.nn.Module):
    def __init__(self, vocab_size, max_seq_length, batch_size, device, enable_packed_data_mode):
        super(BertPretrainingCriterion, self).__init__()
        self.mlm_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.nsp_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1 if enable_packed_data_mode else 0)
        self.vocab_size = vocab_size
        self.masked_lm_offsets = (max_seq_length * torch.arange(batch_size, dtype=torch.int64, device=device)).view(-1, 1)

    def forward(self, prediction_scores, seq_relationship_score, masked_lm_positions, masked_lm_ids, next_sentence_label):
        if masked_lm_positions is not None and masked_lm_ids is not None and next_sentence_label is not None:
            masked_lm_positions_ = masked_lm_positions + self.masked_lm_offsets
            masked_lm_positions_flat = masked_lm_positions_.view(-1)
            masked_lm_ids_flat = masked_lm_ids.view(-1)
            masked_prediction_scores = prediction_scores.view(-1, self.vocab_size)[masked_lm_positions_flat]
            masked_lm_loss = self.mlm_loss_fn(masked_prediction_scores, masked_lm_ids_flat)
            next_sentence_loss = self.nsp_loss_fn(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = (masked_lm_loss + next_sentence_loss).float()
            if self.training:
                return total_loss, None, None
            # Masked Language Model Accuracy
            valid_mask = (masked_lm_ids_flat != 0).int()
            num_valid = valid_mask.sum(dtype=torch.int64)
            mlm_predictions = masked_prediction_scores.argmax(dim=-1)
            mlm_acc = ((mlm_predictions == masked_lm_ids_flat) * valid_mask).sum(dtype=torch.float) / num_valid

            return total_loss, mlm_acc, num_valid
        else: #TODO: Handle this path for dense sequence output as well
            return prediction_scores.item(), seq_relationship_score.item()

def parse_arguments():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain .hdf5 files  for the task.")

    parser.add_argument("--config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The BERT model config")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument('--do_eval',
                        default=False,
                        action='store_true',
                        help='Enable eval every block')
    parser.add_argument("--eval_dir",
                        default=None,
                        type=str,
                        help="The eval data dir. Should contain .hdf5 files  for the task.")
    parser.add_argument("--num_eval_examples",
                        default=10000,
                        type=int,
                        help="number of eval examples to run eval on")
    parser.add_argument("--num_eval_workers",
                        default=8,
                        type=int,
                        help="numer of distributed eval workers")
    parser.add_argument("--eval_batch_size",
                        default=125,
                        type=int,
                        help="Total batch size for training.")

    ## Other parameters
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="The initial checkpoint to start training from.")
    parser.add_argument("--async_checkpoint", dest='async_checkpoint', action='store_true',
                        help="Enables usage of asynchronous checkpoint saving during training.")
    parser.add_argument("--no_async_checkpoint", dest='async_checkpoint', action='store_false',
                        help="Disables usage of asynchronous checkpoint saving during training.")
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_predictions_per_seq",
                        default=80,
                        type=int,
                        help="The maximum total of masked tokens in input sequence")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--lamb_beta_1",
                        default=0.9,
                        type=float,
                        help="Coefficient used for computing running averages of gradient (default: 0.9)")
    parser.add_argument("--lamb_beta_2",
                        default=0.999,
                        type=float,
                        help="Coefficient used for computing running averages of gradient square (default: 0.9)")
    parser.add_argument("--lamb_weight_decay",
                        default=0.01,
                        type=float,
                        help="Coefficient governing how much model weights vanish with each optimization step (default: 0.01)")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps",
                        default=1000,
                        type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.01,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=os.getenv('LOCAL_RANK', -1),
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--loss_scale',
                        type=float, default=0.0,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--log_freq',
                        type=float, default=1.0,
                        help='frequency of logging loss.')
    parser.add_argument('--checkpoint_activations',
                        default=False,
                        action='store_true',
                        help="Whether to use gradient checkpointing")
    parser.add_argument("--resume_from_checkpoint",
                        default=False,
                        action='store_true',
                        help="Whether to resume training from checkpoint.")
    parser.add_argument('--num_steps_per_checkpoint',
                        type=int,
                        default=100,
                        help="Number of update steps until a model checkpoint is saved to disk.")
    parser.add_argument('--skip_checkpoint',
                        default=False,
                        action='store_true',
                        help="Whether to save checkpoints")
    parser.add_argument("--checkpoint_filter",
                        default=None,
                        type=str,
                        help="Defines what to save in checkpoints")
    parser.add_argument('--phase2',
                        default=False,
                        action='store_true',
                        help="Whether to train with seq len 512")
    parser.add_argument('--allreduce_post_accumulation',
                        default=False,
                        action='store_true',
                        help="DEPRECATED. Use: --distribution_strategy allreduce_post_accumulation")
    parser.add_argument('--allreduce_post_accumulation_fp16',
                        default=False,
                        action='store_true',
                        help="Whether to do fp16 allreduce post accumulation.")
    parser.add_argument('--init_loss_scale',
                        type=int,
                        default=2**20,
                        help="Initial loss scaler value")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument('--json-summary', type=str, default="results/dllogger.json",
                        help='If provided, the json summary will be written to'
                             'the specified file.')
    parser.add_argument('--disable_progress_bar',
                        default=False,
                        action='store_true',
                        help='Disable tqdm progress bar')
    parser.add_argument('--steps_this_run', type=int, default=-1,
                        help='If provided, only run this many steps before exiting')
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether to use CPU when available")
    parser.add_argument("--use_habana",
                        action="store_true",
                        help="Whether not to use Habana device when available")
    parser.add_argument('--use_autocast', default='True', type=lambda x: x.lower() == 'true',
                        help='Enable torch.autocast')
    parser.add_argument("--use_fused_lamb",
                        action='store_true',
                        help='use FusedLamb optimizer')
    parser.add_argument("--use_lazy_mode",
                        default='True', type=lambda x: x.lower() == 'true',
                        help='Whether to run model in lazy or eager execution mode, default=True for lazy mode')
    parser.add_argument('--enable_packed_data_mode', default='True', type=lambda x: x.lower() == 'true',
                        help='enable/disable training with packed data. Default is True, --input_dir should be set accordingly')
    parser.add_argument('--tensorboard_dir', default=None, type=str,
                        help='tensorboard directory path: set to track metrics')
    parser.add_argument('--profile_steps', default=None, type=str,
                        help='step number or range of accumulation steps to profile, e.g.: 50,55')
    parser.add_argument('--enable_device_warmup', default='True', type=lambda x: x.lower() == 'true',
                        help='perform warmup sequence. Default is True')
    parser.add_argument('--max_packed_sequences', default=3, type=int,
                        help='maximum sequences packed into a sample. Required to be consistent among all input files. Default = 3. Used only in packed data mode')
    parser.add_argument('--masked_lm_positions_size', default=79, type=int,
                        help='Size of the masked_lm_positions, masked_lm_ids and masked_lm_weights. Required to be consistent among all input files. Default = 79.')
    parser.add_argument('--stop_threshold', default=0.720, type=float, help='MLperf Mask LM accuracy target. Default = 0.720')
    parser.add_argument('--samples_between_eval', default=150080, type=int, help='MLPerf Evaluation frequency in samples. Default = 150080')
    parser.add_argument('--num_warmup_steps', default=0, type=int, help='Number of warmup steps. Default = 0')
    parser.add_argument('--distribution_strategy', '--dist',
                        choices=['allreduce_post_accumulation', 'ddp', 'fastddp', 'gradientbuffer'],
                        default='gradientbuffer',
                        help="distribution strategy used to collect and all-reduce the gradients (default: gradientbuffer)")
    parser.add_argument('--allreduce_dtype',
                        choices=['bf16', 'fp32'],
                        default='fp32',
                        help="dtype of all-reduced gradient data")
    parser.add_argument('--use_hpu_graph', default='False', type=lambda x: x.lower() == 'true',
                        help='use HPU Graph feature to speed up the host op accumulation by recording it once, then replaying with each step')

    parser.set_defaults(async_checkpoint=True)

    args = parser.parse_args()

    if args.steps_this_run < 0:
        args.steps_this_run = args.max_steps

    return args

def unflatten_tensor(flat, tensor_list):
    outputs = []
    offset = 0
    for tensor in tensor_list:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return outputs

def update_tensors(grad_tensors, outputs):
    idx=0
    for grad in grad_tensors:
        grad.copy_(outputs[idx])
        idx+=1
    return outputs

def setup_training(args):

    if args.use_habana:
        device = torch.device("hpu")

        args.n_pu = 1
        from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu
        args.world_size, args.rank, args.local_rank = initialize_distributed_hpu()
        if args.local_rank != -1:
            torch.distributed.init_process_group('hccl',
                    rank=args.rank, world_size=args.world_size)

    elif args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        if device == torch.device("cuda"):
            args.n_pu = torch.cuda.device_count()
        else:
            args.n_pu = 1

        args.allreduce_post_accumulation_fp16 = False
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.n_pu = 1

    if args.gradient_accumulation_steps == 1:
        args.allreduce_post_accumulation_fp16 = False

    os.makedirs(os.path.dirname(args.json_summary), exist_ok=True)

    if is_main_process():
        dllogger.init(backends=[dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,
                                                           filename=args.json_summary),
                                dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE, step_format=format_step)])
    else:
        dllogger.init(backends=[dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,
                                                           filename=args.json_summary + "_" + str(get_rank()))])

    print("device: {} n_pu: {}, distributed training: {}".format(
        device, args.n_pu, bool(args.local_rank != -1)))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, batch size {} should be divisible".format(
            args.gradient_accumulation_steps, args.train_batch_size))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if args.enable_packed_data_mode:
        args.gradient_accumulation_steps = round(args.gradient_accumulation_steps / avg_seq_per_pack)

    if args.do_train:
        if not args.resume_from_checkpoint and os.path.exists(args.output_dir) and (
            os.listdir(args.output_dir) and any([i.startswith('ckpt') for i in os.listdir(args.output_dir)])):
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    if (not args.resume_from_checkpoint or not os.path.exists(args.output_dir)) and is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)

    return device, args

def remap_segmented_model_parameters(model_dict, config):
    res_dict = OrderedDict()
    for k in model_dict:
        if "N_Weight" in k or "iterCnt" in k or "M_Weight" in k or "N_Grad" in k or "M_Grad" in k:
            continue
        elif "intermediate.dense_act" in k or "pooler.dense_act" in k or "transform.dense_act" in k:
            new_k = k
        elif "intermediate.dense" in k or "pooler.dense" in k or "transform.dense" in k:
            new_k = k.replace("dense", "dense_act")
        else:
            new_k = k
        res_dict[new_k] = model_dict[k]
    model_dict.clear()
    return res_dict

def run_eval(args, model, eval_dataloader, device, criterion, num_eval_examples, htcore, first_eval=False, use_cache=False):
    model.eval()
    criterion.eval()

    total_eval_loss = torch.tensor([0.0], dtype=torch.float32, device=device)
    total_eval_mlm_acc = torch.tensor([0.0], dtype=torch.float32, device=device)
    total_masked = torch.tensor([0], dtype=torch.int32, device=device)

    # on first eval, load and cache data on GPU
    if first_eval and use_cache:
        for batch in eval_dataloader:
            #batch = preprocess_batch(args, *batch)
            cached_batches.append([t.to(device, non_blocking=True) for t in batch])

    cached_batches = []
    for batch in eval_dataloader:
        cached_batches.append([t.to(device, non_blocking=True) for t in batch])
    loss_list = []

    with torch.no_grad():
        for idx, batch in enumerate(cached_batches) if use_cache else enumerate(eval_dataloader):
            if not use_cache:
                batch = [t.to(device, non_blocking=True) for t in batch]
                input_ids, segment_ids, input_mask, masked_lm_positions, masked_lm_ids, next_sentence_labels = batch
                prediction_scores, seq_relationship_score = model(input_ids, segment_ids, input_mask)
                loss, mlm_acc, num_masked = criterion(prediction_scores, seq_relationship_score, masked_lm_positions, masked_lm_ids, next_sentence_labels)

                if args.use_autocast:
                    with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=False):
                        total_eval_loss += loss.to(torch.float32) * num_masked
                        total_eval_mlm_acc += mlm_acc.to(torch.float32)* num_masked

                else:
                    total_eval_loss += loss * num_masked
                    total_eval_mlm_acc += mlm_acc * num_masked

                total_masked += num_masked

                if args.use_lazy_mode:
                    htcore.mark_step()

    model.train()
    criterion.train()
    if torch.distributed.is_initialized():
        # Mask out the evaluation metrics obtained by shadow workers.
        if args.rank >= args.num_eval_workers:
            total_eval_loss *= 0
            total_eval_mlm_acc *= 0
            total_masked *= 0

        torch.distributed.all_reduce(total_eval_loss, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_eval_mlm_acc, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_masked, op=torch.distributed.ReduceOp.SUM)
    total_eval_mlm_acc = total_eval_mlm_acc.float()
    total_eval_loss = total_eval_loss.float()
    total_eval_mlm_acc /= total_masked
    total_eval_loss /= total_masked

    return total_eval_loss.item(), total_eval_mlm_acc.item()

def prepare_model_and_optimizer(args, device, worker_init):

    # Prepare model
    config = modeling.BertConfig.from_json_file(args.config_file)

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training
    model = modeling.BertForPreTraining(config)

    checkpoint = None
    global_step = 0
    if args.init_checkpoint:
        checkpoint=torch.load(args.init_checkpoint, map_location="cpu")["model"]
        checkpoint_remapped = remap_segmented_model_parameters(checkpoint, config)
        model.load_state_dict(checkpoint_remapped, strict=True)

    model.to(device)
    # BERT modeling  uses weight sharing between word embedding and prediction decoder.
    # So make sure the storage is pointing properly even after model is moved to device.
    if args.use_habana:
        model.cls.predictions.decoder.weight = model.bert.embeddings.word_embeddings.weight

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.lamb_weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    if args.use_habana:
        if args.use_fused_lamb:
            from fastlamb import FastLamb
            optimizer = FastLamb(optimizer_grouped_parameters, lr=args.learning_rate, betas=(args.lamb_beta_1, args.lamb_beta_2), sharded=False, use_lowp=True)
        else:
            optimizer = NVLAMB(
                        optimizer_grouped_parameters,
                        lr=args.learning_rate)
    else:
        if torch.cuda.is_available():
            optimizer = FusedLAMB(optimizer_grouped_parameters,
                              lr=args.learning_rate)
        else:
            optimizer = NVLAMB(
                        optimizer_grouped_parameters,
                        lr=args.learning_rate)

    lr_scheduler = PolyWarmUpScheduler(optimizer,
                                       warmup=args.warmup_proportion,
                                       total_steps=args.max_steps,
                                       degree=1.0)

    model.checkpoint_activations(args.checkpoint_activations)

    if args.resume_from_checkpoint:
        if args.phase2 or args.init_checkpoint:
            keys = list(checkpoint['optimizer']['state'].keys())
            #Override hyperparameters from previous checkpoint
            for key in keys:
                checkpoint['optimizer']['state'][key]['step'] = global_step
            for iter, item in enumerate(checkpoint['optimizer']['param_groups']):
                checkpoint['optimizer']['param_groups'][iter]['step'] = global_step
                checkpoint['optimizer']['param_groups'][iter]['t_total'] = args.max_steps
                checkpoint['optimizer']['param_groups'][iter]['warmup'] = args.warmup_proportion
                checkpoint['optimizer']['param_groups'][iter]['lr'] = args.learning_rate
        optimizer.load_state_dict(checkpoint['optimizer'])  # , strict=False)

    if args.local_rank != -1:
        if not args.use_habana:
            if args.distribution_strategy == 'allreduce_post_accumulation':
                flat_dist_call([param.data for param in model.parameters()], torch.distributed.broadcast, (0,) )
            elif args.distribution_strategy == 'ddp':
                model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size() * args.gradient_accumulation_steps)
            else:
                assert False, 'Unsupported distribution strategy.'
        else:
            if args.distribution_strategy == 'allreduce_post_accumulation':
                assert args.allreduce_dtype == 'fp32', 'Only all-reduce of dtype float32 is supported with allreduce_post_accumulation distribution strategy.'

            elif args.distribution_strategy == 'ddp':
                model = DDP(model, broadcast_buffers=False, bucket_cap_mb=1500)

                import habana_frameworks.torch.core as htcore

                def grad_norm_hook(
                    process_group, bucket
                ) -> torch.futures.Future[torch.Tensor]:
                    group_to_use = process_group if process_group is not None else torch.distributed.group.WORLD
                    world_size = group_to_use.size()

                    buffer = bucket.buffer()

                    if args.allreduce_dtype == 'bf16':
                        buffer = buffer.to(torch.bfloat16)

                    max_grad_norm = 1.0
                    clip_global_grad_norm = torch.ops.hpu.optimizer_lamb_fused_norm([buffer], max_grad_norm)

                    buffer.div_(clip_global_grad_norm * (world_size * args.gradient_accumulation_steps))

                    fut = torch.distributed.all_reduce(
                        buffer, group=group_to_use, async_op=True
                    ).get_future()

                    def finish_buffer(fut):
                        buffer = fut.value()[0]
                        if args.allreduce_dtype == 'bf16':
                            buffer.to(torch.float32)
                        return buffer

                    return fut.then(finish_buffer)

                model.register_comm_hook(None, grad_norm_hook)

            elif args.distribution_strategy == 'fastddp':
                assert args.gradient_accumulation_steps == 1, '"fastddp" distribution strategy support only 1 accumulation per step (2 in case of packed dataset).'

                from fastddp import FastDistributedDataParallel
                model = FastDistributedDataParallel(model,
                                                    fusion_buffer_dtype=torch.float32 if args.allreduce_dtype == 'fp32' else torch.bfloat16,
                                                    mark_step_on_gradients=[] if args.use_hpu_graph else [292],
                                                    use_hpu_graph=args.use_hpu_graph)

            elif args.distribution_strategy == 'gradientbuffer':
                from gradientbuffer import GradientBuffer
                global gradient_buffer
                gradient_buffer = GradientBuffer(model,
                                                 hook_autograd=False,
                                                 dtype=torch.float32 if args.allreduce_dtype == 'fp32' else torch.bfloat16,
                                                 allow_accumulation=True,
                                                 grad_div_factor=float(get_world_size() * args.gradient_accumulation_steps))

            else:
                assert False

            # Brodacast all the model parameters (it is not necessary when starting from the checkpoint, but it is cheap).
            for param in model.parameters():
                torch.distributed.broadcast(param.data, 0)

    elif args.n_pu > 1:
        model = torch.nn.DataParallel(model)

    # NOTE: eval dataset is unpacked
    #       and uses different batch size
    if args.do_train:
        packed = args.enable_packed_data_mode
        batch_size = args.train_batch_size
    else:
        packed = False
        batch_size = args.eval_batch_size

    criterion = BertPretrainingCriterion(config.vocab_size, args.max_seq_length, batch_size, device, packed)
    return model, optimizer, lr_scheduler, checkpoint, global_step, criterion

def take_optimizer_step(args, optimizer, model, overflow_buf, global_step):
    if not args.use_habana:
        if args.distribution_strategy == 'allreduce_post_accumulation':
            # manually allreduce gradients after all accumulation steps
            # check for Inf/NaN
            # 1. allocate an uninitialized buffer for flattened gradient
            loss_scale = 1
            master_grads = [p.grad for p in amp.master_params(optimizer) if p.grad is not None]
            flat_grad_size = sum(p.numel() for p in master_grads)
            allreduce_dtype = torch.float16 if args.allreduce_post_accumulation_fp16 else torch.float32
            flat_raw = torch.empty(flat_grad_size, device='cuda', dtype=allreduce_dtype)
            # 2. combine unflattening and predivision of unscaled 'raw' gradient
            allreduced_views = apex_C.unflatten(flat_raw, master_grads)
            overflow_buf.zero_()
            amp_C.multi_tensor_scale(65536,
                overflow_buf,
                [master_grads, allreduced_views],
                loss_scale / (get_world_size() * args.gradient_accumulation_steps))
            # 3. sum gradient across ranks. Because of the predivision, this averages the gradient
            torch.distributed.all_reduce(flat_raw)
            # 4. combine unscaling and unflattening of allreduced gradient
            overflow_buf.zero_()
            amp_C.multi_tensor_scale(65536,
                overflow_buf,
                [allreduced_views, master_grads],
                1./loss_scale)
            # 5. call optimizer step function
            optimizer.step()
            global_step += 1
            for param in model.parameters():
                param.grad = None
    else:
        if args.local_rank != -1:
            # Perform the all-reduce of all 398 gradients.
            # This process differs depending on the selected distribution strategy.

            if args.distribution_strategy == 'gradientbuffer':
                global gradient_buffer
                # Gradients are alredy fused and ready to be all-reduced.
                gradient_buffer.all_reduce()
                gradient_buffer.unfuse_gradients()

            elif args.distribution_strategy == 'fastddp':
                # All-reduce the FastDDP's fusion buffer, but do not unfuse the gradients.
                model.all_reduce_gradients(unfuse_gradients=False)

                # Instruct (once) the FastLamb optimizer to fetch gradients directly from FastDDP's fusion buffer.
                if optimizer.external_param_to_grad_map is None:
                    optimizer.external_param_to_grad_map = {}
                    from fastddp import _fusion_views
                    for param, (fused_grad_index, fused_grad) in zip(model.parameters(), sorted(_fusion_views.items())):
                        optimizer.external_param_to_grad_map[param] = fused_grad

            elif args.distribution_strategy == 'ddp':
                pass

            elif args.distribution_strategy == 'allreduce_post_accumulation':
                import habana_frameworks.torch.core as htcore
                grad_tensors = [param.grad for param in model.parameters() if param.grad is not None]
                max_grad_norm = 1.0
                clip_global_grad_norm = torch.ops.hpu.optimizer_lamb_fused_norm(grad_tensors, max_grad_norm)
                for g in reversed(grad_tensors):
                    g.div_(clip_global_grad_norm * (torch.distributed.get_world_size() * args.gradient_accumulation_steps))
                    torch.distributed.all_reduce(g, async_op=True)

        # Take the optimizer's step.
        optimizer.step()

        # Clear the gradients, unless on critical path (FastDDP+FastLamb).
        if args.local_rank == -1 or args.distribution_strategy != 'fastddp':
            optimizer.zero_grad(set_to_none=False)

        global_step += 1

    return global_step

def get_metadata_file_path(input_dir : str) -> str:
    norm_path = os.path.normpath(input_dir)
    head_tail = os.path.split(norm_path)
    metadata_file_name = head_tail[1]
    metadata_file_name = metadata_file_name + '_metadata.json'
    metadata_file_path = os.path.join(head_tail[0],metadata_file_name)
    return metadata_file_path

def read_avg_seq_per_sample(input_dir : str, max_sequence_length) -> float:
    metadata = None
    metadata_file_path = get_metadata_file_path(input_dir)
    print(f"Reading dataset metadata from: {metadata_file_path}")
    if os.path.exists(metadata_file_path):
        file_handle = open(metadata_file_path, mode='r')
        json_content = file_handle.read()
        metadata = json.loads(json_content)
    else:
        avg_seq_per_sample = defaultdict(None, {128 : 1.2, 512 : 2.0})[max_sequence_length]
        if avg_seq_per_sample is None:
            assert f"invalid max_sequence_length"

        print(f"Packed dataset metadata file not accessible, falling back to default values of avg_seq_per_sample={avg_seq_per_sample}")
        return avg_seq_per_sample

    avg_seq_per_sample_key = "avg_seq_per_sample"
    if metadata is not None and avg_seq_per_sample_key in metadata.keys():
        avg_seq_per_sample = metadata[avg_seq_per_sample_key]
    else:
        assert False, f"Key {avg_seq_per_sample_key} not present in packed dataset metadata file: {metadata_file_path}"
    print(f"AVG_SEQ_PER_SAMPLE: {avg_seq_per_sample}")
    return avg_seq_per_sample

def perform_training(args, device, dataset, step_limit, worker_init, htcore, profiler=None, summary_writer=None, checkpoint_queue=None, mllogger=None, mllog=None):

    starting_time = time.time()
    raw_train_start = None

    training_steps = 0
    average_loss = 0.0  # averaged loss every args.log_freq steps
    epoch = 0
    average_training_time_per_step = 0
    average_perf_per_step = 0
    loss_list = []
    iteration = 0

    block_timestamps = []

    ckpt_saver = None
    if checkpoint_queue is not None:
        def save_async_checkpoint(queue):
            while True:
                checkpoint, path = queue.get()
                if checkpoint is not None:
                    print("Saving checkpoint in separate process.")
                    torch.save(checkpoint, path)
                else:
                    return

        ckpt_saver = torch.multiprocessing.Process(target=save_async_checkpoint, args=(checkpoint_queue,))
        ckpt_saver.start()


    # Prepare optimizer
    model, optimizer, lr_scheduler, checkpoint, global_step, criterion = prepare_model_and_optimizer(args, device, worker_init)

    if is_main_process() and not args.enable_device_warmup:
        for model_weight, param in model.named_parameters():
            mllogger.event(key=mllog.constants.WEIGHTS_INITIALIZATION, metadata={'tensor': convert_weight_name(model_weight)})

    if not args.was_optimizer_initialized:
        mllogger.event(key=mllog.constants.OPT_LAMB_BETA_1, value=optimizer.param_groups[0]['betas'][0])
        mllogger.event(key=mllog.constants.OPT_LAMB_BETA_2, value=optimizer.param_groups[0]['betas'][1])
        mllogger.event(key=mllog.constants.OPT_LAMB_WEIGHT_DECAY, value=optimizer.param_groups[0]['weight_decay'])
        mllogger.event(key=mllog.constants.OPT_LAMB_EPSILON, value=optimizer.param_groups[0]['eps'])
        args.was_optimizer_initialized = True

    if not args.enable_device_warmup:
        mllogger.end(key=mllog.constants.INIT_STOP)
        mllogger.start(key=mllog.constants.RUN_START)

    model.train()
    criterion.train()

    if not args.enable_device_warmup:
        mllogger.start(
            key=mllog.constants.BLOCK_START,
            value=epoch + 1,
            metadata={
                mllog.constants.FIRST_EPOCH_NUM: int(iteration * args.num_steps_between_eval),
                mllog.constants.EPOCH_COUNT: int(args.num_steps_between_eval),
        })
    # Note: We loop infinitely over epochs, termination is handled via iteration count
    while True:
        restored_data_loader = None
        if not args.resume_from_checkpoint or epoch > 0 or (args.phase2 and global_step < 1) or args.init_checkpoint:
            if args.enable_packed_data_mode:
                files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if
                        os.path.isfile(os.path.join(args.input_dir, f))] # Packed files have no 'training' pre/postfix.
            else:
                files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if
                        os.path.isfile(os.path.join(args.input_dir, f)) and 'part' in f] # mlperf files have no 'training' pre/postfix
            files.sort()
            num_files = len(files)
            random.Random().shuffle(files)
            f_start_id = 0
        else:
            f_start_id = checkpoint['files'][0]
            files = checkpoint['files'][1:]
            args.resume_from_checkpoint = False
            num_files = len(files)
            # may not exist in all checkpoints
            epoch = checkpoint.get('epoch', 0)
            restored_data_loader = checkpoint.get('data_loader', None)

        if torch.distributed.is_initialized() and get_world_size() > num_files:
            remainder = get_world_size() % num_files
            data_file = files[(f_start_id*get_world_size()+get_rank() + remainder*f_start_id)%num_files]
        else:
            data_file = files[(f_start_id*get_world_size()+get_rank())%num_files]

        train_dataloader = restored_data_loader if restored_data_loader is not None else create_dataset(dataset, args, worker_init, input_file=data_file)
        restored_data_loader = None

        overflow_buf = None
        if args.distribution_strategy == 'allreduce_post_accumulation' and not args.use_habana:
            overflow_buf = torch.cuda.IntTensor([0])

        if args.use_hpu_graph:
            recorded_model_graph = None

        for f_id in range(f_start_id + 1 , len(files)):
            if get_world_size() > num_files:
                data_file = files[(f_id*get_world_size()+get_rank() + remainder*f_id)%num_files]
            else:
                data_file = files[(f_id*get_world_size()+get_rank())%num_files]

            train_iter = tqdm(train_dataloader, desc=dataset.name(), disable=args.disable_progress_bar) if is_main_process() else train_dataloader

            if raw_train_start is None:
                raw_train_start = time.time()

            # Iterate over all the batches of the input dataset.
            for batch in train_iter:

                training_steps += 1

                # Move the input batch tensors to HPU memory.
                batch = [t.to(device) for t in batch]
                if args.enable_packed_data_mode:
                    input_ids, segment_ids, input_mask, positions, masked_lm_positions, masked_lm_ids, next_sentence_positions, next_sentence_labels = batch
                else:
                    input_ids, segment_ids, input_mask, masked_lm_positions, masked_lm_ids, next_sentence_labels = batch

                if args.use_hpu_graph:
                    assert args.use_habana, 'HPU graph mode (`--use_hpu_graph true`) can be used only with HPU devices.'
                    assert args.enable_packed_data_mode, 'HPU graph mode (`--use_hpu_graph true`) can be used only with packed dataset mode.'
                    assert args.distribution_strategy == 'fastddp', 'HPU graph mode (`--use_hpu_graph true`) can be used only with FastDDP distribution strategy.'

                    if recorded_model_graph is None:
                        recorded_input_ids = torch.zeros_like(input_ids)
                        recorded_segment_ids = torch.zeros_like(segment_ids)
                        recorded_input_mask = torch.zeros_like(input_mask)
                        recorded_positions = torch.zeros_like(positions)
                        recorded_next_sentence_positions = torch.zeros_like(next_sentence_positions)
                        recorded_masked_lm_positions = torch.zeros_like(masked_lm_positions)
                        recorded_masked_lm_ids = torch.zeros_like(masked_lm_ids)
                        recorded_next_sentence_labels = torch.zeros_like(next_sentence_labels)

                        import habana_frameworks.torch as ht
                        recorded_model_graph = ht.hpu.HPUGraph()
                        s = ht.hpu.Stream()
                        with ht.hpu.stream(s):
                            recorded_model_graph.capture_begin()
                            with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=args.use_autocast):
                                prediction_scores, seq_relationship_score = model(
                                        input_ids=recorded_input_ids, token_type_ids=recorded_segment_ids, attention_mask=recorded_input_mask, enable_packed_data_mode=args.enable_packed_data_mode,
                                        positions=recorded_positions if args.enable_packed_data_mode else None,
                                        next_sentence_positions=recorded_next_sentence_positions if args.enable_packed_data_mode else None)
                                recorded_loss, _, _ = criterion(
                                    prediction_scores, seq_relationship_score, recorded_masked_lm_positions, recorded_masked_lm_ids, recorded_next_sentence_labels)
                            recorded_loss.backward()
                            recorded_model_graph.capture_end()

                    recorded_input_ids.copy_(input_ids)
                    recorded_segment_ids.copy_(segment_ids)
                    recorded_input_mask.copy_(input_mask)
                    recorded_positions.copy_(positions)
                    recorded_next_sentence_positions.copy_(next_sentence_positions)
                    recorded_masked_lm_positions.copy_(masked_lm_positions)
                    recorded_masked_lm_ids.copy_(masked_lm_ids)
                    recorded_next_sentence_labels.copy_(next_sentence_labels)

                    recorded_model_graph.replay()

                    loss = recorded_loss.detach()

                    divisor = args.gradient_accumulation_steps

                else:
                    with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=args.use_autocast):
                        if args.local_rank != -1 and (args.distribution_strategy == 'ddp') \
                                    and (training_steps % args.gradient_accumulation_steps != 0):
                            with model.no_sync():
                                prediction_scores, seq_relationship_score = model(
                                    input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, enable_packed_data_mode=args.enable_packed_data_mode,
                                    positions=positions if args.enable_packed_data_mode else None,
                                    next_sentence_positions=next_sentence_positions if args.enable_packed_data_mode else None)
                        else:
                            prediction_scores, seq_relationship_score = model(
                                    input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, enable_packed_data_mode=args.enable_packed_data_mode,
                                    positions=positions if args.enable_packed_data_mode else None,
                                    next_sentence_positions=next_sentence_positions if args.enable_packed_data_mode else None)

                        loss,_,_ = criterion(
                            prediction_scores, seq_relationship_score, masked_lm_positions, masked_lm_ids, next_sentence_labels)
                        if args.n_pu > 1:
                            loss = loss.mean()  # mean() to average on multi-pu.

                    divisor = args.gradient_accumulation_steps

                    if not args.use_habana:
                        if args.gradient_accumulation_steps > 1:
                            if not args.allreduce_post_accumulation:
                                # this division was merged into predivision
                                with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=args.use_autocast):
                                    loss = loss / args.gradient_accumulation_steps
                                divisor = 1.0
                    else:
                        # loss is not divided by args.gradient_accumulation_steps, as this division is part of pre-division of all distribution strategies.
                        pass

                    loss.backward()

                    if args.distribution_strategy == 'gradientbuffer':
                        gradient_buffer.fuse_gradients()

                    if args.use_lazy_mode:
                        htcore.mark_step()

                loss_list.append(loss)

                if training_steps % args.gradient_accumulation_steps == 0:
                    lr_scheduler.step()  # learning rate warmup
                    global_step = take_optimizer_step(args, optimizer, model, overflow_buf, global_step)
                elif args.distribution_strategy == 'gradientbuffer':
                    # When using GradientBuffer, gradients have to be cleared after each accumulation, not only after the optimizer's step.
                    optimizer.zero_grad()

                if args.use_lazy_mode:
                    htcore.mark_step()

                if training_steps % args.gradient_accumulation_steps == 0 and profiler is not None:
                    profiler.step()

                global has_training_finished
                finish = global_step >= step_limit or timeout_sent or has_training_finished
                log_status = training_steps % (args.log_freq * args.gradient_accumulation_steps) == 0

                create_checkpoint = (finish or training_steps % (args.num_steps_per_checkpoint * args.gradient_accumulation_steps) == 0) \
                                    and not args.skip_checkpoint \
                                    and checkpoint_queue is not None

                if finish or log_status:
                    for loss_t in loss_list:
                        with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=args.use_autocast):
                            average_loss += loss_t.item()
                    loss_list.clear()
                    train_time = time.time() - starting_time
                    starting_time = time.time()
                    average_training_time_per_step = train_time/(args.gradient_accumulation_steps * args.log_freq)
                    average_perf_per_step = args.train_batch_size*avg_seq_per_pack/average_training_time_per_step

                if finish:
                    train_time_raw = time.time() - raw_train_start
                    last_num_steps = int(training_steps / args.gradient_accumulation_steps) % args.log_freq
                    last_num_steps = args.log_freq if last_num_steps == 0 else last_num_steps
                    with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=args.use_autocast):
                        average_loss = average_loss / (last_num_steps * divisor)
                        average_loss = torch.tensor(average_loss, dtype=torch.float32).to(device)
                        if (torch.distributed.is_initialized()):
                            average_loss /= get_world_size()
                            torch.distributed.barrier()
                            torch.distributed.all_reduce(average_loss)
                        final_loss = average_loss.item()
                    if is_main_process():
                        train_iter.update(1)
                        train_iter.close()
                        dllogger.log(step=(epoch, global_step, ), data={"final_loss": final_loss,
                                                                        "average_training_time_step": average_training_time_per_step,
                                                                        "average_perf_per_step": average_perf_per_step})
                elif log_status:
                    if is_main_process():
                        dllogger.log(step=(epoch, global_step, ), data={"average_loss": average_loss / (args.log_freq * divisor),
                                                                        "step_loss": loss.item() * args.gradient_accumulation_steps / divisor,
                                                                        "learning_rate": optimizer.param_groups[0]['lr'],
                                                                        "average_training_time_step": average_training_time_per_step,
                                                                        "average_perf_per_step": average_perf_per_step})
                        if summary_writer is not None:
                            summary_writer.add_scalar('average_loss', average_loss / (args.log_freq * divisor), global_step)
                            summary_writer.add_scalar('average_training_time_per_step', average_training_time_per_step, global_step)
                            summary_writer.add_scalar('average_perf_per_step', average_perf_per_step, global_step)
                            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    average_loss = 0

                if create_checkpoint:
                    if args.local_rank == 0:
                        # Save a trained model
                        dllogger.log(step="PARAMETER", data={"checkpoint_step": global_step})
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Only save the model it-self
                        output_save_file = os.path.join(args.output_dir, "ckpt-{}.pt".format(global_step))
                        checkpoint_dict ={}
                        if args.do_train:
                            if args.use_habana or args.no_cuda:
                                if args.async_checkpoint:
                                    model_state_dict = OrderedDict({k:v.to('cpu') for k, v in model_to_save.state_dict().items()})
                                else:
                                    model_state_dict = model_to_save.state_dict()
                                checkpoint_dict = {'model': model_state_dict,
                                            'optimizer': optimizer.state_dict(),
                                            'files': [f_id] + files,
                                            'epoch': epoch,
                                            'data_loader': None if global_step >= args.max_steps else train_dataloader}
                            else:
                                checkpoint_dict = {'model': model_to_save.state_dict(),
                                            'optimizer': optimizer.state_dict(),
                                            'master params': list(amp.master_params(optimizer)),
                                            'files': [f_id] + files,
                                            'epoch': epoch,
                                            'data_loader': None if global_step >= args.max_steps else train_dataloader}

                            if args.checkpoint_filter is not None:
                                checkpoint_dict = dict(filter(lambda item: re.match(args.checkpoint_filter, item[0]), checkpoint_dict.items()))

                            if args.async_checkpoint:
                                checkpoint_queue.put((checkpoint_dict, output_save_file))
                            else:
                                torch.save(checkpoint_dict, output_save_file)

                    iteration = training_steps / args.num_steps_per_checkpoint
                    mllogger.end(key=mllog.constants.BLOCK_STOP,
                                    value=int(iteration),
                                    metadata={mllog.constants.FIRST_EPOCH_NUM: int(iteration * args.num_steps_between_eval)})
                    block_timestamps.append(int(time.time()*1e3))
                    if not finish:
                                mllogger.start(key=mllog.constants.BLOCK_START,
                                                value=int(iteration + 1),
                                                metadata={mllog.constants.FIRST_EPOCH_NUM: int(iteration * args.num_steps_between_eval), mllog.constants.EPOCH_COUNT: int(args.num_steps_between_eval)})

                # Exiting the training due to hitting max steps, or being sent a
                # timeout from the cluster scheduler
                if finish:
                    del train_dataloader
                    if ckpt_saver is not None:
                        checkpoint_queue.put((None, None))
                        ckpt_saver.join()
                    return final_loss, train_time_raw, global_step, block_timestamps

            del train_dataloader
            # Make sure pool has finished and switch train_dataloader
            # NOTE: Will block until complete
            train_dataloader = create_dataset(dataset, args, worker_init, input_file=data_file)

        epoch += 1

def main():
    global timeout_sent
    global avg_seq_per_pack
    global has_training_finished
    has_training_finished = False

    e2e_start = time.time()
    args = parse_arguments()

    if is_main_process and args.async_checkpoint:
        checkpoint_queue = torch.multiprocessing.Queue()

    worker_init = None
    if args.enable_packed_data_mode:
        avg_seq_per_pack = read_avg_seq_per_sample(args.input_dir, args.max_seq_length)
    else:
        warnings.warn("--enable_packed_data_mode flag will be deprecated and usage of packed and unpacked dataset"
                      " will be decided based on metadata file availability at input_dir")
        avg_seq_per_pack = 1.0

    device, args = setup_training(args)

    assert not args.allreduce_post_accumulation, "Parameter --allreduce_post_accumulation is deprecated. Please use: --distribution_strategy allreduce_post_accumulation"

    if args.use_habana:
        if args.use_lazy_mode:
            assert os.getenv('PT_HPU_LAZY_MODE') == '1', f"use_lazy_mode == True, but PT_HPU_LAZY_MODE={os.getenv('PT_HPU_LAZY_MODE')}"
            try:
                import habana_frameworks.torch.core as htcore
                import habana_frameworks.torch.hpu as hthpu
                hthpu.enable_dynamic_shape()
            except ImportError:
                assert False, "Could Not import habana_frameworks.torch.core"
        else:
            assert os.getenv('PT_HPU_LAZY_MODE') == '0' or os.getenv('PT_HPU_LAZY_MODE')== '2', f"args.use_lazy_mode == False, but PT_HPU_LAZY_MODE={os.getenv('PT_HPU_LAZY_MODE')}"
    else:
        args.use_lazy_mode = False
        htcore = None

    dllogger.log(step="PARAMETER", data={"Config": [str(args)]})

    training_results = None
    block_timestamps = None
    args.global_batch_size = args.train_batch_size * args.gradient_accumulation_steps * avg_seq_per_pack * torch.distributed.get_world_size()
    args.num_steps_between_eval = math.ceil(args.samples_between_eval / args.global_batch_size)
    mllogger, mllog = get_mllog_mlloger(args)

    if args.do_train:
        mllogger.event(key=mllog.constants.SUBMISSION_BENCHMARK, value=mllog.constants.BERT)
        mllogger.event(key=mllog.constants.SUBMISSION_ORG, value='Habana')
        mllogger.event(key=mllog.constants.SUBMISSION_DIVISION, value='closed')
        mllogger.event(key=mllog.constants.SUBMISSION_PLATFORM, value='gaudi-8')
        mllogger.event(key=mllog.constants.SUBMISSION_STATUS, value='onprem')
        mllogger.event(key=mllog.constants.CACHE_CLEAR)
        mllogger.start(key=mllog.constants.INIT_START)
        mllogger.event(key=mllog.constants.GLOBAL_BATCH_SIZE, value=int(args.global_batch_size))
        mllogger.event(key=mllog.constants.TRAIN_SAMPLES, value=args.global_batch_size * args.max_steps)
        mllogger.event(key=mllog.constants.MAX_SEQUENCE_LENGTH, value=args.max_seq_length)
        mllogger.event(key='max_predictions_per_seq', value=args.max_predictions_per_seq)
        mllogger.event(key=mllog.constants.GRADIENT_ACCUMULATION_STEPS, value=args.gradient_accumulation_steps)
        mllogger.event(key=mllog.constants.OPT_LR_TRAINING_STEPS, value=int(args.max_steps))
        mllogger.event(key=mllog.constants.START_WARMUP_STEP, value=0)
        mllogger.event(key=mllog.constants.OPT_BASE_LR, value=args.learning_rate)
        mllogger.event(key=mllog.constants.EVAL_SAMPLES, value=args.num_eval_examples)
        mllogger.event(key=mllog.constants.NUM_WARMUP_STEPS, value=args.num_warmup_steps)
        mllogger.event(key=mllog.constants.OPT_LR_WARMUP_STEPS, value=args.num_warmup_steps)

        if is_main_process():
            dllogger.log(step="PARAMETER", data={"train_start": True})
            dllogger.log(step="PARAMETER", data={"batch_size_per_pu": args.train_batch_size})
            dllogger.log(step="PARAMETER", data={"learning_rate": args.learning_rate})

        # Provide support for TensorBoard (whether required)
        if args.tensorboard_dir is not None:
            from torch.utils.tensorboard import SummaryWriter
            filename_suffix = f"rank_{args.local_rank if args.local_rank != -1 else 0}"
            summary_writer = SummaryWriter(args.tensorboard_dir, filename_suffix=filename_suffix)
        else:
            summary_writer = None

        # Provide support for Synapse profiler
        if args.profile_steps is not None:
            assert args.tensorboard_dir is not None, "Enabling the profiling with 'profile_steps' parameter also requires specifying 'tensorboard_dir'"
            step_words = args.profile_steps.split(",")
            if len(step_words) == 1:
                step_words = args.profile_steps.split(":")
            start_step = int(step_words[0])
            end_step = int(step_words[1])
            if len(step_words) == 1:
                active_steps = 1
            elif len(step_words) == 2:
                active_steps = end_step - start_step + 1
            else:
                assert False, f"'profile_steps' must contain an integer or a pair of integers separated by ',' or ':'"
            MAX_WARMUP_STEPS = 2
            warmup_steps = min(MAX_WARMUP_STEPS, start_step)
            wait_steps = start_step - warmup_steps

            worker_name = f"rank_{args.local_rank if args.local_rank != -1 else 0}"

            if 'HABANA_PROFILE' in os.environ and os.environ['HABANA_PROFILE'] == 'profile_api' and 'HABANA_PROF_CONFIG' in os.environ and os.environ['HABANA_PROF_CONFIG'] != None:
                profiler = SynapseProfilerAdatper(steps = (start_step, end_step), node_rank = get_rank())
            else:
                profiler = torch.profiler.profile(
                    activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU),
                    schedule=torch.profiler.schedule(wait=wait_steps, warmup=warmup_steps, active=active_steps),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(args.tensorboard_dir, worker_name=worker_name, use_gzip=True),
                    record_shapes=True,
                    with_stack=True)

            profiler.start()
        else:
            profiler = None

        try:
            args.was_optimizer_initialized = False
            if args.enable_device_warmup:
                perform_training(args, device, warmup_dataset, 2, worker_init, htcore, mllog=mllog, mllogger=mllogger)
                args.enable_device_warmup = False
            training_results = perform_training(args, device, pretraining_dataset, args.steps_this_run, worker_init, htcore, profiler, summary_writer, checkpoint_queue, mllog=mllog, mllogger=mllogger)

        finally:
            if profiler is not None:
                profiler.step()
                profiler.stop()

        if training_results:
            final_loss, train_time_raw, global_step, block_timestamps = training_results

            pu_count = args.n_pu

            if torch.distributed.is_initialized():
                pu_count = get_world_size()
            if is_main_process():
                e2e_time = time.time() - e2e_start
                training_perf = args.train_batch_size * args.gradient_accumulation_steps * pu_count * avg_seq_per_pack\
                                * global_step / train_time_raw
                dllogger.log(step=tuple(), data={"e2e_train_time": e2e_time, "training_sequences_per_second": training_perf,
                                                "final_loss": final_loss, "raw_train_time (min)": train_time_raw / 60 })

    if args.do_eval:

        block_number = 1
        block_timestamp = lambda block_number : block_timestamps[int(block_number)-1] if block_timestamps is not None else 0

        # Flags required by eval
        args.do_train = False
        args.allreduce_post_accumulation = True

        config = modeling.BertConfig.from_json_file(args.config_file)
        eval_dataloader = create_eval_dataset(args, worker_init)
        torch.distributed.barrier()

        eval_avg_mlm_accuracy = 0
        for ckpt_file in sorted(filter(lambda file : file.endswith('.pt'), os.listdir(args.output_dir)), key = lambda x: int(x.split("-")[1].split(".")[0])):
            ckpt_file_path = os.path.join(args.output_dir, ckpt_file)

            block_number = int(ckpt_file.split("-")[1].split(".")[0])
            block_number /= int(args.num_steps_per_checkpoint)
            if is_main_process():
                print("Eval for block [", int(block_number), "]:", ckpt_file_path)

            if os.path.isfile(ckpt_file_path) and 'ckpt' in ckpt_file:
                model, _, _, _, _, criterion = prepare_model_and_optimizer(args, device, worker_init)
                checkpoint=torch.load(ckpt_file_path, map_location="cpu")["model"]
                checkpoint_remapped = remap_segmented_model_parameters(checkpoint, config)

                model.load_state_dict(checkpoint_remapped, strict=True)
                eval_iter = tqdm(eval_dataloader, desc="Iteration", disable=args.disable_progress_bar) if is_main_process() else eval_dataloader
                eval_avg_loss, eval_avg_mlm_accuracy = run_eval(args, model, eval_iter, device, criterion, args.num_eval_examples, htcore,
                        first_eval=True, use_cache=False)

                has_training_finished = bool(eval_avg_mlm_accuracy >= args.stop_threshold)
                mllogger.event(key=mllog.constants.EVAL_ACCURACY,
                                    value=eval_avg_mlm_accuracy,
                                    time_ms=block_timestamp(block_number),
                                    metadata={'epoch_num': int((block_number)*args.samples_between_eval), 'epoch_count': int(block_number)})
                if is_main_process():
                    print("eval_loss", eval_avg_loss, "eval_mlm_accuracy", eval_avg_mlm_accuracy)

                if has_training_finished:
                    break

        if eval_avg_mlm_accuracy is not None and has_training_finished is True:
            if eval_avg_mlm_accuracy >= args.stop_threshold:
                    mllogger.end(key=mllog.constants.RUN_STOP, value=eval_avg_mlm_accuracy, time_ms=block_timestamp(block_number), metadata={'epoch_num': int((block_number)*args.samples_between_eval), 'epoch_count': int(block_number), 'status': 'success'})
            else:
                    mllogger.end(key=mllog.constants.RUN_STOP, value=eval_avg_mlm_accuracy, time_ms=block_timestamp(block_number), metadata={'epoch_num': int((block_number)*args.samples_between_eval), 'epoch_count': int(block_number), 'status': 'fail'})
        else:
            mllogger.end(key=mllog.constants.RUN_STOP, value=eval_avg_mlm_accuracy,time_ms=block_timestamp(block_number), metadata=None)

    dllogger.flush()

if __name__ == "__main__":
    main()
