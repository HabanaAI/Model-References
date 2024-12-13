# coding=utf-8
# Copyright (c) 2021, Habana Labs Ltd.  All rights reserved.
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# ==================
import os
import time
import argparse
import random
import h5py
from tqdm import tqdm, trange
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Dataset
import sys
import signal
import glob
import shutil

import modeling
from schedulers import PolyWarmUpScheduler

from utils import (is_main_process, format_step, get_world_size, get_rank, fix_tensor_numpy, get_local_rng_state,
                   set_local_rng_state)

from lamb import NVLAMB
from lans import LANS
from lamb_exp import NVLAMB_EXP
from torch.optim.adam import Adam as Adam
from torch.optim.adamw import AdamW as AdamW

try:
    from apex import optimizers
except ImportError:
    if torch.cuda.is_available():
        raise ImportError("Please install apex from "
                          "https://www.github.com/nvidia/apex")

import dllogger
from concurrent.futures import ProcessPoolExecutor

import deepspeed
from deepspeed.utils import log_dist
from deepspeed.ops.lamb.fused_lamb import FusedLamb as DeepSpeedFusedLamb
from contextlib import nullcontext
import json

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

skipped_steps = 0

# Track whether a SIGTERM (cluster time up) has been handled
timeout_sent = False

OPTIMIZERS_CUDA_ONLY = ('fused_lamb', 'ds_fused_lamb')
OPTIMIZERS_ALL = ('nvlamb', 'nvlamb_exp', 'adam', 'adamw', 'fused_adamw', 'lans') + OPTIMIZERS_CUDA_ONLY


# handle SIGTERM sent from the scheduler and mark so we
# can gracefully save & exit
def signal_handler(sig, frame):
    global timeout_sent
    timeout_sent = True


signal.signal(signal.SIGTERM, signal_handler)

# Dummy class for checkpoint backward compatibility
class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed


def create_pretraining_dataset(input_file, max_pred_length, shared_list, args):
    train_data = pretraining_dataset(input_file=input_file, max_pred_length=max_pred_length)
    train_sampler = RandomSampler(train_data) if not args.disable_random_sampler else None
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  num_workers=0,
                                  pin_memory=True)
    return train_dataloader, input_file


class pretraining_dataset(Dataset):

    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        keys = ['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids',
                'next_sentence_labels']
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):

        [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels] = [
            torch.from_numpy(input[index].astype(np.int64)) if indice < 5 else torch.from_numpy(
                np.asarray(input[index].astype(np.int64))) for indice, input in enumerate(self.inputs)]

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_pred_length
        # store number of  masked tokens in index
        padded_mask_indices = (masked_lm_positions == 0).nonzero()
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        return [input_ids, segment_ids, input_mask,
                masked_lm_labels, next_sentence_labels]


class BertPretrainingCriterion(torch.nn.Module):
    def __init__(self, vocab_size, run_loss_in_fp32=True):
        super(BertPretrainingCriterion, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size
        self.loss_in_fp32 = run_loss_in_fp32

    def forward(self, prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels):
        if self.loss_in_fp32:
            prev_dtype = prediction_scores.dtype
            prediction_scores = prediction_scores.float()
            seq_relationship_score = seq_relationship_score.float()
        masked_lm_loss = self.loss_fn(prediction_scores.view(-1, self.vocab_size), masked_lm_labels.view(-1))
        next_sentence_loss = self.loss_fn(seq_relationship_score.view(-1, 2), next_sentence_labels.view(-1))
        total_loss = masked_lm_loss + next_sentence_loss
        if self.loss_in_fp32:
            total_loss = total_loss.to(prev_dtype)
        return total_loss


def zero_optimization(ds_cfg):
    with open(ds_cfg, 'r') as ds_cfg:
        data = json.load(ds_cfg)
        if not ('zero_optimization' in data.keys()):
            return False
        elif not ('stage' in data['zero_optimization']):
            return False
        return data['zero_optimization']['stage'] > 0

# TODO SW-96497: remove this WA when SW-96431 is resolved
def bfloat16_enabled(ds_cfg):
    with open(ds_cfg, 'r') as ds_cfg:
        data = json.load(ds_cfg)
        if not ('bf16' in data.keys()):
            return False
        elif not ('enabled' in data['bf16']):
            return False
        return data['bf16']['enabled']

def is_cpu_offload_optimizer(ds_cfg):
    with open(ds_cfg, 'r') as ds_cfg:
        data = json.load(ds_cfg)
        if not ('zero_optimization' in data.keys()):
            return False
        elif not ('offload_optimizer' in data['zero_optimization']):
            return False
        elif not ('device' in data['zero_optimization']['offload_optimizer']):
            return False
        offload_device = data['zero_optimization']['offload_optimizer']['device'].lower()
        return offload_device == 'cpu' or offload_device == 'nvme'

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
    parser.add_argument("--bert_model", default="bert-large-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
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
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
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
    parser.add_argument("--scheduler_degree",
                        default=1.0,
                        type=float,
                        help="Degree to use in the PolyWarmUpScheduler after warmup. "
                             "E.g., base_lr * ((1.0 - progress) ** degree)")
    parser.add_argument("--constant_proportion",
                        default=0.0,
                        type=float,
                        help="Proportion of training to perform constant learning rate after the learning rate warmup phase.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--loss_scale',
                        type=float, default=0.0,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--log_freq',
                        type=float, default=1.0,
                        help='frequency of logging loss.')
    parser.add_argument("--resume_from_checkpoint",
                        default=False,
                        action='store_true',
                        help="Whether to resume training from checkpoint.")
    parser.add_argument('--resume_step',
                        type=int,
                        default=-1,
                        help="Step to resume training from.")
    parser.add_argument('--num_steps_per_checkpoint',
                        type=int,
                        default=100,
                        help="Number of update steps until a model checkpoint is saved to disk.")
    parser.add_argument('--skip_checkpoint',
                        default=False,
                        action='store_true',
                        help="Whether to save checkpoints")
    parser.add_argument('--phase2',
                        default=False,
                        action='store_true',
                        help="Whether to train with seq len 512")
    parser.add_argument('--phase1_end_step',
                        type=int,
                        default=7038,
                        help="Number of training steps in Phase1 - seq len 128")
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
    parser.add_argument("--use_env",
                        action='store_true',
                        help="Whether to read local rank from ENVVAR")
    parser.add_argument('--disable_progress_bar',
                        default=False,
                        action='store_true',
                        help='Disable tqdm progress bar')
    parser.add_argument('--steps_this_run', type=int, default=-1,
                        help='If provided, only run this many steps before exiting')
    parser.add_argument("--optimizer", choices=OPTIMIZERS_ALL, default=None,
                        help='configure optimizer. if not configured, optimizer will be taken from config file')
    parser.add_argument("--betas",
                        default=None,
                        nargs='+',
                        type=float,
                        help="Optimizer beta1 beta2. If not provided will use optimizer defaults")
    # TODO: enhance use_lr_scheduler to accept scheduler type
    parser.add_argument("--use_lr_scheduler",
                        action='store_true',
                        help='configure PolyWarmUpScheduler')
    parser.add_argument("--disable_random_sampler",
                        action='store_true',
                        help='if true, will not use RandomSampler - for debugging')
    parser.add_argument("--use_ds_lamb_bias_correction",
                        action='store_true',
                        help='if true, will use DS FusedLamb bias correction mode. '
                             'Applicable only for optimizer=nvlamb_exp')
    parser.add_argument("--run_loss_in_fp32",
                        action='store_true',
                        help='if true, loss calculation will be forced to fp32')
    parser.add_argument("--checkpoint_activations",
                        action='store_true',
                        help='if true, will activate checkpoint activations')
    parser.add_argument("--checkpoint_activations_interval",
                        default=1,
                        type=int,
                        help='Number of layers between checkpoint activations')
    parser.add_argument("--log_model_inputs",
     action="store_true",
      help="If set, log model\'s inputs for configured iterations")

    parser.add_argument("--log_fwd_activations",
     action="store_true",
      help="If set, log model\'s nn.Module forward activations for configured iterations")

    parser.add_argument("--log_bwd_grads",
     action="store_true",
      help="If set, log model\'s nn.Module backward gradients for configured iterations")

    parser.add_argument("--tensor_logger_max_iterations",
     type=int,
     default=0,
     help="Sets the maximum number of iterations to capture. If 0, disable tensor logger")

    parser.add_argument("--tensor_logger_path",
     type=str,
     default=None,
     help="Path for saving tensor logger captured tensors file")

    parser.add_argument("--profile",
     type=str,
     default=None,
     choices=['pt', 'pt-full', 'hltv'],
     help="Enable profiling")

    parser.add_argument("--profile_steps",
     type=str,
     default='2,3',
     help="Which steps to profile. Format: <start step>,<end step>")

    parser.add_argument("--enable_torch_compile",
     default=False,
     action='store_true',
     help='Enable compilation of the model using torch.compile')

    parser.add_argument("--enable_compiled_autograd",
     default=False,
     action='store_true',
     help='Enable compiled autograd')

    parser.add_argument("--enable_torch_compile_optimizer",
     default=False,
     action='store_true',
     help='Enable compilation of the optimizer step with torch compile')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    if args.steps_this_run < 0:
        args.steps_this_run = args.max_steps

    if not args.deepspeed:
        raise ValueError("This script is supported for DeepSpeed only. Please use --deepspeed.")

    if args.use_ds_lamb_bias_correction and not args.optimizer == 'nvlamb_exp':
        raise ValueError("Using --use_ds_lamb_bias_correction requires --optimizer=nvlamb_exp.")

    args.zero_optimization = zero_optimization(args.deepspeed_config)

    # TODO SW-96497: remove this WA when SW-96431 is resolved
    args.bfloat16_enabled = bfloat16_enabled(args.deepspeed_config)

    args.use_hpu = deepspeed.accelerator.get_accelerator().device_name() == "hpu"

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

on_step_begin = []
on_step_end = []

def trigger(phase):
    [f() for f in phase]

def setup_profiler(args, device):
    if args.profile is None:
        return

    start_step, end_step = map(int, args.profile_steps.split(','))
    active_steps = end_step - start_step + 1
    warmup_steps = start_step
    cur_step = 0

    def on_step_begin_fn():
        nonlocal cur_step
        cur_step = cur_step + 1
    on_step_begin.append(on_step_begin_fn)

    def when(cond, clbk):
        def fn():
            if cond():
                clbk()
        return fn


    def is_start_step():
        return cur_step == start_step

    def is_end_step():
        return cur_step == end_step

    def is_capture_step():
        return cur_step>=start_step and cur_step<=end_step

    if args.profile.startswith('pt'):
        schedule = torch.profiler.schedule(wait=0, warmup=0, active=active_steps, repeat=1)
        activities = [torch.profiler.ProfilerActivity.CPU]
        activities.extend([torch.profiler.ProfilerActivity.HPU] if device.type=="hpu" else [])
        activities.extend([torch.profiler.ProfilerActivity.CUDA] if device.type=="cuda" else [])
        full = args.profile == 'pt-full'

        profiler = torch.profiler.profile(
            schedule=schedule,
            activities=activities,
            on_trace_ready=torch.profiler.tensorboard_trace_handler('.', use_gzip=False),
            with_stack=full)

        on_step_begin.append(when(is_start_step, profiler.start))
        on_step_end.append(when(is_capture_step, profiler.step))
        on_step_end.append(when(is_end_step, profiler.stop))

    elif args.profile == 'hltv':
        sys.path.append(os.environ['PYTORCH_MODULES_ROOT_PATH'])
        from topologies.tools import SynapseProfilerApi, TraceType
        api = SynapseProfilerApi()

        on_step_begin.append(when(is_start_step, lambda: api.profiler_start(TraceType.TraceAll, 0)))
        on_step_end.append(when(is_end_step, lambda: hpu.synchronize()))
        on_step_end.append(when(is_end_step, lambda: api.profiler_stop(TraceType.TraceAll, 0)))
        on_step_end.append(when(is_end_step, lambda: api.profiler_get_trace_json(TraceType.TraceAll, 0)))

def setup_training(args):
    if 'WORLD_SIZE' in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])

    if 'RANK' in os.environ:
        args.rank = int(os.environ["RANK"])

    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])

    if os.getenv('MASTER_ADDR') is None:
        os.environ['MASTER_ADDR'] = 'localhost'
    if os.getenv('MASTER_PORT') is None:
        os.environ['MASTER_PORT'] = '12355'

    assert args.local_rank != -1, "Supporting distributed training only, but local_rank is -1"

    if args.profile is not None:
        os.environ['HABANA_PROFILE'] = 'profile_api_with_nics'
        shutil.rmtree('.graph_dumps', ignore_errors=True)

    init_method = None
    if args.use_hpu:
        global hpu
        import habana_frameworks.torch.hpu as hpu
        import habana_frameworks.torch.distributed.hccl
        device = torch.device("hpu")
        dist_backend = "hccl"
    elif args.no_cuda:
        device = torch.device("cpu")
        dist_backend = "gloo"
    else:
        device = torch.device("cuda", args.local_rank)
        dist_backend = "nccl"
        init_method = init_method="env://"

    print(f"Distributed training with backend={dist_backend}, device={device}, local_rank={args.local_rank}")
    deepspeed.init_distributed(dist_backend=dist_backend, init_method=init_method)

    if is_main_process():
        os.makedirs(os.path.dirname(args.json_summary), exist_ok=True)
        dllogger.init(backends=[dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,
                                                           filename=args.json_summary),
                                dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE, step_format=format_step)])
    else:
        dllogger.init(backends=[])

    if not args.do_train:
        raise ValueError(" `do_train`  must be True.")

    if not args.resume_from_checkpoint and os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    if (not args.resume_from_checkpoint or not os.path.exists(args.output_dir)) and is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)

    return device, args


def adjust_phase2_initial_checkpoint(output_dir, tag, adjusted_tag, sharded_optim_states):
    def get_state_files(pattern):
        states_file_pattern = os.path.join(output_dir, str(adjusted_tag), '*_' + pattern + '.pt')
        states_files = glob.glob(states_file_pattern)
        states_files.sort()
        return states_files

    def adjust_optimizer(opt_sd):
        opt_state = opt_sd['state']
        for p in opt_state.values():
            if 'step' in p:
                p['step'] = 0
        param_groups = opt_sd['param_groups']
        for group in param_groups:
            if 'step' in group:
                group['step'] = 0

    def adjust_model_states_ckp(ckp, adjust_optim_states):
        ckp['global_steps'] = 0
        ckp['skipped_steps'] = 0
        ckp['global_samples'] = 0
        ckp_lr_scheduler = ckp.get('lr_scheduler', None)
        if ckp_lr_scheduler:
            # DeepSpeed schedulers mandates last_batch_iteration to exist in sd
            if 'last_batch_iteration' in ckp_lr_scheduler:
                ckp_lr_scheduler['last_batch_iteration'] = 0
            else:
                ckp_lr_scheduler.clear()
        if adjust_optim_states:
            optimizer = ckp['optimizer']
            adjust_optimizer(optimizer)

    def adjust_optim_states_ckp(ckp):
        optimizer_state_dict = ckp['optimizer_state_dict']
        base_optimizer_state = optimizer_state_dict['base_optimizer_state']
        adjust_optimizer(base_optimizer_state)

    src_path = os.path.join(output_dir, tag)
    dst_path = os.path.join(output_dir, adjusted_tag)
    shutil.rmtree(dst_path, ignore_errors=True)
    shutil.copytree(src_path, dst_path, dirs_exist_ok=False)

    files = get_state_files(pattern='model_states')
    for filename in files:
        checkpoint = torch.load(filename)
        adjust_model_states_ckp(checkpoint, adjust_optim_states=(not sharded_optim_states))
        torch.save(checkpoint, filename)

    if sharded_optim_states:
        files = get_state_files(pattern='optim_states')
        for filename in files:
            checkpoint = torch.load(filename)
            adjust_optim_states_ckp(checkpoint)
            torch.save(checkpoint, filename)

    return adjusted_tag


def prepare_model_and_optimizer(args, device, with_cuda, with_hpu):
    # Prepare model
    config = modeling.BertConfig.from_json_file(args.config_file)

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training
    model = modeling.BertForPreTraining(config)
    model.checkpoint_activations(args.checkpoint_activations, args.checkpoint_activations_interval)

    # TODO SW-96497: remove this WA when SW-96431 is resolved
    if args.bfloat16_enabled:
        model.to(dtype=torch.bfloat16, device=device)
    else:
        model.to(device=device)

    # Optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    assert not (args.optimizer in OPTIMIZERS_CUDA_ONLY) or (not args.use_hpu and torch.cuda.is_available()), \
        'Optimizers fused_lamb and ds_fused_lamb require cuda'
    tensor_lr = torch.tensor(args.learning_rate)
    if not is_cpu_offload_optimizer(args.deepspeed_config):
        tensor_lr = tensor_lr.to(device=device)
    optimizer_kwargs = {
        'params': optimizer_grouped_parameters,
        'lr': tensor_lr
    }
    if args.betas is not None:
        assert len(args.betas) == 2, '--betas must include exactly 2 values: beta1 beta2'
        optimizer_kwargs.update({'betas': tuple(args.betas)})
        print(f"Using non-default betas={optimizer_kwargs}")

    if args.optimizer == 'fused_lamb':
        print('Using FusedLamb')
        optimizer = optimizers.FusedLAMB(**optimizer_kwargs)
    elif args.optimizer == 'ds_fused_lamb':
        print('Using DeepSpeed FusedLamb')
        optimizer = DeepSpeedFusedLamb(**optimizer_kwargs)
    elif args.optimizer == 'nvlamb':
        print('Using NVLamb')
        optimizer = NVLAMB(adjust_step=args.zero_optimization, **optimizer_kwargs)
    elif args.optimizer == 'nvlamb_exp':
        print('Using NVLamb Experimental with use_ds_lamb_bias_correction={}'.format(args.use_ds_lamb_bias_correction))
        optimizer = NVLAMB_EXP(max_grad_norm=0., eps=1e-8, adjust_step=args.zero_optimization, use_nvlamb=True,
                               max_trust=10., min_trust=0.01,
                               use_ds_lamb_bias_correction=args.use_ds_lamb_bias_correction, **optimizer_kwargs)
    elif args.optimizer == 'lans':
        print('Using LANS')
        optimizer = LANS(max_grad_norm=0., eps=1e-6, adjust_step=args.zero_optimization, **optimizer_kwargs)
    elif args.optimizer == 'adam':
        print('Using Adam')
        optimizer = Adam(**optimizer_kwargs)
    elif args.optimizer == 'adamw':
        print('Using AdamW')
        optimizer = AdamW(**optimizer_kwargs)
    elif args.optimizer == 'fused_adamw':
        print('Using FusedAdamW')
        from habana_frameworks.torch.hpex.optimizers import FusedAdamW
        optimizer = FusedAdamW(**optimizer_kwargs)
    else:
        print('Optimizer is expected to be configured in deepspeed configuration file')
        optimizer = None

    if optimizer and args.use_lr_scheduler:
        lr_scheduler_args = {
            'warmup': args.warmup_proportion,
            'total_steps': args.max_steps,
            'degree': args.scheduler_degree,
            'constant': args.constant_proportion
        }
        print('Using PolyWarmUpScheduler with args={}'.format(lr_scheduler_args))
        lr_scheduler = PolyWarmUpScheduler(optimizer, **lr_scheduler_args)

    else:
        print('LR Scheduler is expected to be configured in deepspeed configuration file')
        lr_scheduler = None

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=args,
        model_parameters=None if optimizer else optimizer_grouped_parameters,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )

    # Sync params from DeeepSpeed parser
    train_batch_size, train_micro_batch_size_per_gpu, gradient_accumulation_steps = model.get_batch_info()
    args.train_batch_size = train_micro_batch_size_per_gpu()
    args.gradient_accumulation_steps = gradient_accumulation_steps()

    # Checkpointing
    checkpoint, per_worker_checkpoint = None, None
    if not args.resume_from_checkpoint:
        global_step = 0
    else:
        if args.resume_step == -1:
            args.resume_step = 0
            try:
                if os.path.isfile(os.path.join(args.output_dir, 'latest')):
                    with open(os.path.join(args.output_dir, 'latest')) as file:
                        lines = file.readlines()
                        args.resume_step = int(lines[0])
                global_step = args.resume_step

                tag = f"{global_step}"

                # on phase2 1st iteration, we must skip loading lr_scheduler state and reset lr and step in optimizer
                orig_tag = tag
                is_adjust = args.phase2 and (args.phase1_end_step == global_step)
                if is_adjust:
                    tag += '_phase2_adjusted'
                    if is_main_process():
                        adjust_phase2_initial_checkpoint(args.output_dir, orig_tag, tag,
                                                         sharded_optim_states=args.zero_optimization)

                    if torch.distributed.is_initialized():
                        torch.distributed.barrier()

                load_path, checkpoint = model.load_checkpoint(args.output_dir, tag)

                # restore local RNG state for this worker
                per_worker_filename = os.path.join(args.output_dir, orig_tag, f"state_{get_rank()}")
                if not os.path.isfile(per_worker_filename):
                    print(f"WARNING: per-worker checkpoint state file={per_worker_filename} is missing")
                else:
                    per_worker_checkpoint = torch.load(per_worker_filename)
                    set_local_rng_state(per_worker_checkpoint['rng_state'], with_cuda=with_cuda, with_hpu=with_hpu)

                if args.phase2:
                    global_step -= args.phase1_end_step
                if is_main_process():
                    print(f"Loaded from checkpoint {load_path}. Resume step {args.resume_step}, global step {global_step}")
            except:
                print(f"Having --resume_from_checkpoint, but no valid checkpoint found. Starting from scratch.")
                args.resume_from_checkpoint = False

    criterion = BertPretrainingCriterion(config.vocab_size, run_loss_in_fp32=args.run_loss_in_fp32)

    return model, optimizer, lr_scheduler, checkpoint, per_worker_checkpoint, global_step, criterion


def save_common_checkpoint(args, model, epoch, global_step, files):
    # Save a trained model
    dllogger.log(step="PARAMETER", data={"checkpoint_step": global_step})
    if args.resume_step < 0 or not args.phase2:
        tag = f"{global_step}"
    else:
        tag = f"{global_step + args.phase1_end_step}"

    checkpoint_dict = {}
    if args.do_train:
        checkpoint_dict = {
            'files': files,
            'epoch': epoch
        }
    if is_main_process():
        print(f"Saving checkpoint {tag}.")
    model.save_checkpoint(args.output_dir, tag, checkpoint_dict)
    return tag


def save_per_worker_checkpoint(args, tag, n_processed_files, step_in_current_file, with_cuda, with_hpu):
    if not args.do_train:
        return

    checkpoint = {
        'version': 1,
        'n_processed_files': n_processed_files,
        'step_in_current_file': step_in_current_file
    }
    filename = os.path.join(args.output_dir, tag, f"state_{get_rank()}")
    rng_state = get_local_rng_state(with_cuda=with_cuda, with_hpu=with_hpu)
    checkpoint['rng_state'] = rng_state
    torch.save(checkpoint, filename)


def main_train():
    global timeout_sent

    # SW-82670: Calling ::detach() before Tensor::numpy()
    fix_tensor_numpy()

    args = parse_arguments()

    device, args = setup_training(args)

    setup_profiler(args, device)

    seed = args.seed + args.local_rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dllogger.log(step="PARAMETER", data={"Config": [str(args)]})

    # Prepare optimizer
    with_cuda, with_hpu = device.type == 'cuda', device.type == 'hpu'
    model, optimizer, lr_scheduler, checkpoint, per_worker_checkpoint, global_step, criterion = \
        prepare_model_and_optimizer(args, device, with_cuda, with_hpu)

    if is_main_process():
        dllogger.log(step="PARAMETER", data={"SEED": args.seed})

    raw_train_start = None
    if args.do_train:
        if is_main_process():
            dllogger.log(step="PARAMETER", data={"train_start": True})
            dllogger.log(step="PARAMETER", data={"batch_size_per_pu": args.train_batch_size})
            dllogger.log(step="PARAMETER", data={"learning_rate": args.learning_rate})

        model.train()
        average_loss = 0.0  # averaged loss every args.log_freq steps
        epoch = 0
        training_steps = 0
        average_training_time_per_step = 0
        average_perf_per_step = 0
        loss_list = []

        if args.use_hpu:
            import habana_frameworks.torch.core as htcore
            if args.enable_compiled_autograd:
                from habana_frameworks.torch.dynamo.compile_backend.experimental import enable_compiled_autograd
                enable_compiled_autograd()

        if args.enable_torch_compile:
            compile_kwargs = {"dynamic": False}
            if args.enable_torch_compile_optimizer:
                model.compile(compile_optimizer_step=args.enable_torch_compile_optimizer, compile_kwargs=compile_kwargs)
            else:
                model.compile(compile_kwargs=compile_kwargs)

        if device.type == 'cuda':
            pool = ProcessPoolExecutor(1)
        starting_time = time.time()
        # Note: We loop infinitely over epochs, termination is handled via iteration count
        while True:
            skip_steps = 0
            if not args.resume_from_checkpoint or epoch > 0 or (args.phase2 and global_step < 1):
                files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if
                         os.path.isfile(os.path.join(args.input_dir, f)) and 'training' in f]
                files.sort()
                num_files = len(files)
                random.Random(args.seed + epoch).shuffle(files)
                f_start_id = 0
            else:
                # New checkpoint format allows to recover the state of data loader per worker
                if per_worker_checkpoint and 'version' in per_worker_checkpoint:
                    f_start_id = per_worker_checkpoint['n_processed_files']
                    skip_steps = per_worker_checkpoint['step_in_current_file'] + 1
                    log_dist(f"Worker {get_rank()}: processed {f_start_id} files, skipping {skip_steps} micro steps.",
                             ranks=[-1])
                elif 'file_start_id' in checkpoint.keys():
                    f_start_id = checkpoint['file_start_id']
                    skip_steps = checkpoint['step'] + 1
                    log_dist(f"Skipping {skip_steps} micro steps.", ranks=[0])
                else:
                    log_dist(f"Unsupported old checkpoint format. Aborting", ranks=[0])

                files = checkpoint['files']
                epoch = checkpoint.get('epoch', 0)
                args.resume_from_checkpoint = False
                num_files = len(files)

            shared_file_list = {}

            if torch.distributed.is_initialized() and get_world_size() > num_files:
                remainder = get_world_size() % num_files
                data_file = files[(f_start_id*get_world_size()+get_rank() + remainder*f_start_id) % num_files]
            else:
                data_file = files[(f_start_id*get_world_size()+get_rank()) % num_files]

            train_dataloader, data_file = create_pretraining_dataset(data_file, args.max_predictions_per_seq, shared_file_list, args)

            f_id_curr = f_start_id
            for f_id in range(f_start_id + 1, len(files)):
                if get_world_size() > num_files:
                    data_file = files[(f_id*get_world_size()+get_rank() + remainder*f_id) % num_files]
                else:
                    data_file = files[(f_id*get_world_size()+get_rank()) % num_files]

                if device.type == 'cuda':
                    dataset_future = pool.submit(create_pretraining_dataset, data_file, args.max_predictions_per_seq, shared_file_list, args)

                train_iter = tqdm(train_dataloader, desc="Iteration", disable=args.disable_progress_bar) if is_main_process() else train_dataloader

                if (args.tensor_logger_max_iterations > 0):
                    from deepspeed.tools.tensor_logger import TensorLogger, save_logged_tensors
                    tensor_logger = TensorLogger(model,
                        log_activations_enabled=args.log_fwd_activations,
                        max_iterations=args.tensor_logger_max_iterations,
                        log_grads_enabled=args.log_bwd_grads,
                        log_inputs_enabled=args.log_model_inputs,
                        prefix=None)
                else:
                    tensor_logger = None

                if raw_train_start is None:
                    raw_train_start = time.time()

                for step, batch in enumerate(train_iter):
                    # train_iter points to the beginning of the train sequence contained in data_file.
                    # So, we need to skip a number of batches if loaded from a checkpoint to make train_iter point to the correct batch.
                    if skip_steps > 0:
                        did_skip = True
                        skip_steps -= 1
                        continue

                    trigger(on_step_begin)

                    with tensor_logger.log_iteration(step) if tensor_logger else nullcontext():
                        training_steps += 1
                        batch = [t.to(device) for t in batch]
                        input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch

                        prediction_scores, seq_relationship_score = model(
                                    input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

                        loss = criterion(
                            prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels)

                        loss = model.backward(loss)

                    if args.use_hpu:
                        htcore.mark_step()

                    loss_list.append(loss)

                    # deepspeed model step
                    model.step()
                    # increase global step every gradient_accumulation_steps
                    if training_steps % args.gradient_accumulation_steps == 0:
                        global_step += 1

                    if args.use_hpu:
                        htcore.mark_step()

                    if global_step >= args.steps_this_run or timeout_sent or training_steps % (args.log_freq * args.gradient_accumulation_steps) == 0:
                        for loss_t in loss_list:
                            average_loss += loss_t.item()
                        loss_list.clear()
                        train_time = time.time() - starting_time
                        starting_time = time.time()
                        average_training_time_per_step = train_time / args.log_freq
                        average_perf_per_step = args.train_batch_size * args.gradient_accumulation_steps / average_training_time_per_step

                    if global_step >= args.steps_this_run or timeout_sent:
                        train_time_raw = time.time() - raw_train_start
                        last_num_steps = int(training_steps / args.gradient_accumulation_steps) % args.log_freq
                        last_num_steps = args.log_freq if last_num_steps == 0 else last_num_steps
                        average_loss = average_loss / last_num_steps
                        average_loss = torch.tensor(average_loss, dtype=torch.float32).to(device)
                        if (torch.distributed.is_initialized()):
                            average_loss /= get_world_size()
                            torch.distributed.barrier()
                            # TODO (SW-109589) Remove the WA below once using a cached group already creaated by DS (SW-105363).
                            torch.distributed.all_reduce(average_loss, group=model.data_parallel_group)
                        final_loss = average_loss.item()
                        if is_main_process():
                            dllogger.log(step=(epoch, global_step, ), data={"final_loss": final_loss,
                                                                            "average_training_time_step": average_training_time_per_step,
                                                                            "average_perf_per_step": average_perf_per_step})
                    elif training_steps % (args.log_freq * args.gradient_accumulation_steps) == 0:
                        if is_main_process():
                            current_lr = optimizer.param_groups[0]['lr']
                            scalar_lr =  current_lr.item() if torch.is_tensor(current_lr) else current_lr
                            dllogger.log(step=(epoch, global_step, ), data={"average_loss": average_loss / args.log_freq,
                                                                            "step_loss": loss.item() * args.gradient_accumulation_steps,
                                                                            "learning_rate": scalar_lr,
                                                                            "average_training_time_step": average_training_time_per_step,
                                                                            "average_perf_per_step": average_perf_per_step})
                            dllogger.flush()

                        average_loss = 0

                    if global_step >= args.steps_this_run or training_steps % (
                            args.num_steps_per_checkpoint * args.gradient_accumulation_steps) == 0 or timeout_sent:
                        if not args.skip_checkpoint:
                            tag = save_common_checkpoint(args, model, epoch, global_step, files)
                            save_per_worker_checkpoint(args, tag, f_id_curr, step, with_cuda, with_hpu)

                        # Exiting the training due to hitting max steps, or being sent a
                        # timeout from the cluster scheduler
                        if global_step >= args.steps_this_run or timeout_sent:
                            del train_dataloader
                            # save tensor logger file
                            if tensor_logger:
                                save_logged_tensors(tensor_logger, args.tensor_logger_path, get_rank())
                            # thread.join()
                            return args, final_loss, train_time_raw, global_step

                    trigger(on_step_end)

                del train_dataloader
                # thread.join()
                # Make sure pool has finished and switch train_dataloader
                # NOTE: Will block until complete
                if device.type == 'cuda':
                    train_dataloader, data_file = dataset_future.result(timeout=None)
                else:
                    train_dataloader, data_file = create_pretraining_dataset(data_file, args.max_predictions_per_seq, shared_file_list, args)
                f_id_curr = f_id

            epoch += 1


def main():
    now = time.time()
    args, final_loss, train_time_raw, global_step = main_train()
    pu_count = 1
    global_step += args.phase1_end_step if (args.phase2 and args.resume_step > 0) else 0
    if args.resume_step == -1:
        args.resume_step = 0
    if torch.distributed.is_initialized():
        pu_count = get_world_size()
    if is_main_process():
        e2e_time = time.time() - now
        training_perf = \
            args.train_batch_size * args.gradient_accumulation_steps * pu_count \
            * (global_step - args.resume_step + skipped_steps) / train_time_raw
        dllogger.log(step=tuple(), data={"e2e_train_time": e2e_time, "training_sequences_per_second": training_perf,
                                         "final_loss": final_loss, "raw_train_time": train_time_raw})
    dllogger.flush()


if __name__ == "__main__":
    main()
