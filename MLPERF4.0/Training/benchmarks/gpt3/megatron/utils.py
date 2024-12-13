# coding=utf-8
# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
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

"""General utilities."""

import sys

import os
import torch
from torch.nn.parallel import DistributedDataParallel as torchDDP
from megatron import get_args

if (torch.cuda.is_available()):
    from apex.multi_tensor_apply import multi_tensor_applier
    import amp_C

from megatron import get_args
from megatron import print_rank_0
from megatron import get_adlr_autoresume
from megatron import mpu
from megatron.model.module import param_is_not_shared
from megatron.mpu.layers import param_is_not_tensor_parallel_duplicate
from megatron import get_num_microbatches

def unwrap_model(model, module_instances=(torchDDP)):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def calc_params_l2_norm(model):
    """Calculate l2 norm of parameters """
    args = get_args()
    if not isinstance(model, list):
        model = [model]
    # Remove duplicate params.
    params_data = []
    for model_ in model:
        for param in model_.parameters():
            is_not_shared = param_is_not_shared(param)
            is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
            if is_not_shared and is_not_tp_duplicate:
                if args.bf16:
                    params_data.append(param.data.float())
                else:
                    params_data.append(param.data)
    # Calculate norm
    # TODO SW-56092: not having multi_tensor_applier implementation outside CUDA
    if args.device.type == "cuda":
        # Calculate norm
        dummy_overflow_buf = torch.cuda.IntTensor([0])
        norm, _ = multi_tensor_applier(
            amp_C.multi_tensor_l2norm,
            dummy_overflow_buf,
            [params_data],
            False # no per-parameter norm
        )
        norm_2 = norm * norm
    else:
        # less optimized version for now
        norm_2 = 0.0
        for param_tmp in params_data:
            norm_2 += param_tmp.square().sum()

    # Sum across all model-parallel GPUs.
    torch.distributed.all_reduce(norm_2,
                                 op=torch.distributed.ReduceOp.SUM,
                                 group=mpu.get_model_parallel_group())
    return norm_2.item() ** 0.5


def average_losses_across_data_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat(
        [loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses,
                                 group=mpu.get_data_parallel_group())
    averaged_losses = averaged_losses / \
        torch.distributed.get_world_size(group=mpu.get_data_parallel_group())

    return averaged_losses


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(
        torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | reserved: {}'.format(
        torch.cuda.memory_reserved() / mega_bytes)
    string += ' | max reserved: {}'.format(
        torch.cuda.max_memory_reserved() / mega_bytes)
    if mpu.get_data_parallel_rank() == 0:
        print("[Rank {}] {}".format(torch.distributed.get_rank(), string),
              flush=True)


def print_params_min_max_norm(optimizer, iteration):
    """Print min, max, and norm of all parameters."""
    index = 0
    rank = torch.distributed.get_rank()
    string = 'iteration, rank, index, tensor-model-parallel, min, max, norm\n'
    optimizer_ = optimizer.optimizer
    for param_group in optimizer_.param_groups:
        for param in param_group['params']:
            index += 1
            min_ = param.data.min()
            max_ = param.data.max()
            norm = torch.linalg.norm(param.data)
            string += '{:7d}, {:4d}, {:4d}, {:2d}, '.format(
                iteration, rank, index, int(param.tensor_model_parallel))
            string += '{:.6E}, {:.6E}, {:.6E}\n'.format(min_, max_, norm)
    print(string, flush=True)


def check_adlr_autoresume_termination(iteration, model,
                                      optimizer, lr_scheduler):
    """Check for autoresume signal and exit if it is received."""
    from megatron.checkpointing import save_checkpoint

    args = get_args()
    autoresume = get_adlr_autoresume()
    # Add barrier to ensure consistnecy.
    torch.distributed.barrier()
    if autoresume.termination_requested():
        if args.save:
            save_checkpoint(iteration, model, optimizer, lr_scheduler)
        print_rank_0(">>> autoresume termination request found!")
        if torch.distributed.get_rank() == 0:
            autoresume.request_resume()
        print_rank_0(">>> training terminated. Returning")
        sys.exit(0)


def get_ltor_masks_and_position_ids(data,
                                    eod_token,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    eod_mask_loss,
                                    dummy_sample=None,
                                    labels=None):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    if dummy_sample is not None:
        loss_mask[dummy_sample.bool()] = 0.0

    if labels is not None:
        loss_mask[labels == -1] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = (attention_mask < 0.5)

    return attention_mask, loss_mask, position_ids


def get_parameters_in_billions(model):
    gpus_per_model = torch.distributed.get_world_size(group=mpu.get_model_parallel_group())

    approx_parameters_in_billions = sum([sum([p.ds_numel if hasattr(p,'ds_id') else  p.nelement() for p in model_module.parameters()])
                                        for model_module in model])

    return approx_parameters_in_billions*gpus_per_model/(1e9)

def throughput_calculator(model, args, iteration_time, total_iterations):
    gpus_per_model = torch.distributed.get_world_size(group = mpu.get_model_parallel_group())
    batch_size = args.micro_batch_size * get_num_microbatches() * args.data_parallel_size
    samples_per_model = batch_size * args.seq_length
    model_replica_count = torch.distributed.get_world_size() / gpus_per_model
    approx_parameters_in_billions = get_parameters_in_billions(model)
    elapsed_time_per_iter = iteration_time/total_iterations
    samples_per_second = batch_size / elapsed_time_per_iter

    #flops calculator
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    vocab_size = args.padded_vocab_size

    # General TFLOPs formula (borrowed from Equation 3 in Section 5.1 of
    # https://arxiv.org/pdf/2104.04473.pdf).
    # The factor of 4 is when used with activation check-pointing,
    # otherwise it will be 3.
    checkpoint_activations_factor = 4 if args.checkpoint_activations else 3
    flops_per_iteration = (24 * checkpoint_activations_factor * batch_size * args.seq_length * num_layers * (hidden_size**2)) * (1. + (args.seq_length / (6. * hidden_size)) + (vocab_size / (16. * num_layers * hidden_size)))
    tflops = flops_per_iteration / (elapsed_time_per_iter * args.world_size * (10**12))
    return samples_per_second, tflops, approx_parameters_in_billions

def checkpoint_throughput_calculator(model, latency_second):
    approx_parameters_in_billions = get_parameters_in_billions(model)
    checkpoint_multiplier = 14  # fp16 weights (2), fp32 weights (4), fp32 momentum (4), fp32 variance (4)
    checkpoint_GB = approx_parameters_in_billions * checkpoint_multiplier
    GB_per_second = checkpoint_GB / latency_second
    print_rank_0(f"Checkpoint Save GB: {round(checkpoint_GB, 3)}, GB/Sec: {round(GB_per_second,2)}, Latency(second): {round(latency_second, 3)}")


def found_kill_switch():
    args = get_args()
    if args.kill_switch_path is not None and os.path.exists(args.kill_switch_path):
        return True
    else:
        return False
