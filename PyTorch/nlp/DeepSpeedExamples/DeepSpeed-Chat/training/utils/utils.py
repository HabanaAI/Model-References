# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import torch
import random
import numpy as np
from transformers import set_seed, AutoTokenizer
import json
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.accelerator import get_accelerator
import time
from datetime import datetime


def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


class MovingAverage:

    def __init__(self):
        self.count = 0
        self.total = 0
        self.mean = 0

    def update(self, num):
        self.total += num
        self.count += 1
        self.mean = self.total / self.count

        return self.mean


class ExponentialMovingAverage:

    def __init__(self, alpha=0.99):
        self.alpha = alpha
        self.ema = None

    def update(self, num):
        prev_ema = num if self.ema is None else self.ema
        self.ema = self.alpha * prev_ema + (1.0 - self.alpha) * num
        return self.ema

    def get(self):
        return self.ema if self.ema is not None else 0.


def load_hf_tokenizer(model_name_or_path, fast_tokenizer=True):
    if os.path.exists(model_name_or_path):
        # Locally tokenizer loading has some issue, so we need to force download
        # currently we must support offline execution so in case of local model path,
        # also tokenizer will be first tried to downloaded
        model_json = os.path.join(model_name_or_path, "config.json")
        if os.path.exists(model_json):
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                          fast_tokenizer=True)
            except Exception as e:
                print(f"failed loading tokenizer from local path={model_name_or_path} due to exception: {e}")
                model_json_file = json.load(open(model_json))
                model_name = model_json_file["_name_or_path"]
                print(f"falling back to HF path: {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                          fast_tokenizer=True)

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                  fast_tokenizer=True)
    return tokenizer


def save_hf_format(model, tokenizer, args, sub_folder=""):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_dir = os.path.join(args.output_dir, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    for key in list(save_dict.keys()):
        if "lora" in key:
            del save_dict[key]
    torch.save(save_dict, output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)


def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        get_accelerator().manual_seed_all(seed)


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


def get_optimizer_grouped_parameters(model,
                                     weight_decay,
                                     no_decay_name_list=[
                                         "bias", "layernorm.weight", "ln_f.weight"
                                     ]):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n.lower()
                            for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n.lower()
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            0.0,
        },
    ]
    return optimizer_grouped_parameters


def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def moving_average(model, model_ema, beta=0.992, device=None, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    with torch.no_grad():
        for param, param_ema in zip(model.parameters(),
                                    model_ema.parameters()):
            # TODO: use prefiltering for efficiency
            params_to_fetch = _z3_params_to_fetch([param, param_ema
                                                   ]) if zero_stage_3 else []
            should_gather_param = len(params_to_fetch) > 0
            with deepspeed.zero.GatheredParameters(
                    params_to_fetch, enabled=should_gather_param):
                data = param.data
                if device is not None:
                    data = data.to(device)
                param_ema.data.copy_(torch.lerp(data, param_ema.data, beta))


def save_zero_three_model(model_ema, global_rank, save_dir, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    os.makedirs(save_dir, exist_ok=True)
    WEIGHTS_NAME = "pytorch_model.bin"
    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)

    model_to_save = model_ema.module if hasattr(model_ema,
                                                'module') else model_ema
    if not zero_stage_3:
        if global_rank == 0:
            torch.save(model_to_save.state_dict(), output_model_file)
    else:
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():

            if hasattr(v, 'ds_id'):
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v
                                                                            ]),
                                                       enabled=zero_stage_3):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if global_rank == 0 and "lora" not in k:
                output_state_dict[k] = v_p
        if global_rank == 0:
            torch.save(output_state_dict, output_model_file)
        del output_state_dict

def print_stats(epoch, step, steps_per_print, last_time, train_bs, gas, loss, loss_sum, rank):
    if loss_sum is None:
        loss_sum = torch.zeros_like(loss)
    loss_sum += loss
    if step != 0 and step % (steps_per_print * gas) == 0:
        curr_time = time.time()
        sec_per_step = (curr_time - last_time) / steps_per_print
        last_time = curr_time
        samples_per_sec = train_bs / sec_per_step
        opt_step = step / gas
        avg_loss = loss_sum / gas
        print_rank_0(
            f"[{datetime.now()}] epoch: {epoch} | step: {opt_step} | loss: {avg_loss} | time per iter (s): {sec_per_step} | samples per seconds: {samples_per_sec}",
            rank)
    if step != 0 and step % gas == 0:
        loss_sum.zero_()

    return last_time, loss_sum

def is_hpu():
    return get_accelerator().device_name() == "hpu"

def hpu_mark_step():
    if is_hpu():
        import habana_frameworks.torch.core as htcore
        htcore.mark_step()