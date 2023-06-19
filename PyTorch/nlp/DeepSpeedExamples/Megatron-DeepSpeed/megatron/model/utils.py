# coding=utf-8
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

"""Utilities for models."""

import math

import torch

from megatron import mpu, get_args

def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method_normal(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_

def perform_masking(attention_scores, attention_mask):
    if attention_mask.dtype == torch.bool:
        attention_scores.masked_fill_(attention_mask, -10000.0)
    else:
        attention_scores.add_(attention_mask)

def attention_mask_func(attention_scores, attention_mask):
    args = get_args()
    if args.curriculum_learning:
        attention_mask_ = attention_mask
        actual_seqlen = attention_scores.size()[2]
        if actual_seqlen != attention_mask_.size()[2]:
            # attention_mask has size [1, 1, seqlen, seqlen]
            attention_mask_ = attention_mask_[:, :, :actual_seqlen, :actual_seqlen].contiguous()
        perform_masking(attention_scores, attention_mask_)
    else:
        perform_masking(attention_scores, attention_mask)

    return attention_scores


def get_linear_layer(rows, columns, init_method):
    """Simple linear layer with weight initialization."""
    layer = torch.nn.Linear(rows, columns)
    init_method(layer.weight)
    with torch.no_grad():
        layer.bias.zero_()
    return layer

@torch.jit.script
def gelu_impl_gpu(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                       (1.0 + 0.044715 * x * x)))

def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                        (1.0 + 0.044715 * x * x)))

def openai_gelu(x):
    if torch.cuda.is_available():
        return gelu_impl_gpu(x)
    else:
        return gelu_impl(x)

#This is actually Python equivalent of torch.nn.functional.gelu(), also with type hints for ONNX exporter
@torch.jit.script
def erf_gelu_impl_gpu(x):
    return x * 0.5 * (torch.erf(x / 1.41421).to(dtype=x.dtype)+torch.ones_like(x).to(dtype=x.dtype))

def erf_gelu_impl(x):
    return x * 0.5 * (torch.erf(x / 1.41421).to(dtype=x.dtype)+torch.ones_like(x).to(dtype=x.dtype))

def erf_gelu(x):
    if torch.cuda.is_available():
        return erf_gelu_impl_gpu(x)
    else:
        return erf_gelu_impl(x)

def CrossEntropy(output, labels):
    labels, loss_mask = labels[0], labels[1]

    args = get_args()

    losses = mpu.vocab_parallel_cross_entropy(output.contiguous().float(), labels)
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    return loss

class WrapName(torch.nn.Module):
    def __init__(self, name='', module=None, *args, **kwargs):
        super(WrapName, self).__init__()
        self.name = name
        if module is not None:
            value = module(*args, **kwargs)
            setattr(self, self.name, value)
            self.forward_func = value.forward

    def forward(self, inputs):
        return self.forward_func(inputs)
