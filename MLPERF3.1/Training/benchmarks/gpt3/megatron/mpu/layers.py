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


# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch


import math

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from .initialize import get_tensor_model_parallel_rank
from .initialize import get_tensor_model_parallel_world_size
from .initialize import get_tensor_model_parallel_group
from .initialize import get_global_memory_buffer

from .mappings import copy_to_tensor_model_parallel_region
from .mappings import gather_from_tensor_model_parallel_region
from .mappings import gather_from_sequence_parallel_region
from .mappings import reduce_from_tensor_model_parallel_region
from .mappings import scatter_to_tensor_model_parallel_region
from .mappings import reduce_scatter_to_sequence_parallel_region

from .random import get_cuda_rng_tracker
from .utils import divide
from .utils import split_tensor_along_last_dim
from .utils import VocabUtility
from megatron.model.fused_layer_norm import MixedFusedLayerNorm as LayerNorm
from megatron import get_args, mpu
from megatron.global_vars import get_current_device, get_num_microbatches
import deepspeed.runtime.activation_checkpointing.checkpointing as ds_checkpointing
from typing import Optional
import os
import warnings


_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {'tensor_model_parallel': False,
                                      'partition_dim': -1,
                                      'partition_stride': 1}


def param_is_not_tensor_parallel_duplicate(param):
    return (hasattr(param, 'tensor_model_parallel') and
            param.tensor_model_parallel) or (
                get_tensor_model_parallel_rank() == 0)


def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # Set the attributes.
    setattr(tensor, 'tensor_model_parallel', is_parallel)
    setattr(tensor, 'partition_dim', dim)
    setattr(tensor, 'partition_stride', stride)


def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor):
    def maybe_set(attribute, value):
        if not hasattr(tensor, attribute):
            setattr(tensor, attribute, value)
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_set(attribute, _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS[attribute])


def copy_tensor_model_parallel_attributes(destination_tensor, source_tensor):
    def maybe_copy(attribute):
        if hasattr(source_tensor, attribute):
            setattr(destination_tensor, attribute,
                    getattr(source_tensor, attribute))
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_copy(attribute)


def _initialize_affine_weight_gpu(weight, init_method,
                                  partition_dim, stride=1):
    """Initialize affine weight for model parallel on GPU."""

    set_tensor_model_parallel_attributes(tensor=weight,
                                         is_parallel=True,
                                         dim=partition_dim,
                                         stride=stride)

    if ds_checkpointing.is_configured():
        global get_cuda_rng_tracker
        get_cuda_rng_tracker = ds_checkpointing.get_cuda_rng_tracker

    with get_cuda_rng_tracker().fork():
        init_method(weight)


def _initialize_affine_weight_cpu(weight, output_size, input_size,
                                  per_partition_size, partition_dim,
                                  init_method, stride=1,
                                  return_master_weight=False):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    set_tensor_model_parallel_attributes(tensor=weight,
                                         is_parallel=True,
                                         dim=partition_dim,
                                         stride=stride)

    # Initialize master weight
    master_weight = torch.empty(output_size, input_size,
                                dtype=torch.float,
                                requires_grad=False)
    init_method(master_weight)
    args = get_args()
    master_weight = master_weight.to(dtype=args.params_dtype)

    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size,
                              dim=partition_dim)
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None


# This class encapsulates the behavior related to two mechanisms: hpu graph and amax measuring interval
class FP8ModuleRunner():
    def __init__(self, module, hpu_graph_enabled: bool=True, measure_interval: int=1, cache_fp8_weight_fwd=False):
        self.module = module
        self.hpu_graph_enabled = hpu_graph_enabled
        self.measure_interval = measure_interval
        self.cache_fp8_weight_fwd = cache_fp8_weight_fwd
        self.module_with_measurement = None
        self.module_no_measurement = None
        self.run_cnt = 0

    @staticmethod
    def _init_hpu_graph(module, input, weight, bias=None):
        import habana_frameworks.torch as ht
        fp8_meta = module.save_fp8_meta()
        tmp_in = torch.zeros_like(input, requires_grad=input.requires_grad)
        tmp_w = torch.zeros_like(weight, requires_grad=weight.requires_grad)
        tmp_b = torch.zeros_like(bias, requires_grad=bias.requires_grad) if bias is not None else None
        wrapped_module = ht.hpu.ModuleCacher(max_graphs=10)(model=module, inplace=False)
        wrapped_module(tmp_in, tmp_w, tmp_b).cpu()
        wrapped_module.load_fp8_meta(fp8_meta)
        return wrapped_module

    def _is_first_microbatch(self):
        if not self.cache_fp8_weight_fwd:
            return None

        return self.run_cnt % get_num_microbatches() in [1,2]

    def __call__(self, input, weight, bias=None):
        from habana_frameworks.torch.hpex.experimental.transformer_engine.fp8 import (
            set_measurement_mode,
            is_fp8_enabled
        )
        if not is_fp8_enabled():
            return self.module(input, weight, bias)

        self.run_cnt += 1
        measure = self.measure_interval == 1 or self.run_cnt % self.measure_interval == 1
        set_measurement_mode(manual=True, manual_value=measure)

        is_first_microbatch = self._is_first_microbatch()

        # In case of hpu graphs, do not record first iteration
        if not self.hpu_graph_enabled or self.run_cnt == 1:
            return self.module(input, weight, bias, is_first_microbatch=is_first_microbatch)

        assert is_first_microbatch==None, "is_first_microbatch handling not implemented with HPU graphs, turn off either hpu graphs or cache fp8 weight fwd"

        # HPU graphs case
        if measure:
            if self.module_with_measurement is None:
                self.module_with_measurement = self._init_hpu_graph(self.module, input, weight, bias)
            return self.module_with_measurement(input, weight, bias)
        else:
            if self.module_no_measurement is None:
                self.module_no_measurement = self._init_hpu_graph(self.module, input, weight, bias)
            return self.module_no_measurement(input, weight, bias)


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, num_embeddings, embedding_dim,
                 init_method=init.xavier_normal_):
        super(VocabParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set the defaults for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
        # Divide the weight matrix along the vocaburaly dimension.
        self.vocab_start_index, self.vocab_end_index = \
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings, get_tensor_model_parallel_rank(),
                self.tensor_model_parallel_size)
        self.num_embeddings_per_partition = self.vocab_end_index - \
            self.vocab_start_index

        # Allocate weights and initialize.
        args = get_args()

        # only the first stage embedding runs this class' forward. The head's embedding does its own
        # thing, so don't waste memory allocating LN weights.
        self.layer_norm = None
        if mpu.is_pipeline_first_stage() and args.embed_layernorm:
            self.layer_norm = LayerNorm(embedding_dim)

        if args.use_cpu_initialization:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                dtype=args.params_dtype))
            _initialize_affine_weight_cpu(
                self.weight, self.num_embeddings, self.embedding_dim,
                self.num_embeddings_per_partition, 0, init_method)
        else:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                device=get_current_device(), dtype=args.params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=0, stride=1)

    def forward(self, input_):
        if self.tensor_model_parallel_size > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | \
                         (input_ >= self.vocab_end_index)
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
            # Get the embeddings.
        output_parallel = F.embedding(masked_input, self.weight,
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)
        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = reduce_from_tensor_model_parallel_region(output_parallel)

        if self.layer_norm is not None:
            output = self.layer_norm(output)

        return output

class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        world_size = get_tensor_model_parallel_world_size()
        ctx.size = input.size()
        dim_size = list(input.size())
        dim_size[0] = dim_size[0] * world_size

        all_gather_buffer = \
            get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
        torch.distributed.all_gather_into_tensor(
            all_gather_buffer,
            input,
            group=get_tensor_model_parallel_group(),
            async_op=True)

        total_input = all_gather_buffer
        return total_input

    @staticmethod
    def backward(ctx, grad_output):
        sub_grad_output = torch.empty(ctx.size, dtype=grad_output.dtype,
                                    device=get_current_device(),
                                    requires_grad=False)

        # reduce_scatter
        torch.distributed.reduce_scatter_tensor(sub_grad_output, grad_output,
                                        group=get_tensor_model_parallel_group(),
                                        async_op=True)
        return sub_grad_output

def all_gather(input):
    return AllGather().apply(input)

def flatten_input(func):
    def wrapper(input, weight, bias=None):
        if input.dim() > 2:
            input_size = input.size()
            input = torch.flatten(input, start_dim=0, end_dim=input.dim()-2)
            output = func(input, weight, bias)
            output = torch.unflatten(output, dim=0, sizes=input_size[:-1])
        else:
            output = func(input, weight, bias)
        return output
    return wrapper

class VocabParallelProjection(torch.nn.Module):
    """Projection parallelized in the vocabulary dimension.

    This is mainly adapted from VocabParallelEmbedding and parallel_lm_logits.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        parallel_output: whether to output parallel outputs
        init_method: method to initialize weights.
    """

    def __init__(self, num_embeddings, embedding_dim, parallel_output=True,
                 init_method=init.xavier_normal_):
        super(VocabParallelProjection, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.parallel_output = parallel_output
        # Set the defaults for compatibility.
        self._weight = None
        self.bias = None
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
        # Divide the weight matrix along the vocaburaly dimension.
        self.vocab_start_index, self.vocab_end_index = \
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings, get_tensor_model_parallel_rank(),
                self.tensor_model_parallel_size)
        self.num_embeddings_per_partition = self.vocab_end_index - \
            self.vocab_start_index

        # Allocate weights and initialize.
        args = get_args()

        if args.use_cpu_initialization:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                dtype=args.params_dtype))
            _initialize_affine_weight_cpu(
                self.weight, self.num_embeddings, self.embedding_dim,
                self.num_embeddings_per_partition, 0, init_method)
        else:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                device=get_current_device(), dtype=args.params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=0, stride=1)

    def forward(self, input_):
        """LM logits using word projection weights."""
        # Parallel logits.
        input_parallel = mpu.copy_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        if self.bias is None:
            logits_parallel = F.linear(input_parallel, self.weight)
        else:
            logits_parallel = F.linear(input_parallel, self.weight, self.bias)
        # Gather if needed.
        if self.parallel_output:
            return logits_parallel

        return mpu.gather_from_tensor_model_parallel_region(logits_parallel)


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
        sequence_parallel: Indicates that sequence parallelism is used.
    """

    def __init__(self, input_size, output_size,
                 bias=True, gather_output=True,
                 init_method=init.xavier_normal_,
                 stride=1, keep_master_weight_for_test=False,
                 sequence_parallel=False, moe=False,
                 enable_expert_tensor_parallelism=False
                 ):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        if moe and (not enable_expert_tensor_parallelism):
            world_size = 1
            self.is_expert_without_slicing = True
        else:
            world_size = get_tensor_model_parallel_world_size()
            self.is_expert_without_slicing = False

        self.output_size_per_partition = divide(output_size, world_size)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        args = get_args()
        self.sequence_parallel = sequence_parallel

        if args.use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.output_size_per_partition,
                                                self.input_size,
                                                dtype=args.params_dtype))
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight, self.output_size, self.input_size,
                self.output_size_per_partition, 0, init_method,
                stride=stride, return_master_weight=keep_master_weight_for_test)
        else:
            self.weight = Parameter(torch.empty(
                self.output_size_per_partition, self.input_size,
                device=get_current_device(), dtype=args.params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=0, stride=stride)

        if bias:
            if args.use_cpu_initialization:
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition, dtype=args.params_dtype))
            else:
                self.bias = Parameter(torch.zeros(
                    self.output_size_per_partition,
                    device=get_current_device(),
                    dtype=args.params_dtype))
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
        else:
            self.register_parameter('bias', None)

        import habana_frameworks.torch.hpex.experimental.transformer_engine as te
        output_parallel_linear = te.Linear(
            self.input_size,
            self.output_size_per_partition,
            skip_weight_param_allocation=True,
            bias = True,
            minimize_memory=not args.cache_fp8_weight)
        self.output_parallel_linear = FP8ModuleRunner(
            output_parallel_linear,
            args.use_hpu_graphs,
            args.hpu_fp8_measure_interval,
            args.cache_fp8_weight_fwd)

        if self.sequence_parallel:
            if world_size <= 1:
                warnings.warn(
                    "`sequence_parallel_enabled` is set to `True`, "
                    f"but tensor model parallel size is {world_size}. "
                    f"Disabling sequence parallel."
                )
                self.sequence_parallel = False

    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.is_expert_without_slicing or self.sequence_parallel:
            input_parallel = input_
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input_)

        gather_input = lambda x: x
        if self.sequence_parallel:
            gather_input = all_gather

        # Matrix multiply.
        output_parallel = flatten_input(self.output_parallel_linear)(gather_input(input_parallel), self.weight, self.bias)

        if self.gather_output and not self.is_expert_without_slicing:
            # All-gather across the partitions.
            assert not self.sequence_parallel
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        return output


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimization where bias
                       can be fused with other elementwise operations. We skip
                       adding bias but instead return it.
        sequence_parallel: Indicates that sequence parallelism is used.
    """

    def __init__(self, input_size, output_size,
                 bias=True, input_is_parallel=False,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False,
                 sequence_parallel=False,
                 moe=False,
                 enable_expert_tensor_parallelism=False
                 ):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.

        if moe and (not enable_expert_tensor_parallelism):
            world_size = 1
        else:
            world_size = get_tensor_model_parallel_world_size()

        self.is_expert_without_slicing = moe and world_size==1

        self.input_size_per_partition = divide(input_size, world_size)
        self.skip_bias_add = skip_bias_add
        self.sequence_parallel = sequence_parallel

        if self.sequence_parallel and not self.input_is_parallel:
            raise RuntimeError("To enable `sequence_parallel_enabled`, `input_is_parallel` must be `True`")

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        args = get_args()
        self.args = args
        if args.use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.output_size,
                                                self.input_size_per_partition,
                                                dtype=args.params_dtype))
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight, self.output_size, self.input_size,
                self.input_size_per_partition, 1, init_method,
                stride=stride, return_master_weight=keep_master_weight_for_test)
        else:
            self.weight = Parameter(torch.empty(
                self.output_size, self.input_size_per_partition,
                device=get_current_device(), dtype=args.params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=1, stride=stride)
        if bias:
            if args.use_cpu_initialization:
                self.bias = Parameter(torch.empty(self.output_size,
                                                  dtype=args.params_dtype))
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size, device=get_current_device(),
                    dtype=args.params_dtype))
            if self.sequence_parallel:
                setattr(self.bias, 'sequence_parallel', True)

            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

        import habana_frameworks.torch.hpex.experimental.transformer_engine as te
        output_parallel_linear = te.Linear(
            self.input_size_per_partition,
            self.output_size,
            skip_weight_param_allocation=True,
            bias=False,
            minimize_memory=not args.cache_fp8_weight)
        self.output_parallel_linear = FP8ModuleRunner(
            output_parallel_linear,
            args.use_hpu_graphs,
            args.hpu_fp8_measure_interval,
            args.cache_fp8_weight_fwd)


    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel or self.is_expert_without_slicing:
            input_parallel = input_
        else:
            assert not self.sequence_parallel
            input_parallel = scatter_to_tensor_model_parallel_region(input_)

        # Matrix multiply.
        output_parallel = flatten_input(self.output_parallel_linear)(input_parallel, self.weight)

        if self.sequence_parallel:
            output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
        else:
            if self.is_expert_without_slicing:  # non-expert only tensor-parallelism
                output_ = output_parallel
            else:
                output_ = reduce_from_tensor_model_parallel_region(output_parallel)

        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias
