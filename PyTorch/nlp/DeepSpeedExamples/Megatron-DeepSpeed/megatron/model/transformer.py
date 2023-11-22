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

"""Transformer."""
import math
import torch
import torch.nn.functional as F

from megatron import get_args
from megatron import mpu
from .module import MegatronModule
from megatron.model.enums import AttnMaskType, LayerType, AttnType
from megatron.enums import PositionEmbeddingType
from megatron.model import LayerNorm, RMSNorm
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.fused_bias_gelu import bias_gelu_impl
from megatron.model.utils import attention_mask_func, openai_gelu, erf_gelu
from megatron.global_vars import get_current_device
from torch import distributed as dist
import deepspeed
from deepspeed.moe.layer import MoE
from .positional_embeddings import RotaryEmbedding, apply_rotary_pos_emb_torch, apply_rotary_pos_emb
import hashlib

try:
    from einops import rearrange
except ImportError:
    rearrange = None
import habana_frameworks.torch.hpu as ht

# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

def verify_tp_workers(tensor, layer_number, location, hash=False):

    tensor = mpu.gather_from_tensor_model_parallel_region(torch.clone(tensor))

    if mpu.get_tensor_model_parallel_rank() == 0:
        split = mpu.split_tensor_along_last_dim(tensor, mpu.get_tensor_model_parallel_world_size(), True)
        if hash:
            hashed_split = list(map(lambda x: hashlib.md5(x.to('cpu').to(torch.float32).detach().numpy().tobytes()).hexdigest(), split))
        else:
            for t in range(len(split[1:])):
                if not ((hash and hashed_split[0] == hashed_split[t]) or ((not hash) and torch.equal(split[0], split[t + 1]))):
                    torch.save(split[0].detach().cpu(), "inconsistent_tp_0.pt")
                    torch.save(split[t + 1].detach().cpu(), "inconsistent_tp_" + str(t + 1) + ".pt")
                    raise RuntimeError(f"TP workers disagree at layer: " + str(layer_number) + " " + location)

""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
"""

class ParallelMLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    """

    def __init__(self, init_method, output_layer_init_method, moe=False, enable_expert_tensor_parallelism=False):
        super(ParallelMLP, self).__init__()
        args = get_args()

        self.use_swiglu = args.activation_func_type == 'swiglu'

        # Project to 4h.
        self.dense_h_to_4h = mpu.ColumnParallelLinear(
            args.hidden_size,
            args.ffn_hidden_size,
            bias = not args.no_bias,
            gather_output=False,
            init_method=init_method,
            moe=moe,
            enable_expert_tensor_parallelism=enable_expert_tensor_parallelism,
            sequence_parallel=args.sequence_parallel)

        if self.use_swiglu:
            self.dense_h_to_4h_swiglu = mpu.ColumnParallelLinear(
                args.hidden_size,
                args.ffn_hidden_size,
                bias = not args.no_bias,
                gather_output=False,
                init_method=init_method,
                moe=moe,
                enable_expert_tensor_parallelism=enable_expert_tensor_parallelism,
                sequence_parallel=args.sequence_parallel
                )

            self.activation_func = F.silu
        else:
            self.activation_func = F.gelu

        if args.openai_gelu:
            self.activation_func = openai_gelu
        elif args.onnx_safe:
            self.activation_func = erf_gelu

        # Project back to h.
        self.dense_4h_to_h = mpu.RowParallelLinear(
            args.ffn_hidden_size,
            args.hidden_size,
            bias = not args.no_bias,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=False,
            moe=moe,
            enable_expert_tensor_parallelism=enable_expert_tensor_parallelism,
            sequence_parallel=args.sequence_parallel)

    def forward(self, hidden_states):

        # [s, b, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)

        intermediate_parallel = self.activation_func(intermediate_parallel)

        if self.use_swiglu:
            intermediate_parallel = intermediate_parallel * self.dense_h_to_4h_swiglu(hidden_states)

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias


class FusedCoreAttention(MegatronModule):
    def __init__(self, layer_number, attn_mask_type=AttnMaskType.padding):
        super(FusedCoreAttention, self).__init__()
        print("*************** Using FusedSDPA ******************")
        assert rearrange is not None, 'Please install einops first, e.g., with pip install einops'
        args = get_args()

        self.num_attention_heads = args.num_attention_heads

        projection_size = args.kv_channels * args.num_attention_heads

        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(
            projection_size, world_size)
        self.hidden_size_per_attention_head = mpu.divide(
            projection_size, args.num_attention_heads)
        self.num_attention_heads_per_partition = mpu.divide(
            args.num_attention_heads, world_size)

        self.attention_dropout = args.attention_dropout
        self.bf16 = args.bf16

        self.position_embedding_type = args.position_embedding_type
        if self.position_embedding_type == PositionEmbeddingType.rotary:
            self.rotary_emb = RotaryEmbedding(self.hidden_size_per_attention_head,
                                              precision=args.params_dtype)

        self.recompute = args.use_fused_sdpa_with_recompute
        from habana_frameworks.torch.hpex.kernels import FusedSDPA
        self.fused_sdpa = FusedSDPA

    def forward(self, query_layer, key_layer, value_layer, attention_mask,
                layer_past=None, get_key_value=False, alibi=None):

        assert alibi is None, "FusedSDPA is not compatible with Attention with Linear Biases (ALiBi)"

        # Rotary embeddings
        if self.position_embedding_type == PositionEmbeddingType.rotary:
            batch_size = query_layer.shape[1]
            seq_len    = query_layer.shape[0]
            num_heads =  query_layer.shape[2]
            query_layer, key_layer = [rearrange(x, 's b n h -> s (b n) h') for x in [query_layer, key_layer]]
            apply_rotary_fn = apply_rotary_pos_emb_torch if self.bf16 else apply_rotary_pos_emb

            offset = 0
            if layer_past is not None and layer_past.numel() > 0:
                offset = layer_past[0].shape[0]
                seq_len += offset
            cos, sin = self.rotary_emb(value_layer, seq_len=seq_len)
            query_layer, key_layer = apply_rotary_fn(query_layer, key_layer, cos, sin, offset=offset)
            query_layer, key_layer = [rearrange(x, 's (b n) h -> s b n h', b=batch_size, n=num_heads) for x in [query_layer, key_layer]]


        q, k, v = [rearrange(x, 's b n h -> b n s h') for x in [query_layer, key_layer, value_layer]]

        with ht.sdp_kernel(enable_recompute = self.recompute):
            context_layer = self.fused_sdpa.apply(q, k, v, None, self.attention_dropout, True, None)

        context_layer = rearrange(context_layer, 'b n s h -> s b n h')
        context_layer = rearrange(context_layer, 's b n h -> s b (n h)').contiguous()
        return context_layer

class CoreAttention(MegatronModule):
    def __init__(self, layer_number, attn_mask_type=AttnMaskType.padding):
        super(CoreAttention, self).__init__()
        args = get_args()
        self.fp16 = args.fp16
        self.bf16 = args.bf16
        self.position_embedding_type = args.position_embedding_type

        self.apply_query_key_layer_scaling = args.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number+1)
        self.attn_mask_type = attn_mask_type
        self.num_attention_heads = args.num_attention_heads

        projection_size = args.kv_channels * args.num_attention_heads

        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(
            projection_size, world_size)
        self.hidden_size_per_attention_head = mpu.divide(
            projection_size, args.num_attention_heads)
        self.num_attention_heads_per_partition = mpu.divide(
            args.num_attention_heads, world_size)

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16, self.bf16,
            self.attn_mask_type,
            args.masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(args.attention_dropout) if args.attention_dropout != 0 else None

        if self.position_embedding_type == PositionEmbeddingType.rotary:
            self.rotary_emb = RotaryEmbedding(self.hidden_size_per_attention_head,
                                              precision=args.params_dtype)

    def sub_forward(self, query_layer, key_layer, value_layer, attention_mask,
                layer_past, get_key_value, alibi):

        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================
        # [b, np, sq, sk]
        output_size = (query_layer.size(1),
                       query_layer.size(2),
                       query_layer.size(0),
                       key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2],
                                       output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3],
                                   output_size[0] * output_size[1], -1)

        # pre-allocating result tensor: [b * np, sq, sk]
        if alibi is None:
            matmul_result = torch.empty(
                output_size[0]*output_size[1],
                output_size[2],
                output_size[3],
                dtype=query_layer.dtype,
                device=get_current_device())
        else:
            matmul_result = alibi[:output_size[0]*output_size[1], :, :output_size[3]]

        # Rotary embeddings
        if self.position_embedding_type == PositionEmbeddingType.rotary:
            apply_rotary_fn = apply_rotary_pos_emb_torch if self.bf16 else apply_rotary_pos_emb

            seq_len = key_layer.shape[0]
            offset = 0
            if layer_past is not None and layer_past.numel() > 0:
                offset = layer_past[0].shape[0]
                seq_len += offset
            cos, sin = self.rotary_emb(value_layer, seq_len=seq_len)
            query_layer, key_layer = apply_rotary_fn(query_layer, key_layer, cos, sin, offset=offset)

        # Raw attention scores. [b * np, sq, sk]
        beta = 0.0
        if alibi is not None:
            beta = 1.0
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),   # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=beta, alpha=(1.0/self.norm_factor))

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        if get_key_value:
            with torch.no_grad():
                if layer_past is not None:
                    attention_mask = attention_mask[
                        ...,
                        attention_scores.size(3) - 1,
                        :attention_scores.size(3)].unsqueeze(2)
                else:
                    attention_mask = attention_mask[
                        ...,
                        :attention_scores.size(3),
                        :attention_scores.size(3)]

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        if self.attention_dropout is not None:
            with mpu.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1),
                       value_layer.size(2),
                       query_layer.size(0),
                       value_layer.size(3))

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0),
                                       output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                               output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        return context_layer, output_size

    def forward(self, query_layer, key_layer, value_layer, attention_mask,
                layer_past=None, get_key_value=False, alibi=None):

        context_layer, output_size = self.sub_forward(query_layer, key_layer, value_layer,
                                                      attention_mask, layer_past, get_key_value, alibi)

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class ParallelAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(self, init_method,
                 output_layer_init_method, layer_number,
                 attention_type=AttnType.self_attn,
                 attn_mask_type=AttnMaskType.padding):
        super(ParallelAttention, self).__init__()
        args = get_args()

        self.layer_number = layer_number
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.use_gqa = (self.num_attention_heads != self.num_key_value_heads)
        projection_size = args.kv_channels * args.num_attention_heads

        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(projection_size,
                                                    world_size)
        self.hidden_size_per_attention_head = mpu.divide(
            projection_size, args.num_attention_heads)
        self.num_attention_heads_per_partition = mpu.divide(
            args.num_attention_heads, world_size)

        # Per GQA head and per partition values
        if self.use_gqa:
            kv_projection_size = args.kv_channels * args.num_key_value_heads
            self.num_key_value_heads_per_partition = mpu.divide(
                args.num_key_value_heads, world_size)
            self.num_key_value_groups = mpu.divide(
                args.num_attention_heads, args.num_key_value_heads)
            assert self.hidden_size_per_attention_head == mpu.divide(
                kv_projection_size, args.num_key_value_heads)

        # Strided linear layer.
        if attention_type == AttnType.self_attn and not self.use_gqa:
            self.query_key_value = mpu.ColumnParallelLinear(
                args.hidden_size,
                3 * projection_size,
                bias = not args.no_bias,
                gather_output=False,
                init_method=init_method,
                sequence_parallel=args.sequence_parallel)
        elif attention_type == AttnType.self_attn and self.use_gqa:
            self.query = mpu.ColumnParallelLinear(
                args.hidden_size,
                projection_size,
                bias = not args.no_bias,
                gather_output=False,
                init_method=init_method,
                sequence_parallel=args.sequence_parallel)
            self.key_value = mpu.ColumnParallelLinear(
                args.hidden_size,
                2 * kv_projection_size,
                bias = not args.no_bias,
                gather_output=False,
                init_method=init_method,
                sequence_parallel=args.sequence_parallel)
        else:
            assert attention_type == AttnType.cross_attn
            self.query = mpu.ColumnParallelLinear(
                args.hidden_size,
                projection_size,
                bias = not args.no_bias,
                gather_output=False,
                init_method=init_method,
                sequence_parallel=args.sequence_parallel)

            self.key_value = mpu.ColumnParallelLinear(
                args.hidden_size,
                2 * projection_size,
                bias = not args.no_bias,
                gather_output=False,
                init_method=init_method,
                sequence_parallel=args.sequence_parallel)

        self.checkpoint_core_attention = \
            args.checkpoint_activations_granularity == 'selective'

        if args.use_fused_sdpa:
            if args.use_fused_sdpa_with_recompute:
                assert self.checkpoint_core_attention is False, "Please use either use_fused_sdpa_with_recompute or activation checkpointing"
            self.core_attention = FusedCoreAttention(self.layer_number, self.attn_mask_type)
        else:
            self.core_attention = CoreAttention(self.layer_number, self.attn_mask_type)

        # Output.
        self.dense = mpu.RowParallelLinear(
            projection_size,
            args.hidden_size,
            bias = not args.no_bias,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add= False,
            sequence_parallel= args.sequence_parallel)

        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint

    def _checkpointed_attention_forward(
            self, query_layer, key_layer, value_layer, attention_mask, alibi=None):

        """ Forward method with activation checkpointing """

        def custom_forward(*inputs):
            _query_layer = inputs[0]
            _key_layer = inputs[1]
            _value_layer = inputs[2]
            _attention_mask = inputs[3]
            _alibi = inputs[4]
            output_ = self.core_attention(
                _query_layer, _key_layer, _value_layer, _attention_mask,
                layer_past=None, get_key_value=False, alibi=_alibi)
            return output_

        hidden_states = mpu.checkpoint(
            custom_forward,
            query_layer, key_layer, value_layer, attention_mask, alibi)

        return hidden_states

    def repeat_kv(self, hidden_states, n_rep):
        slen, batch, num_key_value_heads_per_partition, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, :, None, :].expand(
            slen, batch, num_key_value_heads_per_partition, n_rep, head_dim)
        return hidden_states.reshape(slen, batch,
                                     num_key_value_heads_per_partition * n_rep,
                                     head_dim)

    def sub_forward(self, hidden_states, attention_mask, layer_past=None,
                get_key_value=False, encoder_output=None, alibi=None):
        # hidden_states: [sq, b, h]

        # =====================
        # Query, Key, and Value
        # =====================

        if self.attention_type == AttnType.self_attn and not self.use_gqa:
            # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
            mixed_x_layer = self.query_key_value(hidden_states)

            # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer,
             key_layer,
             value_layer) = mpu.split_tensor_along_last_dim(mixed_x_layer, 3)
        elif self.attention_type == AttnType.self_attn and self.use_gqa:
            # Attention head [sq, b, h] --> [sq, b, hp]
            query_layer = self.query(hidden_states)
            # [sq, b, hp] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 self.hidden_size_per_attention_head)
            query_layer = query_layer.view(*new_tensor_shape)

            # Attention heads [sq, b, h] --> [sq, b, (np * 2 * hn)]
            mixed_kv_layer = self.key_value(hidden_states)
            # [sq, b, (np * 2 * hn)] --> [sq, b, np, 2 * hn]
            new_tensor_shape = mixed_kv_layer.size()[:-1] + \
                (self.num_key_value_heads_per_partition,
                 2 * self.hidden_size_per_attention_head)
            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)
            # [sq, b, np, 2 * hn] --> 2 [sq, b, np, hn]
            (key_layer,
             value_layer) = mpu.split_tensor_along_last_dim(
                 mixed_kv_layer, 2)

            # Repeat kv
            key_layer = self.repeat_kv(key_layer, self.num_key_value_groups)
            value_layer = self.repeat_kv(value_layer,
                                         self.num_key_value_groups)
            key_layer = key_layer.contiguous()
            value_layer = value_layer.contiguous()
        else:
            assert not self.use_gqa, 'GQA + cross-attn not tested yet'
            # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
            mixed_kv_layer = self.key_value(encoder_output)

            # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
            new_tensor_shape = mixed_kv_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 2 * self.hidden_size_per_attention_head)
            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
            (key_layer,
             value_layer) = mpu.split_tensor_along_last_dim(mixed_kv_layer, 2)

            # Attention head [sq, b, h] --> [sq, b, hp]
            query_layer = self.query(hidden_states)
            # [sq, b, hp] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 self.hidden_size_per_attention_head)
            query_layer = query_layer.view(*new_tensor_shape)

        # ==================================
        # Adjust key and value for inference
        # ==================================

        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer),
                                   key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer),
                                     value_layer), dim=0)
        if get_key_value:
            present = (key_layer, value_layer)

        # ==================================
        # core attention computation
        # ==================================

        if self.checkpoint_core_attention:
            context_layer = self._checkpointed_attention_forward(
                query_layer, key_layer, value_layer, attention_mask, alibi)
        else:
            context_layer = self.core_attention(
                query_layer, key_layer, value_layer, attention_mask,
                layer_past, get_key_value, alibi)

        output, bias = self.dense(context_layer)

        if get_key_value:
            return output, bias, present

        return output, bias, None

    def forward(self, hidden_states, attention_mask, layer_past=None,
                get_key_value=False, encoder_output=None, alibi=None):

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias, present = self.sub_forward(hidden_states, attention_mask, layer_past,
                                                 get_key_value, encoder_output, alibi)

        if get_key_value:
            output = [output, present]

        return output, bias


def bias_dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, Tensor, Tensor, float, bool) -> Tensor
    x = x + bias if bias is not None else x
    if prob == 0:
        out = x
    else:
        out = torch.nn.functional.dropout(x , p=prob, training=training)
    out = residual + out
    return out

def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)
    return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_fused_train(x, bias, residual, prob):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(x, bias, residual, prob):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_add(x, bias, residual, prob, False)


class ParallelTransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(self, init_method, output_layer_init_method,
                 layer_number, layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding, num_experts=1):
        args = get_args()

        super(ParallelTransformerLayer, self).__init__()
        self.layer_number = layer_number
        self.layer_type = layer_type

        self.apply_residual_connection_post_layernorm \
            = args.apply_residual_connection_post_layernorm

        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection

        # Layernorm/RMSNorm on the input data.
        norm_class = RMSNorm if args.layernorm_type == "rmsnorm" else LayerNorm
        self.input_layernorm = norm_class(args.hidden_size,
                                          eps=args.layernorm_epsilon,
                                          sequence_parallel=args.sequence_parallel)

        # Self attention.
        self.attention = ParallelAttention(
            init_method,
            output_layer_init_method,
            layer_number,
            attention_type=AttnType.self_attn,
            attn_mask_type=self_attn_mask_type)
        self.hidden_dropout = args.hidden_dropout
        self.bias_dropout_fusion = args.bias_dropout_fusion

        # Layernorm/RMSNorm on the attention output
        norm_class = RMSNorm if args.layernorm_type == "rmsnorm" else LayerNorm
        self.post_attention_layernorm = norm_class(args.hidden_size,
                                                   eps=args.layernorm_epsilon,
                                                   sequence_parallel=args.sequence_parallel)

        if self.layer_type == LayerType.decoder:
            self.inter_attention = ParallelAttention(
                init_method,
                output_layer_init_method,
                layer_number,
                attention_type=AttnType.cross_attn)
            # Layernorm on the attention output.
            self.post_inter_attention_layernorm = LayerNorm(
                args.hidden_size,
                eps=args.layernorm_epsilon,
                sequence_parallel=args.sequence_parallel)

        self.num_experts = num_experts
        # MLP
        if self.num_experts <= 1:
            self.mlp = ParallelMLP(init_method,
                               output_layer_init_method)
        else:
            enable_expert_tensor_parallelism = args.enable_expert_tensor_parallelism
            self.mlp = MoE(args.hidden_size,
                            ParallelMLP(init_method,
                                output_layer_init_method=output_layer_init_method,
                                moe=True,
                                enable_expert_tensor_parallelism=enable_expert_tensor_parallelism),
                            num_experts=self.num_experts,
                            ep_size=args.moe_expert_parallel_size,
                            k=args.topk,
                            use_residual=(args.mlp_type == 'residual'),
                            capacity_factor=args.moe_train_capacity_factor,
                            eval_capacity_factor=args.moe_eval_capacity_factor,
                            min_capacity=args.moe_min_capacity,
                            drop_tokens=args.moe_token_dropping, use_tutel=args.use_tutel,
                            enable_expert_tensor_parallelism=enable_expert_tensor_parallelism)

        # Alibi
        if args.position_embedding_type == PositionEmbeddingType.alibi:
            assert args.micro_batch_size == args.eval_micro_batch_size, \
                "ParallelTransformerLayer (init) - Unsupported for split micro batch size"
            self.alibi = self._build_alibi_tensor(args.seq_length, args.num_attention_heads, args.micro_batch_size).to(get_current_device())
            if args.params_dtype == torch.float16:
                self.alibi = self.alibi.to(torch.float16)
            elif args.params_dtype == torch.bfloat16:
                self.alibi = self.alibi.to(torch.bfloat16)
        else:
            self.alibi = None

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                layer_past=None, get_key_value=False):
        # hidden_states: [b, s, h]
        args = get_args()
        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        if args.use_hpu and args.verify_tp_workers:
            verify_tp_workers(layernorm_output, self.layer_number, "after input layernorm", args.verify_tp_workers_hash)

        # Self attention.
        attention_output, attention_bias = \
            self.attention(layernorm_output,
                                attention_mask,
                                layer_past=layer_past,
                                get_key_value=get_key_value,
                                alibi=self.alibi)
        if args.use_hpu and args.verify_tp_workers:
            verify_tp_workers(attention_output, self.layer_number, "after attention", args.verify_tp_workers_hash)

        if get_key_value:
            attention_output, presents = attention_output

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # jit scripting for a nn.module (with dropout) is not
        # trigerring the fusion kernel. For now, we use two
        # different nn.functional routines to account for varying
        # dropout semantics during training and inference phases.
        if torch.cuda.is_available() and self.bias_dropout_fusion:
            if self.training:
                bias_dropout_add_func = bias_dropout_add_fused_train
            else:
                bias_dropout_add_func = bias_dropout_add_fused_inference
        else:
            bias_dropout_add_func = get_bias_dropout_add(self.training)

        # re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            layernorm_input = bias_dropout_add_func(
                attention_output,
                attention_bias.expand_as(residual) if attention_bias is not None else None,
                residual,
                self.hidden_dropout)

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        if self.layer_type == LayerType.decoder:
            attention_output, attention_bias = \
                self.inter_attention(layernorm_output,
                                     enc_dec_attn_mask,
                                     encoder_output=encoder_output)
            # residual connection
            if self.apply_residual_connection_post_layernorm:
                residual = layernorm_output
            else:
                residual = layernorm_input

            # re-enable torch grad to enable fused optimization.
            with torch.enable_grad():
                layernorm_input = bias_dropout_add_func(
                    attention_output,
                    attention_bias.expand_as(residual) if attention_bias is not None else None,
                    residual,
                    self.hidden_dropout)

            # Layer norm post the decoder attention
            layernorm_output = self.post_inter_attention_layernorm(layernorm_input)

        # MLP.
        moe_loss = torch.tensor(0.0, device=layernorm_output.device, dtype=layernorm_output.dtype)
        mlp_bias = torch.tensor(0.0, device=layernorm_output.device, dtype=layernorm_output.dtype)


        if args.use_hpu and args.verify_tp_workers:
            verify_tp_workers(layernorm_input, self.layer_number, "before mlp", args.verify_tp_workers_hash)

        if self.num_experts == 1:
            mlp_output, mlp_bias = self.mlp(layernorm_output)
        else:
            mlp_output, moe_loss, _ = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        # re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            #if self.num_experts <= 1:
            output = bias_dropout_add_func(
                    mlp_output,
                    mlp_bias.expand_as(residual) if mlp_bias is not None else None,
                    residual,
                    self.hidden_dropout)
            #else:
            #    output = mlp_output + residual

        if get_key_value:
            output = [output, presents]

        if args.use_hpu and args.verify_tp_workers:
            verify_tp_workers(output, self.layer_number, "after transformer", args.verify_tp_workers_hash)
        return output, moe_loss

    @staticmethod
    def _build_alibi_tensor(max_seq_len, num_attention_heads, batch_size):
        # Based on https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
        """Returns tensor shaped (batch_size * num_attention_heads, 1, max_seq_len)"""

        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = (2 ** (-2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio ** i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][
                                                                   :n - closest_power_of_2]

        slopes = torch.Tensor(get_slopes(num_attention_heads))
        alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_seq_len).unsqueeze(0).unsqueeze(0).expand(
            num_attention_heads, -1, -1)

        #Select the part of the tensor that corresponds to our tensor parallel index.
        tp_world_size = mpu.get_tensor_model_parallel_world_size()
        tp_index = mpu.get_tensor_model_parallel_rank()
        alibi = alibi.reshape((tp_world_size, -1, *alibi.shape[1:]))[tp_index]

        alibi = alibi.repeat(batch_size, 1, 1)
        return alibi


class ParallelTransformerLayerPipe(ParallelTransformerLayer):
    """Extends ParallelTransformerLayer to forward attention_mask through the pipeline.

    Forward has two usages that affect attention mask communication:

    1) forward((input, attn_mask) , **kwargs) -> (output, mask)
       When the attention mask is provided as the second positional
       argument, typical pipeline behavior is used and both the output
       *and* mask are returned in a tuple. This tuple is then forwarded
       to the next stage in the pipeline.

       This version is useful if masks are dynamic.

    2) forward(input, **kwargs) -> output
       When the mask is static over all samples, it is advantageous to
       cache the mask and avoid communicating it.

       If no mask is provided, the module will query `self._args.attn_mask`
       for the mask and only return `super().forward(...)`
    """
    def forward(self, inputs, **kwargs):
        assert torch.is_tensor(inputs) or isinstance(inputs, tuple)
        if torch.is_tensor(inputs) or len(inputs) == 1:
            # No attention mask forwarded, search for args.attn_mask
            if not hasattr(self, '_args'):
                self._args = get_args()
            hidden_states, attention_mask = inputs, self._args.attn_mask
            # HACK: currently MoE model does not support pipeline parallel, so
            # here we just ignore the moe_loss returned by forward()
            return super().forward(hidden_states, attention_mask, **kwargs)[0]
        elif len(inputs) == 2:
            # Attention mask is an activation.
            hidden_states, attention_mask = inputs[0], inputs[1]
            # HACK: currently MoE model does not support pipeline parallel, so
            # here we just ignore the moe_loss returned by forward()
            return super().forward(*inputs, **kwargs)[0], attention_mask
        else:
            raise RuntimeError('Received more inputs than understood.')


class ParallelTransformer(MegatronModule):
    """Transformer class."""

    def __init__(self, init_method, output_layer_init_method,
                 layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 pre_process=True, post_process=True,
                 num_experts=[1]):

        super(ParallelTransformer, self).__init__()
        args = get_args()

        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection
        self.pre_process = pre_process
        self.post_process = post_process
        self.input_tensor = None
        self.ds_inference = args.ds_inference

        # Store activation checkpoiting flag.
        self.checkpoint_activations = args.checkpoint_activations \
                                      and args.checkpoint_activations_granularity == "full"
        self.checkpoint_num_layers = args.checkpoint_num_layers

        # Number of layers.
        assert args.num_layers % mpu.get_pipeline_model_parallel_world_size() == 0, \
            'num_layers must be divisible by pipeline_model_parallel_size'
        self.num_layers = args.num_layers // mpu.get_pipeline_model_parallel_world_size()

        # Transformer layers.
        def build_layer(layer_number, n_e):
            return ParallelTransformerLayer(
                init_method,
                output_layer_init_method,
                layer_number,
                layer_type=layer_type,
                self_attn_mask_type=self_attn_mask_type,
                num_experts=n_e)

        if args.virtual_pipeline_model_parallel_size is not None:
            assert args.num_layers % args.virtual_pipeline_model_parallel_size == 0, \
                'num_layers_per_stage must be divisible by ' \
                'virtual_pipeline_model_parallel_size'
            # Number of layers in each model chunk is the number of layers in the stage,
            # divided by the number of model chunks in a stage.
            self.num_layers = self.num_layers // args.virtual_pipeline_model_parallel_size
            # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0]  [2]  [4]  [6]
            # Stage 1: [1]  [3]  [5]  [7]
            # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0, 1]  [4, 5]
            # Stage 1: [2, 3]  [6, 7]
            offset = mpu.get_virtual_pipeline_model_parallel_rank() * (
                args.num_layers // args.virtual_pipeline_model_parallel_size) + \
                (mpu.get_pipeline_model_parallel_rank() * self.num_layers)
        else:
            # Each stage gets a contiguous set of layers.
            offset = mpu.get_pipeline_model_parallel_rank() * self.num_layers

        assert len(num_experts) == 1 or len(num_experts) == self.num_layers // args.expert_interval, \
        'num_experts must be either a single value or a list of the same length as the number of MoE layers'

        # Create the list of MoE experts
        if len(num_experts) == 1:
            num_experts = num_experts * (self.num_layers // args.expert_interval)

        self.layers = []
        # Build the layers
        for i in range(self.num_layers):
            layer_num = i + 1 + offset
            if layer_num % args.expert_interval == 0:
                n_e = num_experts[(layer_num-1) // args.expert_interval]
            else:
                n_e = 1
            self.layers.append(build_layer(layer_num, n_e))

        self.layers = torch.nn.ModuleList(self.layers)

        if self.post_process:
            # Final layer norm/RMSNorm before output.
            norm_class = RMSNorm if args.layernorm_type == "rmsnorm" else LayerNorm
            self.final_layernorm = norm_class(args.hidden_size, eps=args.layernorm_epsilon)

        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint
    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def _checkpointed_forward(self, hidden_states, attention_mask,
                              encoder_output, enc_dec_attn_mask):
        """Forward method with activation checkpointing."""
        def custom(start, end):
            def custom_forward(*inputs):
                x_ = inputs[0]
                attention_mask = inputs[1]
                encoder_output = inputs[2]
                enc_dec_attn_mask = inputs[3]
                moe_losses = []
                for index in range(start, end):
                    layer = self._get_layer(index)
                    x_, moe_loss = layer(x_, attention_mask, encoder_output, enc_dec_attn_mask)
                    moe_losses.append(moe_loss)
                return (x_, *moe_losses)
            return custom_forward

        moe_losses = []
        # Make sure memory is freed.
        mpu.reset_checkpointed_activations_memory_buffer()
        l = 0
        while l < self.num_layers:
            hidden_states, *local_moe_losses = mpu.checkpoint(
                custom(l, l + self.checkpoint_num_layers),
                hidden_states, attention_mask, encoder_output, enc_dec_attn_mask)
            moe_losses.extend(local_moe_losses)
            l += self.checkpoint_num_layers

        return hidden_states, moe_losses

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(self, hidden_states, attention_mask, layer_past=None,
                get_key_value=False, encoder_output=None, enc_dec_attn_mask=None):

        # Checks.
        if layer_past is not None:
            assert get_key_value, \
                'for not None values in layer_past, ' \
                'expected get_key_value to be set'
        if get_key_value:
            assert not self.checkpoint_activations, \
                'get_key_value does not work with ' \
                'activation checkpointing'

        # Reza's note: DeepSpeed inference does not support transposes
        if not self.ds_inference:
            if self.pre_process:
                # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
                # If the input flag for fp32 residual connection is set, convert for float.
                if self.fp32_residual_connection:
                    hidden_states = hidden_states.transpose(0, 1).contiguous().float()
                # Otherwise, leave it as is.
                else:
                    hidden_states = hidden_states.transpose(0, 1).contiguous()
            else:
                # See set_input_tensor()
                hidden_states = self.input_tensor

            if encoder_output is not None:
                 encoder_output = encoder_output.transpose(0, 1).contiguous()

        moe_losses = []
        if self.checkpoint_activations:
            hidden_states, moe_losses = self._checkpointed_forward(hidden_states,
                                                       attention_mask,
                                                       encoder_output,
                                                       enc_dec_attn_mask)
        else:
            if get_key_value:
                presents = []
            for index in range(self.num_layers):
                layer = self._get_layer(index)
                past = None
                if layer_past is not None:
                    past = layer_past[index]
                hidden_states = layer(hidden_states,
                                      attention_mask,
                                      encoder_output=encoder_output,
                                      enc_dec_attn_mask=enc_dec_attn_mask,
                                      layer_past=past,
                                      get_key_value=get_key_value)
                if not self.ds_inference:
                    hidden_states, moe_loss = hidden_states
                    moe_losses.append(moe_loss)
                if get_key_value:
                    hidden_states, present = hidden_states
                    presents.append(present)

        # Final layer norm.
        if self.post_process:
            if not self.ds_inference:
                # Reverting data format change [s b h] --> [b s h].
                hidden_states = hidden_states.transpose(0, 1).contiguous()
            output = self.final_layernorm(hidden_states)
        else:
            output = hidden_states
        if get_key_value:
            output = [output, presents]

        return (output, *moe_losses)
