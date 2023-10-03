# coding=utf-8
# Copyright 2021 The EleutherAI and HuggingFace Teams. All rights reserved.
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

###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################
# Changes:
# - remove dead code (functions and tensors unused in MLPerf GPT-J benchmark)
# - remove training support
# - remove float16 support
# - remove device-parallelism support
# - use apply_rotary_pos_emb kernel on HPU
# - remove duplicated operations (for example: calculate sin and cos only once)
# - reshape tensors from 4D to 3D for better performance
# - use optimized softmax
# - adjust the code to HPU graphs
# - use optimized kernels for KV cache reorder
# - introduce support for fp8 KV cache
# - remove unnecessary int64 usage (use int32 or bfloat16)

from typing import Optional, Tuple, Union
import numpy as np

import torch
import torch.fx
import torch.utils.checkpoint
from torch import nn

try:
    from habana_frameworks.torch.hpex.kernels import apply_rotary_pos_emb as apply_rotary_pos_emb_hpu
    from habana_frameworks.torch.hpex.kernels import RotaryPosEmbeddingMode
except ImportError:
    print("Not using HPU kernel for apply_rotary_pos_emb")
    apply_rotary_pos_emb_hpu = None

from habana_frameworks.torch.hpex.kernels import CustomSoftmax as FastSoftmax

try:
    in_place_interleave_hpu = torch.ops.hpu.in_place_interleave_
except AttributeError:
    print(f"Not using HPU kernel for in_place_interleave_")
    in_place_interleave_hpu = None

__package__ = 'transformers.models.gptj'

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from .configuration_gptj import GPTJConfig



def create_sinusoidal_positions(num_pos: int, dim: int) -> torch.Tensor:
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(num_pos, dtype=torch.float), inv_freq).float()
    return torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1)


def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb(tensor: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    if apply_rotary_pos_emb_hpu is None:
        return (tensor * cos) + (rotate_every_two(tensor) * sin)
    else:
        return apply_rotary_pos_emb_hpu(tensor, cos, sin, None, 0, RotaryPosEmbeddingMode.PAIRWISE)

class Matmul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)

class BatchMatmul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.bmm(x,y)

class CacheUpdateFp8(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prev, cur, dim, idx):
        orig_cur = cur
        cur_fp8, amax = torch.ops.hpu.cast_to_fp8_v2(cur,None,False, False)
        if prev.shape[0] != cur_fp8.shape[0]:
            assert prev.shape[0] % cur_fp8.shape[0] == 0, f'Cannot update kv-cache. BatchSize changed! {prev.shape[0]} vs {cur_fp8.shape[0]}'
            # Repeat to accomodate bs/beam changes
            repeats = (prev.shape[0] // cur_fp8.shape[0], 1, 1, 1)
            cur_fp8 = torch.ops.hpu.fp8_repeat_v2(cur_fp8, repeats)
            assert prev.shape == cur_fp8.shape, f'Cannot update kv-cache. BatchSize changed! {prev.shape[0]} vs {cur_fp8.shape[0]}'
            # Initialize
            torch.ops.hpu.fp8_copy_(prev, cur_fp8)
            return orig_cur
        else:
            assert cur_fp8.shape[2] == 1, f'Cannot update kv-cache. Unsupported shapes. prev:{prev.shape} cur:{cur_fp8.shape}'
            torch.ops.hpu.fp8_index_copy_(prev, dim, idx - 1, cur_fp8)
            prev_bf16 = torch.ops.hpu.cast_from_fp8(prev, None, cur.dtype)
            return prev_bf16

class CacheUpdate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prev, cur, dim, idx):
        orig_cur = cur
        if prev.shape[0] != cur.shape[0]:
            assert prev.shape[0] % cur.shape[0] == 0, f'Cannot update kv-cache. BatchSize changed! {prev.shape[0]} vs {cur.shape[0]}'
            # Repeat to accomodate bs/beam changes
            cur = cur.repeat(prev.shape[0] // cur.shape[0], 1, 1, 1)
            assert prev.shape == cur.shape, f'Cannot update kv-cache. BatchSize changed! {prev.shape[0]} vs {cur.shape[0]}'
            # Initialize
            prev.copy_(cur)
            return orig_cur
        else:
            assert cur.shape[2] == 1, f'Cannot update kv-cache. Unsupported shapes. prev:{prev.shape} cur:{cur.shape}'
            return prev.index_copy_(dim, idx - 1, cur)


class GPTJAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.matmul_qk = BatchMatmul()
        self.matmul_av = Matmul()

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.past_key = {}
        self.past_value = {}
        self.kv_cache_fp8 = False
        self.v_update = CacheUpdate()
        self.k_update = CacheUpdate()

        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and"
                f" `num_attention_heads`: {self.num_attention_heads})."
            )
        self.register_buffer("inv_scale_attn",
                torch.rsqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(torch.get_default_dtype()),
                persistent=False)
        self.inv_scale_attn_scalar = 1.0 / np.sqrt(self.head_dim)

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.rotary_dim = config.rotary_dim

    def _split_heads(self, tensor, num_attention_heads, attn_head_size, rotary):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        if rotary:
            return tensor
        if len(tensor.shape) == 5:
            return tensor.permute(0, 1, 3, 2, 4)  # (batch, blocks, head, block_length, head_features)
        elif len(tensor.shape) == 4:
            return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        else:
            raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")

    def _merge_heads(self, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        if len(tensor.shape) == 5:
            tensor = tensor.permute(0, 1, 3, 2, 4).contiguous()
        elif len(tensor.shape) == 4:
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
        else:
            raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")
        new_shape = tensor.size()[:-2] + (num_attention_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(
        self,
        query,
        key,
        value,
        attention_mask=None,
        start_end=None,
    ):
        batch_size, query_len, key_len = query.shape[0], query.shape[-2], key.shape[-2]

        # Reshape to 3D tensors
        query = query.reshape((batch_size * self.num_attention_heads, query_len, self.head_dim))
        key = key.reshape((batch_size * self.num_attention_heads, key_len, self.head_dim))
        value = value.reshape((batch_size * self.num_attention_heads, key_len, self.head_dim))

        attn_weights = self.matmul_qk(query, key.transpose(-1, -2))

        if query_len == 1:
            # next token
            attn_weights = attn_weights * self.inv_scale_attn
            attn_weights = attn_weights + attention_mask

            attn_weights = FastSoftmax.apply(attn_weights, 2) # optimized softmax (no LUTs)

        else:
            # first token
            attn_weights = torch.ops.hpu.scaled_masked_triangular_softmax(
                attn_weights,
                start_end,
                self.inv_scale_attn_scalar,
                self.num_attention_heads,
                False, # don't use max
                1 # optimized softmax (no LUTs)
            )

        attn_output = self.matmul_av(attn_weights, value)

        # Reshape back to 4D tensors
        attn_output = attn_output.reshape((batch_size, self.num_attention_heads) + attn_output.shape[1:])
        attn_weights = attn_weights.reshape((batch_size, self.num_attention_heads) + attn_weights.shape[1:])

        return attn_output, attn_weights


    def allocate_kv_cache(self, batch_size, seq_len, kv_cache_fp8):
        if (batch_size, seq_len) not in self.past_key.keys():
            device = self.k_proj.weight.device
            dtype = self.k_proj.weight.dtype
            shape = (batch_size, self.num_attention_heads, seq_len, self.head_dim)
            past_key = torch.empty(shape, dtype=dtype, device=device)
            past_value = torch.empty(shape, dtype=dtype, device=device)
            if kv_cache_fp8:
                self.kv_cache_fp8 = True
                self.past_value[(batch_size, seq_len)], amax = torch.ops.hpu.cast_to_fp8_v2(past_value, None, False, False)
                self.past_key[(batch_size, seq_len)], amax = torch.ops.hpu.cast_to_fp8_v2(past_key, None, False, False)
                self.v_update = CacheUpdateFp8()
                self.k_update = CacheUpdateFp8()

                import habana_frameworks.torch.core as htcore
                htcore.mark_step()
            else:
                self.past_key[(batch_size, seq_len)] = past_key
                self.past_value[(batch_size, seq_len)] = past_value

    def reorder_first_token(self, tensor):
        if in_place_interleave_hpu is not None:
            in_place_interleave_hpu(tensor)
        else:
            shape = tensor.shape
            l = []
            NUM_BEAMS=4
            for i in range(shape[0] // NUM_BEAMS):
                val = tensor[i, :, :, :].clone()
                for i in range(NUM_BEAMS):
                    l.append(val)
            updated = torch.cat(l, 0)
            updated = torch.reshape(updated, shape)
            tensor.copy_(updated)

    def reorder_kv_cache_first_token(self, kv_cache_shape):
        if self.past_key is None or kv_cache_shape not in self.past_key.keys():
            return (None, None)

        self.reorder_first_token(self.past_key[kv_cache_shape])
        self.reorder_first_token(self.past_value[kv_cache_shape])

        return (self.past_key[kv_cache_shape].shape, self.past_value[kv_cache_shape].shape)

    def reorder_kv_cache_next_token(self, start, end, beam_idx, kv_cache_shape):
        if self.past_key is None or kv_cache_shape not in self.past_key.keys():
            return (None, None)

        if self.kv_cache_fp8:
            torch.ops.hpu.fp8_kv_reorder_(self.past_key[kv_cache_shape], start, end, beam_idx)
            torch.ops.hpu.fp8_kv_reorder_(self.past_value[kv_cache_shape], start, end, beam_idx)
        else:
            torch.ops.hpu.kv_reorder_(self.past_key[kv_cache_shape], start, end, beam_idx)
            torch.ops.hpu.kv_reorder_(self.past_value[kv_cache_shape], start, end, beam_idx)

        return (self.past_key[kv_cache_shape].shape, self.past_value[kv_cache_shape].shape)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        token_idx: Optional[torch.Tensor] = None,
        reuse_cache: Optional[bool] = False,
        kv_cache_shape: Tuple[int, int] = None,
        sin: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        start_end: Optional[torch.Tensor] = None,
    ) -> Union[
        Tuple[torch.Tensor, Tuple[torch.Tensor]],
        Optional[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor, ...]]],
    ]:
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.num_attention_heads, self.head_dim, True)
        key = self._split_heads(key, self.num_attention_heads, self.head_dim, True)
        value = self._split_heads(value, self.num_attention_heads, self.head_dim, False)

        k_rot = key[:, :, :, : self.rotary_dim]
        k_pass = key[:, :, :, self.rotary_dim :]

        q_rot = query[:, :, :, : self.rotary_dim]
        q_pass = query[:, :, :, self.rotary_dim :]

        k_rot = apply_rotary_pos_emb(k_rot, sin, cos)
        q_rot = apply_rotary_pos_emb(q_rot, sin, cos)

        key = torch.cat([k_rot, k_pass], dim=-1)
        query = torch.cat([q_rot, q_pass], dim=-1)

        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)

        if layer_past is not None or reuse_cache:
            if reuse_cache:
                past_key, past_value = self.past_key[kv_cache_shape], self.past_value[kv_cache_shape]
            else:
                past_key, past_value = layer_past

            key = self.k_update(past_key, key, -2, token_idx)
            value = self.v_update(past_value, value, -2, token_idx)

        if use_cache is True:
            if reuse_cache:
                present = (key.shape, value.shape)
            else:
                present = (key, value)
        else:
            present = None

        # compute self-attention: V x Softmax(QK^T)
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, start_end)

        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class GPTJMLP(nn.Module):
    def __init__(self, intermediate_size, config):  # in MLP: intermediate_size= 4 * embed_dim
        super().__init__()
        embed_dim = config.n_embd

        self.fc_in = nn.Linear(embed_dim, intermediate_size)
        self.fc_out = nn.Linear(intermediate_size, embed_dim)

        self.act = ACT2FN["quick_gelu"]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[torch.FloatTensor]) -> torch.FloatTensor:
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPTJBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPTJAttention(config)
        self.mlp = GPTJMLP(inner_dim, config)

    def allocate_kv_cache(self, batch_size, seq_len, kv_cache_fp8):
        self.attn.allocate_kv_cache(batch_size, seq_len, kv_cache_fp8)

    def reorder_kv_cache_first_token(self, kv_cache_shape):
        return self.attn.reorder_kv_cache_first_token(kv_cache_shape)

    def reorder_kv_cache_next_token(self, start, end, beam_idx, kv_cache_shape):
        return self.attn.reorder_kv_cache_next_token(start, end, beam_idx, kv_cache_shape)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        token_idx: Optional[torch.Tensor] = None,
        reuse_cache: Optional[bool] = None,
        kv_cache_shape: Tuple[int, int] = None,
        sin: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        start_end: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states=hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            token_idx=token_idx,
            reuse_cache=reuse_cache,
            kv_cache_shape=kv_cache_shape,
            sin=sin,
            cos=cos,
            start_end=start_end,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_output + feed_forward_hidden_states + residual

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions)


class GPTJPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPTJConfig
    base_model_prefix = "transformer"
    is_parallelizable = True
    _no_split_modules = ["GPTJBlock"]
    _skip_keys_device_placement = "past_key_values"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear,)):
            # Slightly different from Mesh Transformer JAX which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class GPTJModel(GPTJPreTrainedModel):
    config_class = GPTJConfig
    base_model_prefix = "transformer"
    is_parallelizable = True
    _no_split_modules = ["GPTJBlock"]
    _skip_keys_device_placement = "past_key_values"

    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.n_embd
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim, dtype=torch.bfloat16)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPTJBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.register_buffer("embed_positions",
            create_sinusoidal_positions(self.config.max_position_embeddings, self.config.rotary_dim),
            persistent=False)
        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear,)):
            # Slightly different from Mesh Transformer JAX which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def allocate_kv_cache(self, batch_size, seq_len, kv_cache_fp8):
        for layer in self.h:
            layer.allocate_kv_cache(batch_size, seq_len, kv_cache_fp8)

    def reorder_kv_cache_first_token(self, kv_cache_shape):
        return tuple(layer.reorder_kv_cache_first_token(kv_cache_shape) for layer in self.h)

    def reorder_kv_cache_next_token(self, start, end, beam_idx, kv_cache_shape):
        return tuple(layer.reorder_kv_cache_next_token(start, end, beam_idx, kv_cache_shape) for layer in self.h)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        token_idx: Optional[torch.Tensor] = None,
        reuse_cache: Optional[bool] = None,
        kv_cache_shape: Tuple[int, int] = None,
        sin: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        start_end: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        # Attention mask.
        if attention_mask is not None:
            # TODO: try get value from GPTJAttention
            num_attention_heads = 16
            attention_mask = torch.repeat_interleave(
                attention_mask, num_attention_heads, 0, output_size=num_attention_heads*batch_size)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        hidden_states = inputs_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states=hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                token_idx=token_idx,
                reuse_cache=reuse_cache,
                kv_cache_shape=kv_cache_shape,
                sin=sin,
                cos=cos,
                start_end=start_end,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class GPTJForCausalLM(GPTJPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"h\.\d+\.attn\.masked_bias", r"h\.\d+\.attn\.bias"]
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPTJModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "token_type_ids": token_type_ids,
            }
        )

        return model_inputs

    def allocate_kv_cache(self, batch_size, seq_len, kv_cache_fp8):
        self.transformer.allocate_kv_cache(batch_size, seq_len, kv_cache_fp8)

    def reorder_kv_cache_first_token(self, kv_cache_shape):
        return self.transformer.reorder_kv_cache_first_token(kv_cache_shape)

    def reorder_kv_cache_next_token(self, start, end, beam_idx, kv_cache_shape):
        return self.transformer.reorder_kv_cache_next_token(start, end, beam_idx, kv_cache_shape)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        token_idx: Optional[torch.Tensor] = None,
        reuse_cache: Optional[bool] = None,
        trim_logits: Optional[bool] = None,
        kv_cache_shape: Tuple[int, int] = None,
        sin: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        start_end: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            token_idx=token_idx,
            reuse_cache=reuse_cache,
            kv_cache_shape=kv_cache_shape,
            sin=sin,
            cos=cos,
            start_end=start_end,
        )
        hidden_states = transformer_outputs[0]
        _, seq_len, _ = hidden_states.shape
        if seq_len > 1 and trim_logits:
            if token_idx is not None:
                hidden_states = hidden_states.index_select(1, token_idx - 1)
            else:
                hidden_states = hidden_states[:, -1, :]

        # make sure sampling in fp16 works correctly and
        # compute loss in fp32 to match with mesh-tf version
        # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return output

        return CausalLMOutputWithPast(
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
        [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )

