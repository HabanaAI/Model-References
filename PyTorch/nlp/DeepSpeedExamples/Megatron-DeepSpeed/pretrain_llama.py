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

"""Pretrain LLaMA"""

import torch
from megatron import get_args
from megatron import print_rank_0
from megatron import mpu
from megatron import get_tokenizer
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.model import LLaMAModel, LLaMAModelPipe
from megatron.training import pretrain
from megatron.global_vars import get_current_device
from megatron.enums import PositionEmbeddingType
import deepspeed
from deepspeed.runtime.utils import see_memory_usage
from pretrain_gpt import git_ds_info, pretrain, train_valid_test_datasets_provider, forward_step, get_batch_pipe
import os

def model_provider(pre_process=True, post_process=True, parallel_output=True):
    """Build the model."""

    print_rank_0('building LLaMA model ...')
    see_memory_usage(f"Before Building Model", force=True)

    args = get_args()
    if args.use_hpu:
        os.environ['DEEPSPEED_HPU_SYNC_INSIDE_INIT'] = "1"
        os.environ['DEEPSPEED_SYNC_MICRO_BATCH_STEP'] = "1"

    with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
                             remote_device=None if args.remote_device == 'none' else args.remote_device,
                             config_dict_or_path=args.deepspeed_config,
                             enabled=args.zero_stage == 3,
                             mpu=mpu):
        if args.deepspeed and not args.no_pipeline_parallel:

            # verify --deepspeed_activation_checkpointing
            # mandatory! otherwise the model uses fork() mapping to Megatron's RNGStatesTrackerSingleton
            # while LLaMAModelPipe uses DS checkpoint activations that uses DS's RNGStatesTracker
            if args.checkpoint_activations and args.checkpoint_activations_granularity == "full":
                assert args.deepspeed_activation_checkpointing, \
                    "Flag --deepspeed_activation_checkpointing is mandatory when using LLaMAModelPipe" \
                    " with checkpoint activations granularity full."

            model = LLaMAModelPipe(
                num_tokentypes=0,
                parallel_output=parallel_output
            )
            # This is a hack to give us a reference to get_batch_pipe from within training.py
            # We need to call model.set_batch_fn after deepspeed.initialize
            model._megatron_batch_fn = get_batch_pipe

            # Predompute the attention mask and store it in args. This avoids having to
            # pipeline it as an activation during training. The mask is constant, and thus
            # we can reuse it.
            current_device = get_current_device()
            attention_mask = torch.tril(torch.ones(
                (1, args.seq_length, args.seq_length), device=current_device)).view(
                    1, 1, args.seq_length, args.seq_length)

            # Convert attention mask to binary:
            attention_mask = (attention_mask < 0.5)
            if args.fp16:
                attention_mask = attention_mask.half()
            elif args.bf16:
                attention_mask = attention_mask.bfloat16()

            if args.mask_tensor_adding:
                args.attn_mask = attention_mask * -10000.0
            else:
                args.attn_mask = attention_mask.to(torch.bool)

        else:
            model = LLaMAModel(
                num_tokentypes=0,
                parallel_output=parallel_output,
                pre_process=pre_process,
                post_process=post_process
            )
    see_memory_usage(f"After Building Model", force=True)
    return model

def llama_argument_handler(parser):
    parser.set_defaults(
        position_embedding_type='rotary',
        max_position_embeddings=None,
        no_bias=True,
        layernorm_type='rmsnorm',
        activation_func_type='swiglu',
        ffn_hidden_coeff=(2/3)*4)
    return parser

if __name__ == "__main__":
    git_ds_info()
    pretrain(train_valid_test_datasets_provider, model_provider, forward_step, extra_args_provider=llama_argument_handler,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
