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

"""LLaMA model."""

import torch

from megatron import get_args
from megatron import mpu
from .module import MegatronModule, fp32_to_float16

from .enums import AttnMaskType
from megatron.enums import PositionEmbeddingType
from .language_model import parallel_lm_logits
from .language_model import get_language_model
from .utils import init_method_normal, scaled_init_method_normal, WrapName

from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec
from megatron.model import RMSNorm, LayerNorm, CrossEntropy
from megatron.model.module import float16_to_fp32
from .language_model import EmbeddingPipe
from .transformer import ParallelTransformerLayerPipe


def logits_loss(lm_output, labels, fp16_lm_cross_entropy):

    if labels is None:
        return lm_output
    else:
        if fp16_lm_cross_entropy:
            assert lm_output.dtype == torch.half
            loss = mpu.vocab_parallel_cross_entropy(lm_output, labels)
        else:
            loss = mpu.vocab_parallel_cross_entropy(lm_output.float(), labels)
        return loss


class LLaMAModel(MegatronModule):
    """LLaMA Language model."""

    def __init__(self,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True,
                 return_moe_loss=True):
        super(LLaMAModel, self).__init__()
        args = get_args()

        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.return_moe_loss = return_moe_loss
        if args.no_scaled_init:
            scaled_init_method = init_method_normal(args.init_method_std)
        else:
            scaled_init_method = scaled_init_method_normal(args.init_method_std, args.num_layers)
        assert args.position_embedding_type == PositionEmbeddingType.rotary, 'LLaMA should use rotary positional embeddings'
        self.language_model, self._language_model_key = get_language_model(
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            encoder_attn_mask_type=AttnMaskType.causal,
            init_method=init_method_normal(args.init_method_std),
            scaled_init_method=scaled_init_method,
            num_experts=args.num_experts,
            pre_process=self.pre_process,
            post_process=self.post_process,
            use_position=False)

        self.vocab_projection = mpu.layers.VocabParallelProjection(
            args.padded_vocab_size,
            args.hidden_size,
            parallel_output=True,
            init_method=init_method_normal(args.init_method_std)
            )

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(self, input_ids, position_ids, attention_mask, labels=None,
                tokentype_ids=None, layer_past=None, get_key_value=False,
                forward_method_parallel_output=None, curriculum_seqlen=None):
        args = get_args()
        if curriculum_seqlen is not None:
            args.curriculum_seqlen = curriculum_seqlen
            if curriculum_seqlen < input_ids.size()[1]:
                # seqlen-based curriculum learning
                # input_ids, position_ids, labels have size [batch size, seqlen]
                input_ids = input_ids[:, :curriculum_seqlen].contiguous()
                position_ids = position_ids[:, :curriculum_seqlen].contiguous()
                if labels is not None:
                    labels = labels[:, :curriculum_seqlen].contiguous()

                # attention_mask has size [1, 1, seqlen, seqlen]
                attention_mask = attention_mask[:, :, :curriculum_seqlen, :curriculum_seqlen].contiguous()
        else:
            if args.curriculum_learning:
                # If got a None input, need to reset curriculum_seqlen on user side
                args.curriculum_seqlen = args.seq_length

        lm_output, *moe_losses = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            layer_past=layer_past,
            get_key_value=get_key_value)

        if self.post_process:
            if get_key_value:
                lm_output, presents = lm_output
            lm_output = self.vocab_projection(lm_output)
            lm_output = logits_loss(lm_output, labels, self.fp16_lm_cross_entropy)
            if get_key_value:
                lm_output = [lm_output, presents]

        if self.return_moe_loss:
            return (lm_output, *moe_losses)
        else:
            return lm_output

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):

        state_dict_ = {}
        language_model_state_dict = self.language_model.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)
        # MoE states need to be handled separately by DeepSpeed engine, thus
        # moving them to the top level dictionary
        if "moe_state_dict" in language_model_state_dict:
            for key in list(language_model_state_dict["moe_state_dict"].keys()):
                state_dict_[key] = language_model_state_dict["moe_state_dict"].pop(key)
            del language_model_state_dict["moe_state_dict"]
        state_dict_[self._language_model_key] = language_model_state_dict
        # Save word_embeddings.
        if self.post_process and not self.pre_process:
            state_dict_[self._word_embeddings_for_head_key] \
                = self.word_embeddings.state_dict(destination, prefix, keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Load word_embeddings.
        if self.post_process and not self.pre_process:
            self.word_embeddings.load_state_dict(
                state_dict[self._word_embeddings_for_head_key], strict=strict)
        # Gather MoE states and move under language model
        moe_state_dict = {}
        for key in list(state_dict.keys()):
            if 'expert' in key and 'moe.gate.wg.weight' not in key:
                moe_state_dict[key] = state_dict.pop(key)
        if self._language_model_key in state_dict:
            state_dict = state_dict[self._language_model_key]
        if len(moe_state_dict) > 0:
            state_dict["moe_state_dict"] = moe_state_dict
        self.language_model.load_state_dict(state_dict, strict=strict)


class LLaMAModelPipe(PipelineModule,MegatronModule):
    """LLaMA Language model."""

    def __init__(self,
                 num_tokentypes=0,
                 parallel_output=True):
        args = get_args()
        self.parallel_output = parallel_output

        init_method = init_method_normal(args.init_method_std)

        if args.no_scaled_init:
            scaled_init_method = init_method_normal(args.init_method_std)
        else:
            scaled_init_method = scaled_init_method_normal(args.init_method_std, args.num_layers)

        self.specs = []

        def _to_float16(inputs):
            if args.fp16:
                return fp32_to_float16(inputs, lambda v: v.half())
            elif args.bf16:
                return fp32_to_float16(inputs, lambda v: v.bfloat16())
            else:
                return inputs

        self.specs.append(_to_float16)

        # Embedding layer
        # assert args.position_embedding_type == PositionEmbeddingType.rotary, 'LLaMA should use rotary positional embeddings'
        self.specs.append(LayerSpec(EmbeddingPipe,
                                        args.hidden_size,
                                        args.padded_vocab_size,
                                        args.max_position_embeddings,
                                        args.hidden_dropout,
                                        init_method=init_method,
                                        num_tokentypes=num_tokentypes,
                                        use_position=False))

        if args.fp32_residual_connection:
            self.specs.append(lambda x: x.transpose(0, 1).contiguous().float())
        else:
            self.specs.append(lambda x: x.transpose(0, 1).contiguous())

        for layer_idx in range(args.num_layers):
            self.specs.append(
                LayerSpec(ParallelTransformerLayerPipe,
                    init_method=init_method,
                    output_layer_init_method=scaled_init_method,
                    layer_number=layer_idx,
                    self_attn_mask_type=AttnMaskType.causal))


        # Undo data format change
        self.specs.append(lambda x: x.transpose(0, 1).contiguous())

        # Final RMSNorm after transformer layers
        assert args.layernorm_type=='rmsnorm', 'LLaMA model should use RMSNorm'
        self.specs.append(
            LayerSpec(WrapName, 'final_rmsnorm', 
                      RMSNorm,
                      args.hidden_size,
                      eps=args.layernorm_epsilon))

        self.specs.append(
            LayerSpec(WrapName, 'vocab_parallel_projection',
                      mpu.layers.VocabParallelProjection,
                      args.padded_vocab_size,
                      args.hidden_size,
                      init_method=init_method)
        )

        # Convert to fp32 if needed
        if args.fp16 or args.bf16:
            self.specs.append(float16_to_fp32)

        if args.checkpoint_activations and args.checkpoint_activations_granularity == "full":
            interval = args.checkpoint_num_layers
        else:
            interval = 0

        from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
        topo = PipeModelDataParallelTopology(num_pp=mpu.get_pipeline_model_parallel_world_size(),
                                             num_mp=mpu.get_tensor_model_parallel_world_size(),
                                             num_dp=mpu.get_data_parallel_world_size())

        super().__init__(layers=self.specs,
                         loss_fn=CrossEntropy,
                         topology=topo,
                         activation_checkpoint_interval=interval,
                         partition_method='type:transformer')
