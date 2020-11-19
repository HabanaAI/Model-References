# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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


import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, ids_tensor


if is_torch_available():
    import torch
    from transformers import (
        OpenAIGPTConfig,
        OpenAIGPTModel,
        OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST,
        OpenAIGPTLMHeadModel,
        OpenAIGPTDoubleHeadsModel,
    )


class OpenAIGPTModelTester:
    def __init__(
        self, parent,
    ):
        self.parent = parent
        self.batch_size = 13
        self.seq_length = 7
        self.is_training = True
        self.use_token_type_ids = True
        self.use_labels = True
        self.vocab_size = 99
        self.hidden_size = 32
        self.num_hidden_layers = 5
        self.num_attention_heads = 4
        self.intermediate_size = 37
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 512
        self.type_vocab_size = 16
        self.type_sequence_label_size = 2
        self.initializer_range = 0.02
        self.num_labels = 3
        self.num_choices = 4
        self.scope = None

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = OpenAIGPTConfig(
            vocab_size=self.vocab_size,
            n_embd=self.hidden_size,
            n_layer=self.num_hidden_layers,
            n_head=self.num_attention_heads,
            # intermediate_size=self.intermediate_size,
            # hidden_act=self.hidden_act,
            # hidden_dropout_prob=self.hidden_dropout_prob,
            # attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            n_positions=self.max_position_embeddings,
            n_ctx=self.max_position_embeddings
            # type_vocab_size=self.type_vocab_size,
            # initializer_range=self.initializer_range
        )

        head_mask = ids_tensor([self.num_hidden_layers, self.num_attention_heads], 2)

        return (
            config,
            input_ids,
            head_mask,
            token_type_ids,
            sequence_labels,
            token_labels,
            choice_labels,
        )

    def check_loss_output(self, result):
        self.parent.assertListEqual(list(result["loss"].size()), [])

    def create_and_check_openai_gpt_model(self, config, input_ids, head_mask, token_type_ids, *args):
        model = OpenAIGPTModel(config=config)
        model.to(torch_device)
        model.eval()

        model(input_ids, token_type_ids=token_type_ids, head_mask=head_mask)
        model(input_ids, token_type_ids=token_type_ids)
        (sequence_output,) = model(input_ids)

        result = {"sequence_output": sequence_output}
        self.parent.assertListEqual(
            list(result["sequence_output"].size()), [self.batch_size, self.seq_length, self.hidden_size],
        )

    def create_and_check_lm_head_model(self, config, input_ids, head_mask, token_type_ids, *args):
        model = OpenAIGPTLMHeadModel(config)
        model.to(torch_device)
        model.eval()

        loss, lm_logits = model(input_ids, token_type_ids=token_type_ids, labels=input_ids)

        result = {"loss": loss, "lm_logits": lm_logits}

        self.parent.assertListEqual(list(result["loss"].size()), [])
        self.parent.assertListEqual(
            list(result["lm_logits"].size()), [self.batch_size, self.seq_length, self.vocab_size],
        )

    def create_and_check_double_lm_head_model(self, config, input_ids, head_mask, token_type_ids, *args):
        model = OpenAIGPTDoubleHeadsModel(config)
        model.to(torch_device)
        model.eval()

        loss, lm_logits, mc_logits = model(input_ids, token_type_ids=token_type_ids, labels=input_ids)

        result = {"loss": loss, "lm_logits": lm_logits}

        self.parent.assertListEqual(list(result["loss"].size()), [])
        self.parent.assertListEqual(
            list(result["lm_logits"].size()), [self.batch_size, self.seq_length, self.vocab_size],
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            head_mask,
            token_type_ids,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "head_mask": head_mask,
        }

        return config, inputs_dict


@require_torch
class OpenAIGPTModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (
        (OpenAIGPTModel, OpenAIGPTLMHeadModel, OpenAIGPTDoubleHeadsModel) if is_torch_available() else ()
    )
    all_generative_model_classes = (
        (OpenAIGPTLMHeadModel,) if is_torch_available() else ()
    )  # TODO (PVP): Add Double HeadsModel when generate() function is changed accordingly

    def setUp(self):
        self.model_tester = OpenAIGPTModelTester(self)
        self.config_tester = ConfigTester(self, config_class=OpenAIGPTConfig, n_embd=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_openai_gpt_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_openai_gpt_model(*config_and_inputs)

    def test_openai_gpt_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(*config_and_inputs)

    def test_openai_gpt_double_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_double_lm_head_model(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = OpenAIGPTModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


@require_torch
class OPENAIGPTModelLanguageGenerationTest(unittest.TestCase):
    @slow
    def test_lm_generate_openai_gpt(self):
        model = OpenAIGPTLMHeadModel.from_pretrained("openai-gpt")
        model.to(torch_device)
        input_ids = torch.tensor([[481, 4735, 544]], dtype=torch.long, device=torch_device)  # the president is
        expected_output_ids = [
            481,
            4735,
            544,
            246,
            963,
            870,
            762,
            239,
            244,
            40477,
            244,
            249,
            719,
            881,
            487,
            544,
            240,
            244,
            603,
            481,
        ]  # the president is a very good man. " \n " i\'m sure he is, " said the

        output_ids = model.generate(input_ids, do_sample=False)
        self.assertListEqual(output_ids[0].tolist(), expected_output_ids)
