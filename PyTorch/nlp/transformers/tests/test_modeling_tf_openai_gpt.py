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

from transformers import OpenAIGPTConfig, is_tf_available
from transformers.testing_utils import require_tf, slow

from .test_configuration_common import ConfigTester
from .test_modeling_tf_common import TFModelTesterMixin, ids_tensor


if is_tf_available():
    import tensorflow as tf
    from transformers.modeling_tf_openai import (
        TFOpenAIGPTModel,
        TFOpenAIGPTLMHeadModel,
        TFOpenAIGPTDoubleHeadsModel,
        TF_OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST,
    )


class TFOpenAIGPTModelTester:
    def __init__(
        self, parent,
    ):
        self.parent = parent
        self.batch_size = 13
        self.seq_length = 7
        self.is_training = True
        self.use_token_type_ids = True
        self.use_input_mask = True
        self.use_labels = True
        self.use_mc_token_ids = True
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

        input_mask = None
        if self.use_input_mask:
            input_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        mc_token_ids = None
        if self.use_mc_token_ids:
            mc_token_ids = ids_tensor([self.batch_size, self.num_choices], self.seq_length)

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
            input_mask,
            head_mask,
            token_type_ids,
            mc_token_ids,
            sequence_labels,
            token_labels,
            choice_labels,
        )

    def create_and_check_openai_gpt_model(self, config, input_ids, input_mask, head_mask, token_type_ids, *args):
        model = TFOpenAIGPTModel(config=config)
        inputs = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": token_type_ids}
        sequence_output = model(inputs)[0]

        inputs = [input_ids, input_mask]
        sequence_output = model(inputs)[0]

        sequence_output = model(input_ids)[0]

        result = {
            "sequence_output": sequence_output.numpy(),
        }
        self.parent.assertListEqual(
            list(result["sequence_output"].shape), [self.batch_size, self.seq_length, self.hidden_size]
        )

    def create_and_check_openai_gpt_lm_head(self, config, input_ids, input_mask, head_mask, token_type_ids, *args):
        model = TFOpenAIGPTLMHeadModel(config=config)
        inputs = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": token_type_ids}
        prediction_scores = model(inputs)[0]
        result = {
            "prediction_scores": prediction_scores.numpy(),
        }
        self.parent.assertListEqual(
            list(result["prediction_scores"].shape), [self.batch_size, self.seq_length, self.vocab_size]
        )

    def create_and_check_openai_gpt_double_head(
        self, config, input_ids, input_mask, head_mask, token_type_ids, mc_token_ids, *args
    ):
        model = TFOpenAIGPTDoubleHeadsModel(config=config)

        multiple_choice_inputs_ids = tf.tile(tf.expand_dims(input_ids, 1), (1, self.num_choices, 1))
        multiple_choice_input_mask = tf.tile(tf.expand_dims(input_mask, 1), (1, self.num_choices, 1))
        multiple_choice_token_type_ids = tf.tile(tf.expand_dims(token_type_ids, 1), (1, self.num_choices, 1))

        inputs = {
            "input_ids": multiple_choice_inputs_ids,
            "mc_token_ids": mc_token_ids,
            "attention_mask": multiple_choice_input_mask,
            "token_type_ids": multiple_choice_token_type_ids,
        }
        lm_logits, mc_logits = model(inputs)[:2]
        result = {"lm_logits": lm_logits.numpy(), "mc_logits": mc_logits.numpy()}
        self.parent.assertListEqual(
            list(result["lm_logits"].shape), [self.batch_size, self.num_choices, self.seq_length, self.vocab_size]
        )
        self.parent.assertListEqual(list(result["mc_logits"].shape), [self.batch_size, self.num_choices])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()

        (
            config,
            input_ids,
            input_mask,
            head_mask,
            token_type_ids,
            mc_token_ids,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs

        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_tf
class TFOpenAIGPTModelTest(TFModelTesterMixin, unittest.TestCase):

    all_model_classes = (
        (TFOpenAIGPTModel, TFOpenAIGPTLMHeadModel, TFOpenAIGPTDoubleHeadsModel) if is_tf_available() else ()
    )
    all_generative_model_classes = (
        (TFOpenAIGPTLMHeadModel,) if is_tf_available() else ()
    )  # TODO (PVP): Add Double HeadsModel when generate() function is changed accordingly

    def setUp(self):
        self.model_tester = TFOpenAIGPTModelTester(self)
        self.config_tester = ConfigTester(self, config_class=OpenAIGPTConfig, n_embd=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_openai_gpt_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_openai_gpt_model(*config_and_inputs)

    def test_openai_gpt_lm_head(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_openai_gpt_lm_head(*config_and_inputs)

    def test_openai_gpt_double_head(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_openai_gpt_double_head(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in TF_OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = TFOpenAIGPTModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


@require_tf
class TFOPENAIGPTModelLanguageGenerationTest(unittest.TestCase):
    @slow
    def test_lm_generate_openai_gpt(self):
        model = TFOpenAIGPTLMHeadModel.from_pretrained("openai-gpt")
        input_ids = tf.convert_to_tensor([[481, 4735, 544]], dtype=tf.int32)  # the president is
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
        self.assertListEqual(output_ids[0].numpy().tolist(), expected_output_ids)
