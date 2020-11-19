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
from .test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch
    from transformers import (
        MobileBertConfig,
        MobileBertModel,
        MobileBertForMaskedLM,
        MobileBertForNextSentencePrediction,
        MobileBertForPreTraining,
        MobileBertForQuestionAnswering,
        MobileBertForSequenceClassification,
        MobileBertForTokenClassification,
        MobileBertForMultipleChoice,
    )


class MobileBertModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=64,
        embedding_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

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

        config = MobileBertConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            embedding_size=self.embedding_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
        )

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def prepare_config_and_inputs_for_decoder(self):
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = self.prepare_config_and_inputs()

        config.is_decoder = True
        encoder_hidden_states = floats_tensor([self.batch_size, self.seq_length, self.hidden_size])
        encoder_attention_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

        return (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        )

    def check_loss_output(self, result):
        self.parent.assertListEqual(list(result["loss"].size()), [])

    def create_and_check_mobilebert_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = MobileBertModel(config=config)
        model.to(torch_device)
        model.eval()
        sequence_output, pooled_output = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        sequence_output, pooled_output = model(input_ids, token_type_ids=token_type_ids)
        sequence_output, pooled_output = model(input_ids)

        result = {
            "sequence_output": sequence_output,
            "pooled_output": pooled_output,
        }
        self.parent.assertListEqual(
            list(result["sequence_output"].size()), [self.batch_size, self.seq_length, self.hidden_size]
        )
        self.parent.assertListEqual(list(result["pooled_output"].size()), [self.batch_size, self.hidden_size])

    def create_and_check_mobilebert_model_as_decoder(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        model = MobileBertModel(config)
        model.to(torch_device)
        model.eval()
        sequence_output, pooled_output = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        sequence_output, pooled_output = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            encoder_hidden_states=encoder_hidden_states,
        )
        sequence_output, pooled_output = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)

        result = {
            "sequence_output": sequence_output,
            "pooled_output": pooled_output,
        }
        self.parent.assertListEqual(
            list(result["sequence_output"].size()), [self.batch_size, self.seq_length, self.hidden_size]
        )
        self.parent.assertListEqual(list(result["pooled_output"].size()), [self.batch_size, self.hidden_size])

    def create_and_check_mobilebert_for_masked_lm(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = MobileBertForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        loss, prediction_scores = model(
            input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels
        )
        result = {
            "loss": loss,
            "prediction_scores": prediction_scores,
        }
        self.parent.assertListEqual(
            list(result["prediction_scores"].size()), [self.batch_size, self.seq_length, self.vocab_size]
        )
        self.check_loss_output(result)

    def create_and_check_mobilebert_for_next_sequence_prediction(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = MobileBertForNextSentencePrediction(config=config)
        model.to(torch_device)
        model.eval()
        loss, seq_relationship_score = model(
            input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, next_sentence_label=sequence_labels,
        )
        result = {
            "loss": loss,
            "seq_relationship_score": seq_relationship_score,
        }
        self.parent.assertListEqual(list(result["seq_relationship_score"].size()), [self.batch_size, 2])
        self.check_loss_output(result)

    def create_and_check_mobilebert_for_pretraining(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = MobileBertForPreTraining(config=config)
        model.to(torch_device)
        model.eval()
        loss, prediction_scores, seq_relationship_score = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            labels=token_labels,
            next_sentence_label=sequence_labels,
        )
        result = {
            "loss": loss,
            "prediction_scores": prediction_scores,
            "seq_relationship_score": seq_relationship_score,
        }
        self.parent.assertListEqual(
            list(result["prediction_scores"].size()), [self.batch_size, self.seq_length, self.vocab_size]
        )
        self.parent.assertListEqual(list(result["seq_relationship_score"].size()), [self.batch_size, 2])
        self.check_loss_output(result)

    def create_and_check_mobilebert_for_question_answering(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = MobileBertForQuestionAnswering(config=config)
        model.to(torch_device)
        model.eval()
        loss, start_logits, end_logits = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
        )
        result = {
            "loss": loss,
            "start_logits": start_logits,
            "end_logits": end_logits,
        }
        self.parent.assertListEqual(list(result["start_logits"].size()), [self.batch_size, self.seq_length])
        self.parent.assertListEqual(list(result["end_logits"].size()), [self.batch_size, self.seq_length])
        self.check_loss_output(result)

    def create_and_check_mobilebert_for_sequence_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = MobileBertForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        loss, logits = model(
            input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=sequence_labels
        )
        result = {
            "loss": loss,
            "logits": logits,
        }
        self.parent.assertListEqual(list(result["logits"].size()), [self.batch_size, self.num_labels])
        self.check_loss_output(result)

    def create_and_check_mobilebert_for_token_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = MobileBertForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        loss, logits = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        result = {
            "loss": loss,
            "logits": logits,
        }
        self.parent.assertListEqual(list(result["logits"].size()), [self.batch_size, self.seq_length, self.num_labels])
        self.check_loss_output(result)

    def create_and_check_mobilebert_for_multiple_choice(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_choices = self.num_choices
        model = MobileBertForMultipleChoice(config=config)
        model.to(torch_device)
        model.eval()
        multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_token_type_ids = token_type_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_input_mask = input_mask.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        loss, logits = model(
            multiple_choice_inputs_ids,
            attention_mask=multiple_choice_input_mask,
            token_type_ids=multiple_choice_token_type_ids,
            labels=choice_labels,
        )
        result = {
            "loss": loss,
            "logits": logits,
        }
        self.parent.assertListEqual(list(result["logits"].size()), [self.batch_size, self.num_choices])
        self.check_loss_output(result)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class MobileBertModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (
        (
            MobileBertModel,
            MobileBertForMaskedLM,
            MobileBertForMultipleChoice,
            MobileBertForNextSentencePrediction,
            MobileBertForPreTraining,
            MobileBertForQuestionAnswering,
            MobileBertForSequenceClassification,
            MobileBertForTokenClassification,
        )
        if is_torch_available()
        else ()
    )

    def setUp(self):
        self.model_tester = MobileBertModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MobileBertConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_mobilebert_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mobilebert_model(*config_and_inputs)

    def test_mobilebert_model_as_decoder(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_mobilebert_model_as_decoder(*config_and_inputs)

    def test_mobilebert_model_as_decoder_with_default_input_mask(self):
        # This regression test was failing with PyTorch < 1.3
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        ) = self.model_tester.prepare_config_and_inputs_for_decoder()

        input_mask = None

        self.model_tester.create_and_check_mobilebert_model_as_decoder(
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        )

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mobilebert_for_masked_lm(*config_and_inputs)

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mobilebert_for_multiple_choice(*config_and_inputs)

    def test_for_next_sequence_prediction(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mobilebert_for_next_sequence_prediction(*config_and_inputs)

    def test_for_pretraining(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mobilebert_for_pretraining(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mobilebert_for_question_answering(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mobilebert_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mobilebert_for_token_classification(*config_and_inputs)


def _long_tensor(tok_lst):
    return torch.tensor(tok_lst, dtype=torch.long, device=torch_device,)


TOLERANCE = 1e-3


@require_torch
class MobileBertModelIntegrationTests(unittest.TestCase):
    @slow
    def test_inference_no_head(self):
        model = MobileBertModel.from_pretrained("google/mobilebert-uncased").to(torch_device)
        input_ids = _long_tensor([[101, 7110, 1005, 1056, 2023, 11333, 17413, 1029, 102]])
        with torch.no_grad():
            output = model(input_ids)[0]
        expected_shape = torch.Size((1, 9, 512))
        self.assertEqual(output.shape, expected_shape)
        expected_slice = torch.tensor(
            [
                [
                    [-2.4736526e07, 8.2691656e04, 1.6521838e05],
                    [-5.7541704e-01, 3.9056022e00, 4.4011507e00],
                    [2.6047359e00, 1.5677652e00, -1.7324188e-01],
                ]
            ],
            device=torch_device,
        )

        # MobileBERT results range from 10e0 to 10e8. Even a 0.0000001% difference with a value of 10e8 results in a
        # ~1 difference, it's therefore not a good idea to measure using addition.
        # Here, we instead divide the expected result with the result in order to obtain ~1. We then check that the
        # result is held between bounds: 1 - TOLERANCE < expected_result / result < 1 + TOLERANCE
        lower_bound = torch.all((expected_slice / output[..., :3, :3]) >= 1 - TOLERANCE)
        upper_bound = torch.all((expected_slice / output[..., :3, :3]) <= 1 + TOLERANCE)

        self.assertTrue(lower_bound and upper_bound)
