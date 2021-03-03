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

from transformers import AlbertConfig, is_tf_available
from transformers.testing_utils import require_tf, slow

from .test_configuration_common import ConfigTester
from .test_modeling_tf_common import TFModelTesterMixin, ids_tensor


if is_tf_available():
    from transformers.modeling_tf_albert import (
        TFAlbertModel,
        TFAlbertForPreTraining,
        TFAlbertForMaskedLM,
        TFAlbertForSequenceClassification,
        TFAlbertForQuestionAnswering,
        TF_ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    )


class TFAlbertModelTester:
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
        embedding_size=16,
        hidden_size=32,
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
        self.batch_size = 13
        self.seq_length = 7
        self.is_training = True
        self.use_input_mask = True
        self.use_token_type_ids = True
        self.use_labels = True
        self.vocab_size = 99
        self.embedding_size = 16
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

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = AlbertConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
        )

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def create_and_check_albert_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = TFAlbertModel(config=config)
        # inputs = {'input_ids': input_ids,
        #           'attention_mask': input_mask,
        #           'token_type_ids': token_type_ids}
        # sequence_output, pooled_output = model(**inputs)
        inputs = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": token_type_ids}
        sequence_output, pooled_output = model(inputs)

        inputs = [input_ids, input_mask]
        sequence_output, pooled_output = model(inputs)

        sequence_output, pooled_output = model(input_ids)

        result = {
            "sequence_output": sequence_output.numpy(),
            "pooled_output": pooled_output.numpy(),
        }
        self.parent.assertListEqual(
            list(result["sequence_output"].shape), [self.batch_size, self.seq_length, self.hidden_size]
        )
        self.parent.assertListEqual(list(result["pooled_output"].shape), [self.batch_size, self.hidden_size])

    def create_and_check_albert_for_pretraining(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = TFAlbertForPreTraining(config=config)
        inputs = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": token_type_ids}
        prediction_scores, sop_scores = model(inputs)
        result = {
            "prediction_scores": prediction_scores.numpy(),
            "sop_scores": sop_scores.numpy(),
        }
        self.parent.assertListEqual(
            list(result["prediction_scores"].shape), [self.batch_size, self.seq_length, self.vocab_size]
        )
        self.parent.assertListEqual(list(result["sop_scores"].shape), [self.batch_size, self.num_labels])

    def create_and_check_albert_for_masked_lm(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = TFAlbertForMaskedLM(config=config)
        inputs = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": token_type_ids}
        (prediction_scores,) = model(inputs)
        result = {
            "prediction_scores": prediction_scores.numpy(),
        }
        self.parent.assertListEqual(
            list(result["prediction_scores"].shape), [self.batch_size, self.seq_length, self.vocab_size]
        )

    def create_and_check_albert_for_sequence_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = TFAlbertForSequenceClassification(config=config)
        inputs = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": token_type_ids}
        (logits,) = model(inputs)
        result = {
            "logits": logits.numpy(),
        }
        self.parent.assertListEqual(list(result["logits"].shape), [self.batch_size, self.num_labels])

    def create_and_check_albert_for_question_answering(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = TFAlbertForQuestionAnswering(config=config)
        inputs = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": token_type_ids}
        start_logits, end_logits = model(inputs)
        result = {
            "start_logits": start_logits.numpy(),
            "end_logits": end_logits.numpy(),
        }
        self.parent.assertListEqual(list(result["start_logits"].shape), [self.batch_size, self.seq_length])
        self.parent.assertListEqual(list(result["end_logits"].shape), [self.batch_size, self.seq_length])

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


@require_tf
class TFAlbertModelTest(TFModelTesterMixin, unittest.TestCase):

    all_model_classes = (
        (
            TFAlbertModel,
            TFAlbertForPreTraining,
            TFAlbertForMaskedLM,
            TFAlbertForSequenceClassification,
            TFAlbertForQuestionAnswering,
        )
        if is_tf_available()
        else ()
    )

    def setUp(self):
        self.model_tester = TFAlbertModelTester(self)
        self.config_tester = ConfigTester(self, config_class=AlbertConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_albert_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_albert_model(*config_and_inputs)

    def test_for_pretraining(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_albert_for_pretraining(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_albert_for_masked_lm(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_albert_for_sequence_classification(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_albert_for_question_answering(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in TF_ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = TFAlbertModel.from_pretrained(model_name)
            self.assertIsNotNone(model)
