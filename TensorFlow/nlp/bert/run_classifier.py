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

###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
#
# Changes:
# - Added stdout/stderr flush in main
# - Added default values for environment variables:
#   - HABANA_INITIAL_WORKSPACE_SIZE_MB
#   - TF_DISABLE_MKL
#   - TF_BF16_CONVERSION
#   - TF_PRELIMINARY_CLUSTER_SIZE
#   - TF_DISABLE_SCOPED_ALLOCATOR
###############################################################################

"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import json
import os
import sys
import socket
import TensorFlow.nlp.bert.utils.habana_hooks as habana_hooks
import TensorFlow.nlp.bert.modeling as modeling
import TensorFlow.nlp.bert.optimization as optimization
import TensorFlow.nlp.bert.data_preprocessing.tokenization as tokenization
import tensorflow as tf

from TensorFlow.common.tb_utils import write_hparams_v1
from TensorFlow.common.horovod_helpers import hvd, hvd_init, horovod_enabled
from TensorFlow.common.str2bool import condition_env_var

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, '..'))

from TensorFlow.common.library_loader import load_habana_module

flags = tf.compat.v1.flags

FLAGS = flags.FLAGS


def init_flags():
  # Required parameters
  flags.DEFINE_string(
      "data_dir", None,
      "The input data dir. Should contain the .tsv files (or other data files) "
      "for the task.")

  flags.DEFINE_string(
      "bert_config_file", None,
      "The config json file corresponding to the pre-trained BERT model. "
      "This specifies the model architecture.")

  flags.DEFINE_string("task_name", None, "The name of the task to train.")

  flags.DEFINE_string("vocab_file", None,
                      "The vocabulary file that the BERT model was trained on.")

  flags.DEFINE_string(
      "output_dir", None,
      "The output directory where the model checkpoints will be written.")

  # Other parameters

  flags.DEFINE_string(
      "init_checkpoint", None,
      "Initial checkpoint (usually from a pre-trained BERT model).")

  flags.DEFINE_bool(
      "do_lower_case", True,
      "Whether to lower case the input text. Should be True for uncased "
      "models and False for cased models.")

  flags.DEFINE_integer(
      "max_seq_length", 128,
      "The maximum total input sequence length after WordPiece tokenization. "
      "Sequences longer than this will be truncated, and sequences shorter "
      "than this will be padded.")

  flags.DEFINE_bool("do_train", False, "Whether to run training.")

  flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

  flags.DEFINE_bool(
      "do_predict", False,
      "Whether to run the model in inference mode on the test set.")

  flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

  flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

  flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

  flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

  flags.DEFINE_float("num_train_epochs", 3.0,
                     "Total number of training epochs to perform.")

  flags.DEFINE_float(
      "warmup_proportion", 0.1,
      "Proportion of training to perform linear learning rate warmup for. "
      "E.g., 0.1 = 10% of training.")
  flags.DEFINE_float(
      "dropout_before_logits", 0.1,
      "Dropout rate before logits layer, defaults to 0.1")
  flags.DEFINE_integer("save_checkpoints_steps", 1000,
                       "How often to save the model checkpoint.")

  flags.DEFINE_integer("save_summary_steps", 1,
                     "How often to save the summary data.")

  flags.DEFINE_integer("iterations_per_loop", 1000,
                       "How many steps to make in each estimator call.")

  flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

  flags.DEFINE_string(
      "tpu_name", None,
      "The Cloud TPU to use for training. This should be either the name "
      "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
      "url.")

  flags.DEFINE_string(
      "tpu_zone", None,
      "[Optional] GCE zone where the Cloud TPU is located in. If not "
      "specified, we will attempt to automatically detect the GCE project from "
      "metadata.")

  flags.DEFINE_string(
      "gcp_project", None,
      "[Optional] Project name for the Cloud TPU-enabled project. If not "
      "specified, we will attempt to automatically detect the GCE project from "
      "metadata.")

  flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

  flags.DEFINE_integer(
      "num_tpu_cores", 8,
      "Only used if `use_tpu` is True. Total number of TPU cores to use.")

  flags.DEFINE_bool("cpu_only", False, "Whether to run on CPU.")

  flags.DEFINE_bool("use_horovod", False, "Run training using horovod.")

  flags.DEFINE_bool('enable_scoped_allocator', False, "Enable scoped allocator optimization")

  flags.DEFINE_bool('horovod_hierarchical_allreduce', False, "Enables hierarchical allreduce in Horovod. "
                    "The environment variable `HOROVOD_HIERARCHICAL_ALLREDUCE` will be set to `1`.")

  flags.DEFINE_string("bf16_config_path", None, "Defines config for tensor converson to bf16 data type")

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.io.gfile.GFile(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


class XnliProcessor(DataProcessor):
  """Processor for the XNLI data set."""

  def __init__(self):
    self.language = "zh"

  def get_train_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(
        os.path.join(data_dir, "multinli",
                     "multinli.train.%s.tsv" % self.language))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "train-%d" % (i)
      text_a = tokenization.convert_to_unicode(line[0])
      text_b = tokenization.convert_to_unicode(line[1])
      label = tokenization.convert_to_unicode(line[2])
      if label == tokenization.convert_to_unicode("contradictory"):
        label = tokenization.convert_to_unicode("contradiction")
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_dev_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(os.path.join(data_dir, "xnli.dev.tsv"))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "dev-%d" % (i)
      language = tokenization.convert_to_unicode(line[0])
      if language != tokenization.convert_to_unicode(self.language):
        continue
      text_a = tokenization.convert_to_unicode(line[6])
      text_b = tokenization.convert_to_unicode(line[7])
      label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]


class MnliProcessor(DataProcessor):
  """Processor for the MultiNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
        "dev_matched")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
      text_a = tokenization.convert_to_unicode(line[8])
      text_b = tokenization.convert_to_unicode(line[9])
      if set_type == "test":
        label = "contradiction"
      else:
        label = tokenization.convert_to_unicode(line[-1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class MrpcProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[3])
      text_b = tokenization.convert_to_unicode(line[4])
      if set_type == "test":
        label = "0"
      else:
        label = tokenization.convert_to_unicode(line[0])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class ColaProcessor(DataProcessor):
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      # Only the test set has a header
      if set_type == "test" and i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      if set_type == "test":
        text_a = tokenization.convert_to_unicode(line[1])
        label = "0"
      else:
        text_a = tokenization.convert_to_unicode(line[3])
        label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.compat.v1.logging.info("*** Example ***")
    tf.compat.v1.logging.info("guid: %s" % (example.guid))
    tf.compat.v1.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.compat.v1.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.compat.v1.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.compat.v1.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.compat.v1.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.compat.v1.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature([feature.label_id])
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.io.FixedLenFeature([], tf.int64),
      "is_real_example": tf.io.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.io.parse_single_example(serialized=record, features=name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, dtype=tf.int32)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      if horovod_enabled():
        d = d.shard(hvd.size(), hvd.rank())
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings, dropout_before_logits=0.1):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1]

  output_weights = tf.compat.v1.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.compat.v1.get_variable(
      "output_bias", [num_labels], initializer=tf.compat.v1.zeros_initializer())

  with tf.compat.v1.variable_scope("loss_scope"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, rate=dropout_before_logits)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(input_tensor=one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(input_tensor=per_example_loss)

    return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, dropout_before_logits=0.1):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.compat.v1.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.compat.v1.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(input=label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings, dropout_before_logits)

    tvars = tf.compat.v1.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
       ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.compat.v1.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.compat.v1.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.compat.v1.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(input=logits, axis=-1, output_type=tf.int32)
        accuracy = tf.compat.v1.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        return {
            "accuracy": accuracy,
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example])
      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.compat.v1.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    features.append(feature)
  return features


def main(_):

  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

  processors = {
      "cola": ColaProcessor,
      "mnli": MnliProcessor,
      "mrpc": MrpcProcessor,
      "xnli": XnliProcessor,
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.io.gfile.makedirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  model_dir = FLAGS.output_dir
  if horovod_enabled():
    model_dir = os.path.join(FLAGS.output_dir, "worker_" + str(hvd.rank()))

  # The Scoped Allocator Optimization is enabled by default unless disabled by a flag.
  if FLAGS.enable_scoped_allocator:
    from tensorflow.core.protobuf import rewriter_config_pb2  # pylint: disable=import-error

    session_config = tf.compat.v1.ConfigProto()
    session_config.graph_options.rewrite_options.scoped_allocator_optimization = rewriter_config_pb2.RewriterConfig.ON

    enable_op = session_config.graph_options.rewrite_options.scoped_allocator_opts.enable_op
    del enable_op[:]
    enable_op.append("HorovodAllreduce")
  else:
    session_config = None

  is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.compat.v1.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=model_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      save_summary_steps=FLAGS.save_summary_steps,
      tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host),
      session_config=session_config)

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None

  train_batch_size = FLAGS.train_batch_size
  if horovod_enabled():
    train_batch_size = train_batch_size * hvd.size()

  if FLAGS.do_train:
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    num_train_steps = int(
        len(train_examples) / train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  start_index = 0
  end_index = len(train_examples)
  per_worker_filenames = [os.path.join(FLAGS.output_dir, "train.tf_record")]
  worker_id = 0

  if horovod_enabled():
    per_worker_filenames = [os.path.join(FLAGS.output_dir, "train.tf_record_{}".format(i)) for i in range(hvd.size())]
    num_examples_per_rank = len(train_examples) // hvd.size()
    remainder = len(train_examples) % hvd.size()
    worker_id = hvd.rank()
    if worker_id < remainder:
      start_index = worker_id * (num_examples_per_rank + 1)
      end_index = start_index + num_examples_per_rank + 1
    else:
      start_index = worker_id * num_examples_per_rank + remainder
      end_index = start_index + (num_examples_per_rank)

  learning_rate = FLAGS.learning_rate
  if horovod_enabled():
    learning_rate = learning_rate * hvd.size()

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu,
      dropout_before_logits=FLAGS.dropout_before_logits)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  write_hparams_v1(FLAGS.output_dir, {
    'batch_size': FLAGS.train_batch_size,
    **{x: getattr(FLAGS, x) for x in FLAGS}
  })

  if FLAGS.do_train:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    file_based_convert_examples_to_features(
        train_examples[start_index:end_index], label_list, FLAGS.max_seq_length, tokenizer, per_worker_filenames[worker_id])
    tf.compat.v1.logging.info("***** Running training *****")
    tf.compat.v1.logging.info("  Num examples = %d", len(train_examples))
    tf.compat.v1.logging.info("  Per-worker batch size = %d", FLAGS.train_batch_size)
    tf.compat.v1.logging.info("  Total batch size = %d", train_batch_size)
    tf.compat.v1.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=per_worker_filenames,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)

    train_hooks = [habana_hooks.PerfLoggingHook(batch_size=train_batch_size, mode="train")]
    if horovod_enabled():
      train_hooks.append(hvd.BroadcastGlobalVariablesHook(0))

    if "range" == os.environ.get("HABANA_SYNAPSE_LOGGER", "False").lower():
      from TensorFlow.common.horovod_helpers import SynapseLoggerHook
      begin = 30
      end = begin + 10
      print("Begin: {}".format(begin))
      print("End: {}".format(end))
      train_hooks.append(SynapseLoggerHook(list(range(begin, end)), False))

    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps, hooks=train_hooks)

  if FLAGS.do_eval:
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    num_actual_eval_examples = len(eval_examples)
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on. These do NOT count towards the metric (all tf.metrics
      # support a per-instance weight, and these get a weight of 0.0).
      while len(eval_examples) % FLAGS.eval_batch_size != 0:
        eval_examples.append(PaddingInputExample())

    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    file_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

    tf.compat.v1.logging.info("***** Running evaluation *****")
    tf.compat.v1.logging.info("  Num examples = %d (%d actual, %d padding)",
                              len(eval_examples), num_actual_eval_examples,
                              len(eval_examples) - num_actual_eval_examples)
    tf.compat.v1.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    if FLAGS.use_tpu:
      assert len(eval_examples) % FLAGS.eval_batch_size == 0
      eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)

    eval_hooks = [habana_hooks.PerfLoggingHook(batch_size=FLAGS.eval_batch_size, mode="eval")]
    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps, hooks=eval_hooks)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.io.gfile.GFile(output_eval_file, "w") as writer:
      tf.compat.v1.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.compat.v1.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_predict:
    predict_examples = processor.get_test_examples(FLAGS.data_dir)
    num_actual_predict_examples = len(predict_examples)
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on.
      while len(predict_examples) % FLAGS.predict_batch_size != 0:
        predict_examples.append(PaddingInputExample())

    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples, label_list,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file)

    tf.compat.v1.logging.info("***** Running prediction*****")
    tf.compat.v1.logging.info("  Num examples = %d (%d actual, %d padding)",
                              len(predict_examples), num_actual_predict_examples,
                              len(predict_examples) - num_actual_predict_examples)
    tf.compat.v1.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn)

    output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
    with tf.io.gfile.GFile(output_predict_file, "w") as writer:
      num_written_lines = 0
      tf.compat.v1.logging.info("***** Predict results *****")
      for (i, prediction) in enumerate(result):
        probabilities = prediction["probabilities"]
        if i >= num_actual_predict_examples:
          break
        output_line = "\t".join(
            str(class_probability)
            for class_probability in probabilities) + "\n"
        writer.write(output_line)
        num_written_lines += 1
    assert num_written_lines == num_actual_predict_examples


if __name__ == "__main__":
  init_flags()
  tf.compat.v1.enable_resource_variables()

  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")

  if FLAGS.bf16_config_path is not None:
    os.environ['TF_BF16_CONVERSION'] = FLAGS.bf16_config_path

  bert_config = json.load(open(FLAGS.bert_config_file, 'r'))
  if 'hidden_size' in bert_config and bert_config['hidden_size'] == 768:
    # cluster slicing optimization tested for bert base, works well with this size
    os.environ.setdefault('TF_PRELIMINARY_CLUSTER_SIZE', '1000')

  os.environ.setdefault('HABANA_INITIAL_WORKSPACE_SIZE_MB', '15437')
  os.environ.setdefault("TF_DISABLE_MKL", "1")

  if FLAGS.use_horovod:
    if FLAGS.horovod_hierarchical_allreduce:
      os.environ['HOROVOD_HIERARCHICAL_ALLREDUCE'] = "1"
    assert not FLAGS.cpu_only, "Horovod without HPU is not supported in helpers."
    hvd_init()

  if not FLAGS.cpu_only:
    log_info_devices = load_habana_module()
    print(f"Devices:\n {log_info_devices}")

  host_name = socket.gethostname()
  vmulti = os.environ.get('MULTI_HLS_IPS')
  vhcl = os.environ.get('HCL_CONFIG_PATH')
  v1 = os.environ.get('HABANA_INITIAL_WORKSPACE_SIZE_MB')
  v2 = os.environ.get('TF_BF16_CONVERSION')
  print(f"{host_name}: MULTI_HLS_IPS = {vmulti}")
  print(f"{host_name}: HCL_CONFIG_PATH = {vhcl}")
  print(f"{host_name}: HABANA_INITIAL_WORKSPACE_SIZE_MB = {v1}")
  print(f"{host_name}: TF_BF16_CONVERSION = {v2}")
  sys.stdout.flush()
  sys.stderr.flush()

  tf.compat.v1.app.run()
