# coding=utf-8
# Copyright 2021 The Tensor2Tensor Authors.
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

"""Modalities, which specify a feature's domain.

T2TModel applies a default transformation to each feature according to its
modality. Override them by specifying a model's
hparams.{bottom,loss,top,weights_fn}.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range  # pylint: disable=redefined-builtin

from TensorFlow.nlp.transformer.layers import common_attention
from TensorFlow.nlp.transformer.layers import common_layers

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp


class ModalityType(object):
  """Types of modalities."""

  IDENTITY = "identity"  # identity top and bottom
  IDENTITY_SYMBOL = "identity_symbol"  # symbol with identity top and bottom
  SYMBOL = "symbol"


  @staticmethod
  def get_choices():
    return [
        ModalityType.IDENTITY,
        ModalityType.IDENTITY_SYMBOL,
        ModalityType.SYMBOL,
    ]


def class_label_targets_bottom(x, model_hparams, vocab_size):
  with tf.variable_scope("class_label_modality_%d_%d" % (
      vocab_size, model_hparams.hidden_size)):
    return tf.zeros([common_layers.shape_list(x)[0],
                     1,
                     1,
                     model_hparams.hidden_size])


def identity_bottom(x, model_hparams, vocab_size):
  del model_hparams, vocab_size  # unused arg
  return tf.cast(x, tf.float32)


def make_targets_bottom(bottom):
  def targets_bottom(x, model_hparams, vocab_size):
    with tf.variable_scope("targets_bottom"):
      return bottom(x, model_hparams, vocab_size)
  return targets_bottom


def real_bottom(x, model_hparams, vocab_size):
  del vocab_size  # unused arg
  with tf.variable_scope("real"):
    return tf.layers.dense(
        tf.cast(x, tf.float32), model_hparams.hidden_size, name="bottom")


def get_weights(model_hparams, vocab_size, hidden_dim=None):
  """Create or get concatenated embedding or softmax variable.

  Args:
    model_hparams: HParams, model hyperparmeters.
    vocab_size: int, vocabulary size.
    hidden_dim: dim of the variable. Defaults to _model_hparams' hidden_size

  Returns:
     a list of num_shards Tensors.
  """
  if hidden_dim is None:
    hidden_dim = model_hparams.hidden_size
  num_shards = model_hparams.symbol_modality_num_shards
  shards = []
  for i in range(num_shards):
    shard_size = (vocab_size // num_shards) + (
        1 if i < vocab_size % num_shards else 0)
    var_name = "weights_%d" % i
    shards.append(
        tf.get_variable(
            var_name, [shard_size, hidden_dim],
            initializer=tf.random_normal_initializer(0.0, hidden_dim**-0.5)))
  if num_shards == 1:
    ret = shards[0]
  else:
    ret = tf.concat(shards, 0)
  # Convert ret to tensor.
  if not tf.executing_eagerly():
    ret = common_layers.convert_gradient_to_tensor(ret)
  return ret


def _symbol_bottom_simple(x, model_hparams, vocab_size, name, reuse):
  """Bottom transformation for symbols."""
  with tf.variable_scope(name, reuse=reuse):
    # Ensure the inputs are 3-D
    if len(x.get_shape()) == 4:
      x = tf.squeeze(x, axis=3)
    while len(x.get_shape()) < 3:
      x = tf.expand_dims(x, axis=-1)

    var = get_weights(model_hparams, vocab_size)
    x = common_layers.dropout_no_scaling(
        x, 1.0 - model_hparams.symbol_dropout)
    ret = common_layers.gather(var, x)
    if model_hparams.multiply_embedding_mode == "sqrt_depth":
      ret *= model_hparams.hidden_size**0.5
    ret *= tf.expand_dims(
        common_layers.cast_like(tf.not_equal(x, 0), ret), -1)
    return ret


def symbol_bottom(x, model_hparams, vocab_size):
  if (model_hparams.shared_embedding_and_softmax_weights or
      model_hparams.get("shared_embedding")):
    return _symbol_bottom_simple(
        x, model_hparams, vocab_size, "shared", reuse=None)
  return _symbol_bottom_simple(
      x, model_hparams, vocab_size, "input_emb", reuse=None)


def symbol_targets_bottom(x, model_hparams, vocab_size):
  if (model_hparams.shared_embedding_and_softmax_weights or
      model_hparams.get("shared_embedding")):
    try:
      return _symbol_bottom_simple(
          x, model_hparams, vocab_size, "shared", reuse=True)
    except ValueError:
      # perhaps there were no inputs, and this is a new variable.
      return _symbol_bottom_simple(
          x, model_hparams, vocab_size, "shared", reuse=None)
  else:
    return _symbol_bottom_simple(
        x, model_hparams, vocab_size, "target_emb", reuse=None)


# Loss transformations, applied to target features


def generic_loss(top_out, targets, model_hparams, vocab_size, weights_fn):
  """Compute loss numerator and denominator for one shard of output."""
  del vocab_size  # unused arg
  logits = top_out
  logits = common_attention.maybe_upcast(logits, hparams=model_hparams)
  cutoff = getattr(model_hparams, "video_modality_loss_cutoff", 0.0)

  return common_layers.padded_cross_entropy(
      logits,
      targets,
      model_hparams.label_smoothing,
      cutoff=cutoff,
      weights_fn=weights_fn)


# Top transformations, applied to target features


def is_pointwise(func):
  """Decorator for whether the function is pointwise.

  An example of a pointwise function is a linear layer followed by
  a softmax. Given a tensor [batch, length, height, depth] it operates
  only on the last axis, on every point in [batch, length, height] fully
  independently. In contrast, a classifier that first averages over length
  and height is not pointwise, as it depends on the whole field. It is useful
  to know if top functions are pointwise to speed up decoding in certain models.

  Args:
    func: Function to decorate.

  Returns:
    Original function with an attribute pointwise set to True.
  """
  func.pointwise = True
  return func


def identity_top(body_output, targets, model_hparams, vocab_size):
  del targets, model_hparams, vocab_size  # unused arg
  return body_output


@is_pointwise
def symbol_top(body_output, targets, model_hparams, vocab_size):
  """Generate logits.

  Args:
    body_output: A Tensor with shape
      [batch, p0, p1, model_hparams.hidden_size].
    targets: Unused.
    model_hparams: HParams, model hyperparmeters.
    vocab_size: int, vocabulary size.

  Returns:
    logits: A Tensor with shape  [batch, p0, p1, ?, vocab_size].
  """
  del targets  # unused arg
  if model_hparams.shared_embedding_and_softmax_weights:
    scope_name = "shared"
    reuse = tf.AUTO_REUSE
  else:
    scope_name = "softmax"
    reuse = False
  with tf.variable_scope(scope_name, reuse=reuse):
    body_output_shape = common_layers.shape_list(body_output)
    var = get_weights(model_hparams, vocab_size, body_output_shape[-1])
    if (model_hparams.factored_logits and
        model_hparams.mode == tf.estimator.ModeKeys.TRAIN):
      # insert channels dimension
      body_output = tf.expand_dims(body_output, 3)
      return common_layers.FactoredTensor(body_output, var)
    else:
      body_output = tf.reshape(body_output, [-1, body_output_shape[-1]])
      logits = tf.matmul(body_output, var, transpose_b=True)
      return tf.reshape(logits,
                        body_output_shape[:-1] + [1, vocab_size])


# Utility functions similar to tf.keras for default transformations


def get_bottom(modality_type, value=None):
  """Gets default bottom transformation; if none available, return value."""
  if modality_type == ModalityType.SYMBOL:
    return symbol_bottom
  elif modality_type in (ModalityType.IDENTITY,
                         ModalityType.IDENTITY_SYMBOL):
    return identity_bottom
  return value


def get_loss(modality_type, value=None):
  """Gets default loss transformation; if none available, return value."""
  if modality_type in (ModalityType.IDENTITY,
                       ModalityType.IDENTITY_SYMBOL,
                       ModalityType.SYMBOL):
    return generic_loss
  return value


def get_name(modality_type, value=None):
  """Gets default name for transformations; if none available, return value."""
  # For legacy reasons, modalities vary in their naming scheme. Future plans are
  # to remove any need for get_name. We do not recommend using it.
  if modality_type == ModalityType.IDENTITY:
    return lambda model_hparams, vocab_size: "identity_modality"
  elif modality_type == ModalityType.SYMBOL:
    def name(model_hparams, vocab_size):
      return "symbol_modality_%d_%d" % (vocab_size, model_hparams.hidden_size)
    return name
  return value


def get_targets_bottom(modality_type, value=None):
  """Gets default bottom transformation for targets; if none, return value."""
  if modality_type == ModalityType.SYMBOL:
    return symbol_targets_bottom
  elif modality_type == ModalityType.IDENTITY_SYMBOL:
    return identity_bottom
  elif modality_type == ModalityType.IDENTITY:
    return make_targets_bottom(identity_bottom)
  return value


def get_top(modality_type, value=None):
  """Gets default top transformation; if none available, return value."""
  if modality_type in (ModalityType.IDENTITY,
                       ModalityType.IDENTITY_SYMBOL):
    return identity_top
  elif modality_type == ModalityType.SYMBOL:
    return symbol_top
  return value


def get_weights_fn(modality_type, value=None):
  """Gets default weights function; if none available, return value."""
  if modality_type in (ModalityType.IDENTITY_SYMBOL,
                       ModalityType.SYMBOL):
    return common_layers.weights_nonzero
  elif modality_type in ModalityType.get_choices():
    return common_layers.weights_all
  return value
