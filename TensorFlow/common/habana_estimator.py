# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
#
# Two functions taken from tensorflow_estimator/python/estimator/util.py:
#     - parse_iterator_result
#     - parse_input_fn_result - extended to set iterator on HPU
#
###############################################################################
import tensorflow as tf
import logging
from tensorflow.python.training import training
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator import util as estimator_util
from tensorflow.python.framework import ops


class HabanaEstimator(tf.estimator.Estimator):

  def _assert_members_are_not_overridden(self):
    """you bet"""
    pass


  def _get_features_and_labels_from_input_fn(self, input_fn, mode):
    """Extracts the `features` and labels from return values of `input_fn`."""

    from tensorflow.python.data.ops import dataset_ops
    class _DatasetInitializerHook(training.SessionRunHook):
      """Creates a SessionRunHook that initializes the passed iterator."""

      def __init__(self, iterator):
        self._iterator = iterator

      def begin(self):
        self._initializer = self._iterator.initializer

      def after_create_session(self, session, coord):
        del coord
        session.run(self._initializer)

    def parse_input_fn_result(result):
      input_hooks = []
      if isinstance(result, dataset_ops.DatasetV2):
        device = "/device:HPU:0"
        with tf.device(device):
          iterator = dataset_ops.make_initializable_iterator(result)
          result = iterator.get_next()
        input_hooks.append(_DatasetInitializerHook(iterator))
      return parse_iterator_result(result) + (input_hooks,)


    def parse_iterator_result(result):
      """Gets features, labels from result."""
      if isinstance(result, (list, tuple)):
        if len(result) != 2:
          raise ValueError(
              'input_fn should return (features, labels) as a len 2 tuple.')
        return result[0], result[1]
      return result, None


    return parse_input_fn_result(
        self._call_input_fn(input_fn, mode))
  def _get_iterator_from_input_fn(self, input_fn, mode, distribution=None):
    """Calls `input_fn` and returns an iterator."""
    if distribution is not None:
      # pylint: disable=g-long-lambda
      iterator = distribution.make_input_fn_iterator(lambda input_context: self._call_input_fn(input_fn, mode,
                                                                                               input_context))
      input_hooks = [estimator_util.DistributedIteratorInitializerHook(iterator)]
    else:
      result = self._call_input_fn(input_fn, mode)
      device = "/device:HPU:0"
      with ops.device(device):
        iterator = result.make_initializable_iterator()
      input_hooks = [estimator_util._DatasetInitializerHook(iterator)]  # pylint: disable=protected-access
    return iterator, input_hooks

  def _call_input_fn(self, input_fn, mode, input_context=None):
    """Calls the input function.

    Args:
      input_fn: The input function.
      mode: `tf.estimator.ModeKeys`

    Returns:
      The return value of the passed `input_fn`, which should be one of:

        * A 'tf.data.Dataset' object: Outputs of `Dataset` object must be a
            tuple `(features, labels)` with same constraints as below.
        * A tuple `(features, labels)`: Where `features` is a `Tensor` or a
          dictionary of string feature name to `Tensor` and `labels` is a
          `Tensor` or a dictionary of string label name to `Tensor`. Both
          `features` and `labels` are consumed by `model_fn`. They should
          satisfy the expectation of `model_fn` from inputs.

    Raises:
      ValueError: if `input_fn` takes invalid arguments.
    """
    input_fn_args = function_utils.fn_args(input_fn)
    kwargs = {}
    if 'mode' in input_fn_args:
      kwargs['mode'] = mode
    if 'params' in input_fn_args:
      kwargs['params'] = self.params
    if 'config' in input_fn_args:
      kwargs['config'] = self.config
    if input_context and 'input_context' in input_fn_args:
      logging.info('The `input_fn` accepts an `input_context` which will '
                   'be given by DistributionStrategy')
      kwargs['input_context'] = input_context
    #with ops.device('/cpu:0'):
    return input_fn(**kwargs)



