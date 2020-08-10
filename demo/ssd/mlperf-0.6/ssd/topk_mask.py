# Copyright 2018 Google. All Rights Reserved.
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

"""Efficient implementation of topk_mask for TPUs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def topk_mask(score, k):
  """Efficient implementation of topk_mask for TPUs.

  This is a more efficient implementation of the following snippet with support
  for higher rank tensors. It has the limitation that it only supports float32
  as element type. The mask only contains k elements even if other elements
  have the same value as the kth largest.

  def topk_mask(score, k):
    _, indices = tf.nn.top_k(score, k=k)
    return tf.scatter_nd(tf.expand_dims(indices, -1), tf.ones(k),
                         tf.squeeze(score).shape.as_list())

  The implementation binary searches for the kth value along each row of the
  input and once the kth value is found it creates the mask via a single select
  instruction. This approach is more than 100x faster on TPUs for large inputs
  compared with the above snippet.

  Args:
    score: 1-D or higher Tensor with last dimension at least k.
    k: Number of top elements to look for along the last dimension (along each
      row for matrices).
  """
  last_dim_size = score.get_shape().as_list()[-1]

  # Choose top k+epsilon where epsilon is the number of times the k'th largest i
  # element is present in the input.
  topk_mask_with_duplicate = topk_mask_internal(score, k)
  # Calculate the number of redudant duplicate values to discard.
  select_num = tf.cast(
      tf.reduce_sum(topk_mask_with_duplicate, axis=-1, keepdims=True), tf.int32)
  redudant_num = select_num - k

  # softmax cross entropy value range [0, 1].
  # k's largest value is the smallest value being selected.
  k_th_value = tf.reduce_min(
      tf.where(
          tf.cast(topk_mask_with_duplicate, tf.bool), score,
          tf.ones_like(score) * 2.0),
      axis=-1,
      keepdims=True)
  # Mask to indicate if score equals k th largest value.
  equal_k_th_value = tf.equal(score, k_th_value)
  # Creates a tensor wherer the value is 1 if the value is equal to kth largest
  # value, otherwise, 0.
  k_th_value = tf.where(equal_k_th_value, tf.ones_like(score, dtype=tf.int32),
                        tf.zeros_like(score, dtype=tf.int32))
  index = tf.range(last_dim_size)

  k_th_value_index = tf.multiply(k_th_value, index)

  duplicate_mask = topk_mask_internal(
      tf.cast(k_th_value_index, tf.float32), redudant_num)

  return tf.where(
      tf.cast(duplicate_mask, tf.bool), tf.zeros_like(topk_mask_with_duplicate),
      topk_mask_with_duplicate)


def topk_mask_internal(score, k):
  """Efficient implementation of topk_mask for TPUs.

  This is a more efficient implementation of the following snippet with support
  for higher rank tensors. It has the limitation that it only supports float32
  as element type. The mask may contain more than k elements if other elements
  have the same value as the kth largest.

  The implementation binary searches for the kth value along each row of the
  input and once the kth value is found it creates the mask via a single select
  instruction. This approach is more than 100x faster on TPUs for large inputs
  compared with the above snippet.

  Args:
    score: 1-D or higher Tensor with last dimension at least k.
    k: Number of top elements to look for along the last dimension (along each
      row for matrices).
  """

  def larger_count(data, limit):
    """Number of elements larger than limit along the most minor dimension.

    Args:
      data: Rn tensor with the data to compare.
      limit: Rn tensor with last dimension being 1 and rest of the dimensions
          being same as for data.

    Returns:
      Rn tensor with same shape as limit and int32 as element type containing
      the number of elements larger then limit inside data.
    """
    return tf.reduce_sum(
        tf.cast(data > tf.broadcast_to(limit, data.shape), tf.int32),
        axis=-1, keepdims=True)

  # Predicate specifying if the kth value is negative or positive.
  kth_negative = (larger_count(score, 0.0) < k)

  # Value of the sign bit for each row.
  limit_sign = tf.where(kth_negative,
                        tf.broadcast_to(1, kth_negative.shape),
                        tf.broadcast_to(0, kth_negative.shape))

  # Initial value for the binary search with the sign bit set.
  next_value = tf.bitwise.left_shift(limit_sign, 31)

  def cond(bit_index, _):
    return bit_index >= 0

  def body(bit_index, value):
    """Body for the while loop executing the binary search.

    Args:
      bit_index: Index of the bit to be updated next.
      value: Current value of the binary search separator. Stored as an int32
          but bitcasted to a float32 for comparison.
    Returns:
      The updated value of bit_index and value
    """

    # Calculate new value via `new_value = value | (1 << bit_index)`
    new_value = tf.bitwise.bitwise_or(
        value, tf.bitwise.left_shift(1, bit_index))

    # Calculate number of values larger than new_value
    larger = larger_count(score, tf.bitcast(new_value, tf.float32))

    # Update next_value based on new_value. For positive numbers new_value is
    # larger than value while for negative numbers it is the other way around.
    next_value = tf.where(tf.logical_xor(larger >= k, kth_negative),
                          new_value, value)
    return bit_index - 1, next_value

  # Executes a binary search for the value of the limits. We run the loop 31
  # times to calculate the 31 bits of the float32 value (the sign is calculated
  # separately).
  _, limit = tf.while_loop(cond, body, (30, next_value))

  # Create a mask by comparing the individual values to the kth value and then
  # selecting zero or one accordingly.
  return tf.where(
      score >= tf.broadcast_to(tf.bitcast(limit, tf.float32), score.shape),
      tf.ones(score.shape), tf.zeros(score.shape))
