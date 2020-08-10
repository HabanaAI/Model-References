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
"""SSD (via ResNet34) model definition.

Distributed batch normaliaztion.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import moving_averages
import ssd_constants


def _ring_2d(m, n):
  """Ring-order of a mxn mesh.

  Args:
    m: an integer
    n: an integer

  Returns:
    a list of mxn pairs
  """
  if m == 1:
    return [(0, i) for i in range(n)]
  if n == 1:
    return [(i, 0) for i in range(m)]
  if m % 2 != 0:
    tf.logging.warning('Odd dimension')
    return [(i % m, i // m) for i in range(n * m)]
  ret = [(0, 0)]
  for i in range(m // 2):
    for j in range(1, n):
      ret.append((2 * i, j))
    for j in range(n - 1, 0, -1):
      ret.append((2 * i + 1, j))
  for i in range(m - 1, 0, -1):
    ret.append((i, 0))
  return ret


def _ring_2d_id(m, n):
  ring_2d = _ring_2d(m, n)
  ret = []
  for index in ring_2d:
    row, col = index
    ret.append(row * n * 2 + col * 2)
    ret.append(row * n * 2 + col * 2 + 1)
  return ret


def _ring_2d_tile_id(m, n, num_cores_per_row):
  ring_2d = _ring_2d(m, n)
  ret = []
  for index in ring_2d:
    row, col = index
    ret.append(row * num_cores_per_row + col * 2)
    ret.append(row * num_cores_per_row + col * 2 + 1)
  return ret


def _ring_2d_id_sp(m, n):
  # The rings of replicas goes vertically first.
  ring_2d = _ring_2d(n, m)
  ret = []
  for index in ring_2d:
    # Transpose to make it correct.
    col, row = index
    ret.append(row * n + col)

  return ret


def _ring_2d_tile_id_sp(m, n, num_replicas_per_row):
  ring_2d = _ring_2d(m, n)
  ret = []
  for index in ring_2d:
    row, col = index
    ret.append(row * num_replicas_per_row + col)
  return ret


def spatial_partitioning_group_assignment(physical_shape, tile_shape,
                                          num_groups):
  """Create group assignment for spatial partitioning."""
  # Number of rows and columns of the TPU replica topology.
  physical_shape_row, physical_shape_col = physical_shape
  tile_shape_row, tile_shape_col = tile_shape
  first_group = _ring_2d_tile_id_sp(tile_shape_row, tile_shape_col,
                                    physical_shape_col)

  group_assignment = []
  logical_to_physical = dict(
      zip(
          _ring_2d_id_sp(physical_shape_row, physical_shape_col),
          [i for i in range(physical_shape_col * physical_shape_row)]))

  replicas_per_row_in_tile = tile_shape_col
  tiles_per_col = physical_shape_col // tile_shape_col
  for i in range(num_groups):
    offset = i // tiles_per_col * tiles_per_col * tile_shape_row + i % tiles_per_col
    new_group_logical = [
        y + offset * replicas_per_row_in_tile for y in first_group
    ]
    new_group_physical = [
        logical_to_physical[index] for index in new_group_logical
    ]
    group_assignment.append(new_group_physical)

  return group_assignment


def normal_group_assignment(physical_shape, tile_shape, num_groups):
  """Create group assignment for TPU cores."""

  # Nmber of rows and cols of TPU chip topology. Each chip has 2 cores.
  physical_shape_row, physical_shape_col = physical_shape
  # Number of rows and columns in each TPU chip subgroup. Each chip has 2 cores.
  tile_shape_row, tile_shape_col = tile_shape
  first_group = _ring_2d_tile_id(tile_shape_row, tile_shape_col,
                                 physical_shape_col * 2)

  group_assignment = []
  logical_to_physical = dict(
      zip(
          _ring_2d_id(physical_shape_row, physical_shape_col),
          [i for i in range(physical_shape_col * physical_shape_row * 2)]))

  cores_per_row_in_tile = tile_shape_col * 2
  for i in range(num_groups):
    tiles_per_col = physical_shape_col // tile_shape_col
    offset = i // tiles_per_col * tiles_per_col * tile_shape_row + i % tiles_per_col
    new_group_logical = [
        y + offset * cores_per_row_in_tile for y in first_group
    ]
    new_group_physical = [
        logical_to_physical[index] for index in new_group_logical
    ]
    group_assignment.append(new_group_physical)
  return group_assignment


def cross_replica_average(inputs,
                          num_shards=None,
                          num_shards_per_group=None,
                          physical_shape=None,
                          tile_shape=None,
                          use_spatial_partitioning=False):
  """Customized cross replica sum op."""
  # if num_shards_per_group is defined, apply distributed batch norm.
  group_assignment = None

  if num_shards_per_group > 0:
    if num_shards % num_shards_per_group != 0:
      raise ValueError(
          'num_shards: %d mod num_shards_per_group: %d, should be 0' %
          (num_shards, num_shards_per_group))

  num_groups = num_shards // num_shards_per_group

  if physical_shape is not None and tile_shape is not None:
    if use_spatial_partitioning:
      group_assignment = spatial_partitioning_group_assignment(
          physical_shape, tile_shape, num_groups)
    else:
      group_assignment = normal_group_assignment(physical_shape, tile_shape,
                                                 num_groups)
  else:
    group_assignment = [[  # pylint: disable=g-complex-comprehension
        x for x in range(num_shards) if x // num_shards_per_group == y
    ] for y in range(num_groups)]

  return tpu_ops.cross_replica_sum(inputs, group_assignment) / math_ops.cast(
      num_shards_per_group, inputs.dtype)


def distributed_batch_norm(inputs,
                           decay=ssd_constants.BATCH_NORM_DECAY,
                           epsilon=ssd_constants.BATCH_NORM_EPSILON,
                           is_training=True,
                           gamma_initializer=None,
                           num_shards=None,
                           distributed_group_size=4,
                           physical_shape=None,
                           tile_shape=None,
                           use_spatial_partitioning=False,
                           scope=None):
  """Adds a Batch Normalization layer from http://arxiv.org/abs/1502.03167.

  Note: When is_training is True the moving_mean and moving_variance need to be
  updated, by default the update_ops are placed in `tf.GraphKeys.UPDATE_OPS` so
  they need to be added as a dependency to the `train_op`, example:

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
      updates = tf.group(*update_ops)
      total_loss = control_flow_ops.with_dependencies([updates], total_loss)

  One can set updates_collections=None to force the updates in place, but that
  can have speed penalty, especially in distributed settings.

  Args:
    inputs: A tensor with 2 or more dimensions, where the first dimension has
      `batch_size`. The normalization is over all but the last dimension if
    decay: Decay for the moving average. Reasonable values for `decay` are close
      to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc.
        Lower `decay` value (recommend trying `decay`=0.9) if model experiences
        reasonably good training performance but poor validation and/or test
        performance.
    epsilon: Small float added to variance to avoid dividing by zero.
    is_training: Whether or not the layer is in training mode. In training mode
      it would accumulate the statistics of the moments into `moving_mean` and
      `moving_variance` using an exponential moving average with the given
      `decay`. When it is not in training mode then it would use the values of
      the `moving_mean` and the `moving_variance`.
    gamma_initializer:  Initializers for gamma.
    num_shards: Number of shards that participate in the global reduction.
      Default is set to None, that will skip the cross replica sum in and
      normalize across local examples only.
    distributed_group_size: Number of replicas to normalize across in the
      distributed batch normalization.
    physical_shape: A tuple of TPU slice shape, for example (8, 16) represents
      8x16.
    tile_shape: Distributed batch norm group tile shape, for example (4, 2)
      represents 4 * 4 * 2 cores per group.
    use_spatial_partitioning: if use spatial partitioning.
    scope: Optional scope for `variable_scope`.

  Returns:
    A `Tensor` representing the output of the operation.
     Raises:

  Raises:
    ValueError: If input shape is not fully defined.
  """

  with tf.variable_scope(scope, 'batch_normalization', [inputs], reuse=None):
    inputs = tf.convert_to_tensor(inputs)
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]
    if not params_shape.is_fully_defined():
      raise ValueError('Inputs %s has undefined `C` dimension %s.' %
                       (inputs.name, params_shape))
    # Allocate parameters for the beta and gamma of the normalization.
    beta = tf.get_variable(
        'beta',
        shape=params_shape,
        dtype=tf.float32,
        initializer=tf.zeros_initializer(),
        trainable=True)
    gamma = tf.get_variable(
        'gamma',
        dtype=tf.float32,
        shape=params_shape,
        initializer=gamma_initializer,
        trainable=True)
    # Disable partition setting for moving_mean and moving_variance
    # as assign_moving_average op below doesn't support partitioned variable.
    scope = tf.get_variable_scope()
    partitioner = scope.partitioner
    scope.set_partitioner(None)
    moving_mean = tf.get_variable(
        'moving_mean',
        shape=params_shape,
        dtype=tf.float32,
        initializer=tf.zeros_initializer(),
        trainable=False)
    moving_variance = tf.get_variable(
        'moving_variance',
        shape=params_shape,
        initializer=tf.ones_initializer(),
        trainable=False)
    # Restore scope's partitioner setting.
    scope.set_partitioner(partitioner)

    # Add cross replica sum to do subset mean and variance calculation
    # First compute mean and variance
    if is_training:
      # Execute a distributed batch normalization
      axis = 3
      inputs_dtype = inputs.dtype
      inputs = tf.cast(inputs, tf.float32)
      ndims = len(inputs_shape)
      reduction_axes = [i for i in range(ndims) if i != axis]
      counts, mean_ss, variance_ss, _ = tf.nn.sufficient_statistics(
          inputs, reduction_axes, keep_dims=False)
      mean_variance_ss = tf.concat([mean_ss, variance_ss], 0)
      mean_variance_ss = cross_replica_average(
          inputs=mean_variance_ss,
          num_shards=num_shards,
          num_shards_per_group=distributed_group_size,
          physical_shape=physical_shape,
          tile_shape=tile_shape,
          use_spatial_partitioning=use_spatial_partitioning)
      num_elements = tf.reduce_prod(mean_ss.get_shape())
      mean_ss = tf.slice(mean_variance_ss, [0], [num_elements])
      variance_ss = tf.slice(mean_variance_ss, [num_elements], [num_elements])

      mean, variance = tf.nn.normalize_moments(
          counts, mean_ss, variance_ss, shift=None)
      outputs = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma,
                                          epsilon)
      outputs = tf.cast(outputs, inputs_dtype)
    else:
      outputs, mean, variance = tf.nn.fused_batch_norm(
          inputs,
          gamma,
          beta,
          mean=moving_mean,
          variance=moving_variance,
          epsilon=epsilon,
          is_training=False,
          data_format='NHWC')

    if is_training:
      update_moving_mean = moving_averages.assign_moving_average(
          moving_mean,
          tf.cast(mean, moving_mean.dtype),
          decay,
          zero_debias=False)
      update_moving_variance = moving_averages.assign_moving_average(
          moving_variance,
          tf.cast(variance, moving_variance.dtype),
          decay,
          zero_debias=False)
      tf.add_to_collection('update_ops', update_moving_mean)
      tf.add_to_collection('update_ops', update_moving_variance)

    outputs.set_shape(inputs_shape)
    return outputs
