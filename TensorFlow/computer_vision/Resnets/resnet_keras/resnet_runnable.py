# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# List of changes:
# - added profiling callbacks support

# Copyright (C) 2020 HabanaLabs, Ltd.
# All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
#

"""Runs a ResNet model on the ImageNet dataset using custom training loops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from TensorFlow.common.modeling import performance
from TensorFlow.common.training import grad_utils
from TensorFlow.common.training import standard_runnable
from TensorFlow.common.training import utils
from TensorFlow.utils.flags import core as flags_core
from TensorFlow.common.image_classification.resnet import common
from TensorFlow.common.image_classification.resnet import imagenet_preprocessing
from TensorFlow.common.image_classification.resnet import resnet_model


class ResnetRunnable(standard_runnable.StandardTrainable,
                     standard_runnable.StandardEvaluable):
  """Implements the training and evaluation APIs for Resnet model."""

  def __init__(self, flags_obj, time_callback, epoch_steps, profiler_callback):
    standard_runnable.StandardTrainable.__init__(self,
                                                 flags_obj.use_tf_while_loop,
                                                 flags_obj.use_tf_function)
    standard_runnable.StandardEvaluable.__init__(self,
                                                 flags_obj.use_tf_function)

    self.strategy = tf.distribute.get_strategy()
    self.flags_obj = flags_obj
    self.dtype = flags_core.get_tf_dtype(flags_obj)
    self.time_callback = time_callback
    self.profiler_callback = profiler_callback

    # Input pipeline related
    batch_size = flags_obj.batch_size
    if batch_size % self.strategy.num_replicas_in_sync != 0:
      raise ValueError(
          'Batch size must be divisible by number of replicas : {}'.format(
              self.strategy.num_replicas_in_sync))

    # As auto rebatching is not supported in
    # `experimental_distribute_datasets_from_function()` API, which is
    # required when cloning dataset to multiple workers in eager mode,
    # we use per-replica batch size.
    self.batch_size = int(batch_size / self.strategy.num_replicas_in_sync)

    if self.flags_obj.use_synthetic_data:
      self.input_fn = common.get_synth_input_fn(
          height=imagenet_preprocessing.DEFAULT_IMAGE_SIZE,
          width=imagenet_preprocessing.DEFAULT_IMAGE_SIZE,
          num_channels=imagenet_preprocessing.NUM_CHANNELS,
          num_classes=imagenet_preprocessing.NUM_CLASSES,
          dtype=self.dtype,
          drop_remainder=True)
    else:
      self.input_fn = imagenet_preprocessing.input_fn

    self.model = resnet_model.resnet50(
        num_classes=imagenet_preprocessing.NUM_CLASSES,
        batch_size=flags_obj.batch_size,
        use_l2_regularizer=not flags_obj.single_l2_loss_op)

    lr_schedule = common.PiecewiseConstantDecayWithWarmup(
        batch_size=flags_obj.batch_size,
        epoch_size=imagenet_preprocessing.NUM_IMAGES['train'],
        warmup_epochs=common.LR_SCHEDULE[0][1],
        boundaries=list(p[1] for p in common.LR_SCHEDULE[1:]),
        multipliers=list(p[0] for p in common.LR_SCHEDULE),
        compute_lr_on_cpu=True)
    self.optimizer = common.get_optimizer(lr_schedule)
    # Make sure iterations variable is created inside scope.
    self.global_step = self.optimizer.iterations

    use_graph_rewrite = flags_obj.fp16_implementation == 'graph_rewrite'
    if use_graph_rewrite and not flags_obj.use_tf_function:
      raise ValueError('--fp16_implementation=graph_rewrite requires '
                       '--use_tf_function to be true')
    self.optimizer = performance.configure_optimizer(
        self.optimizer,
        use_float16=self.dtype == tf.float16,
        use_graph_rewrite=use_graph_rewrite,
        loss_scale=flags_core.get_loss_scale(flags_obj, default_for_fp16=128))

    self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        'train_accuracy', dtype=tf.float32)
    self.test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        'test_accuracy', dtype=tf.float32)

    self.checkpoint = tf.train.Checkpoint(
        model=self.model, optimizer=self.optimizer)

    # Handling epochs.
    self.epoch_steps = epoch_steps
    self.epoch_helper = utils.EpochHelper(epoch_steps, self.global_step)

  def build_train_dataset(self):
    """See base class."""
    return utils.make_distributed_dataset(
        self.strategy,
        self.input_fn,
        is_training=True,
        data_dir=self.flags_obj.data_dir,
        batch_size=self.batch_size,
        parse_record_fn=imagenet_preprocessing.parse_record,
        datasets_num_private_threads=self.flags_obj
        .datasets_num_private_threads,
        dtype=common.get_dl_type(self.flags_obj),
        drop_remainder=True,
        experimental_preloading=self.flags_obj.experimental_preloading)

  def build_eval_dataset(self):
    """See base class."""
    return utils.make_distributed_dataset(
        self.strategy,
        self.input_fn,
        is_training=False,
        data_dir=self.flags_obj.data_dir,
        batch_size=self.batch_size,
        parse_record_fn=imagenet_preprocessing.parse_record,
        dtype=common.get_dl_type(self.flags_obj),
        experimental_preloading=self.flags_obj.experimental_preloading)

  def train_loop_begin(self):
    """See base class."""
    # Reset all metrics
    self.train_loss.reset_states()
    self.train_accuracy.reset_states()

    self._epoch_begin()
    self.time_callback.on_batch_begin(self.epoch_helper.batch_index)
    if self.profiler_callback is not None:
      self.profiler_callback.on_batch_begin(self.epoch_helper.batch_index)

  def train_step(self, iterator):
    """See base class."""

    def step_fn(inputs):
      """Function to run on the device."""
      images, labels = inputs
      with tf.GradientTape() as tape:
        logits = self.model(images, training=True)

        prediction_loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels, logits)
        loss = tf.reduce_sum(prediction_loss) * (1.0 /
                                                 self.flags_obj.batch_size)
        num_replicas = self.strategy.num_replicas_in_sync

        if self.flags_obj.single_l2_loss_op:
          l2_loss = resnet_model.L2_WEIGHT_DECAY * 2 * tf.add_n([
              tf.nn.l2_loss(v)
              for v in self.model.trainable_variables
              if 'bn' not in v.name
          ])

          loss += (l2_loss / num_replicas)
        else:
          loss += (tf.reduce_sum(self.model.losses) / num_replicas)

      grad_utils.minimize_using_explicit_allreduce(
          tape, self.optimizer, loss, self.model.trainable_variables)
      self.train_loss.update_state(loss)
      self.train_accuracy.update_state(labels, logits)

    self.strategy.run(step_fn, args=(next(iterator),))

  def train_loop_end(self):
    """See base class."""
    metrics = {
        'train_loss': self.train_loss.result(),
        'train_accuracy': self.train_accuracy.result(),
    }
    self.time_callback.on_batch_end(self.epoch_helper.batch_index - 1)
    if self.profiler_callback is not None:
      self.profiler_callback.on_batch_end(self.epoch_helper.batch_index - 1)
    self._epoch_end()
    return metrics

  def eval_begin(self):
    """See base class."""
    self.test_loss.reset_states()
    self.test_accuracy.reset_states()

  def eval_step(self, iterator):
    """See base class."""

    def step_fn(inputs):
      """Function to run on the device."""
      images, labels = inputs
      logits = self.model(images, training=False)
      loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
      loss = tf.reduce_sum(loss) * (1.0 / self.flags_obj.batch_size)
      self.test_loss.update_state(loss)
      self.test_accuracy.update_state(labels, logits)

    self.strategy.run(step_fn, args=(next(iterator),))

  def eval_end(self):
    """See base class."""
    return {
        'test_loss': self.test_loss.result(),
        'test_accuracy': self.test_accuracy.result()
    }

  def _epoch_begin(self):
    if self.epoch_helper.epoch_begin():
      self.time_callback.on_epoch_begin(self.epoch_helper.current_epoch)
      if self.profiler_callback is not None:
        self.profiler_callback.on_epoch_begin(self.epoch_helper.current_epoch)

  def _epoch_end(self):
    if self.epoch_helper.epoch_end():
      self.time_callback.on_epoch_end(self.epoch_helper.current_epoch)
      if self.profiler_callback is not None:
        self.profiler_callback.on_epoch_end(self.epoch_helper.current_epoch)
