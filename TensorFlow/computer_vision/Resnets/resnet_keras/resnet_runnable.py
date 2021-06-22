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
# - use of batch size calculation utility function for horovod/distributed usage

# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company

"""Runs a ResNet model on the ImageNet dataset using custom training loops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops

from TensorFlow.common.modeling import performance
from TensorFlow.common.training import grad_utils
from TensorFlow.common.training import standard_runnable
from TensorFlow.common.training import utils
from local_flags import core as flags_core
from TensorFlow.computer_vision.common import imagenet_preprocessing
from TensorFlow.computer_vision.Resnets.resnet_keras import common
from TensorFlow.computer_vision.Resnets.resnet_keras import resnet_model
from TensorFlow.computer_vision.Resnets.resnet_keras.common import adjust_batch_size
from TensorFlow.common.horovod_helpers import hvd, horovod_enabled

class ResnetRunnable(standard_runnable.StandardTrainable,
                     standard_runnable.StandardEvaluable):
  """Implements the training and evaluation APIs for Resnet model."""

  def __init__(self, flags_obj, time_callback, train_steps, epoch_steps, profiler_callback):
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
    self.first_step = True

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
        batch_size=batch_size,
        use_l2_regularizer=not flags_obj.single_l2_loss_op)

    self.use_lars_optimizer = self.flags_obj.optimizer == 'LARS'
    self.optimizer = common.get_optimizer(flags_obj,
                                          adjust_batch_size(flags_obj.batch_size),
                                          train_steps)
    # Make sure iterations variable is created inside scope.
    self.global_step = self.optimizer.iterations

    self.one_hot = False
    self.label_smoothing = flags_obj.label_smoothing
    if self.label_smoothing and self.label_smoothing > 0:
      self.one_hot = True

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
    if self.one_hot:
      self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(
          'train_accuracy', dtype=tf.float32)
    else:
      self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
          'train_accuracy', dtype=tf.float32)
    self.test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    if self.one_hot:
      self.test_accuracy = tf.keras.metrics.CategoricalAccuracy(
          'test_accuracy', dtype=tf.float32)
    else:
      self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
          'test_accuracy', dtype=tf.float32)

    self.checkpoint = tf.train.Checkpoint(
        model=self.model, optimizer=self.optimizer)

    self.local_loss_mean = tf.keras.metrics.Mean("local_loss_min", dtype=tf.float32)

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
        experimental_preloading=self.flags_obj.experimental_preloading,
        use_distributed_eval=self.flags_obj.use_distributed_eval)

  def get_prediction_loss(self, labels, logits, training=True):
    if self.one_hot:
      return tf.keras.losses.categorical_crossentropy(
          labels, logits, label_smoothing=self.label_smoothing)
    else:
      return tf.keras.losses.sparse_categorical_crossentropy(labels, logits)

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
      if self.one_hot:
        labels = tf.cast(labels, tf.int32)
        labels = tf.one_hot(labels, 1001)
        labels = tf.squeeze(labels)

      with tf.GradientTape() as tape:
        logits = self.model(images, training=True)

        prediction_loss = self.get_prediction_loss(labels, logits)

        loss = tf.reduce_sum(prediction_loss) * (1.0 /
                                                 self.flags_obj.batch_size)

        if not self.use_lars_optimizer:
          num_replicas = self.strategy.num_replicas_in_sync

          if self.flags_obj.single_l2_loss_op:
            l2_loss = self.flags_obj.weight_decay * tf.add_n([
                tf.nn.l2_loss(v)
                for v in self.model.trainable_variables
                if ('bn' not in v.name)
            ])

            loss += (l2_loss / num_replicas)
          else:
            loss += (tf.reduce_sum(self.model.losses) / num_replicas)

      if horovod_enabled():
        tape = hvd.DistributedGradientTape(tape)
        grads = tape.gradient(loss, self.model.trainable_variables)
        grads_and_vars = zip(grads, self.model.trainable_variables)

        self.optimizer.apply_gradients(
          grads_and_vars, experimental_aggregate_gradients=False)

        tf.cond(self.global_step == 1,
          lambda: hvd.broadcast_variables(self.model.variables + self.optimizer.variables(),
                                          root_rank=0),
          lambda: tf.constant(True))
      else:
        grad_utils.minimize_using_explicit_allreduce(
          tape, self.optimizer, loss, self.model.trainable_variables)

      self.train_loss.update_state(loss)
      self.train_accuracy.update_state(labels, logits)

    self.strategy.run(step_fn, args=(next(iterator),))

  def train_loop_end(self):
    """See base class."""
    metrics = {
        'loss': self.train_loss.result(),
        'accuracy': self.train_accuracy.result(),
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
      if self.one_hot:
        labels = tf.cast(labels, tf.int32)
        labels = tf.one_hot(labels, 1001)
        labels = tf.squeeze(labels)

      logits = self.model(images, training=False)
      loss = self.get_prediction_loss(labels, logits, training=False)

      loss = tf.reduce_sum(loss) * (1.0 / self.flags_obj.batch_size)
      self.test_loss.update_state(loss)
      self.test_accuracy.update_state(labels, logits)

    self.strategy.run(step_fn, args=(next(iterator),))

  def eval_end(self):
    """See base class."""

    if self.flags_obj.use_distributed_eval and horovod_enabled():
      test_accuracy = hvd.allreduce(self.test_accuracy.result())
    else:
      test_accuracy = self.test_accuracy.result()

    return {
        'test_loss': self.test_loss.result(),
        'test_accuracy': test_accuracy
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
