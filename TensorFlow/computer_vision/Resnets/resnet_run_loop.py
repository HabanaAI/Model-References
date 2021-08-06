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
###############################################################################
# List of changes:
# - disabled support for LARS
# - changed default model type to resnext
# - changed default weight_decay to 6.103515625e-05
# - changed default momentum to 0.875
# - changed default label_smoothing to 0.1
# - changed default warmup_epochs to 8
# - changed default use_cosine_lr to True
# - added logic for setting save_checkpoint_steps
"""Contains utility and supporting functions for ResNet.

  This module contains ResNet code which does not directly build layers. This
includes dataset management, hyperparameter and optimizer code, and argument
parsing. Code for defining the ResNet layers can be found in resnet_model.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math
import multiprocessing
import os

from absl import flags
import tensorflow as tf

from TensorFlow.computer_vision.common import imagenet_preprocessing
from TensorFlow.computer_vision.Resnets import resnet_model
from TensorFlow.computer_vision.Resnets.utils import export
from TensorFlow.computer_vision.Resnets.utils.flags import core as flags_core
from TensorFlow.computer_vision.Resnets.utils.logs import hooks_helper
from TensorFlow.computer_vision.Resnets.utils.logs import logger
from TensorFlow.computer_vision.Resnets.utils.misc import distribution_utils
from TensorFlow.computer_vision.Resnets.utils.misc import model_helpers
from TensorFlow.common.horovod_helpers import hvd, horovod_enabled
from TensorFlow.common.habana_estimator import HabanaEstimator
from TensorFlow.common.str2bool import condition_env_var
from TensorFlow.common.tb_utils import ExamplesPerSecondEstimatorHook, write_hparams_v1
from TensorFlow.computer_vision.Resnets import imagenet_main

_NUM_EXAMPLES_NAME = "num_examples"


################################################################################
# Functions for input processing.
################################################################################
def process_record_dataset(dataset,
                           is_training,
                           batch_size,
                           shuffle_buffer,
                           parse_record_fn,
                           num_epochs=1,
                           dtype=tf.float32,
                           datasets_num_private_threads=None,
                           drop_remainder=False,
                           tf_data_experimental_slack=False,
                           experimental_preloading=False):
  """Given a Dataset with raw records, return an iterator over the records.

  Args:
    dataset: A Dataset representing raw records
    is_training: A boolean denoting whether the input is for training.
    batch_size: The number of samples per batch.
    shuffle_buffer: The buffer size to use when shuffling records. A larger
      value results in better randomness, but smaller values reduce startup
      time and use less memory.
    parse_record_fn: A function that takes a raw record and returns the
      corresponding (image, label) pair.
    num_epochs: The number of epochs to repeat the dataset.
    dtype: Data type to use for images/features.
    datasets_num_private_threads: Number of threads for a private
      threadpool created for all datasets computation.
    drop_remainder: A boolean indicates whether to drop the remainder of the
      batches. If True, the batch dimension will be static.
    tf_data_experimental_slack: Whether to enable tf.data's
      `experimental_slack` option.

  Returns:
    Dataset of (image, label) pairs ready for iteration.
  """
  # Defines a specific size thread pool for tf.data operations.
  if datasets_num_private_threads:
    options = tf.data.Options()
    options.experimental_threading.private_threadpool_size = (
        datasets_num_private_threads)
    dataset = dataset.with_options(options)
    tf.compat.v1.logging.info('datasets_num_private_threads: %s',
                              datasets_num_private_threads)


  if not experimental_preloading:
    # Disable intra-op parallelism to optimize for throughput instead of latency.
    options = tf.data.Options()
    options.experimental_threading.max_intra_op_parallelism = 1
    dataset = dataset.with_options(options)
    # Prefetches a batch at a time to smooth out the time taken to load input
    # files for shuffling and processing.
    dataset = dataset.prefetch(buffer_size=batch_size)

  if is_training:
    # Shuffles records before repeating to respect epoch boundaries.
    dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=flags.FLAGS.shuffle_seed)
  else:
    dataset = dataset.take(imagenet_main.NUM_IMAGES['validation'])

  if horovod_enabled():
    # Repeats the dataset. Due to sharding in multinode, training is related
    # directly to the number of max iterations not to number of epochs.
    dataset = dataset.repeat()
  else:
    # Repeats the dataset for the number of epochs to train.
    dataset = dataset.repeat(num_epochs)

  num_parallel_calls = 16 if horovod_enabled() else flags.FLAGS.dataset_parallel_calls

  # Parses the raw records into images and labels.
  dataset = dataset.map(
      lambda value: parse_record_fn(value, is_training, dtype),
      num_parallel_calls=num_parallel_calls, deterministic=False)
  dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

  # Operations between the final prefetch and the get_next call to the iterator
  # will happen synchronously during run time. We prefetch here again to
  # background all of the above processing work and keep it out of the
  # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
  # allows DistributionStrategies to adjust how many batches to fetch based
  # on how many devices are present.

  if experimental_preloading:
    device = "/device:HPU:0"
    dataset = dataset.apply(tf.data.experimental.prefetch_to_device(device))
  else:
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    if tf_data_experimental_slack:
      options = tf.data.Options()
      options.experimental_slack = True
      dataset = dataset.with_options(options)

  return dataset


def get_synth_input_fn(height, width, num_channels, num_classes,
                       dtype=tf.float32):
  """Returns an input function that returns a dataset with random data.

  This input_fn returns a data set that iterates over a set of random data and
  bypasses all preprocessing, e.g. jpeg decode and copy. The host to device
  copy is still included. This used to find the upper throughput bound when
  tunning the full input pipeline.

  Args:
    height: Integer height that will be used to create a fake image tensor.
    width: Integer width that will be used to create a fake image tensor.
    num_channels: Integer depth that will be used to create a fake image tensor.
    num_classes: Number of classes that should be represented in the fake labels
      tensor
    dtype: Data type for features/images.

  Returns:
    An input_fn that can be used in place of a real one to return a dataset
    that can be used for iteration.
  """
  # pylint: disable=unused-argument
  def input_fn(is_training, data_dir, batch_size, *args, **kwargs):
    """Returns dataset filled with random data."""
    # Synthetic input should be within [0, 255].
    inputs = tf.random.truncated_normal(
        [batch_size] + [height, width, num_channels],
        dtype=dtype,
        mean=127,
        stddev=60,
        name='synthetic_inputs')

    labels = tf.random.uniform(
        [batch_size],
        minval=0,
        maxval=num_classes - 1,
        dtype=tf.int32,
        name='synthetic_labels')
    data = tf.data.Dataset.from_tensors((inputs, labels)).repeat()
    data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return data

  return input_fn


def image_bytes_serving_input_fn(image_shape, dtype=tf.float32):
  """Serving input fn for raw jpeg images."""

  def _preprocess_image(image_bytes):
    """Preprocess a single raw image."""
    # Bounding box around the whole image.
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=dtype, shape=[1, 1, 4])
    height, width, num_channels = image_shape
    image = imagenet_preprocessing.preprocess_image(
        image_bytes, bbox, height, width, num_channels, is_training=False)
    return image

  image_bytes_list = tf.compat.v1.placeholder(
      shape=[None], dtype=tf.string, name='input_tensor')
  images = tf.map_fn(
      _preprocess_image, image_bytes_list, back_prop=False, dtype=dtype)
  return tf.estimator.export.TensorServingInputReceiver(
      images, {'image_bytes': image_bytes_list})


def override_flags_and_set_envars_for_gpu_thread_pool(flags_obj):
  """Override flags and set env_vars for performance.

  These settings exist to test the difference between using stock settings
  and manual tuning. It also shows some of the ENV_VARS that can be tweaked to
  squeeze a few extra examples per second.  These settings are defaulted to the
  current platform of interest, which changes over time.

  On systems with small numbers of cpu cores, e.g. under 8 logical cores,
  setting up a gpu thread pool with `tf_gpu_thread_mode=gpu_private` may perform
  poorly.

  Args:
    flags_obj: Current flags, which will be adjusted possibly overriding
    what has been set by the user on the command-line.
  """
  cpu_count = multiprocessing.cpu_count()
  tf.compat.v1.logging.info('Logical CPU cores: %s', cpu_count)

  # Sets up thread pool for each GPU for op scheduling.
  per_gpu_thread_count = 1
  total_gpu_thread_count = per_gpu_thread_count * flags_obj.num_gpus
  os.environ['TF_GPU_THREAD_MODE'] = flags_obj.tf_gpu_thread_mode
  os.environ['TF_GPU_THREAD_COUNT'] = str(per_gpu_thread_count)
  tf.compat.v1.logging.info('TF_GPU_THREAD_COUNT: %s',
                            os.environ['TF_GPU_THREAD_COUNT'])
  tf.compat.v1.logging.info('TF_GPU_THREAD_MODE: %s',
                            os.environ['TF_GPU_THREAD_MODE'])

  # Reduces general thread pool by number of threads used for GPU pool.
  main_thread_count = cpu_count - total_gpu_thread_count
  flags_obj.inter_op_parallelism_threads = main_thread_count

  # Sets thread count for tf.data. Logical cores minus threads assign to the
  # private GPU pool along with 2 thread per GPU for event monitoring and
  # sending / receiving tensors.
  num_monitoring_threads = 2 * flags_obj.num_gpus
  flags_obj.datasets_num_private_threads = (cpu_count - total_gpu_thread_count
                                            - num_monitoring_threads)


################################################################################
# Functions for running training/eval/validation loops for the model.
################################################################################
def learning_rate_with_decay(
    batch_size, batch_denom, num_images, boundary_epochs, train_epochs,
    decay_rates, base_lr=0.1, warmup=False, warmup_epochs=5, use_cosine_lr=False):
  """Get a learning rate that decays step-wise as training progresses.

  Args:
    batch_size: the number of examples processed in each training batch.
    batch_denom: this value will be used to scale the base learning rate.
      `0.1 * batch size` is divided by this number, such that when
      batch_denom == batch_size, the initial learning rate will be 0.1.
    num_images: total number of images that will be used for training.
    boundary_epochs: list of ints representing the epochs at which we
      decay the learning rate.
    decay_rates: list of floats representing the decay rates to be used
      for scaling the learning rate. It should have one more element
      than `boundary_epochs`, and all elements should have the same type.
    base_lr: Initial learning rate scaled based on batch_denom.
    warmup: Run a 5 epoch warmup to the initial lr.
  Returns:
    Returns a function that takes a single argument - the number of batches
    trained so far (global_step)- and returns the learning rate to be used
    for training the next batch.
  """
  initial_learning_rate = base_lr * batch_size / batch_denom
  batches_per_epoch = num_images / batch_size

  # Reduce the learning rate at certain epochs.
  # CIFAR-10: divide by 10 at epoch 100, 150, and 200
  # ImageNet: divide by 10 at epoch 30, 60, 80, and 90
  boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
  vals = [initial_learning_rate * decay for decay in decay_rates]

  def learning_rate_fn(global_step):
    """Builds scaled learning rate function with 5 epoch warm up."""
    if use_cosine_lr:
      tf.compat.v1.logging.info("Using cosine decay learning rate schedule")
      decay_steps = int(batches_per_epoch * train_epochs)
      lr = tf.compat.v1.train.cosine_decay(
        initial_learning_rate, global_step, decay_steps)
    else:
      lr = tf.compat.v1.train.piecewise_constant(global_step, boundaries, vals)

    if warmup:
      tf.compat.v1.logging.info("Using warmup for %d epochs", warmup_epochs)
      warmup_steps = int(batches_per_epoch * warmup_epochs)
      warmup_lr = (
          initial_learning_rate * tf.cast(global_step, tf.float32) / tf.cast(
              warmup_steps, tf.float32))
      return tf.cond(pred=global_step < warmup_steps,
                     true_fn=lambda: warmup_lr,
                     false_fn=lambda: lr)
    return lr

  return learning_rate_fn


def resnet_model_fn(features, labels, mode, model_class,
                    resnet_size, weight_decay, learning_rate_fn, momentum,
                    data_format, resnet_version, loss_scale,
                    loss_filter_fn=None, model_type=resnet_model.DEFAULT_MODEL_TYPE,
                    dtype=resnet_model.DEFAULT_DTYPE,
                    fine_tune=False, label_smoothing=0.0):
  """Shared functionality for different resnet model_fns.

  Initializes the ResnetModel representing the model layers
  and uses that model to build the necessary EstimatorSpecs for
  the `mode` in question. For training, this means building losses,
  the optimizer, and the train op that get passed into the EstimatorSpec.
  For evaluation and prediction, the EstimatorSpec is returned without
  a train op, but with the necessary parameters for the given mode.

  Args:
    features: tensor representing input images
    labels: tensor representing class labels for all input images
    mode: current estimator mode; should be one of
      `tf.estimator.ModeKeys.TRAIN`, `EVALUATE`, `PREDICT`
    model_class: a class representing a TensorFlow model that has a __call__
      function. We assume here that this is a subclass of ResnetModel.
    resnet_size: A single integer for the size of the ResNet model.
    weight_decay: weight decay loss rate used to regularize learned variables.
    learning_rate_fn: function that returns the current learning rate given
      the current global_step
    momentum: momentum term used for optimization
    data_format: Input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.
    resnet_version: Integer representing which version of the ResNet network to
      use. See README for details. Valid values: [1, 2]
    loss_scale: The factor to scale the loss for numerical stability. A detailed
      summary is present in the arg parser help text.
    loss_filter_fn: function that takes a string variable name and returns
      True if the var should be included in loss calculation, and False
      otherwise. If None, batch_normalization variables will be excluded
      from the loss.
    dtype: the TensorFlow dtype to use for calculations.
    fine_tune: If True only train the dense layers(final layers).
    label_smoothing: If greater than 0 then smooth the labels.

  Returns:
    EstimatorSpec parameterized according to the input params and the
    current mode.
  """
  # Uncomment the following lines if you want to write images to summary,
  # we turned it off for performance reason

  # Generate a summary node for the images
  # tf.compat.v1.summary.image('images',
  #     (features, tf.cast(features, tf.float32)) [features.dtype == tf.bfloat16],
  #     max_outputs=6)

  # Disabling bf16 data loading for ResNeXt is a workaround that improves performance.
  # It turns out we get the best performance when dtype == bf16 and data_loader_image_type == fp32.
  # TODO SW-32809: Investigate why some variables were executed on bf16 in ResNeXt and check if
  #                registering them for bf16 execution on HPU is feasible
  # Once SW-32809 is done, we should probably restore the original asserion (commented out below)

  # # Checks that features/images have same data type being used for calculations.
  # assert features.dtype == dtype

  # Until then, we just print out a warning:
  if features.dtype != dtype:
    tf.compat.v1.logging.warning("Features/images dtype differ from dtype used for calculations")

  model = model_class(resnet_size, data_format, resnet_version=resnet_version,
                      model_type=model_type, dtype=dtype)

  logits = model(features, mode == tf.estimator.ModeKeys.TRAIN)

  # This acts as a no-op if the logits are already in fp32 (provided logits are
  # not a SparseTensor). If dtype is is low precision, logits must be cast to
  # fp32 for numerical stability.
  logits = tf.cast(logits, tf.float32)

  predictions = {
      'classes': tf.argmax(input=logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    # Return the predictions and the specification for serving a SavedModel
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'predict': tf.estimator.export.PredictOutput(predictions)
        })

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  labels = tf.cast(labels, tf.int32)

  if label_smoothing != 0.0:
    one_hot_labels = tf.one_hot(labels, 1001)
    cross_entropy = tf.compat.v1.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=one_hot_labels,
        label_smoothing=label_smoothing)
  else:
    cross_entropy = tf.compat.v1.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels)

  # Create a tensor named cross_entropy for logging purposes.
  tf.compat.v1.summary.scalar('cross_entropy', cross_entropy)
  tf.identity(cross_entropy, name='cross_entropy')

  # If no loss_filter_fn is passed, assume we want the default behavior,
  # which is that batch_normalization variables are excluded from loss.
  def exclude_batch_norm(name):
    return 'batch_normalization' not in name

  loss = cross_entropy

  loss_filter_fn = loss_filter_fn or exclude_batch_norm
  # Add weight decay to the loss.
  l2_loss = weight_decay * tf.add_n(
      # loss is computed using fp32 for numerical stability.
      [
          tf.nn.l2_loss(tf.cast(v, tf.float32))
          for v in tf.compat.v1.trainable_variables()
          if loss_filter_fn(v.name)
      ])
  tf.compat.v1.summary.scalar('l2_loss', l2_loss)

  loss += l2_loss

  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.compat.v1.train.get_or_create_global_step()

    learning_rate = learning_rate_fn(global_step)

    # Create a tensor named learning_rate for logging purposes
    tf.compat.v1.summary.scalar('learning_rate', learning_rate)
    tf.identity(learning_rate, name='learning_rate')

    optimizer = tf.compat.v1.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=momentum
    )

    fp16_implementation = getattr(flags.FLAGS, 'fp16_implementation', None)
    if fp16_implementation == 'graph_rewrite':
      optimizer = (
          tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(
              optimizer, loss_scale=loss_scale))

    if horovod_enabled():
      optimizer = hvd.DistributedOptimizer(optimizer)

    def _dense_grad_filter(gvs):
      """Only apply gradient updates to the final layer.

      This function is used for fine tuning.

      Args:
        gvs: list of tuples with gradients and variable info
      Returns:
        filtered gradients so that only the dense layer remains
      """
      return [(g, v) for g, v in gvs if 'dense' in v.name]

    if loss_scale != 1 and fp16_implementation != 'graph_rewrite':
      # When computing fp16 gradients, often intermediate tensor values are
      # so small, they underflow to 0. To avoid this, we multiply the loss by
      # loss_scale to make these tensor values loss_scale times bigger.
      scaled_grad_vars = optimizer.compute_gradients(loss * loss_scale)

      if fine_tune:
        scaled_grad_vars = _dense_grad_filter(scaled_grad_vars)

      # Once the gradient computation is complete we can scale the gradients
      # back to the correct scale before passing them to the optimizer.
      unscaled_grad_vars = [(grad / loss_scale, var)
                            for grad, var in scaled_grad_vars]
      minimize_op = optimizer.apply_gradients(unscaled_grad_vars, global_step)
    else:
      grad_vars = optimizer.compute_gradients(loss*loss_scale)
      if fine_tune:
        grad_vars = _dense_grad_filter(grad_vars)
      grad_vars = [(grad / loss_scale, var) for grad, var in grad_vars]
      minimize_op = optimizer.apply_gradients(grad_vars, global_step)

    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    train_op = tf.group(minimize_op, update_ops)
  else:
    train_op = None

  accuracy = tf.compat.v1.metrics.accuracy(labels, predictions['classes'],
                                           name='accuracy_op')
  accuracy_top_5 = tf.compat.v1.metrics.mean(
      tf.nn.in_top_k(predictions=logits, targets=labels, k=5, name='top_5_op'))
  metrics = {'accuracy': accuracy,
             'accuracy_top_5': accuracy_top_5}

  # Create a tensor named train_accuracy for logging purposes
  tf.compat.v1.summary.scalar('accuracy', accuracy[1])
  tf.compat.v1.summary.scalar('accuracy_top_5', accuracy_top_5[1])
  tf.identity(accuracy[1], name='accuracy')
  tf.identity(accuracy_top_5[1], name='accuracy_top_5')

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)


def resnet_main(
    flags_obj, model_function, input_function, dataset_name, shape=None):
  """Shared main loop for ResNet Models.

  Args:
    flags_obj: An object containing parsed flags. See define_resnet_flags()
      for details.
    model_function: the function that instantiates the Model and builds the
      ops for train/eval. This will be passed directly into the estimator.
    input_function: the function that processes the dataset and returns a
      dataset that the estimator can train on. This will be wrapped with
      all the relevant flags for running and passed to estimator.
    dataset_name: the name of the dataset for training and evaluation. This is
      used for logging purpose.
    shape: list of ints representing the shape of the images used for training.
      This is only used if flags_obj.export_dir is passed.

  Returns:
     Dict of results of the run.  Contains the keys `eval_results` and
    `train_hooks`. `eval_results` contains accuracy (top_1) and accuracy_top_5.
    `train_hooks` is a list the instances of hooks used during training.
  """

  experimental_preloading = flags_obj.experimental_preloading

  model_helpers.apply_clean(flags.FLAGS)

  # Ensures flag override logic is only executed if explicitly triggered.
  if flags_obj.tf_gpu_thread_mode:
    override_flags_and_set_envars_for_gpu_thread_pool(flags_obj)

  # Configures cluster spec for distribution strategy.
  num_workers = distribution_utils.configure_cluster(flags_obj.worker_hosts,
                                                     flags_obj.task_index)

  # Creates session config. allow_soft_placement = True, is required for
  # multi-GPU and is not harmful for other modes.
  session_config = tf.compat.v1.ConfigProto(
      inter_op_parallelism_threads=flags_obj.inter_op_parallelism_threads,
      intra_op_parallelism_threads=flags_obj.intra_op_parallelism_threads,
      allow_soft_placement=not experimental_preloading)

  if horovod_enabled():
    # The Scoped Allocator Optimization is enabled by default unless disabled by a flag.
    if flags_obj.enable_scoped_allocator                                  :
      from tensorflow.core.protobuf import rewriter_config_pb2  # pylint: disable=import-error
      session_config.graph_options.rewrite_options.scoped_allocator_optimization = rewriter_config_pb2.RewriterConfig.ON
      enable_op = session_config.graph_options.rewrite_options.scoped_allocator_opts.enable_op
      del enable_op[:]
      enable_op.append("HorovodAllreduce")

  distribution_strategy = distribution_utils.get_distribution_strategy(
      distribution_strategy=flags_obj.distribution_strategy,
      num_gpus=flags_core.get_num_gpus(flags_obj),
      num_workers=num_workers,
      all_reduce_alg=flags_obj.all_reduce_alg,
      num_packs=flags_obj.num_packs)

  global_batch_size = flags_obj.batch_size * (hvd.size() if horovod_enabled() else 1)
  steps_per_epoch = (imagenet_main.NUM_IMAGES['train'] // global_batch_size)

  if flags_obj.disable_checkpoints:
    if flags_obj.save_checkpoint_steps != None:
      tf.compat.v1.logging.warning(
        "--disable_checkpoints passed together with non-None value of --save_checkpoint_steps. Disabling checkpoints anyway.")
    save_checkpoints_steps = None
  elif flags_obj.save_checkpoint_steps is None:
    save_checkpoints_steps = steps_per_epoch
  else:
    save_checkpoints_steps = flags_obj.save_checkpoint_steps

  # Creates a `RunConfig` that checkpoints every 24 hours which essentially
  # results in checkpoints determined only by `epochs_between_evals`.
  run_config = tf.estimator.RunConfig(
      train_distribute=distribution_strategy,
      session_config=session_config,
      log_step_count_steps=flags_obj.display_steps,
      save_checkpoints_secs=None,
      save_checkpoints_steps=save_checkpoints_steps,
      save_summary_steps=flags_obj.save_summary_steps)

  # Initializes model with all but the dense layer from pretrained ResNet.
  # if flags_obj.pretrained_model_checkpoint_path is not None:
  #   warm_start_settings = tf.estimator.WarmStartSettings(
  #       flags_obj.pretrained_model_checkpoint_path,
  #       vars_to_warm_start='^(?!.*dense)')
  # else:
  #   warm_start_settings = None
  warm_start_settings = None

  model_dir=flags_obj.model_dir

  if horovod_enabled():
    model_dir="{}/worker_{}".format(flags_obj.model_dir, hvd.rank())

  if experimental_preloading:
    SelectedEstimator = HabanaEstimator
  else:
    SelectedEstimator = tf.estimator.Estimator

  eval_batch_size = flags_obj.batch_size

  model_params = {
    'resnet_size': int(flags_obj.resnet_size),
    'data_format': flags_obj.data_format,
    'batch_size': flags_obj.batch_size,
    'resnet_version': int(flags_obj.resnet_version),
    'model_type': flags_obj.model_type,
    'loss_scale': flags_core.get_loss_scale(flags_obj,
                                            default_for_fp16=128),
    'dtype': flags_core.get_tf_dtype(flags_obj),
    'fine_tune': flags_obj.fine_tune,
    'num_workers': num_workers,
    'train_epochs': flags_obj.train_epochs,
    'warmup_epochs': flags_obj.warmup_epochs,
    'use_cosine_lr': flags_obj.use_cosine_lr,
  }

  classifier = SelectedEstimator(
      model_fn=model_function, model_dir=model_dir, config=run_config,
      warm_start_from=warm_start_settings, params=model_params)

  if model_dir:
    write_hparams_v1(model_dir, model_params)

  run_params = {
      'batch_size': flags_obj.batch_size,
      'dtype': flags_core.get_tf_dtype(flags_obj),
      'resnet_size': flags_obj.resnet_size,
      'resnet_version': flags_obj.resnet_version,
      'model_type': flags_obj.model_type,
      'synthetic_data': flags_obj.use_synthetic_data,
      'train_epochs': flags_obj.train_epochs,
      'num_workers': num_workers,
  }

  if flags_obj.use_synthetic_data:
    dataset_name = dataset_name + '-synthetic'

  benchmark_logger = logger.get_benchmark_logger()
  benchmark_logger.log_run_info('resnet', dataset_name, run_params,
                                test_id=flags_obj.benchmark_test_id)

  train_hooks = hooks_helper.get_train_hooks(
      flags_obj.hooks,
      model_dir=model_dir,
      batch_size=flags_obj.batch_size)

  if horovod_enabled():

    if "tf_profiler_hook" not in flags_obj.hooks and os.environ.get("TF_RANGE_TRACE", False):
      from TensorFlow.common.utils import RangeTFProfilerHook
      begin = (imagenet_main.NUM_IMAGES["train"] // (flags_obj.batch_size * hvd.size()) + 100)
      train_hooks.append(RangeTFProfilerHook(begin,20, "./rank-{}".format(hvd.rank())))

    if "synapse_logger_hook" not in flags_obj.hooks and "range" == os.environ.get("HABANA_SYNAPSE_LOGGER", "False").lower():
      from TensorFlow.common.horovod_helpers import SynapseLoggerHook
      begin = (imagenet_main.NUM_IMAGES["train"] // (flags_obj.batch_size * hvd.size()) + 100)
      end = begin + 100
      print("Begin: {}".format(begin))
      print("End: {}".format(end))
      train_hooks.append(SynapseLoggerHook(list(range(begin, end)), False))
    train_hooks.append(hvd.BroadcastGlobalVariablesHook(0))


  def input_fn_train(num_epochs, input_context=None, is_training=True):
    return input_function(
        is_training=is_training,
        data_dir=flags_obj.data_dir,
        batch_size=distribution_utils.per_replica_batch_size(
            flags_obj.batch_size, flags_core.get_num_gpus(flags_obj)),
        num_epochs=num_epochs,
        dtype=flags_core.get_dl_type(flags_obj),
        datasets_num_private_threads=flags_obj.datasets_num_private_threads,
        input_context=input_context, experimental_preloading=experimental_preloading)

  def input_fn_eval():
    return input_function(
        is_training=False,
        data_dir=flags_obj.data_dir,
        batch_size=distribution_utils.per_replica_batch_size(
            eval_batch_size, flags_core.get_num_gpus(flags_obj)),
        num_epochs=1,
        dtype=flags_core.get_dl_type(flags_obj), experimental_preloading=experimental_preloading)

  train_epochs = (0 if flags_obj.eval_only or not flags_obj.train_epochs else
                  flags_obj.train_epochs)

  max_train_steps = flags_obj.max_train_steps
  if max_train_steps is None:
    max_train_steps = steps_per_epoch * (train_epochs + flags_obj.train_offset)

  max_eval_steps = flags_obj.max_eval_steps
  if max_eval_steps is None:
    max_eval_steps = (imagenet_main.NUM_IMAGES['validation'] + eval_batch_size - 1) // eval_batch_size

  if flags_obj.dummy_epoch:
    dummy_classifier = SelectedEstimator(
        model_fn=model_function, model_dir=model_dir+'/dummy_epoch', config=run_config,
        warm_start_from=warm_start_settings, params={
            'resnet_size': int(flags_obj.resnet_size),
            'data_format': flags_obj.data_format,
            'batch_size': flags_obj.batch_size,
            'resnet_version': int(flags_obj.resnet_version),
            'model_type': flags_obj.model_type,
            'loss_scale': flags_core.get_loss_scale(flags_obj,
                                                    default_for_fp16=128),
            'dtype': flags_core.get_tf_dtype(flags_obj),
            'fine_tune': flags_obj.fine_tune,
            'num_workers': num_workers,
            'train_epochs': flags_obj.train_epochs,
            'warmup_epochs': flags_obj.warmup_epochs,
            'use_cosine_lr': flags_obj.use_cosine_lr,
    })

    tf.compat.v1.logging.info('Running 2 dummy iters before main training')
    dummy_classifier.train(
        input_fn=lambda input_context=None: input_fn_train(
            1, input_context=input_context, is_training=False),
        max_steps=2)

  train_hooks.append(ExamplesPerSecondEstimatorHook(
    batch_size=flags_obj.batch_size, every_n_steps=flags_obj.save_summary_steps,
    output_dir=model_dir, extra_metrics={'global_examples/sec': global_batch_size}))

  use_train_and_evaluate = flags_obj.use_train_and_evaluate or num_workers > 1
  if use_train_and_evaluate:
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda input_context=None: input_fn_train(
            train_epochs, input_context=input_context),
        hooks=train_hooks,
        max_steps=max_train_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_eval)
    tf.compat.v1.logging.info('Starting to train and evaluate.')
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
    # tf.estimator.train_and_evalute doesn't return anything in multi-worker
    # case.
    eval_results = {}
  else:
    if train_epochs == 0:
      # If --eval_only is set, perform a single loop with zero train epochs.
      schedule, n_loops = [0], 1
    else:
      # Compute the number of times to loop while training. All but the last
      # pass will train for `epochs_between_evals` epochs, while the last will
      # train for the number needed to reach `training_epochs`. For instance if
      #   train_epochs = 25 and epochs_between_evals = 10
      # schedule will be set to [10, 10, 5]. That is to say, the loop will:
      #   Train for 10 epochs and then evaluate.
      #   Train for another 10 epochs and then evaluate.
      #   Train for a final 5 epochs (to reach 25 epochs) and then evaluate.
      n_loops = math.ceil(train_epochs / flags_obj.epochs_between_evals)
      schedule = [flags_obj.epochs_between_evals for _ in range(int(n_loops))]
      schedule[-1] = train_epochs - sum(schedule[:-1])  # over counting.

    final_step = 0

    for cycle_index, num_train_epochs in enumerate(schedule):
      tf.compat.v1.logging.info('Starting cycle: %d/%d', cycle_index,
                                int(n_loops))

      if num_train_epochs:
        # Since we are calling classifier.train immediately in each loop, the
        # value of num_train_epochs in the lambda function will not be changed
        # before it is used. So it is safe to ignore the pylint error here
        # pylint: disable=cell-var-from-loop
        final_step += num_train_epochs * steps_per_epoch
        classifier.train(
            input_fn=lambda input_context=None: input_fn_train(
                num_train_epochs, input_context=input_context),
            hooks=train_hooks,
            max_steps=max_train_steps if max_train_steps < final_step else final_step)

      # max_eval_steps is associated with testing and profiling.
      # As a result it is frequently called with synthetic data,
      # which will iterate forever. Passing steps=max_eval_steps
      # allows the eval (which is generally unimportant in those circumstances)
      # to terminate. Note that eval will run for max_eval_steps each loop,
      # regardless of the global_step count.
      if flags_obj.get_flag_value("return_before_eval", False):
        return {}
      if flags_obj.get_flag_value("disable_eval", False):
        eval_results = None
        continue
      tf.compat.v1.logging.info('Starting to evaluate.')
      eval_results = classifier.evaluate(input_fn=input_fn_eval,
                                         steps=max_eval_steps)

      benchmark_logger.log_evaluation_result(eval_results)

      if flags_obj.stop_threshold:
        if horovod_enabled():
          past_treshold = tf.cast(model_helpers.past_stop_threshold(
              flags_obj.stop_threshold, eval_results['accuracy']), tf.float32)
          global_past_treshold = tf.math.greater(
              hvd.allreduce(past_treshold, op=hvd.Sum), tf.zeros(1, tf.float32))
          if global_past_treshold.eval(session=tf.compat.v1.Session()):
            break
        else:
          if model_helpers.past_stop_threshold(
              flags_obj.stop_threshold, eval_results['accuracy']):
            break

  if flags_obj.export_dir is not None:
    # Exports a saved model for the given classifier.
    export_dtype = flags_core.get_tf_dtype(flags_obj)
    if flags_obj.image_bytes_as_serving_input:
      input_receiver_fn = functools.partial(
          image_bytes_serving_input_fn, shape, dtype=export_dtype)
    else:
      input_receiver_fn = export.build_tensor_serving_input_receiver_fn(
          shape, batch_size=flags_obj.batch_size, dtype=export_dtype)
    classifier.export_savedmodel(flags_obj.export_dir, input_receiver_fn,
                                 strip_default_attrs=True)

  stats = {}
  stats['eval_results'] = eval_results
  stats['train_hooks'] = train_hooks

  return stats


def define_resnet_flags(resnet_size_choices=None, dynamic_loss_scale=False,
                        fp16_implementation=False):
  """Add flags and validators for ResNet."""
  flags_core.define_base(clean=True, train_epochs=True,
                         epochs_between_evals=True, stop_threshold=True,
                         num_gpu=True, hooks=True, export_dir=True,
                         distribution_strategy=True, disable_checkpoints=True)
  flags_core.define_performance(num_parallel_calls=False,
                                inter_op=True,
                                intra_op=True,
                                synthetic_data=True,
                                dtype=True,
                                data_loader_image_type = True,
                                all_reduce_alg=True,
                                num_packs=True,
                                tf_gpu_thread_mode=True,
                                datasets_num_private_threads=True,
                                dynamic_loss_scale=dynamic_loss_scale,
                                fp16_implementation=fp16_implementation,
                                loss_scale=True,
                                tf_data_experimental_slack=True,
                                max_train_steps=True,
                                max_eval_steps=True)
  flags_core.define_image()
  flags_core.define_benchmark()
  flags_core.define_distribution()
  flags_core.define_experimental()
  flags.adopt_module_key_flags(flags_core)

  flags.DEFINE_integer(
      name='train_offset', default=0,
      help='Number of train epochs before the train loop - for MLPER.')
  flags.DEFINE_enum(
      name='resnet_version', short_name='rv', default='1',
      enum_values=['1', '2'],
      help=flags_core.help_wrap(
          'Version of ResNet. (1 or 2) See README.md for details.'))
  flags.DEFINE_bool(
      name='return_before_eval', short_name='rbv', default=False,
      help=flags_core.help_wrap(
          'If True exit training before evaluation (for testing)'))
  flags.DEFINE_bool(
      name='disable_eval', default=False,
      help=flags_core.help_wrap(
          'If True disables evaluation step after each training epoch'))
  flags.DEFINE_bool(
      name='fine_tune', short_name='ft', default=False,
      help=flags_core.help_wrap(
          'If True do not train any parameters except for the final layer.'))
  flags.DEFINE_string(
      name='pretrained_model_checkpoint_path', short_name='pmcp', default=None,
      help=flags_core.help_wrap(
          'If not None initialize all the network except the final layer with '
          'these values'))
  flags.DEFINE_boolean(
      name='eval_only', default=False,
      help=flags_core.help_wrap('Skip training and only perform evaluation on '
                                'the latest checkpoint.'))
  flags.DEFINE_boolean(
      name='image_bytes_as_serving_input', default=False,
      help=flags_core.help_wrap(
          'If True exports savedmodel with serving signature that accepts '
          'JPEG image bytes instead of a fixed size [HxWxC] tensor that '
          'represents the image. The former is easier to use for serving at '
          'the expense of image resize/cropping being done as part of model '
          'inference. Note, this flag only applies to ImageNet and cannot '
          'be used for CIFAR.'))
  flags.DEFINE_boolean(
      name='use_train_and_evaluate', default=False,
      help=flags_core.help_wrap(
          'If True, uses `tf.estimator.train_and_evaluate` for the training '
          'and evaluation loop, instead of separate calls to `classifier.train '
          'and `classifier.evaluate`, which is the default behavior.'))
  flags.DEFINE_float(
      name='label_smoothing', default=0.1,
      help=flags_core.help_wrap(
          'Label smoothing parameter used in the softmax_cross_entropy'))
  flags.DEFINE_integer(
      name='warmup_epochs', default=8,
      help=flags_core.help_wrap(
          'The number of epochs used for LR warmup'))
  flags.DEFINE_float(
      name='weight_decay', default=6.103515625e-05,
      help=flags_core.help_wrap(
          'Weight decay coefficiant for l2 regularization.'))
  flags.DEFINE_bool(
      name='disable_warmup', default=False,
      help=flags_core.help_wrap(
          'If True disable warmup phase for learning rate.'))
  flags.DEFINE_bool(
      name='use_cosine_lr', default=True,
      help=flags_core.help_wrap(
          'Use cosine decay learning rate schedule'))
  flags.DEFINE_float(
      name='momentum', default=0.875,
      help=flags_core.help_wrap(
          'Momentum value used for optimization.'))
  flags.DEFINE_enum(
      name='model_type', default='resnext',
      enum_values=['resnet', 'resnext'],
      help=flags_core.help_wrap(
          'Residual model type.'))
  flags.DEFINE_integer(
      name='save_summary_steps', default=1,
      help=flags_core.help_wrap(
          'How often to save the summary data.'))

  flags.DEFINE_bool(
      name='enable_scoped_allocator', default=False,
      help=flags_core.help_wrap('Enable scoped allocator optimization'))

  choice_kwargs = dict(
      name='resnet_size', short_name='rs', default='101',
      help=flags_core.help_wrap('The size of the ResNet model to use.'))

  if resnet_size_choices is None:
    flags.DEFINE_string(**choice_kwargs)
  else:
    flags.DEFINE_enum(enum_values=resnet_size_choices, **choice_kwargs)
