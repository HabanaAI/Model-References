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
# - added logic for bf16 mixed precision setting
# - conditionally set TF_DISABLE_MKL=1 and TF_ALLOW_CONTROL_EDGES_IN_HABANA_OPS=1
"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from absl import app as absl_app
from absl import flags
import tensorflow as tf

from TensorFlow.computer_vision.common import imagenet_preprocessing
from TensorFlow.computer_vision.Resnets import resnet_model
from TensorFlow.computer_vision.Resnets import resnet_run_loop
from TensorFlow.computer_vision.Resnets.utils.flags import core as flags_core
from TensorFlow.computer_vision.Resnets.utils.logs import logger
from TensorFlow.common.debug import dump_callback

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, '..'))

from TensorFlow.common.horovod_helpers import hvd, hvd_init, horovod_enabled, synapse_logger_init
from TensorFlow.common.utils import disable_session_recovery

from TensorFlow.common.library_loader import load_habana_module  # noqa

DEFAULT_IMAGE_SIZE = 224
NUM_CHANNELS = 3
NUM_CLASSES = 1001

NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

_NUM_TRAIN_FILES = 1024
_SHUFFLE_BUFFER = 10000

DATASET_NAME = 'ImageNet'

###############################################################################
# Data processing
###############################################################################
def get_filenames(is_training, data_dir):
  """Return filenames for dataset."""
  if flags.FLAGS.mini_imagenet:
    if is_training:
      return [
          os.path.join(data_dir, 'imagenet_train_subset/imagenet_train_subset-%05d-of-01024' % i)
          for i in range(_NUM_TRAIN_FILES)]
    else:
      return [
          os.path.join(data_dir, 'imagenet_val_subset/imagenet_val_subset-%05d-of-00128' % i)
          for i in range(128)]
  else:
    if is_training:
      return [
          os.path.join(data_dir, 'img_train/img_train-%05d-of-01024' % i)
          for i in range(_NUM_TRAIN_FILES)]
    else:
      return [
          os.path.join(data_dir, 'img_val/img_val-%05d-of-00128' % i)
          for i in range(128)]


def _parse_example_proto(example_serialized):
  """Parses an Example proto containing a training example of an image.

  The output of the build_image_data.py image preprocessing script is a dataset
  containing serialized Example protocol buffers. Each Example proto contains
  the following fields (values are included as examples):

    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/class/synset: 'n03623198'
    image/class/text: 'knee pad'
    image/object/bbox/xmin: 0.1
    image/object/bbox/xmax: 0.9
    image/object/bbox/ymin: 0.2
    image/object/bbox/ymax: 0.6
    image/object/bbox/label: 615
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>

  Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.

  Returns:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int32 containing the label.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
  """
  # Dense features in Example proto.
  feature_map = {
      'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
      'image/class/label': tf.io.FixedLenFeature([], dtype=tf.int64,
                                                 default_value=-1),
      'image/class/text': tf.io.FixedLenFeature([], dtype=tf.string,
                                                default_value=''),
  }
  sparse_float32 = tf.io.VarLenFeature(dtype=tf.float32)
  # Sparse features in Example proto.
  feature_map.update(
      {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                   'image/object/bbox/ymin',
                                   'image/object/bbox/xmax',
                                   'image/object/bbox/ymax']})

  features = tf.io.parse_single_example(serialized=example_serialized, features=feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)

  xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
  ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
  xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
  ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

  # Note that we impose an ordering of (y, x) just to make life difficult.
  bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

  # Force the variable number of bounding boxes into the shape
  # [1, num_boxes, coords].
  bbox = tf.expand_dims(bbox, 0)
  bbox = tf.transpose(a=bbox, perm=[0, 2, 1])

  return features['image/encoded'], label, bbox


def parse_record(raw_record, is_training, dtype):
  """Parses a record containing a training example of an image.

  The input record is parsed into a label and image, and the image is passed
  through preprocessing steps (cropping, flipping, and so on).

  Args:
    raw_record: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
    is_training: A boolean denoting whether the input is for training.
    dtype: data type to use for images/features.

  Returns:
    Tuple with processed image tensor and one-hot-encoded label tensor.
  """
  image_buffer, label, bbox = _parse_example_proto(raw_record)

  image = imagenet_preprocessing.preprocess_image(
      image_buffer=image_buffer,
      bbox=bbox,
      output_height=DEFAULT_IMAGE_SIZE,
      output_width=DEFAULT_IMAGE_SIZE,
      num_channels=NUM_CHANNELS,
      is_training=is_training)
  image = tf.cast(image, dtype)

  return image, label


def input_fn(is_training,
             data_dir,
             batch_size,
             num_epochs=1,
             dtype=tf.float32,
             datasets_num_private_threads=None,
             parse_record_fn=parse_record,
             input_context=None,
             drop_remainder=False,
             tf_data_experimental_slack=False,
             experimental_preloading=False,
             dataset_fn=None):
  """Input function which provides batches for train or eval.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
    dtype: Data type to use for images/features
    datasets_num_private_threads: Number of private threads for tf.data.
    parse_record_fn: Function to use for parsing the records.
    input_context: A `tf.distribute.InputContext` object passed in by
      `tf.distribute.Strategy`.
    drop_remainder: A boolean indicates whether to drop the remainder of the
      batches. If True, the batch dimension will be static.
    tf_data_experimental_slack: Whether to enable tf.data's
      `experimental_slack` option.

  Returns:
    A dataset that can be used for iteration.
  """
  if dataset_fn is None:
    filenames = get_filenames(is_training, data_dir)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
  else:
    dataset = dataset_fn()

  if is_training and horovod_enabled():
    dataset = dataset.shard(hvd.size(), hvd.rank())

  if input_context:
    tf.compat.v1.logging.info(
        'Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d' % (
            input_context.input_pipeline_id, input_context.num_input_pipelines))
    dataset = dataset.shard(input_context.num_input_pipelines,
                            input_context.input_pipeline_id)

  if is_training:
    # Shuffle the input files
    dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES, seed=flags.FLAGS.shuffle_seed)

  # Convert to individual records.
  # cycle_length = 10 means that up to 10 files will be read and deserialized in
  # parallel. You may want to increase this number if you have a large number of
  # CPU cores.
  dataset = dataset.interleave(
      tf.data.TFRecordDataset,
      cycle_length=flags.FLAGS.interleave_cycle_length,
      num_parallel_calls=flags.FLAGS.dataset_parallel_calls)

  return resnet_run_loop.process_record_dataset(
      dataset=dataset,
      is_training=is_training,
      batch_size=batch_size,
      shuffle_buffer=_SHUFFLE_BUFFER,
      parse_record_fn=parse_record_fn,
      num_epochs=num_epochs,
      dtype=dtype,
      datasets_num_private_threads=datasets_num_private_threads,
      drop_remainder=drop_remainder,
      tf_data_experimental_slack=tf_data_experimental_slack,
      experimental_preloading=experimental_preloading
  )


def get_synth_input_fn(dtype):
  return resnet_run_loop.get_synth_input_fn(
      DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, NUM_CHANNELS, NUM_CLASSES,
      dtype=dtype)


###############################################################################
# Running the model
###############################################################################
class ImagenetModel(resnet_model.Model):
  """Model class with appropriate defaults for Imagenet data."""

  def __init__(self, resnet_size, data_format=None, num_classes=NUM_CLASSES,
               resnet_version=resnet_model.DEFAULT_VERSION,
               model_type=resnet_model.DEFAULT_MODEL_TYPE,
               dtype=resnet_model.DEFAULT_DTYPE):
    """These are the parameters that work for Imagenet data.

    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      resnet_version: Integer representing which version of the ResNet network
        to use. See README for details. Valid values: [1, 2]
      dtype: The TensorFlow dtype to use for calculations.
    """

    # For bigger models, we want to use "bottleneck" layers
    if resnet_size < 50:
      bottleneck = False
    else:
      bottleneck = True

    super(ImagenetModel, self).__init__(
        resnet_size=resnet_size,
        bottleneck=bottleneck,
        num_classes=num_classes,
        num_filters=64,
        kernel_size=7,
        conv_stride=2,
        first_pool_size=3,
        first_pool_stride=2,
        block_sizes=_get_block_sizes(resnet_size),
        block_strides=[1, 2, 2, 2],
        model_type=model_type,
        resnet_version=resnet_version,
        data_format=data_format,
        dtype=dtype
    )


def _get_block_sizes(resnet_size):
  """Retrieve the size of each block_layer in the ResNet model.

  The number of block layers used for the Resnet model varies according
  to the size of the model. This helper grabs the layer set we want, throwing
  an error if a non-standard size has been selected.

  Args:
    resnet_size: The number of convolutional layers needed in the model.

  Returns:
    A list of block sizes to use in building the model.

  Raises:
    KeyError: if invalid resnet_size is received.
  """
  choices = {
      18: [2, 2, 2, 2],
      34: [3, 4, 6, 3],
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
      200: [3, 24, 36, 3]
  }

  try:
    return choices[resnet_size]
  except KeyError:
    err = ('Could not find layers for selected Resnet size.\n'
           'Size received: {}; sizes allowed: {}.'.format(
               resnet_size, choices.keys()))
    raise ValueError(err)


def imagenet_model_fn(features, labels, mode, params):
  """Our model_fn for ResNet to be used with our Estimator."""

  # Warmup and higher lr may not be valid for fine tuning with small batches
  # and smaller numbers of training images.
  if params['fine_tune'] or ('disable_warmup' in params and params['disable_warmup']):
    warmup = False
    base_lr = .1
  else:
    warmup = True
    base_lr = .128

  # According to https://arxiv.org/abs/1706.02677 and our internal experiments,
  # the best accuracy results for more than 16 devices are achieved when base_lr == 0.1
  if horovod_enabled() and hvd.size() > 16:
    base_lr = .1

  # Used for ResNeXt101-32x4d
  if params['use_cosine_lr']:
    base_lr = .256

  if horovod_enabled():
    total_batch_size = params['batch_size'] * hvd.size()
  else:
    total_batch_size = params['batch_size'] * params.get('num_workers', 1)

  learning_rate_fn = resnet_run_loop.learning_rate_with_decay(
      batch_size=total_batch_size,
      batch_denom=256, num_images=NUM_IMAGES['train'],
      boundary_epochs=[30, 60, 80, 90],
      train_epochs=params['train_epochs'],
      decay_rates=[1, 0.1, 0.01, 0.001, 1e-4],
      warmup=warmup,
      warmup_epochs=params['warmup_epochs'],
      base_lr=base_lr,
      use_cosine_lr=params['use_cosine_lr'])

  return resnet_run_loop.resnet_model_fn(
      features=features,
      labels=labels,
      mode=mode,
      model_class=ImagenetModel,
      resnet_size=params['resnet_size'],
      weight_decay=flags.FLAGS.weight_decay,
      learning_rate_fn=learning_rate_fn,
      momentum=flags.FLAGS.momentum,
      data_format=params['data_format'],
      resnet_version=params['resnet_version'],
      loss_scale=params['loss_scale'],
      loss_filter_fn=None,
      model_type=params['model_type'],
      dtype=params['dtype'],
      fine_tune=params['fine_tune'],
      label_smoothing=flags.FLAGS.label_smoothing
  )


def define_imagenet_flags():
  resnet_run_loop.define_resnet_flags(
      resnet_size_choices=['18', '34', '50', '101', '152', '200'],
      dynamic_loss_scale=True,
      fp16_implementation=False)
  flags.DEFINE_bool(
      name='use_horovod', short_name='hvd', default=False,
      help=flags_core.help_wrap(
          'Run training using horovod.'))
  flags.adopt_module_key_flags(resnet_run_loop)
  flags_core.set_defaults(train_epochs=10)  # 90


def run_imagenet(flags_obj):
  """Run ResNet ImageNet training and eval loop.

  Args:
    flags_obj: An object containing parsed flag values.

  Returns:
    Dict of results of the run.  Contains the keys `eval_results` and
      `train_hooks`. `eval_results` contains accuracy (top_1) and
      accuracy_top_5. `train_hooks` is a list the instances of hooks used during
      training.
  """
  input_function = (flags_obj.use_synthetic_data and
                    get_synth_input_fn(flags_core.get_tf_dtype(flags_obj)) or
                    input_fn)

  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

  if flags.FLAGS.dtype == 'bf16':
    os.environ['TF_BF16_CONVERSION'] = flags.FLAGS.bf16_config_path

  os.environ.setdefault("TF_DISABLE_MKL", "1")
  os.environ.setdefault("TF_ALLOW_CONTROL_EDGES_IN_HABANA_OPS", "1")

  if flags_obj.use_horovod:
    assert flags_obj.no_hpu == False, "Horovod without HPU is not supported in helpers."
    hvd_init()
  else:
    synapse_logger_init()

  if not flags_obj.no_hpu:
    log_info_devices = load_habana_module()
    print(f"Devices:\n {log_info_devices}")

  result = resnet_run_loop.resnet_main(
      flags_obj, imagenet_model_fn, input_function, DATASET_NAME,
      shape=[DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, NUM_CHANNELS])

  return result


def main(_):
  with dump_callback(), logger.benchmark_context(flags.FLAGS), disable_session_recovery():
    run_imagenet(flags.FLAGS)


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  # Opt-in to TF 2.0 feature - improved variables with a well defined memory model
  tf.compat.v1.enable_resource_variables()

  flags.DEFINE_boolean('mini_imagenet', False, 'mini ImageNet')

  define_imagenet_flags()

  print("Script arguments:", " ".join(sys.argv[1:]))
  absl_app.run(main)
