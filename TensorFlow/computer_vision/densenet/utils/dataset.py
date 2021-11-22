"""dataset.py

This module implements functions for reading ImageNet (ILSVRC2012)
dataset in TFRecords format.
"""
###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################
# List of changes:
# - Removed dataset.prefetch() from get_dataset() function
# Modified the pipeline in order to used using fused jpeg decode and crop for training
# Set value of num_parallel calls in interleave() and map() calls to
#     tf.data.experimental.AUTOTUNE

import os
from functools import partial

import tensorflow as tf

from config import config
from utils.image_processing import preprocess_for_eval, preprocess_for_train


def decode_jpeg(image_buffer, scope=None):
    """Decode a JPEG string into one 3-D float image Tensor.

    Args:
        image_buffer: scalar string Tensor.
        scope: Optional scope for name_scope.
    Returns:
        3-D float Tensor with values ranging from [0, 1).
    """
    with tf.compat.v1.name_scope(values=[image_buffer], name=scope,
                       default_name='decode_jpeg'):
        # Decode the string as an RGB JPEG.
        # Note that the resulting image contains an unknown height
        # and width that is set dynamically by decode_jpeg. In other
        # words, the height and width of image is unknown at compile-i
        # time.
        image = tf.image.decode_jpeg(image_buffer, channels=3)

        # After this point, all image pixels reside in [0,1)
        # until the very end, when they're rescaled to (-1, 1).
        # The various adjust_* ops all require this range for dtype
        # float.
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image


def _parse_fn(example_serialized, is_training):
    """Helper function for parse_fn_train() and parse_fn_valid()

    Each Example proto (TFRecord) contains the following fields:

    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/class/synset: 'n03623198'
    image/class/text: 'knee pad'
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>

    Args:
        example_serialized: scalar Tensor tf.string containing a
                            serialized Example protocol buffer.
        is_training: training (True) or validation (False).

    Returns:
        image_buffer: Tensor tf.string containing the contents of
        a JPEG file.
        label: Tensor tf.int32 containing the label.
        text: Tensor tf.string containing the human-readable label.
    """
    feature_map = {
        'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/class/label': tf.io.FixedLenFeature([], dtype=tf.int64,
                                                default_value=-1),
        'image/class/text': tf.io.FixedLenFeature([], dtype=tf.string,
                                               default_value=''),
    }
    parsed = tf.io.parse_single_example(serialized=example_serialized, features=feature_map)

    if is_training:
      bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                       dtype=tf.float32,
                       shape=[1, 1, 4])
      image_buffer = parsed['image/encoded']
      # We pass undecoded image buffer for train in order to perform
      # fused decode and crop
      image = preprocess_for_train(image_buffer, 224, 224, bbox, fast_mode=True)
    else:
      image = decode_jpeg(parsed['image/encoded'])
      image = preprocess_for_eval(image, 224, 224)

    # The label in the tfrecords is 1~1000 (0 not used).
    # So I think the minus 1 (of class label) is needed below.
    label = tf.one_hot(parsed['image/class/label'] - 1, 1000, dtype=tf.float32)
    return (image, label)


def get_dataset(tfrecords_dir, subset, batch_size):
    """Read TFRecords files and turn them into a TFRecordDataset."""
    if subset == 'train':
      tfrecords_dir = os.path.join(tfrecords_dir, "img_train")
      subset = 'img_train'
      is_training = True
    elif subset == 'validation':
      tfrecords_dir = os.path.join(tfrecords_dir, "img_val")
      subset = 'img_val'
      is_training = False

    files = tf.io.matching_files(os.path.join(tfrecords_dir, '%s-*' % subset))
    shards = tf.data.Dataset.from_tensor_slices(files)
    shards = shards.shuffle(tf.cast(tf.shape(input=files)[0], tf.int64))
    shards = shards.repeat()
    dataset = shards.interleave(tf.data.TFRecordDataset,
                                cycle_length=10,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=8192)

    parser = partial(_parse_fn, is_training=is_training)

    dataset = dataset.map(map_func=parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

    return dataset