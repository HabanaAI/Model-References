###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################
"""dataset.py

This module implements functions for reading ImageNet (ILSVRC2012)
dataset in TFRecords format.
"""

import os
from functools import partial

import tensorflow as tf

from config import config
from utils.image_processing import preprocess_image, resize_and_rescale_image


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
    parsed = tf.io.parse_single_example(
        serialized=example_serialized, features=feature_map)
    image = decode_jpeg(parsed['image/encoded'])
    if config.DATA_AUGMENTATION:
        image = preprocess_image(image, 384, 384, is_training=is_training)
    else:
        image = resize_and_rescale_image(image, 384, 384)
    # The label in the tfrecords is 1~1000 (0 not used).
    # So I think the minus 1 (of class label) is needed below.
    label = tf.one_hot(parsed['image/class/label'] - 1, 1000, dtype=tf.float32)
    return (image, label)


def get_dataset(tfrecords_dir, subset, batch_size, is_training, distributed):
    """Read TFRecords files and turn them into a TFRecordDataset.

    Args:
        tfrecords_dir: dataset directory
        subset: pattern to detect subset in dataset directory
        batch_size: Global batch size
        is_training (bool): use True if dataset will be used for training
        distributed (bool): use True if used in distributed environment

    Returns:
        TFRecordDataset: Dataset.
    """
    filenames = tf.io.matching_files(
        os.path.join(tfrecords_dir, '%s-*' % subset))
    ds = tf.data.Dataset.from_tensor_slices(filenames)

    # Sharding should be used only for training and in distributed environments.
    if distributed and is_training:
        from utils.distribution_utils import comm_size, comm_rank
        ds = ds.shard(comm_size(), comm_rank())

    if is_training:
        num_files = tf.cast(tf.shape(input=filenames)[0], tf.int64)
        ds = ds.shuffle(buffer_size=num_files)

    ds = ds.interleave(tf.data.TFRecordDataset, cycle_length=10)

    if is_training:
        ds = ds.shuffle(buffer_size=10000)
    ds = ds.repeat()

    parser = partial(_parse_fn, is_training=is_training)
    ds = ds.map(map_func=parser,
                num_parallel_calls=config.NUM_DATA_WORKERS, deterministic=False)
    ds = ds.batch(batch_size=batch_size, drop_remainder=True)

    # Sharding is already done, so disable autosharding.
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

    return ds.with_options(options)
