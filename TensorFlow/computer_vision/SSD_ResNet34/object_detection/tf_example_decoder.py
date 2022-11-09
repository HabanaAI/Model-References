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
# Changes:
# - Ported to TF 2.2
# - Changed tf.contrib.slim import to tf_slim
# - Formatted with autopep8

"""Tensorflow Example proto decoder for object detection.

A decoder to decode string tensors containing serialized tensorflow.Example
protos for object detection.
"""
import tensorflow.compat.v1 as tf

import tf_slim as slim
slim_example_decoder = slim.tfexample_decoder


class TfExampleDecoder(object):
    """Tensorflow Example proto decoder."""

    def __init__(self):
        """Constructor sets keys_to_features and items_to_handlers."""
        self.keys_to_features = {
            'image/encoded':
                tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format':
                tf.FixedLenFeature((), tf.string, default_value='jpeg'),
            'image/filename':
                tf.FixedLenFeature((), tf.string, default_value=''),
            'image/key/sha256':
                tf.FixedLenFeature((), tf.string, default_value=''),
            'image/source_id':
                tf.FixedLenFeature((), tf.string, default_value=''),
            'image/height':
                tf.FixedLenFeature((), tf.int64, 1),
            'image/width':
                tf.FixedLenFeature((), tf.int64, 1),
            # Object boxes and classes.
            'image/object/bbox/xmin':
                tf.VarLenFeature(tf.float32),
            'image/object/bbox/xmax':
                tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymin':
                tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymax':
                tf.VarLenFeature(tf.float32),
            'image/object/class/label':
                tf.VarLenFeature(tf.int64),
            'image/object/class/text':
                tf.VarLenFeature(tf.string),
            'image/object/area':
                tf.VarLenFeature(tf.float32),
            'image/object/is_crowd':
                tf.VarLenFeature(tf.int64),
            'image/object/difficult':
                tf.VarLenFeature(tf.int64),
            'image/object/group_of':
                tf.VarLenFeature(tf.int64),
            'image/object/weight':
                tf.VarLenFeature(tf.float32),
        }
        self.items_to_handlers = {
            'image': slim_example_decoder.Image(
                image_key='image/encoded', format_key='image/format', channels=3),
            'source_id': (
                slim_example_decoder.Tensor('image/source_id')),
            'key': (
                slim_example_decoder.Tensor('image/key/sha256')),
            'filename': (
                slim_example_decoder.Tensor('image/filename')),
            # Object boxes and classes.
            'groundtruth_boxes': (
                slim_example_decoder.BoundingBox(
                    ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/')),
        }
        label_handler = slim_example_decoder.Tensor('image/object/class/label')
        self.items_to_handlers['groundtruth_classes'] = label_handler
        CLASS_INV_MAP = (
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
            22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
            44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
            64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87,
            88, 89, 90)
        _MAP = {j: i for i, j in enumerate(CLASS_INV_MAP)}
        CLASS_MAP = tuple(_MAP.get(i, -1) for i in range(max(CLASS_INV_MAP) + 1))
        self.CLASS_MAP = tf.convert_to_tensor(CLASS_MAP)

    def decode(self, tf_example_string_tensor):
        """Decodes serialized tensorflow example and returns a tensor dictionary.

        Args:
          tf_example_string_tensor: a string tensor holding a serialized tensorflow
            example proto.

        Returns:
          A dictionary of the following tensors.
          image - 3D uint8 tensor of shape [None, None, 3]
            containing image.
          source_id - string tensor containing original
            image id.
          key - string tensor with unique sha256 hash key.
          filename - string tensor with original dataset
            filename.
          groundtruth_boxes - 2D float32 tensor of shape
            [None, 4] containing box corners.
          groundtruth_classes - 1D int64 tensor of shape
        """
        serialized_example = tf.reshape(tf_example_string_tensor, shape=[])
        decoder = slim_example_decoder.TFExampleDecoder(self.keys_to_features,
                                                        self.items_to_handlers)
        keys = sorted(decoder.list_items())

        tensors = decoder.decode(serialized_example, items=keys)
        tensor_dict = dict(zip(keys, tensors))
        tensor_dict['image'].set_shape([None, None, 3])
        tensor_dict['groundtruth_classes'] = tf.reshape(tensor_dict['groundtruth_classes'], [-1, 1])
        tensor_dict['groundtruth_classes'] = tf.gather(self.CLASS_MAP, tensor_dict['groundtruth_classes'])
        tensor_dict['groundtruth_classes'] = tf.cast(tensor_dict['groundtruth_classes'], dtype=tf.float32)

        return tensor_dict


class TfExampleSegmentationDecoder(object):
    """Tensorflow Example proto decoder."""

    def __init__(self):
        """Constructor sets keys_to_features and items_to_handlers."""
        self.keys_to_features = {
            'image/encoded':
                tf.FixedLenFeature((), tf.string, default_value=''),
            'image/filename':
                tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format':
                tf.FixedLenFeature((), tf.string, default_value='jpeg'),
            'image/height':
                tf.FixedLenFeature((), tf.int64, default_value=0),
            'image/width':
                tf.FixedLenFeature((), tf.int64, default_value=0),
            'image/segmentation/class/encoded':
                tf.FixedLenFeature((), tf.string, default_value=''),
            'image/segmentation/class/format':
                tf.FixedLenFeature((), tf.string, default_value='png'),
        }
        self.items_to_handlers = {
            'image': slim_example_decoder.Image(
                image_key='image/encoded', format_key='image/format', channels=3),
            'labels_class': slim_example_decoder.Image(
                image_key='image/segmentation/class/encoded',
                format_key='image/segmentation/class/format',
                channels=1)
        }

    def decode(self, tf_example_string_tensor):
        """Decodes serialized tensorflow example and returns a tensor dictionary.

        Args:
          tf_example_string_tensor: a string tensor holding a serialized tensorflow
            example proto.

        Returns:
          A dictionary of the following tensors.
          image - 3D uint8 tensor of shape [None, None, 3] containing image.
          labels_class - 2D unit8 tensor of shape [None, None] containing
            pixel-wise class labels.
        """
        serialized_example = tf.reshape(tf_example_string_tensor, shape=[])
        decoder = slim_example_decoder.TFExampleDecoder(self.keys_to_features,
                                                        self.items_to_handlers)
        keys = sorted(decoder.list_items())
        keys = ['image', 'labels_class']

        tensors = decoder.decode(serialized_example, items=keys)
        tensor_dict = dict(zip(keys, tensors))
        tensor_dict['image'].set_shape([None, None, 3])
        return tensor_dict
