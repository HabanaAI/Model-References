"""
Copyright (c) 2017-2020 Intel Corporation

This software and the related documents are Intel copyrighted materials, and your use of them is governed
by the express license under which they were provided to you (License).
Unless the License provides otherwise, you may not use, modify, copy, publish, distribute, disclose
or transmit this software or the related documents without Intel's prior written permission.
This software and the related documents are provided as is, with no express or implied warranties,
other than those that are expressly stated in the License.

"""
from tensorflow.python import keras


class ResizeNearestNeighbor(keras.layers.Layer):
    """
        Resize tensor to out_shape
        Support only FP32 data types!
    """
    def __init__(self, out_shape, align_corners=False, **kwargs):
        """
            out_shape(list): [new_height,new_width] - The output size
        """
        self.out_shape = out_shape
        self.align_corners = align_corners
        super(ResizeNearestNeighbor, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ResizeNearestNeighbor, self).build(input_shape)

    def call(self, input):
        import tensorflow as tf
        return tf.image.resize(input, self.out_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[0], self.out_shape[0], self.out_shape[1], input_shape[3])

    def get_config(self):
        config = super(ResizeNearestNeighbor, self).get_config()
        config.update({'out_shape': [int(self.out_shape[0]), int(self.out_shape[1])]})
        config.update({'align_corners': self.align_corners})
        return config

