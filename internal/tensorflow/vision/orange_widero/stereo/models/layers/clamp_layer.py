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


class Clamp(keras.layers.Layer):
    '''
    A custom implementation of the Clamp maffe layer:
    the values of the input are clipped between min_val and max_bal
    Supports only FP32 data type!
    '''
    def __init__(self, min_val=0.0, max_val=1.0, **kwargs):
        self.min_val = min_val
        self.max_val = max_val

        super(Clamp, self).__init__(**kwargs)

    def build(self, input_shape):

        super(Clamp, self).build(input_shape)  # Be sure to call this at the end

    def get_config(self):
        config = {
            'min_val': self.min_val,
            'max_val': self.max_val
        }
        base_config = super(Clamp, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        import tensorflow as tf
        return tf.clip_by_value(inputs, self.min_val, self.max_val)

    def compute_output_shape(self, input_shape):
        return input_shape

