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


class Slice(keras.layers.Layer):
    """
    a layer wrapper to K.slice
    Supprt only FP32 data type!
    """
    def __init__(self, start, size, **kwargs):
        """
        :param dims: a list of output channels for the split layer
        """
        self.start = start
        self.size = size

        super(Slice, self).__init__(**kwargs)

    def build(self, input_shape):

        super(Slice, self).build(input_shape)  # Be sure to call this at the end

    def get_config(self):
        config = {
            'start': self.start,
            'size': self.size
        }
        base_config = super(Slice, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        import tensorflow as tf
        return tf.slice(inputs,self.start,self.size)

    def compute_output_shape(self, input_shape):
        out_shape = []
        for ishp,st,sz in zip(input_shape,self.start,self.size):
            if ishp is None:
                out_shape.append(ishp)
            elif sz < 0:
                out_shape.append(ishp - st)
            else:
                out_shape.append(sz)
        return [tuple(out_shape)]


