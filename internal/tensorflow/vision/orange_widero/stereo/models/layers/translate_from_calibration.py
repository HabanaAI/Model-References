"""
Copyright (c) 2017-2020 Intel Corporation

This software and the related documents are Intel copyrighted materials, and your use of them is governed
by the express license under which they were provided to you (License).
Unless the License provides otherwise, you may not use, modify, copy, publish, distribute, disclose
or transmit this software or the related documents without Intel's prior written permission.
This software and the related documents are provided as is, with no express or implied warranties,
other than those that are expressly stated in the License.

"""

import tensorflow as tf
from tensorflow.python import keras


class TranslateFromCalibration(keras.layers.Layer):
    """
    Generate translation parameters from set-up calibration.
    """
    def __init__(self, steps, **kwargs):
        self._steps = steps
        super(TranslateFromCalibration, self).__init__(**kwargs)

    def call(self, inputs):

        assert isinstance(inputs, list)
        assert len(inputs) == 3

        origin = inputs[0]
        focal = inputs[1]
        cntr_srnd_t = inputs[2]

        scale, trans_x, trans_y = [], [], []
        for i, d in enumerate(self._steps):
            D = tf.constant(1., dtype=tf.float32) / (d ** -1 + cntr_srnd_t[:, :, 2]) if d != 0 else 0
            scale.append(tf.constant(1., dtype=tf.float32) - D * cntr_srnd_t[:, :, 2])
            trans_x.append(D * cntr_srnd_t[:, :, 2] * origin[:, 0] + D * focal[:, 0] * cntr_srnd_t[:, :, 0])
            trans_y.append(D * cntr_srnd_t[:, :, 2] * origin[:, 1] + D * focal[:, 1] * cntr_srnd_t[:, :, 1])

        scale = tf.stack(scale, axis=1)[:, :, 0]
        trans_x = tf.stack(trans_x, axis=1)[:, :, 0]
        trans_y = tf.stack(trans_y, axis=1)[:, :, 0]

        return [scale, trans_x, trans_y]

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], len(self._steps), 1)
        return [output_shape] * 3

    def get_config(self):
        config = super(TranslateFromCalibration, self).get_config()
        config.update({
            'steps': self._steps
        })
        return config
