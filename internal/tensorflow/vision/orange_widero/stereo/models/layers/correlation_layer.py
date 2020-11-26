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
import tensorflow_addons as tfa

class Correlation(keras.layers.Layer):

    def __init__(self, **kwargs):
        self.interpolation = kwargs.get('interpolation', 'NEAREST')
        if self.interpolation != 'NEAREST':
            print("WARNING: {} interpolation is only supported during training".format(self.interpolation))
        kwargs.pop('interpolation', None)
        super(Correlation, self).__init__(**kwargs)

    def call(self, inputs):

        assert isinstance(inputs, list)
        assert len(inputs) == 5

        cntr = inputs[0]
        srnd = inputs[1]

        steps_cnt = inputs[2].shape[-1]

        scale_split = tf.split(inputs[2], steps_cnt, axis=-1)
        trans_x_split = tf.split(inputs[3], steps_cnt, axis=-1)
        trans_y_split = tf.split(inputs[4], steps_cnt, axis=-1)

        corrs = []

        for i in range(steps_cnt):
            scale = scale_split[i]
            trans_x = trans_x_split[i]
            trans_y = trans_y_split[i]

            # Translations to transforms
            transforms = tf.transpose(a=tf.stack([scale, tf.zeros(tf.shape(input=scale)), trans_x,
                                                tf.zeros(tf.shape(input=scale)), scale, trans_y,
                                                tf.zeros(tf.shape(input=scale)), tf.zeros(tf.shape(input=scale))])[:, :, 0],
                                      perm=[1, 0])

            # Actual image transform
            warped_srnd = tfa.image.transform(images=srnd,
                                                     transforms=transforms,
                                                     interpolation=self.interpolation)
            corr = tf.reduce_sum(input_tensor=cntr * warped_srnd, axis=-1, keepdims=True)
            corrs.append(corr)

        concat = tf.concat(corrs, axis=-1)

        return concat

    def compute_output_shape(self, input_shape):

        assert isinstance(input_shape, list)

        output_shape = input_shape[0]
        steps_cnt = input_shape[2][-1]
        output_shape[-1] = steps_cnt
        return output_shape

    def get_config(self):
        config = super(Correlation, self).get_config()
        return config
