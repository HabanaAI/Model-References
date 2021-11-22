# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
###############################################################################
# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
###############################################################################
# Changes:
# - script migration to Tensorflow 2.x version
# - tf.contrib.layers.instance_norm, tf.contrib.layers.group_norm replaced with its tensorflow_addons counterparts
# - added support for HabanaInstanceNormalization
# - added seed setting possibility to glorot_uniform_initializer operations

import tensorflow as tf
import tensorflow_addons as tfa

from runtime.arguments import parse_args


params = parse_args()


def _normalization(inputs, name, mode):
    training = mode == tf.estimator.ModeKeys.TRAIN
    if name == 'instancenorm':
        instance_norm_block = tfa.layers.InstanceNormalization
        if not params.no_hpu:
            from habana_frameworks.tensorflow.ops.instance_norm import HabanaInstanceNormalization
            instance_norm_block = HabanaInstanceNormalization
        gamma_initializer = tf.compat.v1.constant_initializer(1.0)
        return instance_norm_block(gamma_initializer=gamma_initializer, epsilon=1e-6)(inputs, training=training)

    if name == 'groupnorm':
        return tfa.layers.GroupNormalization(groups=16, axis=-1)(inputs, training=training)

    if name == 'batchnorm':
        return tf.compat.v1.keras.layers.BatchNormalization(axis=-1,
                                                            trainable=True,
                                                            virtual_batch_size=None)(inputs, training=training)
    elif name == 'none':
        return inputs
    else:
        raise ValueError('Invalid normalization layer')


def _activation(x, activation):
    if activation == 'relu':
        return tf.nn.relu(x)
    elif activation == 'leaky_relu':
        return tf.nn.leaky_relu(x, alpha=0.01)
    elif activation == 'sigmoid':
        return tf.nn.sigmoid(x)
    elif activation == 'softmax':
        return tf.nn.softmax(x)
    elif activation == 'none':
        return x
    else:
        raise ValueError("Unknown activation {}".format(activation))


def convolution(x,
                out_channels,
                kernel_size=3,
                stride=1,
                mode=tf.estimator.ModeKeys.TRAIN,
                normalization='batchnorm',
                activation='leaky_relu',
                transpose=False):

    conv = tf.keras.layers.Conv3DTranspose if transpose else tf.keras.layers.Conv3D
    regularizer = None

    x = conv(filters=out_channels,
             kernel_size=kernel_size,
             strides=stride,
             activation=None,
             padding='same',
             data_format='channels_last',
             kernel_initializer=tf.compat.v1.glorot_uniform_initializer(seed=params.seed),
             kernel_regularizer=regularizer,
             bias_initializer=tf.compat.v1.zeros_initializer(),
             bias_regularizer=regularizer)(x)

    x = _normalization(x, normalization, mode)

    return _activation(x, activation)


def upsample_block(x, skip_connection, out_channels, normalization, mode):
    x = convolution(x, kernel_size=2, out_channels=out_channels, stride=2,
                    normalization='none', activation='none', transpose=True)
    x = tf.keras.layers.Concatenate(axis=-1)([x, skip_connection])

    x = convolution(x, out_channels=out_channels, normalization=normalization, mode=mode)
    x = convolution(x, out_channels=out_channels, normalization=normalization, mode=mode)
    return x


def input_block(x, out_channels, normalization, mode):
    x = convolution(x, out_channels=out_channels, normalization=normalization, mode=mode)
    x = convolution(x, out_channels=out_channels, normalization=normalization, mode=mode)
    return x


def downsample_block(x, out_channels, normalization, mode):
    x = convolution(x, out_channels=out_channels, normalization=normalization, mode=mode, stride=2)
    return convolution(x, out_channels=out_channels, normalization=normalization, mode=mode)


def linear_block(x, out_channels, mode, activation='leaky_relu', normalization='none'):
    x = convolution(x, out_channels=out_channels, normalization=normalization, mode=mode)
    return convolution(x, out_channels=out_channels, activation=activation, mode=mode, normalization=normalization)


def output_layer(x, out_channels, activation):
    x = tf.keras.layers.Conv3D(out_channels,
                               kernel_size=3,
                               activation=None,
                               padding='same',
                               kernel_regularizer=None,
                               kernel_initializer=tf.compat.v1.glorot_uniform_initializer(seed=params.seed),
                               bias_initializer=tf.compat.v1.zeros_initializer(),
                               bias_regularizer=None)(x)
    return _activation(x, activation)
