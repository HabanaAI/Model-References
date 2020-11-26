import tensorflow as tf

try:
    from menta.src.layers import ZeroPadding2D, Cropping2D,\
        Conv2D, Conv2DTranspose, ReLU, BatchNormalization, Add, Concatenate, UpSampling2D
    from menta.src.initializers import glorot_normal, Ones
    print("Using MENTA layers")
except ImportError:
    from tensorflow.keras.layers import ZeroPadding2D, Cropping2D,\
        Conv2D, Conv2DTranspose, ReLU, BatchNormalization, Add, Concatenate, UpSampling2D
    from tensorflow.keras.initializers import glorot_normal, Ones
    print("Using keras layers")


def scalar_expand(s, m):
    return tf.reshape(s, [-1] + [1] * (len(m.shape.as_list()) - 1))


def batch_norm_activation(input, bn_layer, activation_layer, training, batch_norm, activation, bn_first):
    if bn_first:
        out = bn_layer(input, training=training) if batch_norm else input
        out = activation_layer(out) if activation else out
    else:
        out = activation_layer(input) if activation else input
        out = bn_layer(out, training=training) if batch_norm else out
    if batch_norm:
        for x in bn_layer.updates:
            tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, x)
    return out


class KerasLikeLayer(object):
    """
    A thin wrapper for inner keras/menta layers. Used for building object oriented architectures
    and at the same time exposing the inner keras/menta layers of the model.
    """
    def __init__(self, name, **kwargs):
        self.name = name

    def __call__(self, *args, **kwargs):
        with tf.compat.v1.name_scope(self.name):
            outputs = self.call(*args, **kwargs)
        return outputs

    def call(self, inputs, **kwargs):
        # TODO: use abc module
        raise NotImplementedError('subclasses must override call()')


class Conv2DWrap(KerasLikeLayer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', data_format=None,
                 dilation_rate=(1, 1), activation=None, use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, name='Conv2D', **kwargs):
        super(Conv2DWrap, self).__init__(name=name, **kwargs)
        self.conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format,
                           dilation_rate=dilation_rate, activation=activation, use_bias=use_bias,
                           kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                           kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                           activity_regularizer=activity_regularizer,
                           kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, name=name)
        self.already_called = False

    def call(self, inputs):
        conv = self.conv(inputs)
        # This code is not needed is keras training. REGULARIZATION_LOSSES are not used in keras
        if not self.already_called:
            for loss in self.conv.losses:
                tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES, loss)
            self.already_called = True
        return conv


class Conv2DTransposeWrap(KerasLikeLayer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', data_format=None,
                 activation=None, use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, name='ConvTranspose2D', **kwargs):
        super(Conv2DTransposeWrap, self).__init__(name=name, **kwargs)
        self.upconv = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                      data_format=data_format, activation=activation, use_bias=use_bias,
                                      kernel_initializer=kernel_initializer, bias_initializer=bias_constraint,
                                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                      activity_regularizer=activity_regularizer,
                                      kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, name=name)
        self.already_called = False

    def call(self, inputs):
        upconv = self.upconv(inputs)
        # This code is not needed is keras training. REGULARIZATION_LOSSES are not used in keras
        if not self.already_called:
            for loss in self.upconv.losses:
                tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES, loss)
            self.already_called = True
        return upconv
