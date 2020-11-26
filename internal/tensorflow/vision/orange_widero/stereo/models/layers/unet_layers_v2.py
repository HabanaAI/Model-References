from stereo.models.layers.keras_layers import Concatenate, ReLU
from tensorflow.keras.initializers import glorot_normal
from stereo.models.layers.keras_layers import KerasLikeLayer, Conv2DWrap
from stereo.models.layers.keras_layers import UpSampling2D
from stereo.models.layers.resize_nn_layer import ResizeNearestNeighbor


class DownConvLayer(KerasLikeLayer):

    def __init__(self, ch_out, kernel_size=3, activation='relu', name='down_conv_layer',
                 activation_output=True, regularizer=None, **kwargs):
        assert activation is None or activation == 'relu'

        super(DownConvLayer, self).__init__(name=name, **kwargs)

        self.conv1 = Conv2DWrap(filters=ch_out, kernel_size=kernel_size, name=name + '_conv1',
                                strides=[1, 1], padding="same", data_format='channels_last',
                                activation=activation, kernel_initializer=glorot_normal(),
                                kernel_regularizer=regularizer)
        self.conv2 = Conv2DWrap(filters=ch_out, kernel_size=kernel_size, name=name + '_conv2',
                                strides=[2, 2], padding="same", data_format='channels_last',
                                activation=activation if activation_output else None,
                                kernel_initializer=glorot_normal(),
                                kernel_regularizer=regularizer)

    def call(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        output, to_skip = conv2, conv1
        return output, to_skip


class UpConvLayer(KerasLikeLayer):

    def __init__(self, ch_out, kernel_size=3, activation='relu', name='up_conv_layer',
                 activation_output=True, regularizer=None, concat_skip_last=True,
                 input_shape=None, **kwargs):
        assert activation is None or activation == 'relu'

        super(UpConvLayer, self).__init__(name=name, **kwargs)

        if input_shape is not None:
            self.upsamp = ResizeNearestNeighbor((input_shape[1]*2, input_shape[2]*2))
        else:
            self.upsamp = UpSampling2D(size=(2, 2), data_format='channels_last',
                                       interpolation='nearest', name=name + '_upsamp')
        self.concat = Concatenate(name=name + '_concat', axis=3)
        self.conv1 = Conv2DWrap(filters=ch_out, kernel_size=kernel_size, name=name + '_conv1',
                                strides=[1, 1], padding="same", data_format='channels_last',
                                activation=activation, kernel_initializer=glorot_normal(),
                                kernel_regularizer=regularizer)
        self.conv2 = Conv2DWrap(filters=ch_out, kernel_size=kernel_size, name=name + '_conv2',
                                strides=[1, 1], padding="same", data_format='channels_last',
                                activation=activation if activation_output else None,
                                kernel_initializer=glorot_normal(),
                                kernel_regularizer=regularizer)
        self.concat_skip_last = concat_skip_last

    def call(self, inputs):
        inputs, from_skip = inputs[0], inputs[1]
        upsamp = self.upsamp(inputs)
        concat = self.concat([upsamp, from_skip]) if self.concat_skip_last else \
                 self.concat([from_skip, upsamp])
        conv1 = self.conv1(concat)
        conv2 = self.conv2(conv1)
        return conv2
