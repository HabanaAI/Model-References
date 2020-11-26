from stereo.models.layers.keras_layers import ReLU, BatchNormalization, Concatenate
from stereo.models.layers.keras_layers import glorot_normal
from stereo.models.layers.keras_layers import KerasLikeLayer, Conv2DWrap, batch_norm_activation, Conv2DTransposeWrap

from tensorflow.keras.layers import UpSampling2D


class DownConvLayer(KerasLikeLayer):
    """combination of conv stride 1, conv stride 2"""

    def __init__(self,
                 kernel_size, ch_out, activation='relu', name='down_conv_layer',
                 batch_norm=False, batch_renorm=False, training=False, bn_first=True,
                 activation_and_bn_output=True, regularizer=None,
                 **kwargs):
        assert activation is None or activation == 'relu'

        super(DownConvLayer, self).__init__(name=name, **kwargs)

        self.conv1 = Conv2DWrap(filters=ch_out, kernel_size=kernel_size, name=name + '_dcl1',
                                strides=[1, 1], padding="same", data_format='channels_last',
                                kernel_initializer=glorot_normal(),
                                kernel_regularizer=regularizer)
        self.conv2 = Conv2DWrap(filters=ch_out, kernel_size=kernel_size, name=name + '_dcl2',
                                strides=[2, 2], padding="same", data_format='channels_last',
                                kernel_initializer=glorot_normal(),
                                kernel_regularizer=regularizer)

        self.relu1 = ReLU(name=name + '_relu1')
        self.relu2 = ReLU(name=name + '_relu2')

        if batch_norm:
            self.bn1 = BatchNormalization(name=name + '_bn1', renorm=batch_renorm, fused=True, trainable=training)
            self.bn2 = BatchNormalization(name=name + '_bn2', renorm=batch_renorm, fused=True, trainable=training)
        else:
            self.bn1, self.bn2 = None, None

        self.activation = activation is not None
        self.batch_norm = batch_norm
        self.training = training
        self.bn_first = bn_first
        self.activation_and_bn_output = activation_and_bn_output

    def call(self, inputs):
        dcl1 = self.conv1(inputs)
        dcl1 = batch_norm_activation(dcl1, bn_layer=self.bn1, activation_layer=self.relu1,
                                     training=self.training, batch_norm=self.batch_norm, activation=self.activation,
                                     bn_first=self.bn_first)
        dcl2 = self.conv2(dcl1)
        if self.activation_and_bn_output:
            dcl2 = batch_norm_activation(dcl2, bn_layer=self.bn2, activation_layer=self.relu2,
                                         training=self.training, batch_norm=self.batch_norm, activation=self.activation,
                                         bn_first=self.bn_first)
        return dcl2, dcl1


class UpConvLayer(KerasLikeLayer):
    """combination of upconv stride 2, concat with skip and conv stride 1"""

    def __init__(self,
                 ch_out, activation='relu', name='up_conv_layer',
                 batch_norm=False, batch_renorm=False, training=False, bn_first=True,
                 activation_and_bn_output=True, regularizer=None,
                 upconv_kernel_sz=5, conv_kernel_sz=3, upsample_nearest_instead=False,
                 **kwargs):

        assert activation is None or activation == 'relu'

        super(UpConvLayer, self).__init__(name=name, **kwargs)

        if not upsample_nearest_instead:
            self.upconv1 = Conv2DTransposeWrap(filters=ch_out, kernel_size=upconv_kernel_sz, name=name+'_ucl1',
                                               strides=[2, 2], padding="same", data_format='channels_last',
                                               kernel_initializer=glorot_normal(),
                                               kernel_regularizer=regularizer)
        else:
            self.upconv1 = UpSampling2D(name=name+'_upsamp')

        self.conv1 = Conv2DWrap(filters=ch_out, kernel_size=conv_kernel_sz, name=name + '_ucl2',
                                strides=[1, 1], padding="same", data_format='channels_last',
                                kernel_initializer=glorot_normal(),
                                kernel_regularizer=regularizer)

        self.relu1 = ReLU(name=name + '_relu1')
        self.relu2 = ReLU(name=name + '_relu2')

        if batch_norm:
            self.bn1 = BatchNormalization(name=name + '_bn1', renorm=batch_renorm, fused=True, trainable=training)
            self.bn2 = BatchNormalization(name=name + '_bn2', renorm=batch_renorm, fused=True, trainable=training)
        else:
            self.bn1, self.bn2 = None, None

        self.concat = Concatenate(name=name + '_concat', axis=3)

        self.activation = activation is not None
        self.batch_norm = batch_norm
        self.training = training
        self.bn_first = bn_first
        self.activation_and_bn_output = activation_and_bn_output

    def call(self, inputs):
        """inp, from_skip = inputs[0], inputs[1] """
        inp, from_skip = inputs[0], inputs[1]
        ucl1 = self.upconv1(inp)
        ucl1 = batch_norm_activation(ucl1, bn_layer=self.bn1, activation_layer=self.relu1,
                                     training=self.training, batch_norm=self.batch_norm, activation=self.activation,
                                     bn_first=self.bn_first)
        concat = self.concat([ucl1, from_skip])
        ucl2 = self.conv1(concat)

        if self.activation_and_bn_output:
            ucl2 = batch_norm_activation(ucl2, bn_layer=self.bn2, activation_layer=self.relu2,
                                         training=self.training, batch_norm=self.batch_norm,
                                         activation=self.activation,
                                         bn_first=self.bn_first)
        return ucl2
