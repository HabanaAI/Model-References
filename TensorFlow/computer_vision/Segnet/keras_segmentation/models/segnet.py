# Copyright (c) 2021 Habana Labs Ltd., an Intel Company
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

from .config import IMAGE_ORDERING
from .model_utils import get_segmentation_model
from .vgg16 import get_vgg_encoder
#from .mobilenet import get_mobilenet_encoder
from .basic_models import vanilla_encoder
#from .resnet50 import get_resnet50_encoder
import tensorflow as tf
import numpy as np

def replace_upsample(upsample_size, input):
    channels = input.shape[-1]
    new_wt = np.zeros(list(upsample_size) + [channels, channels])
    for channel_id in range(channels):
        new_wt[:,:,channel_id, channel_id] = 1
    new_bias = np.zeros([channels])
    init = tf.keras.initializers.Constant(new_wt.flatten())
    layer = Conv2DTranspose(filters=channels, kernel_size=upsample_size, strides=upsample_size, bias_initializer='zeros', kernel_initializer=init)
    layer.trainable = False
    return layer(input)

def segnet_decoder(f, n_classes, n_up=3, use_upsampling=False):

    assert n_up >= 2

    o = f
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o) if use_upsampling else replace_upsample((2, 2), o)
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    for _ in range(n_up-2):
        o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o) if use_upsampling else replace_upsample((2, 2), o)
        o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
        o = (Conv2D(128, (3, 3), padding='valid',
             data_format=IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o) if use_upsampling else replace_upsample((2, 2), o)
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(64, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = Conv2D(n_classes, (3, 3), padding='same',
               data_format=IMAGE_ORDERING)(o)

    return o


def _segnet(n_classes, encoder,  input_height=416, input_width=608,
            encoder_level=3, batch_size=2, use_upsampling=False, loss_type=0):

    img_input, levels = encoder(
        input_height=input_height,  input_width=input_width, batch_size=batch_size)

    feat = levels[encoder_level]
    o = segnet_decoder(feat, n_classes, n_up=3, use_upsampling=use_upsampling)
    model = get_segmentation_model(img_input, o, loss_type)

    return model


def segnet(n_classes, input_height=416, input_width=608, encoder_level=3, use_upsampling=False, loss_type=0):

    model = _segnet(n_classes, vanilla_encoder,  input_height=input_height,
                    input_width=input_width, encoder_level=encoder_level, use_upsampling=False, loss_type=loss_type)
    model.model_name = "segnet"
    return model


def vgg_segnet(n_classes, input_height=416, input_width=608, encoder_level=3, batch_size=2, use_upsampling=False, loss_type=0):

    model = _segnet(n_classes, get_vgg_encoder,  input_height=input_height,
                    input_width=input_width, encoder_level=encoder_level, batch_size=batch_size, use_upsampling=False, loss_type=loss_type)
    model.model_name = "vgg_segnet"
    return model

if __name__ == '__main__':
    m = vgg_segnet(101)
    m = segnet(101)
    # m = mobilenet_segnet( 101 )
    # from tensorflow.keras.utils import plot_model
    # plot_model( m , show_shapes=True , to_file='model.png')
