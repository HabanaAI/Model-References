import tensorflow as tf
from tensorflow import identity
from stereo.interfaces.implements import implements_format
from stereo.models.layers.unet_layers import DownConvLayer, UpConvLayer
from stereo.models.layers.keras_layers import Conv2DWrap
from stereo.models.arch_utils import pad_image, pad_im_sz_to_levels, image_size_of_format


BOX_PARAMS_NUM = 6


arch_format = ("arch", "top_view")
@implements_format(*arch_format)
def arch(inp, origin=[350, 150], batch_norm=False, training=False, bn_first=True,
         regularization=None):
    """
    Input through u-net pathway.
    """
    im_sz_padded = pad_im_sz_to_levels(image_size_of_format(inp), 5)
    padded_inp, origin_new, cropping = pad_image(inp, origin, im_sz_padded)
    layer1, _ = DownConvLayer(kernel_size=3, ch_out=32, name='layer1', batch_norm=batch_norm, training=training,
                              bn_first=bn_first, regularizer=regularization)(padded_inp)
    layer2, _ = DownConvLayer(kernel_size=3, ch_out=64, name='layer2', batch_norm=batch_norm, training=training,
                              bn_first=bn_first, regularizer=regularization)(layer1)
    layer3, _ = DownConvLayer(kernel_size=3, ch_out=128, name='layer3', batch_norm=batch_norm, training=training,
                              bn_first=bn_first, regularizer=regularization)(layer2)
    layer4, _ = DownConvLayer(kernel_size=3, ch_out=256, name='layer4', batch_norm=batch_norm, training=training,
                              bn_first=bn_first, regularizer=regularization)(layer3)
    layer5, skip = DownConvLayer(kernel_size=3, ch_out=512, name='layer5', batch_norm=batch_norm, training=training,
                                 bn_first=bn_first, regularizer=regularization)(layer4)
    last_layer = UpConvLayer(ch_out=256, name='layer4_up', batch_norm=batch_norm, training=training,
                             bn_first=bn_first, regularizer=regularization, deconv_kernel_sz=4)([layer5, skip])

    out = Conv2DWrap(kernel_size=3, filters=BOX_PARAMS_NUM + 1, kernel_regularizer=regularization)(last_layer)

    _ = identity(inp, "input_image")
    _ = identity(out, name='out')

    return out
