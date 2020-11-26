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
    layer1, _ = DownConvLayer(kernel_size=3, ch_out=64, name='layer1', batch_norm=batch_norm, training=training,
                              bn_first=bn_first, regularizer=regularization)(padded_inp)
    layer2, _ = DownConvLayer(kernel_size=3, ch_out=128, name='layer2', batch_norm=batch_norm, training=training,
                              bn_first=bn_first, regularizer=regularization)(layer1)
    layer3, _ = DownConvLayer(kernel_size=3, ch_out=256, name='layer3', batch_norm=batch_norm, training=training,
                              bn_first=bn_first, regularizer=regularization)(layer2)
    layer4, _ = DownConvLayer(kernel_size=3, ch_out=512, name='layer4', batch_norm=batch_norm, training=training,
                              bn_first=bn_first, regularizer=regularization)(layer3)
    layer5, skip = DownConvLayer(kernel_size=3, ch_out=1024, name='layer5', batch_norm=batch_norm, training=training,
                                 bn_first=bn_first, regularizer=regularization)(layer4)
    layer4_up = UpConvLayer(ch_out=512, name='layer4_up', batch_norm=batch_norm, training=training,
                            bn_first=bn_first, regularizer=regularization, deconv_kernel_sz=4)([layer5, skip])
    last_layer = Conv2DWrap(kernel_size=3, filters=1024, name='layer5', kernel_regularizer=regularization)(layer4_up)

    all_preds = Conv2DWrap(kernel_size=3, filters=BOX_PARAMS_NUM+1, name='boxes_params_preds',
                           kernel_regularizer=regularization)(last_layer)
    box_params = all_preds[..., :4]
    orientation_params = all_preds[..., 4:6]
    orientation_params = 2 * (tf.nn.sigmoid(orientation_params) - 0.5)  # cos and sin between -1 to 1
    confidence = all_preds[..., -1]
    confidence = tf.nn.sigmoid(confidence, name='confidence')
    out = tf.concat([box_params, orientation_params, tf.expand_dims(confidence, -1)], axis=-1)

    _ = identity(inp, "input_image")
    _ = identity(box_params, name='boxes_params')
    _ = identity(orientation_params, name='orientation_params')
    _ = identity(confidence, name='confidence')
    _ = identity(out, name='out')

    return out
