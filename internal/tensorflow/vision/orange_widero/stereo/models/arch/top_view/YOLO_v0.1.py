import tensorflow as tf
from tensorflow import identity
from stereo.interfaces.implements import implements_format
from stereo.models.layers.unet_layers import DownConvLayer
from stereo.models.layers.keras_layers import Conv2DWrap
from stereo.models.arch_utils import pad_image, pad_im_sz_to_levels, image_size_of_format


BOX_PARAMS_NUM = 5


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
    layer5 = Conv2DWrap(kernel_size=3, filters=1024, name='layer5', kernel_regularizer=regularization)(layer4)

    boxes_params_preds = Conv2DWrap(kernel_size=3, filters=BOX_PARAMS_NUM, name='boxes_params_preds',
                                    kernel_regularizer=regularization)(layer5)
    confidence_preds = Conv2DWrap(kernel_size=3, filters=1, name='confidence_preds',
                                  kernel_regularizer=regularization)(layer5)
    confidence_preds = tf.nn.sigmoid(confidence_preds, name='confidence_preds_sigmoid')
    out = tf.concat([boxes_params_preds, confidence_preds], axis=-1)

    _ = identity(inp, "input_image")
    _ = identity(boxes_params_preds, name='boxes_params_preds')
    _ = identity(confidence_preds, name='confidence_preds')
    _ = identity(out, name='out')

    return out
