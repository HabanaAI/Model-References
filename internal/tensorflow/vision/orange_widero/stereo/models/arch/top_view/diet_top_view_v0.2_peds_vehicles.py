import tensorflow as tf
from tensorflow import identity
from tensorflow.keras.layers import Dense
from stereo.interfaces.implements import implements_format
from stereo.models.layers.unet_layers import DownConvLayer, UpConvLayer
from stereo.models.arch_utils import pad_image, unpad_image, pad_im_sz_to_levels, \
    image_size_of_format, color_categories, color_confidence

arch_format = ("arch", "top_view")
@implements_format(*arch_format)
def arch(inp, origin=[350, 150], back_frames=1, batch_norm=False, training=False, bn_first=True,
         regularization=None):
    """
    Input through u-net pathway.
    """
    inp = tf.slice(inp, [0, 0, 0, 0], [-1, -1, -1, back_frames])
    im_sz_padded = pad_im_sz_to_levels(image_size_of_format(inp), 6)
    inp, origin_new, cropping = pad_image(inp, origin, im_sz_padded)
    unet_dn_l1, to_skip_l0 = DownConvLayer(kernel_size=3, ch_out=8, name='unet_dn_l1', batch_norm=batch_norm,
                                           training=training, bn_first=bn_first)(inp)

    unet_dn_l2, to_skip_l1 = DownConvLayer(kernel_size=3, ch_out=16, name='unet_dn_l2', batch_norm=batch_norm,
                                           training=training, bn_first=bn_first, regularizer=regularization)(
        unet_dn_l1)

    unet_dn_l3, to_skip_l2 = DownConvLayer(kernel_size=3, ch_out=96, name='unet_dn_l3', batch_norm=batch_norm,
                                           training=training, bn_first=bn_first)(unet_dn_l2)

    unet_dn_l4, to_skip_l3 = DownConvLayer(kernel_size=3, ch_out=192, name='unet_dn_l4', batch_norm=batch_norm,
                                           training=training, bn_first=bn_first, regularizer=regularization)(
        unet_dn_l3)

    unet_up_l3 = UpConvLayer(ch_out=48, name='unet_up_l3', batch_norm=batch_norm, training=training,
                             deconv_kernel_sz=4,
                             bn_first=bn_first, regularizer=regularization)([unet_dn_l4, to_skip_l3])

    unet_up_l2 = UpConvLayer(ch_out=24, name='unet_up_l2', batch_norm=batch_norm, training=training,
                             deconv_kernel_sz=4,
                             bn_first=bn_first, regularizer=regularization)([unet_up_l3, to_skip_l2])
    unet_up_l1 = UpConvLayer(ch_out=12, name='unet_up_l1', batch_norm=batch_norm, training=training,
                             deconv_kernel_sz=4,
                             bn_first=bn_first, regularizer=regularization)([unet_up_l2, to_skip_l1])
    unet_up_l0 = UpConvLayer(ch_out=2, name='unet_up_l0', batch_norm=batch_norm, training=training,
                             deconv_kernel_sz=4,
                             activation_and_bn_output=False, regularizer=regularization)([unet_up_l1, to_skip_l0])
    out = tf.nn.sigmoid(unet_up_l0)
    out = unpad_image(out, cropping)
    _ = identity(out, name='out')
    _ = identity(inp, name='input_image')
    _ = identity(tf.slice(inp, [0, 0, 0, 0], [-1, -1, -1, 1]), name='current_img')
    _ = identity(tf.slice(inp, [0, 0, 0, back_frames - 1], [-1, -1, -1, 1]), name=('%s_back_img' % (back_frames - 1)))

    return out
