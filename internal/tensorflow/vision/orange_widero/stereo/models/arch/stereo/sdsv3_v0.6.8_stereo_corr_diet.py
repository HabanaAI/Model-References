from tensorflow import identity, reshape
from tensorflow.keras.layers import maximum, minimum, concatenate, add
from tensorflow.keras.backend import name_scope, ones_like
from tensorflow.keras.initializers import glorot_normal

from stereo.interfaces.implements import implements_format
from stereo.models.layers.correlation_map import correlation_map
from stereo.models.layers.unet_layers import DownConvLayer, UpConvLayer
from stereo.models.layers.mvs_layers import MonoFeaturesLayer, CorrFeaturesLayer
from stereo.models.layers.keras_layers import Conv2DWrap
from stereo.models.arch_utils import pad_image, unpad_image, pad_im_sz_to_levels, image_size_of_format


arch_format = ("arch", "four_views")
@implements_format(*arch_format)
def arch(I_cntr, I_srnd_0, I_srnd_1, I_srnd_2, T_cntr_srnd, focal, origin, steps,
         min_Z=None, max_Z=None, batch_norm=False, training=False, bn_first=True, regularization=None):

    im_sz_padded = pad_im_sz_to_levels(image_size_of_format(I_cntr), 6)
    I_cntr, origin_new, cropping = pad_image(I_cntr, origin, im_sz_padded)
    I_srnd_0, origin_new, cropping = pad_image(I_srnd_0, origin, im_sz_padded)
    I_srnd_1, origin_new, cropping = pad_image(I_srnd_1, origin, im_sz_padded)
    I_srnd_2, origin_new, cropping = pad_image(I_srnd_2, origin, im_sz_padded)
    origin = origin_new

    mono_features_l2, to_skip_l0, to_skip_l1 = MonoFeaturesLayer(name='mono_features_l2', batch_norm=batch_norm,
                                                                 training=training, bn_first=bn_first,
                                                                 regularizer=regularization)(I_cntr)
    corr_features_func = CorrFeaturesLayer(name='corr_features', batch_norm=batch_norm, training=training,
                                           bn_first=bn_first, regularizer=regularization,
                                           activation_and_bn_output=False)
    corr_features_cntr_l2 = corr_features_func(I_cntr)
    corr_features_srnd_0_l2 = corr_features_func(I_srnd_0)
    corr_features_srnd_1_l2 = corr_features_func(I_srnd_1)
    corr_features_srnd_2_l2 = corr_features_func(I_srnd_2)

    corr_scores_srnd_0_l2 = correlation_map(corr_features_srnd_0_l2, corr_features_cntr_l2, origin / 4, focal / 4,
                                            T_cntr_srnd[:, 0, :], steps, name='corr_scores_srnd_0_l2')
    corr_scores_srnd_1_l2 = correlation_map(corr_features_srnd_1_l2, corr_features_cntr_l2, origin / 4, focal / 4,
                                            T_cntr_srnd[:, 1, :], steps, name='corr_scores_srnd_1_l2')
    corr_scores_srnd_2_l2 = correlation_map(corr_features_srnd_2_l2, corr_features_cntr_l2, origin / 4, focal / 4,
                                            T_cntr_srnd[:, 2, :], steps, name='corr_scores_srnd_2_l2')
    corr_scores_l2 = add([corr_scores_srnd_0_l2, corr_scores_srnd_1_l2, corr_scores_srnd_2_l2], name='corr_scores_l2')

    concat_l2 = concatenate([mono_features_l2, corr_scores_l2], axis=3, name='concat_l2')

    unet_dn_l3, to_skip_l2 = DownConvLayer(kernel_size=3, ch_out=96,  name='unet_dn_l3', batch_norm=batch_norm,
                                           training=training, bn_first=bn_first, regularizer=regularization)(concat_l2)
    unet_dn_l4, to_skip_l3 = DownConvLayer(kernel_size=3, ch_out=192, name='unet_dn_l4', batch_norm=batch_norm,
                                           training=training, bn_first=bn_first, regularizer=regularization)(unet_dn_l3)
    unet_dn_l5, to_skip_l4 = DownConvLayer(kernel_size=3, ch_out=384, name='unet_dn_l5', batch_norm=batch_norm,
                                           training=training, bn_first=bn_first, regularizer=regularization)(unet_dn_l4)
    unet_dn_l6, to_skip_l5 = DownConvLayer(kernel_size=3, ch_out=384, name='unet_dn_l6', batch_norm=batch_norm,
                                           training=training, bn_first=bn_first, regularizer=regularization)(unet_dn_l5)

    unet_up_l5 = UpConvLayer(ch_out=384, name='unet_up_l5', batch_norm=batch_norm, training=training,
                             bn_first=bn_first, regularizer=regularization)([unet_dn_l6, to_skip_l5])
    unet_up_l4 = UpConvLayer(ch_out=192,  name='unet_up_l4', batch_norm=batch_norm, training=training,
                             bn_first=bn_first, regularizer=regularization)([unet_up_l5, to_skip_l4])
    unet_up_l3 = UpConvLayer(ch_out=96,  name='unet_up_l3', batch_norm=batch_norm, training=training,
                             bn_first=bn_first, regularizer=regularization)([unet_up_l4, to_skip_l3])
    unet_up_l2 = UpConvLayer(ch_out=48,  name='unet_up_l2', batch_norm=batch_norm, training=training,
                             bn_first=bn_first, regularizer=regularization)([unet_up_l3, to_skip_l2])
    unet_up_l1 = UpConvLayer(ch_out=24,  name='unet_up_l1', batch_norm=batch_norm, training=training,
                             bn_first=bn_first, regularizer=regularization)([unet_up_l2, to_skip_l1])
    unet_up_l0 = UpConvLayer(ch_out=8,   name='unet_up_l0', batch_norm=batch_norm, training=training,
                             activation_and_bn_output=False, regularizer=regularization)([unet_up_l1, to_skip_l0])

    unet_up_l0_ch1 = Conv2DWrap(filters=1, kernel_size=3, name='unet_up_l0_ch1', strides=[1, 1], padding="same",
                                data_format='channels_last', activation=None, kernel_initializer=glorot_normal(),
                                kernel_regularizer=regularization)(unet_up_l0)

    min_inv_Z = 1 / max_Z * ones_like(unet_up_l0_ch1)
    max_inv_Z = 1 / min_Z * ones_like(unet_up_l0_ch1)
    out_padded = minimum([maximum([unet_up_l0_ch1, min_inv_Z]), max_inv_Z])
    out = unpad_image(out_padded, cropping)
    out = identity(out, name='out')
    return out
