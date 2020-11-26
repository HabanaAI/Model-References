from tensorflow import identity, cast, argmax, float32, expand_dims, split
from tensorflow.keras.layers import maximum, minimum, concatenate
from tensorflow.keras.backend import name_scope, ones_like

from stereo.interfaces.implements import implements_format
from stereo.models.correlation_map import correlation_map
from stereo.models.arch_utils import pad_image, unpad_image, pad_im_sz_to_levels, \
    image_size_of_format
from stereo.models.unet_layers import DownConvLayer, UpConvLayer
from stereo.models.mvs_layers_v3 import MonoFeaturesLayer
from stereo.models.mvs_layers import CorrFeaturesLayer
from stereo.models.keras_layers import Conv2DWrap
from tensorflow.keras.initializers import glorot_normal


def pad_images(images, origin, num_levels=6):
    im_sz_padded = pad_im_sz_to_levels(image_size_of_format(images[0]), num_levels)
    for i, image in enumerate(images):
        images[i], origin_new, cropping = pad_image(image, origin, im_sz_padded)
    return images, origin_new, cropping


def corr_scores(I_cntr, I_srnd_0, I_srnd_1, I_srnd_2, T_cntr_srnd, steps, focal, origin,
                batch_norm, training, bn_first, regularization):
    corr_features_func = CorrFeaturesLayer(name='corr_features', batch_norm=batch_norm, training=training,
                                           bn_first=bn_first, regularizer=regularization,
                                           activation_and_bn_output=False)
    images = concatenate([I_cntr, I_srnd_0, I_srnd_1, I_srnd_2], axis=0)

    (corr_features_cntr_l2, corr_features_srnd_0_l2,
     corr_features_srnd_1_l2, corr_features_srnd_2_l2) = split(corr_features_func(images), 4, axis=0)

    corr_scores_srnd_0_l2 = correlation_map(corr_features_srnd_0_l2, corr_features_cntr_l2, origin / 4, focal / 4,
                                            T_cntr_srnd[:, 0, :], steps, name='corr_scores_srnd_0_l2')
    corr_scores_srnd_1_l2 = correlation_map(corr_features_srnd_1_l2, corr_features_cntr_l2, origin / 4, focal / 4,
                                            T_cntr_srnd[:, 1, :], steps, name='corr_scores_srnd_1_l2')
    corr_scores_srnd_2_l2 = correlation_map(corr_features_srnd_2_l2, corr_features_cntr_l2, origin / 4, focal / 4,
                                            T_cntr_srnd[:, 2, :], steps, name='corr_scores_srnd_2_l2')
    corr_scores_l2 = corr_scores_srnd_0_l2 + corr_scores_srnd_1_l2 + corr_scores_srnd_2_l2

    return corr_scores_l2


arch_format = ("arch", "four_views")
@implements_format(*arch_format)
def arch(I_cntr, I_srnd_0, I_srnd_1, I_srnd_2, T_cntr_srnd, focal, origin, steps,
         min_Z=None, max_Z=None, batch_norm=False, training=False, bn_first=True, regularization=None):

    [I_cntr, I_srnd_0, I_srnd_1, I_srnd_2], origin, cropping = pad_images([I_cntr, I_srnd_0, I_srnd_1, I_srnd_2],
                                                                          origin)

    mono_l3, mono_skip_l2, mono_skip_l1, mono_skip_l0 = MonoFeaturesLayer(name='mono_features_l3', batch_norm=batch_norm,
                                                                          training=training, bn_first=bn_first,
                                                                          regularizer=regularization)(I_cntr)

    corr_scores_l2 = corr_scores(I_cntr, I_srnd_0, I_srnd_1, I_srnd_2, T_cntr_srnd, steps, focal, origin,
                                 batch_norm, training, bn_first, regularization)
    corr_scores_l3, corr_scores_l2_to_skip = DownConvLayer(kernel_size=3, ch_out=8,  name='corr_scores_l3',
                                                           batch_norm=batch_norm, training=training, bn_first=bn_first,
                                                           regularizer=regularization)(corr_scores_l2)
    corr_scores_argmax = identity(cast(expand_dims(argmax(corr_scores_l2, axis=3), axis=3), float32),name='corr_argmax')
    corr_scores_argmax_l3, corr_scores_argmax_l2 = DownConvLayer(kernel_size=3, ch_out=8,  name='corr_scores_argmax_l3',
                                                                 batch_norm=batch_norm, training=training,
                                                                 bn_first=bn_first,
                                                                 regularizer=regularization)(corr_scores_argmax)
    concat_l3 = concatenate([mono_l3, corr_scores_l3, corr_scores_argmax_l3], axis=3, name='concat_l3')

    unet_dn_l4, to_skip_l3 = DownConvLayer(kernel_size=3, ch_out=96, ch_mid=48, name='unet_dn_l4', batch_norm=batch_norm,
                                           training=training, bn_first=bn_first, regularizer=regularization)(concat_l3)
    unet_dn_l5, to_skip_l4 = DownConvLayer(kernel_size=3, ch_out=192, ch_mid=96, name='unet_dn_l5', batch_norm=batch_norm,
                                           training=training, bn_first=bn_first, regularizer=regularization)(unet_dn_l4)
    unet_dn_l6, to_skip_l5 = DownConvLayer(kernel_size=3, ch_out=384, ch_mid=192, name='unet_dn_l6', batch_norm=batch_norm,
                                           training=training, bn_first=bn_first, regularizer=regularization)(unet_dn_l5)

    unet_up_l5 = UpConvLayer(ch_out=192, name='unet_up_l5', batch_norm=batch_norm, training=training,
                             bn_first=bn_first, regularizer=regularization)([unet_dn_l6, to_skip_l5])
    unet_up_l4 = UpConvLayer(ch_out=96,  name='unet_up_l4', batch_norm=batch_norm, training=training,
                             bn_first=bn_first, regularizer=regularization)([unet_up_l5, to_skip_l4])
    unet_up_l3 = UpConvLayer(ch_out=48,  name='unet_up_l3', batch_norm=batch_norm, training=training,
                             bn_first=bn_first, regularizer=regularization)([unet_up_l4, to_skip_l3])
    concat_l2 = concatenate([mono_skip_l2, corr_scores_l2_to_skip, corr_scores_argmax_l2], axis=3, name='concat_l2')
    unet_up_l2 = UpConvLayer(ch_out=24,  name='unet_up_l2', batch_norm=batch_norm, training=training,
                             bn_first=bn_first, regularizer=regularization)([unet_up_l3, concat_l2])
    unet_up_l1 = UpConvLayer(ch_out=16,  name='unet_up_l1', batch_norm=batch_norm, training=training,
                             bn_first=bn_first, regularizer=regularization)([unet_up_l2, mono_skip_l1])
    unet_up_l0 = UpConvLayer(ch_out=8,   name='unet_up_l0', batch_norm=batch_norm, training=training,
                             regularizer=regularization)([unet_up_l1, mono_skip_l0])

    out = Conv2DWrap(filters=1, kernel_size=3, name='final_conv', kernel_initializer=glorot_normal(),
                     padding='same', data_format='channels_last',
                     kernel_regularizer=regularization)(unet_up_l0)
    out = minimum([maximum([out, 1 / max_Z * ones_like(out)]),
                   1 / min_Z * ones_like(out)])  # this should be called clipped_l0
    out = unpad_image(out, cropping)
    out = identity(out, name='out')
    return out
