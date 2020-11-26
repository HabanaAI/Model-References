from tensorflow import identity, stop_gradient, cast, float32, exp
from tensorflow import minimum as tf_min
from tensorflow.keras.layers import maximum, minimum, concatenate, Dropout
from tensorflow.keras.backend import name_scope, ones_like
from stereo.interfaces.implements import implements_format
from stereo.models.correlation_map import correlation_map
from stereo.models.arch_utils import pad_image, unpad_image, pad_im_sz_to_levels, \
    image_size_of_format
from stereo.models.layers.keras_layers import scalar_expand
from stereo.models.unet_layers import DownConvLayer, UpConvLayer
from stereo.models.mvs_layers import MonoFeaturesLayer, CorrFeaturesLayer, AdversarialEncoder


def reverse_gradient_scale(x, scale=1.0, drop_out_rate=0.0):
    y = -scale * x + stop_gradient((1 + scale) * x)
    if drop_out_rate > 0:
        y = Dropout(drop_out_rate)(inputs=y, training=True)
    return y

def calc_scale(global_step, num_steps):
    return (2 / (1 + exp(-10 * tf_min(cast(global_step, float32)/num_steps, 1.0)))) - 1

arch_format = ("arch", "four_views_domain")
@implements_format(*arch_format)
def arch(I_cntr, I_srnd_0, I_srnd_1, I_srnd_2, T_cntr_srnd, focal, origin, steps, domain,
         min_Z=None, max_Z=None, batch_norm=False, training=False, bn_first=True, regularization=None,
         adv_gradient_scale=1.0, adv_drop_out_rate=0.0, adv_domain=None):


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
    corr_scores_l2 = maximum([corr_scores_srnd_0_l2, corr_scores_srnd_1_l2, corr_scores_srnd_2_l2], name='corr_scores_l2')

    concat_l2 = concatenate([mono_features_l2, corr_scores_l2], axis=3, name='concat_l2')

    unet_dn_l3, to_skip_l2 = DownConvLayer(kernel_size=3, ch_out=96,  name='unet_dn_l3', batch_norm=batch_norm,
                                           training=training, bn_first=bn_first, regularizer=regularization)(concat_l2)
    unet_dn_l4, to_skip_l3 = DownConvLayer(kernel_size=3, ch_out=192, name='unet_dn_l4', batch_norm=batch_norm,
                                           training=training, bn_first=bn_first, regularizer=regularization)(unet_dn_l3)
    unet_dn_l5, to_skip_l4 = DownConvLayer(kernel_size=3, ch_out=384, name='unet_dn_l5', batch_norm=batch_norm,
                                           training=training, bn_first=bn_first, regularizer=regularization)(unet_dn_l4)
    unet_dn_l6, to_skip_l5 = DownConvLayer(kernel_size=3, ch_out=384, name='unet_dn_l6', batch_norm=batch_norm,
                                           training=training, bn_first=bn_first, regularizer=regularization)(unet_dn_l5)

    unet_up_l5 = UpConvLayer(ch_out=192, name='unet_up_l5', batch_norm=batch_norm, training=training,
                             bn_first=bn_first, regularizer=regularization)([unet_dn_l6, to_skip_l5])
    unet_up_l4 = UpConvLayer(ch_out=96,  name='unet_up_l4', batch_norm=batch_norm, training=training,
                             bn_first=bn_first, regularizer=regularization)([unet_up_l5, to_skip_l4])
    unet_up_l3 = UpConvLayer(ch_out=48,  name='unet_up_l3', batch_norm=batch_norm, training=training,
                             bn_first=bn_first, regularizer=regularization)([unet_up_l4, to_skip_l3])
    unet_up_l2 = UpConvLayer(ch_out=24,  name='unet_up_l2', batch_norm=batch_norm, training=training,
                             bn_first=bn_first, regularizer=regularization)([unet_up_l3, to_skip_l2])
    unet_up_l1 = UpConvLayer(ch_out=12,  name='unet_up_l1', batch_norm=batch_norm, training=training,
                             bn_first=bn_first, regularizer=regularization)([unet_up_l2, to_skip_l1])
    unet_up_l0 = UpConvLayer(ch_out=1,   name='unet_up_l0', batch_norm=batch_norm, training=training,
                             activation_and_bn_output=False, regularizer=regularization)([unet_up_l1, to_skip_l0])


    out = minimum([maximum([unet_up_l0, 1/max_Z*ones_like(unet_up_l0)]),
                   1/min_Z*ones_like(unet_up_l0)])  # this should be called clipped_l0
    out = unpad_image(out, cropping)
    out = identity(out, name='out')

    adversarial_encoder = AdversarialEncoder()
    adv_list = [concat_l2, unet_dn_l6, to_skip_l5, to_skip_l4, to_skip_l3, to_skip_l2, to_skip_l1, to_skip_l0]
    adv_reverse_list = []
    if adv_domain is not None:
        mask = tf.cast(tf.equal(domain, adv_domain), tf.float32)
    for adv_ in adv_list:
        if adv_domain is not None:
            scale = adv_gradient_scale * scalar_expand(mask, adv_)
        else:
            scale = adv_gradient_scale
        adv_reverse_list.append(reverse_gradient_scale(adv_, scale, adv_drop_out_rate))
    adv = adversarial_encoder(adv_reverse_list)

    return {'out': out, 'adv': adv}
