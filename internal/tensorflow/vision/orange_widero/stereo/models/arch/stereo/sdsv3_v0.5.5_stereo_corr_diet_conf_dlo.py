import tensorflow as tf
from menta.src.layers import ZeroPadding2D, Cropping2D,\
    Maximum, Concatenate, Slice, Clamp, Identity
from menta.src import regularizers

from stereo.interfaces.implements import implements_format
from stereo.models.layers.correlation_map_dlo import correlation_map
from stereo.models.arch_utils import im_padding, corr_steps
from stereo.models.layers.unet_layers import DownConvLayer, UpConvLayer
from stereo.models.layers.mvs_layers import MonoFeaturesLayer, CorrFeaturesLayer

UNET_LEVELS = 6

arch_format = ("arch", "four_views_dlo")
@implements_format(*arch_format)
def arch(I_cntr, I_srnd_0, I_srnd_1, I_srnd_2, T_cntr_srnd, focal_l2, origin_l2,
         min_Z=None, max_Z=None, batch_norm=False, training=False, bn_first=True,
         regularization=None, reg_constant=0.01,
         corr_steps_kwargs=None):
    steps = corr_steps(**corr_steps_kwargs)
    padding = im_padding(I_cntr, UNET_LEVELS)
    I_cntr = ZeroPadding2D(padding, data_format='channels_last')(I_cntr)
    I_srnd_0 = ZeroPadding2D(padding, data_format='channels_last')(I_srnd_0)
    I_srnd_1 = ZeroPadding2D(padding, data_format='channels_last')(I_srnd_1)
    I_srnd_2 = ZeroPadding2D(padding, data_format='channels_last')(I_srnd_2)

    regularizer = getattr(regularizers, regularization)(reg_constant)

    mono_features_l2, to_skip_l0, to_skip_l1 = MonoFeaturesLayer(name='mono_features_l2', batch_norm=batch_norm,
                                                                 training=training, bn_first=bn_first,
                                                                 regularizer=regularizer)(I_cntr)
    corr_features_func = CorrFeaturesLayer(name='corr_features', batch_norm=batch_norm, training=training,
                                           bn_first=bn_first, regularizer=regularizer, activation_and_bn_output=False)
    corr_features_cntr_l2 = corr_features_func(I_cntr)
    corr_features_srnd_0_l2 = corr_features_func(I_srnd_0)
    corr_features_srnd_1_l2 = corr_features_func(I_srnd_1)
    corr_features_srnd_2_l2 = corr_features_func(I_srnd_2)

    T_cntr_srnd0 = Slice([0, 0, 0], [-1, 1, -1])(T_cntr_srnd)
    T_cntr_srnd1 = Slice([0, 1, 0], [-1, 1, -1])(T_cntr_srnd)
    T_cntr_srnd2 = Slice([0, 2, 0], [-1, 1, -1])(T_cntr_srnd)
    corr_scores_srnd_0_l2 = correlation_map(corr_features_cntr_l2, corr_features_srnd_0_l2, origin_l2, focal_l2,
                                            T_cntr_srnd0, steps, name='corr_scores_srnd_0_l2')
    corr_scores_srnd_1_l2 = correlation_map(corr_features_cntr_l2, corr_features_srnd_1_l2, origin_l2, focal_l2,
                                            T_cntr_srnd1, steps, name='corr_scores_srnd_1_l2')
    corr_scores_srnd_2_l2 = correlation_map(corr_features_cntr_l2, corr_features_srnd_2_l2, origin_l2, focal_l2,
                                            T_cntr_srnd2, steps, name='corr_scores_srnd_2_l2')
    corr_scores_l2 = Maximum(name='corr_scores_l2')([corr_scores_srnd_0_l2, Maximum()([corr_scores_srnd_1_l2, corr_scores_srnd_2_l2])])

    # correlation argmax
    _ = tf.expand_dims(tf.cast(tf.argmax(input=corr_scores_l2, axis=3), dtype=tf.float32),
                       axis=-1, name='corr_scores_argmax')

    concat_l2 = Concatenate(name='concat_l2', axis=3)([mono_features_l2, corr_scores_l2])

    unet_dn_l3, to_skip_l2 = DownConvLayer(kernel_size=3, ch_out=96,  name='unet_dn_l3', batch_norm=batch_norm,
                                           training=training, bn_first=bn_first, regularizer=regularizer)(concat_l2)
    unet_dn_l4, to_skip_l3 = DownConvLayer(kernel_size=3, ch_out=192, name='unet_dn_l4', batch_norm=batch_norm,
                                           training=training, bn_first=bn_first, regularizer=regularizer)(unet_dn_l3)
    unet_dn_l5, to_skip_l4 = DownConvLayer(kernel_size=3, ch_out=384, name='unet_dn_l5', batch_norm=batch_norm,
                                           training=training, bn_first=bn_first, regularizer=regularizer)(unet_dn_l4)
    unet_dn_l6, to_skip_l5 = DownConvLayer(kernel_size=3, ch_out=384, name='unet_dn_l6', batch_norm=batch_norm,
                                           training=training, bn_first=bn_first, regularizer=regularizer)(unet_dn_l5)
    unet_up_l5 = UpConvLayer(ch_out=192, name='unet_up_l5', batch_norm=batch_norm, training=training,
                             bn_first=bn_first, regularizer=regularizer)([unet_dn_l6, to_skip_l5])
    unet_up_l4 = UpConvLayer(ch_out=96,  name='unet_up_l4', batch_norm=batch_norm, training=training,
                             bn_first=bn_first, regularizer=regularizer)([unet_up_l5, to_skip_l4])
    unet_up_l3 = UpConvLayer(ch_out=48,  name='unet_up_l3', batch_norm=batch_norm, training=training,
                             bn_first=bn_first, regularizer=regularizer)([unet_up_l4, to_skip_l3])
    unet_up_l2 = UpConvLayer(ch_out=24,  name='unet_up_l2', batch_norm=batch_norm, training=training,
                             bn_first=bn_first, regularizer=regularizer)([unet_up_l3, to_skip_l2])
    unet_up_l1 = UpConvLayer(ch_out=12,  name='unet_up_l1', batch_norm=batch_norm, training=training,
                             bn_first=bn_first, regularizer=regularizer)([unet_up_l2, to_skip_l1])
    unet_up_l0 = UpConvLayer(ch_out=1,   name='unet_up_l0', batch_norm=batch_norm, training=training,
                             activation_and_bn_output=False, regularizer=regularizer)([unet_up_l1, to_skip_l0])

    # confidence
    conf_up_l1 = UpConvLayer(ch_out=12, name='conf_up_l1', batch_norm=batch_norm, training=training,
                             bn_first=bn_first, regularizer=regularizer)([unet_up_l2, to_skip_l1])
    conf_up_l0 = UpConvLayer(ch_out=1, name='conf_up_l0', batch_norm=False, training=training,
                             activation='relu', regularizer=regularizer)([conf_up_l1, to_skip_l0])
    pred_inv_depth = unet_up_l0
    pred_error = conf_up_l0

    pred_inv_depth = Clamp(min_val=1/max_Z, max_val=1/min_Z)(pred_inv_depth)
    out = Cropping2D(padding, data_format='channels_last')(pred_inv_depth)
    out_conf = Cropping2D(padding, data_format='channels_last')(pred_error)

    _ = tf.identity(out, name='out')
    _ = tf.identity(out_conf, name='out_conf')

    out = Identity(name='out')(out)
    out_conf = Identity(name='out_conf')(out_conf)
    return {'out': out, 'out_conf': out_conf}
