import tensorflow as tf
from stereo.models.layers.identity_layer import Identity
from stereo.models.layers.slice_layer import Slice
from tensorflow.keras.layers import Add, Concatenate
from tensorflow.keras import regularizers

from stereo.interfaces.implements import implements_format
from stereo.models.layers.correlation_map_dlo import correlation_map
from stereo.models.arch_utils import corr_steps
from stereo.models.layers.unet_layers_v2 import DownConvLayer, UpConvLayer
from stereo.models.layers.mvs_layers_v2 import MonoFeaturesLayer, CorrFeaturesLayer
from stereo.models.layers.keras_layers import Conv2DWrap, glorot_normal


arch_format = ("arch", "four_views_dlo_cropped")
@implements_format(*arch_format)
def arch(I_cntr, I_srnd_0, I_srnd_1, I_srnd_2, T_cntr_srnd, focal_l2, origin_l2,
         training=False, regularization=None, reg_constant=0.01, corr_steps_kwargs=None):

    steps = corr_steps(**corr_steps_kwargs)
    regularizer = getattr(regularizers, regularization)(reg_constant)

    T_cntr_srnd0 = Slice([0, 0, 0], [-1, 1, -1])(T_cntr_srnd)
    T_cntr_srnd1 = Slice([0, 1, 0], [-1, 1, -1])(T_cntr_srnd)
    T_cntr_srnd2 = Slice([0, 2, 0], [-1, 1, -1])(T_cntr_srnd)

    mono_features_l2, to_skip_l0, to_skip_l1 = MonoFeaturesLayer(name='mono_features_l2', regularizer=regularizer)(I_cntr)
    corr_features_func = CorrFeaturesLayer(name='corr_features', regularizer=regularizer, activation_and_bn_output=False, input_shape=I_cntr.shape.as_list())
    corr_features_cntr_l2 = corr_features_func(I_cntr)
    corr_features_srnd_0_l2 = corr_features_func(I_srnd_0)
    corr_features_srnd_1_l2 = corr_features_func(I_srnd_1)
    corr_features_srnd_2_l2 = corr_features_func(I_srnd_2)

    corr_scores_srnd_0_l2 = correlation_map(corr_features_cntr_l2, corr_features_srnd_0_l2, origin_l2, focal_l2,
                                            T_cntr_srnd0, steps, name='corr_scores_srnd_0_l2')
    corr_scores_srnd_1_l2 = correlation_map(corr_features_cntr_l2, corr_features_srnd_1_l2, origin_l2, focal_l2,
                                            T_cntr_srnd1, steps, name='corr_scores_srnd_1_l2')
    corr_scores_srnd_2_l2 = correlation_map(corr_features_cntr_l2, corr_features_srnd_2_l2, origin_l2, focal_l2,
                                            T_cntr_srnd2, steps, name='corr_scores_srnd_2_l2')
    corr_scores_l2 = Add(name='corr_scores_l2')([corr_scores_srnd_0_l2, Add()([corr_scores_srnd_1_l2, corr_scores_srnd_2_l2])])

    # correlation argmax
    _ = tf.expand_dims(tf.cast(tf.argmax(input=corr_scores_l2, axis=3), dtype=tf.float32),
                       axis=-1, name='corr_scores_argmax')

    concat_l2 = Concatenate(name='concat_l2', axis=3)([mono_features_l2, corr_scores_l2])

    unet_dn_l3, to_skip_l2 = DownConvLayer(kernel_size=3, ch_out=64,  name='unet_dn_l3', regularizer=regularizer)(concat_l2)
    unet_dn_l4, to_skip_l3 = DownConvLayer(kernel_size=3, ch_out=96, name='unet_dn_l4', regularizer=regularizer)(unet_dn_l3)
    unet_dn_l5, to_skip_l4 = DownConvLayer(kernel_size=3, ch_out=144, name='unet_dn_l5', regularizer=regularizer)(unet_dn_l4)
    unet_dn_l6, to_skip_l5 = DownConvLayer(kernel_size=3, ch_out=216, name='unet_dn_l6', regularizer=regularizer)(unet_dn_l5)
    unet_up_l5 = UpConvLayer(ch_out=144, name='unet_up_l5', regularizer=regularizer, input_shape=unet_dn_l6.shape.as_list())([unet_dn_l6, to_skip_l5])
    unet_up_l4 = UpConvLayer(ch_out=96,  name='unet_up_l4', regularizer=regularizer, input_shape=unet_up_l5.shape.as_list())([unet_up_l5, to_skip_l4])
    unet_up_l3 = UpConvLayer(ch_out=48,  name='unet_up_l3', regularizer=regularizer, input_shape=unet_up_l4.shape.as_list())([unet_up_l4, to_skip_l3])
    unet_up_l2 = UpConvLayer(ch_out=24,  name='unet_up_l2', regularizer=regularizer, input_shape=unet_up_l3.shape.as_list())([unet_up_l3, to_skip_l2])
    unet_up_l1 = UpConvLayer(ch_out=16,  name='unet_up_l1', regularizer=regularizer, input_shape=unet_up_l2.shape.as_list())([unet_up_l2, to_skip_l1])
    unet_up_l0 = UpConvLayer(ch_out=8,   name='unet_up_l0', regularizer=regularizer, input_shape=unet_up_l1.shape.as_list())([unet_up_l1, to_skip_l0])
    unet_up_l0_ch1 = Conv2DWrap(filters=1, kernel_size=3, name='unet_up_l0_ch1', strides=[1, 1], padding="same",
                                data_format='channels_last', kernel_initializer=glorot_normal(),
                                kernel_regularizer=regularizer)(unet_up_l0)
    # confidence branch
    conf_up_l1 = UpConvLayer(ch_out=16, name='conf_up_l1', regularizer=regularizer, input_shape=unet_up_l2.shape.as_list(), concat_skip_last=False)([unet_up_l2, to_skip_l1])
    conf_up_l0 = UpConvLayer(ch_out=8,  name='conf_up_l0', regularizer=regularizer, input_shape=conf_up_l1.shape.as_list(), concat_skip_last=False)([conf_up_l1, to_skip_l0])
    conf_up_l0_ch1 = Conv2DWrap(filters=1, kernel_size=3, name='conf_up_l0_ch1', strides=[1, 1], padding="same",
                                data_format='channels_last', kernel_initializer=glorot_normal(),
                                kernel_regularizer=regularizer)(conf_up_l0)

    out = unet_up_l0_ch1  # pred_tapered_inv_depth
    out_conf = conf_up_l0_ch1  # pred_error
    _ = tf.identity(out, name='out')
    _ = tf.identity(out_conf, name='out_conf')
    out = Identity(name='out')(out)
    out_conf = Identity(name='out_conf')(out_conf)

    return {'out': out, 'out_conf': out_conf}
