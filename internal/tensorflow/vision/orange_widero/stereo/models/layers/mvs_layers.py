from stereo.models.layers.keras_layers import KerasLikeLayer
from stereo.models.layers.unet_layers import DownConvLayer, UpConvLayer
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np


class MonoFeaturesLayer(KerasLikeLayer):
    def __init__(self, batch_norm=False, batch_renorm=False, training=False, bn_first=True, name='mono_features',
                 regularizer=None, **kwargs):
        super(MonoFeaturesLayer, self).__init__(name=name, **kwargs)
        self.mono_l1 = DownConvLayer(kernel_size=7, ch_out=8, activation='relu', name='mono_l1',
                                     batch_norm=batch_norm, batch_renorm=batch_renorm, training=training,
                                     bn_first=bn_first, regularizer=regularizer)
        self.mono_l2 = DownConvLayer(kernel_size=5, ch_out=16, activation='relu', name='mono_l2',
                                     batch_norm=batch_norm, batch_renorm=batch_renorm,
                                     training=training, bn_first=bn_first, regularizer=regularizer)

    def call(self, inputs):
        mono_l1, to_skip_l0 = self.mono_l1(inputs)
        mono_l2, to_skip_l1 = self.mono_l2(mono_l1)
        return mono_l2, to_skip_l0, to_skip_l1


class CorrFeaturesLayer(KerasLikeLayer):
    def __init__(self, batch_norm=False, batch_renorm=False, training=False, bn_first=True,
                 activation_and_bn_output=False, name='corr_features', regularizer=None, **kwargs):
        super(CorrFeaturesLayer, self).__init__(name=name, **kwargs)

        self.corr_dcl1 = DownConvLayer(kernel_size=7, ch_out=8, activation='relu', name='corr_l1',
            batch_norm=batch_norm, batch_renorm=batch_renorm, training=training, bn_first=bn_first, regularizer=regularizer)
        self.corr_dcl2 = DownConvLayer(kernel_size=5, ch_out=16, activation='relu', name='corr_l2',
            batch_norm=batch_norm, batch_renorm=batch_renorm, training=training, bn_first=bn_first, regularizer=regularizer)
        self.corr_dcl3 = DownConvLayer(kernel_size=3, ch_out=32, activation='relu', name='corr_l3',
            batch_norm=batch_norm, batch_renorm=batch_renorm, training=training, bn_first=bn_first, regularizer=regularizer)
        self.corr_ucl = UpConvLayer(ch_out=64, activation='relu', name='corr_l2_up',
            batch_norm=batch_norm, batch_renorm=batch_renorm, training=training, bn_first=bn_first, regularizer=regularizer,
            activation_and_bn_output=activation_and_bn_output)

    def call(self, inputs):
        corr_l1, _ = self.corr_dcl1(inputs)
        corr_l2, _ = self.corr_dcl2(corr_l1)
        corr_l3, to_skip_l2 = self.corr_dcl3(corr_l2)
        corr_l2_up = self.corr_ucl([corr_l3, to_skip_l2])
        return corr_l2_up


class AdversarialEncoder(KerasLikeLayer):

    def __init__(self, name='adversarial_encoder', **kwargs):
        super(AdversarialEncoder, self).__init__(name=name, **kwargs)
        self.adv_dn_1 = DownConvLayer(kernel_size=3, ch_out=16, name='adv_dn_l1')
        self.adv_dn_2 = DownConvLayer(kernel_size=3, ch_out=16, name='adv_dn_l2')
        self.adv_dn_3 = DownConvLayer(kernel_size=3, ch_out=32, name='adv_dn_l3')
        self.adv_dn_4 = DownConvLayer(kernel_size=3, ch_out=64, name='adv_dn_l4')
        self.adv_dn_5 = DownConvLayer(kernel_size=3, ch_out=64, name='adv_dn_l5')
        self.adv_dn_6 = DownConvLayer(kernel_size=3, ch_out=64, name='adv_dn_l6')
        self.adv_dn_7 = DownConvLayer(kernel_size=3, ch_out=64, name='adv_dn_l7')
        self.adv_pre = layers.Dense(128, activation='relu', name='adv_dense_1')
        self.adv = layers.Dense(1, name='adv_dense_2')

    def call(self, inputs):
        concat_l2, unet_dn_l6, to_skip_l5, to_skip_l4, to_skip_l3, to_skip_l2, to_skip_l1, to_skip_l0 = inputs
        adv_dn_1, _ = self.adv_dn_1(to_skip_l0)
        adv_dn_2, _ = self.adv_dn_2(layers.concatenate([adv_dn_1, to_skip_l1], axis=3))
        adv_dn_3, _ = self.adv_dn_3(layers.concatenate([adv_dn_2, concat_l2, to_skip_l2], axis=3))
        adv_dn_4, _ = self.adv_dn_4(layers.concatenate([adv_dn_3, to_skip_l3], axis=3))
        adv_dn_5, _ = self.adv_dn_5(layers.concatenate([adv_dn_4, to_skip_l4], axis=3))
        adv_dn_6, _ = self.adv_dn_6(layers.concatenate([adv_dn_5, to_skip_l5], axis=3))
        adv_dn_7, _ = self.adv_dn_7(layers.concatenate([adv_dn_6, unet_dn_l6], axis=3))
        adv_dn_7_len = np.prod(adv_dn_7.shape.as_list()[1:])
        adv_pre = self.adv_pre(tf.reshape(adv_dn_7, [-1, adv_dn_7_len]))
        adv = self.adv(adv_pre)
        return adv
