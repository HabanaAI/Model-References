from stereo.models.layers.keras_layers import KerasLikeLayer
from stereo.models.layers.unet_layers_v2 import DownConvLayer, UpConvLayer


class MonoFeaturesLayer(KerasLikeLayer):
    def __init__(self, name='mono_features', regularizer=None, **kwargs):
        super(MonoFeaturesLayer, self).__init__(name=name, **kwargs)
        self.mono_l1 = DownConvLayer(ch_out=8, activation='relu', name='mono_l1', regularizer=regularizer)
        self.mono_l2 = DownConvLayer(ch_out=16, activation='relu', name='mono_l2', regularizer=regularizer)

    def call(self, inputs):
        mono_l1, to_skip_l0 = self.mono_l1(inputs)
        mono_l2, to_skip_l1 = self.mono_l2(mono_l1)
        return mono_l2, to_skip_l0, to_skip_l1


class CorrFeaturesLayer(KerasLikeLayer):
    def __init__(self, name='corr_features', regularizer=None, input_shape=None, **kwargs):
        super(CorrFeaturesLayer, self).__init__(name=name, **kwargs)

        self.corr_dcl1 = DownConvLayer(ch_out=8, activation='relu', name='corr_l1', regularizer=regularizer)
        self.corr_dcl2 = DownConvLayer(ch_out=16, activation='relu', name='corr_l2', regularizer=regularizer)
        self.corr_dcl3 = DownConvLayer(ch_out=24, activation='relu', name='corr_l3', regularizer=regularizer)
        self.corr_dcl4 = DownConvLayer(ch_out=32, activation='relu', name='corr_l4', regularizer=regularizer)
        self.corr_ucl1 = UpConvLayer(ch_out=48, activation='relu', name='corr_l3_up', regularizer=regularizer,
                                     input_shape=(input_shape[0], input_shape[1] // 16, input_shape[2] // 16, input_shape[3]))
        self.corr_ucl2 = UpConvLayer(ch_out=64, activation='relu', name='corr_l2_up', regularizer=regularizer,
                                     input_shape=(input_shape[0], input_shape[1] // 8, input_shape[2] // 8, input_shape[3]),
                                     activation_output=False)


    def call(self, inputs):
        corr_l1, _ = self.corr_dcl1(inputs)
        corr_l2, _ = self.corr_dcl2(corr_l1)
        corr_l3, to_skip_l2 = self.corr_dcl3(corr_l2)
        corr_l4, to_skip_l3 = self.corr_dcl4(corr_l3)
        corr_l3_up = self.corr_ucl1([corr_l4, to_skip_l3])
        corr_l2_up = self.corr_ucl2([corr_l3_up, to_skip_l2])
        return corr_l2_up
