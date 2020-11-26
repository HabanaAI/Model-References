from stereo.models.layers.identity_layer import Identity
from stereo.models.layers.translate_from_calibration import TranslateFromCalibration
from stereo.models.layers.correlation_layer import Correlation as CorrelationDLO

from stereo.models.layers.keras_layers import KerasLikeLayer


class Correlation(KerasLikeLayer):
    def __init__(self, steps, name="corr_map", **kwargs):
        super(Correlation, self).__init__(name=name, **kwargs)
        self.invZ_steps = steps.astype('float32')

    def call(self, inputs):
        cntr = inputs[0]
        srnd = inputs[1]
        origin = inputs[2]
        focal = inputs[3]
        T12 = inputs[4]

        transform_params = TranslateFromCalibration(steps=self.invZ_steps)([origin, focal, T12])
        scale = Identity(name=self.name+'_scale')(transform_params[0])
        trans_x = Identity(name=self.name+'_trans_x')(transform_params[1])
        trans_y = Identity(name=self.name+'_trans_y')(transform_params[2])
        corrs_concat = CorrelationDLO()([cntr, srnd, scale, trans_x, trans_y])
        return corrs_concat


def correlation_map(cntr, srnd, origin, focal, T12, steps, name="corr_map"):
    return Correlation(steps=steps, name=name)([cntr, srnd, origin, focal, T12])
