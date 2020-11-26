import tensorflow as tf
from stereo.models.calibration_utils import ResidualRotationWarpLayer

arch_format = ("arch", "anonymous")


def arch(I_srnd_0, I_srnd_1, I_srnd_2, focal, origin, q_stddev=0.0):

    I_srnd_0 = ResidualRotationWarpLayer(name='residual_rotation_srnd_0', stddev=q_stddev)([I_srnd_0, focal, origin])
    I_srnd_1 = ResidualRotationWarpLayer(name='residual_rotation_srnd_1', stddev=q_stddev)([I_srnd_1, focal, origin])
    I_srnd_2 = ResidualRotationWarpLayer(name='residual_rotation_srnd_2', stddev=q_stddev)([I_srnd_2, focal, origin])

    _ = tf.identity(I_srnd_0, name='I_srnd_0')
    _ = tf.identity(I_srnd_1, name='I_srnd_1')
    _ = tf.identity(I_srnd_2, name='I_srnd_2')

    return {'out': I_srnd_0, 'I_srnd_0': I_srnd_0, 'I_srnd_1': I_srnd_1, 'I_srnd_2': I_srnd_2}
