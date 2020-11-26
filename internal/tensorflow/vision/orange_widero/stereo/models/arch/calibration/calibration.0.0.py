import numpy as np
import tensorflow as tf
import imp
from os.path import join
from stereo.models.arch_utils import corr_steps
from stereo.common.general_utils import tree_base, get_args_names
from stereo.models.calibration_utils import ResidualRotationWarpLayer

arch_format = ("arch", "anonymous")

def arch(I_cntr, I_srnd_0, I_srnd_1, I_srnd_2, T_cntr_srnd, focal, origin,
         num_steps_corr=None, min_Z_corr=None, max_Z_corr=None, min_delta_Z_corr=None,
         min_Z=None, max_Z=None, batch_norm=False, isTraining=False, bn_first=True, regularization=None,
         q_stddev=0.0, arch_base_name=None,
         R0_trainable=True, R1_trainable=True, R2_trainable=True, mask_untrainable=False):

    I_srnd_0 = ResidualRotationWarpLayer(name='residual_rotation_srnd_0',
                                         stddev=q_stddev, trainable=R0_trainable)([I_srnd_0, focal])
    I_srnd_1 = ResidualRotationWarpLayer(name='residual_rotation_srnd_1',
                                         stddev=q_stddev, trainable=R1_trainable)([I_srnd_1, focal])
    I_srnd_2 = ResidualRotationWarpLayer(name='residual_rotation_srnd_2',
                                         stddev=q_stddev, trainable=R2_trainable)([I_srnd_2, focal])

    if mask_untrainable:
        I_srnd_0 = I_srnd_0 if R0_trainable else tf.zeros_like(I_srnd_0)
        I_srnd_1 = I_srnd_1 if R1_trainable else tf.zeros_like(I_srnd_1)
        I_srnd_2 = I_srnd_2 if R2_trainable else tf.zeros_like(I_srnd_2)

    im_sz_cntr = I_cntr.shape.as_list()[1:3]
    im_sz_srnd = I_srnd_0.shape.as_list()[1:3]

    margin_x = (im_sz_srnd[1] - im_sz_cntr[1])/2
    margin_y = (im_sz_srnd[0] - im_sz_cntr[0])/2

    if margin_x > 0 and margin_y > 0:
        I_srnd_0 = I_srnd_0[:, margin_y:-margin_y, margin_x:-margin_x, :]
        I_srnd_1 = I_srnd_1[:, margin_y:-margin_y, margin_x:-margin_x, :]
        I_srnd_2 = I_srnd_2[:, margin_y:-margin_y, margin_x:-margin_x, :]

    steps = corr_steps(num_steps_corr, min_Z_corr, max_Z_corr, min_delta_Z_corr).astype(np.float32)
    arch_base = imp.load_source('arch',
                                join(tree_base(), 'stereo/models/arch/stereo', arch_base_name + '.py')).arch._original
    args_names = get_args_names(arch_base)
    kargs = {}
    for arg in args_names:
        kargs[arg] = locals()[arg]

    _ = tf.identity(I_srnd_0, name='I_srnd_0')
    _ = tf.identity(I_srnd_1, name='I_srnd_1')
    _ = tf.identity(I_srnd_2, name='I_srnd_2')

    return {'out': arch_base(**kargs), 'I_srnd_0': I_srnd_0, 'I_srnd_1': I_srnd_1, 'I_srnd_2': I_srnd_2}
