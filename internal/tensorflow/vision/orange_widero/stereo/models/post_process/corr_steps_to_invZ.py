from stereo.models.arch_utils import corr_steps
import tensorflow as tf
import numpy as np

def post_process(out, num_steps_corr, min_Z_corr, max_Z_corr, min_delta_Z_corr):
    steps = corr_steps(num_steps_corr, min_Z_corr, max_Z_corr, min_delta_Z_corr)
    dyn_input_shape = tf.shape(input=out)
    batch = dyn_input_shape[0]
    steps_t = tf.tile(tf.constant(np.r_[steps, 1e-5].reshape(1,1,-1,1).astype(np.float32)), [batch, 1, 1, 1])
    ind = tf.concat([out, tf.zeros_like(out)], axis=3)
    return tf.contrib.resampler.resampler(steps_t, ind)
