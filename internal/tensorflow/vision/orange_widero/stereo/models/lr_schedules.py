"""
This file contains learning rate schedules for TF Estimator training.
For keras of other tensorflow training, prefer the learning rate classes of tf.keras.optimizers.schedules.
"""

import numpy as np
import tensorflow as tf
from stereo.data.map_func.np_tf_imports import np_tf_func


def exponential_decay(step, learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True):
    use_numpy = not tf.is_tensor(step)

    step = np_tf_func('to_float', pred_mode=use_numpy)(step)
    decay_steps = np_tf_func('to_float', pred_mode=use_numpy)(decay_steps)

    p = step / decay_steps
    if staircase:
        p = np_tf_func('floor', pred_mode=use_numpy)(p)
    lr = learning_rate * (decay_rate ** p)
    return lr


def cosine_decay_restarts(step, learning_rate, first_decay_steps=100000, t_mul=2.0, m_mul=1.0, alpha=0.0):
    use_numpy = not tf.is_tensor(step)

    step = np_tf_func('to_float', pred_mode=use_numpy)(step)
    first_decay_steps = np_tf_func('to_float', pred_mode=use_numpy)(first_decay_steps)
    completed_fraction = step / first_decay_steps

    def compute_step(completed_fraction, geometric=False):
        if geometric:
            i_restart = np_tf_func('floor', pred_mode=use_numpy)(
                np_tf_func('log', pred_mode=use_numpy)(1.0 - completed_fraction * (1.0 - t_mul)) /
                np_tf_func('log', pred_mode=use_numpy)(t_mul))

            sum_r = (1.0 - t_mul ** i_restart) / (1.0 - t_mul)
            completed_fraction = (completed_fraction - sum_r) / t_mul ** i_restart

        else:
            i_restart = np_tf_func('floor', pred_mode=use_numpy)(completed_fraction)
            completed_fraction -= i_restart

        return i_restart, completed_fraction

    i_restart, completed_fraction = \
        compute_step(completed_fraction, geometric=False) if t_mul == 1.0 else \
            compute_step(completed_fraction, geometric=True)

    m_fac = m_mul ** i_restart
    if use_numpy:
        cosine_decayed = 0.5 * m_fac * (1.0 + np.cos(np.pi * completed_fraction))
    else:
        cosine_decayed = 0.5 * m_fac * (1.0 + tf.math.cos(np.pi * completed_fraction))
    decayed = (1 - alpha) * cosine_decayed + alpha

    lr = learning_rate * decayed
    return lr


# def cosine_decay_with_warm_up(learning_rate, step, total_steps=1800000, warmup_steps=10000,
#                               hold_base_rate_steps=2.0, alpha=0.0):
#     """
#     Based on https://medium.com/@scorrea92/cosine-learning-rate-decay-e8b50aa455b
#     """
#     if total_steps < warmup_steps:
#         raise ValueError('total_steps must be larger or equal to warmup_steps.')
#     learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
#             np.pi * (global_step - warmup_steps - hold_base_rate_steps) /
#             float(total_steps - warmup_steps - hold_base_rate_steps))
#         )
#     if hold_base_rate_steps > 0:
#         learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
#                                  learning_rate, learning_rate_base)
#     if warmup_steps > 0:
#         if learning_rate_base < warmup_learning_rate:
#             raise ValueError('learning_rate_base must be larger or equal to '
#                              'warmup_learning_rate.')
#         slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
#         warmup_rate = slope * global_step + warmup_learning_rate
#         learning_rate = np.where(global_step < warmup_steps, warmup_rate,
#                                  learning_rate)
#     return learning_rate


# def exponential_decay(step, learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True):
#     return tf.compat.v1.train.exponential_decay(learning_rate=learning_rate, global_step=step,
#                                                 decay_steps=decay_steps,
#                                                 decay_rate=decay_rate, staircase=staircase)
#
#
# def cosine_decay_restarts(learning_rate, step, first_decay_steps=100000, t_mul=2.0, m_mul=1.0, alpha=0.0):
#     return tf.compat.v1.train.cosine_decay_restarts(learning_rate=learning_rate, global_step=step,
#                                                     first_decay_steps=first_decay_steps, t_mul=t_mul, m_mul=m_mul,
#                                                     alpha=alpha)
