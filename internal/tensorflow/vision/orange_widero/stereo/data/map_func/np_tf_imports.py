from functools import partial
import numpy as np
import tensorflow as tf


def np_tf_func(name, pred_mode=False):
    if name == 'expand_dims':
        if pred_mode:
            return np.expand_dims
        else:
            return tf.expand_dims
    if name == 'to_float':
        if pred_mode:
            return np.cast[np.float32]
        else:
            return partial(tf.cast, dtype=tf.float32)
    if name == 'equal':
        if pred_mode:
            return np.equal
        else:
            return tf.equal
    if name == 'as_string':
        if pred_mode:
            return str
        else:
            return tf.as_string
    if name == 'len':
        if pred_mode:
            return lambda a: a.shape[0]
        else:
            return lambda a: a.shape.as_list()[0]
    if name == 'concat':
        if pred_mode:
            return np.concatenate
        else:
            return tf.concat
    if name == 'zeros_like':
        if pred_mode:
            return np.zeros_like
        else:
            return tf.zeros_like
    if name == 'ones_like':
        if pred_mode:
            return np.ones_like
        else:
            return tf.ones_like
    if name == 'im_sz':
        if pred_mode:
            return lambda a: a.shape[1:]
        else:
            return lambda a: a.shape.as_list()[1:]
    if name == 'im_sz_HWC':
        if pred_mode:
            return lambda a: a.shape[:2]
        else:
            return lambda a: a.shape.as_list()[:2]
    if name == 'range':
        if pred_mode:
            return partial(np.arange, dtype='float32')
        else:
            return partial(tf.range, dtype=tf.float32)
    if name == 'meshgrid':
        if pred_mode:
            return np.meshgrid
        else:
            return tf.meshgrid
    if name == 'transpose':
        if pred_mode:
            return np.transpose
        else:
            return tf.transpose
    if name == 'scalar':
        if pred_mode:
            return lambda x: np.array([x], dtype=np.float32)
        else:
            return lambda x: tf.constant(np.array([x], dtype=np.float32))
    if name == 'floor':
        if pred_mode:
            return np.floor
        else:
            return tf.math.floor
    if name == 'log':
        if pred_mode:
            return np.log
        else:
            return tf.math.log
    if name == 'pad':
        if pred_mode:
            return np.pad
        else:
            return tf.pad
