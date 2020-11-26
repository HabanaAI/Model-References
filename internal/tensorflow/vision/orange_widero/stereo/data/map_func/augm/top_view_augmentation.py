from tensorflow.image import flip_left_right
from stereo.data.map_func.map_func_utils import rotate_image
import tensorflow as tf


def map_func(features, pred_mode=False, flip_rate=0.5, rotate=True, rotation_bounds=(-20, 20)):
    if pred_mode:
        return features
    do_flip = tf.random.uniform((), 0, 1) < flip_rate
    features['inp'] = tf.cond(pred=do_flip, true_fn=lambda: flip_left_right(features['inp']), false_fn=lambda: features['inp'])
    features['label'] = tf.cond(pred=do_flip, true_fn=lambda: flip_left_right(features['label']), false_fn=lambda: features['label'])
    if rotate:
        rotation_angle = tf.random.uniform((), *rotation_bounds)
        features['inp'] = rotate_image(features['inp'], rotation_angle, fill_value=-1)
        features['label'] = rotate_image(features['label'], rotation_angle)
    return features
