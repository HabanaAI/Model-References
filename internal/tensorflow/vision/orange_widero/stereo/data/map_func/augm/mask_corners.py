import tensorflow as tf
from stereo.data.map_func.map_func_utils import xy


def mask_corner(image, meshgrid_x, margin, side):
    if side == 'left':
        mask = meshgrid_x < margin
    else:
        assert side == 'right'
        mask = meshgrid_x > margin

    return tf.cast(mask, tf.float32) * image

def map_func(features, pred_mode=False, margin_range=5):
    if pred_mode:
        return features
    features = xy(features, im_name='I_cntr', pred_mode=pred_mode)

    margin_left = -tf.random.uniform(shape=(), minval=0, maxval=margin_range)
    margin_right = tf.random.uniform(shape=(), minval=0, maxval=margin_range)

    features['I_srnd_0'] = mask_corner(features['I_srnd_0'], features['x'], margin_left, 'left')
    features['I_srnd_1'] = mask_corner(features['I_srnd_1'], features['x'], margin_right, 'right')

    return features
