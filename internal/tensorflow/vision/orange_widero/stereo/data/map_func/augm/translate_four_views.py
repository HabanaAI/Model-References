import tensorflow as tf
from tensorflow.contrib.image import translate
from stereo.data.map_func.map_func_utils import views_eval
from stereo.data.map_func.map_func_utils import xy


def map_func(features, pred_mode=False, trans_bounds_xy=[[-20, 20], [-40, 40]]):

    if pred_mode:
        return features

    tx = tf.random.uniform((), minval=trans_bounds_xy[0][0], maxval=trans_bounds_xy[0][1], dtype=tf.int32)
    ty = tf.random.uniform((), minval=trans_bounds_xy[1][0], maxval=trans_bounds_xy[1][1], dtype=tf.int32)
    translation = tf.cast(tf.stack([tx, ty]), dtype=tf.float32)
    features['I_cntr'] = translate(features['I_cntr'], translation)
    features['I_srnd_0'] = translate(features['I_srnd_0'], translation)
    features['I_srnd_1'] = translate(features['I_srnd_1'], translation)
    features['I_srnd_2'] = translate(features['I_srnd_2'], translation)
    features['im_lidar_inv'] = translate(features['im_lidar_inv'], translation)
    features['im_lidar_short_inv'] = translate(features['im_lidar_short_inv'], translation)
    features['im_mask'] = 1.0 - translate(1.0 - features['im_mask'], translation)
    features['photo_loss_im_mask'] = 1.0 - translate(1.0 - features['photo_loss_im_mask'], translation)

    features['origin'] = features['origin'] + translation
    features.pop('x', None)
    features.pop('y', None)
    features = xy(features, im_name='I_cntr', pred_mode=pred_mode)

    return features
