import tensorflow as tf


def map_func(features, pred_mode=False, noise_std=0.05, a_range=(0.8, 1.2), b_range=(-0.2, 0.2)):
    if pred_mode:
        return features
    for im_name in ['I_cntr', 'I_srnd_0', 'I_srnd_1', 'I_srnd_2']:
        noise_image = tf.random.normal(features[im_name].shape) * noise_std
        a = tf.random.uniform(shape=(), minval=a_range[0], maxval=a_range[1])
        b = tf.random.uniform(shape=(), minval=b_range[0], maxval=b_range[1])
        features[im_name] = tf.clip_by_value(a * (features[im_name] + noise_image) + b, 0, 1)
    return features
