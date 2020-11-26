from stereo.data.map_func.np_tf_imports import np_tf_func


def map_func(features, domain, pred_mode=False):
    if 'sim_depth_inv' not in features.keys():
        features['sim_depth_inv'] = features['im_lidar_inv']
    features['domain'] = np_tf_func('scalar', pred_mode)(domain)
    return features