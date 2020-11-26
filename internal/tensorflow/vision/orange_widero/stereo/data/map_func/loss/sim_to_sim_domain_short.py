from stereo.data.map_func.map_func_utils import sim_depth_inv
from stereo.data.map_func.np_tf_imports import np_tf_func


def map_func(features, domain, consts=None, pred_mode=False):
    features = sim_depth_inv(features, pred_mode=pred_mode)
    features['domain'] = np_tf_func('scalar', pred_mode)(domain)
    features['im_lidar_short_inv'] = np_tf_func('zeros_like', pred_mode)(features['sim_depth_inv'])
    return features