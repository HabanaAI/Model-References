def map_func(features, pred_mode=False):
    if 'sim_depth_inv' not in features.keys():
        features['sim_depth_inv'] = features['im_lidar_inv']
    return features