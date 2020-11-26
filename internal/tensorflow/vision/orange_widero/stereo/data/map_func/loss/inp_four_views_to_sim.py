from stereo.data.map_func.map_func_utils import sim_depth_inv


def map_func(features, pred_mode=False):
    return sim_depth_inv(features, pred_mode, depth_name='im_lidar')
