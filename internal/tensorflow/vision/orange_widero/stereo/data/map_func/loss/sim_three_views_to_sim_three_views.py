from stereo.data.map_func.map_func_utils import sim_depth_inv


def map_func(features, consts=None, pred_mode=False):
    features = sim_depth_inv(features, pred_mode=pred_mode)
    return features