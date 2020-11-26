from stereo.data.map_func.map_func_utils import xy


def map_func(features, consts=None, pred_mode=False):
    features = xy(features, pred_mode=pred_mode)
    return features