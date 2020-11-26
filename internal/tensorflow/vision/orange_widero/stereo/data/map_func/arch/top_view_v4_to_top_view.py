from stereo.data.map_func.map_func_utils import sample_id


def map_func(features, pred_mode=False):
    features = sample_id(features, pred_mode=pred_mode)
    return features
