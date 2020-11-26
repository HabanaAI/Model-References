from copy import copy
from stereo.data.map_func.map_func_utils import steps, sample_id


def map_func(features, num_steps_corr, min_Z_corr, max_Z_corr, min_delta_Z_corr, poly_deg=7, pred_mode=False):
    features = steps(features, num_steps_corr, min_Z_corr, max_Z_corr, min_delta_Z_corr, poly_deg, pred_mode=pred_mode)
    features = sample_id(features, pred_mode=pred_mode)
    return features
