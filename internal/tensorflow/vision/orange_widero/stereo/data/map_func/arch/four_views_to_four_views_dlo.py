from stereo.data.map_func.map_func_utils import sample_id
from stereo.data.map_func.np_tf_imports import np_tf_func
from stereo.models.arch_utils import im_padding


def map_func(features, pred_mode=False):

    features = sample_id(features, pred_mode=pred_mode)
    # This is to avoid addition and division with a const at architecture itself
    UNET_LEVELS = 6
    corr_down_levels = 2
    origin = features.get('origin', None)

    if origin is not None:
        padding = im_padding(features['I_cntr'], UNET_LEVELS)
        features['origin_l2'] = (origin + [padding[1][0], padding[0][0]]) / (2**corr_down_levels)
        features['origin_l2'] = np_tf_func('to_float', pred_mode)(features['origin_l2'])

    focal = features.get('focal', None)
    if focal is not None:
        features['focal_l2'] = focal / (2 ** corr_down_levels)
        features['focal_l2'] = np_tf_func('to_float', pred_mode)(features['focal_l2'])

    return features
