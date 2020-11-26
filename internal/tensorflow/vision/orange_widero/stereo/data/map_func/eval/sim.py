from stereo.data.map_func.np_tf_imports import np_tf_func
from stereo.data.map_func.map_func_utils import xy

def map_func(features, pred_mode=False):
    if 'ground_truth' in features.keys():
        return features

    features = xy(features)
    features['center_im'] = np_tf_func('expand_dims', pred_mode)(features['inp'][0, :, :], -1)
    features['inp_ims'] = np_tf_func('transpose', pred_mode)(features['inp'][1:, :, :], [1, 2, 0])
    features['ground_truth'] = np_tf_func('expand_dims', pred_mode)(features['sim_depth'], -1)

    return features