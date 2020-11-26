from stereo.data.map_func.map_func_utils import inp_to_views
from stereo.data.map_func.np_tf_imports import np_tf_func

def map_func(features, num_steps_corr, min_Z_corr, max_Z_corr, min_delta_Z_corr, domain, poly_deg=7, pred_mode=False):
    features = inp_to_views(features, num_steps_corr, min_Z_corr, max_Z_corr, min_delta_Z_corr, poly_deg,
                           four_views=True, add_fake=False, pred_mode=pred_mode)
    features['domain'] = np_tf_func('scalar', pred_mode)(domain)
    return features
