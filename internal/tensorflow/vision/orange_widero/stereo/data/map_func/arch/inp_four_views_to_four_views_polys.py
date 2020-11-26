from stereo.data.map_func.map_func_utils import inp_to_views

def map_func(features, num_steps_corr, min_Z_corr, max_Z_corr, min_delta_Z_corr, poly_deg=7, pred_mode=False):
    return inp_to_views(features, num_steps_corr, min_Z_corr, max_Z_corr, min_delta_Z_corr, poly_deg,
                        four_views=True, add_fake=False, pred_mode=pred_mode)
