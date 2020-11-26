from stereo.data.map_func.map_func_utils import xy, blur_kernels, split_inp


def map_func(features, consts, blur_kernels_names, pred_mode=False):
    features = xy(features, pred_mode=pred_mode)
    features = split_inp(features, four_views=True, add_fake=False, pred_mode=pred_mode)
    features = blur_kernels(features, consts, blur_kernels_names, add_fake=False)

    return features