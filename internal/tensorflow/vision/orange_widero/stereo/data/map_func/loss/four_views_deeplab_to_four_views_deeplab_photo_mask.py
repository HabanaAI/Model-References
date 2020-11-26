from stereo.data.map_func.map_func_utils import xy, blur_kernels


def map_func(features, consts, blur_kernels_names, pred_mode=False):
    features = xy(features, im_name='I_cntr', pred_mode=pred_mode)
    features = blur_kernels(features, consts, blur_kernels_names)
    return features
