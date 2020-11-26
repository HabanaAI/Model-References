from stereo.data.map_func.map_func_utils import xy, blur_kernels, split_inp, im_lidar_inv


def map_func(features, consts, blur_kernels_names, pred_mode=False):
    features = xy(features, pred_mode=pred_mode)
    features = split_inp(features, four_views=False, add_fake=False, pred_mode=pred_mode)
    features = blur_kernels(features, consts, blur_kernels_names)
    features = im_lidar_inv(features, pred_mode=pred_mode)

    return features