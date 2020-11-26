from stereo.data.map_func.np_tf_imports import np_tf_func
from stereo.data.map_func.map_func_utils import xy, blur_kernels


def map_func(features, consts, blur_kernels_names, pred_mode=False):
    features = xy(features, im_name='I_cntr', pred_mode=pred_mode)
    features = blur_kernels(features, consts, blur_kernels_names)
    ones_like = np_tf_func('ones_like', pred_mode)
    features['deeplab'] = ones_like(features['I_cntr'], dtype='uint8') * 255
    return features
