from stereo.data.map_func.map_func_utils import inp_eval


def map_func(features, pred_mode=False):
    features = inp_eval(features, pred_mode=pred_mode)
    return features