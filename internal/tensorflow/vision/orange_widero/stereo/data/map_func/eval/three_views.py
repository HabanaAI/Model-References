from stereo.data.map_func.map_func_utils import views_eval


def map_func(features, pred_mode=False):
    features = views_eval(features, pred_mode=pred_mode)
    return features