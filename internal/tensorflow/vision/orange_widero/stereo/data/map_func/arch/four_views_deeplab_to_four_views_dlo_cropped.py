from stereo.data.map_func.arch.four_views_to_four_views_dlo_cropped import map_func as map_func_


def map_func(features, pred_mode=False):
    return map_func_(features, pred_mode=pred_mode)
