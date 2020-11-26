import imp
from os.path import join
from stereo.common.general_utils import tree_base
import numpy as np

from stereo.interfaces.implements import get_arch_format, get_loss_format, load_dataset_attributes
from stereo.common.s3_utils import my_open


def load_map_funcs(conf, section, load_arch=True, load_loss=True, load_eval=True, load_augm=False, pred_mode=False):
    dataset_name = conf['data_params']['datasets'][section]
    dataset_attributes = load_dataset_attributes(dataset_name)[0]
    arch_map_func, loss_map_func, eval_map_func, augm_map_func = None, None, None, None
    if load_arch:
        arch_kwargs = conf['model_params']['arch'].get('map_func_kwargs', {section: None}).get(section, None)
        arch_map_func = load_arch_map_func(dataset_attributes['format_name'], get_arch_format(conf['model_params']),
                                           kwargs=arch_kwargs,
                                           pred_mode=pred_mode)
    if load_loss:
        loss_kwargs = conf['model_params']['loss'].get('map_func_kwargs', {section: None}).get(section, None)
        consts_path = conf['model_params']['loss'].get('map_func_kwargs', {'consts_path': None}).get('consts_path', None)
        loss_map_func = load_loss_map_func(dataset_attributes['format_name'], get_loss_format(conf['model_params']),
                                           consts_path=consts_path,
                                           kwargs=loss_kwargs,
                                           pred_mode=pred_mode)
    if load_eval:
        eval_map_func = load_eval_map_func(dataset_attributes['format_name'], pred_mode=pred_mode)

    if load_augm:
        augmentation_conf = conf['data_params']['augmentation'].get(section, {})
        augmentation_name = augmentation_conf.get('name', None)
        augm_kwargs = augmentation_conf.get('kwargs', {})
        augm_map_func = load_augm_map_func(augmentation_name,
                                           kwargs=augm_kwargs)

    return arch_map_func, loss_map_func, eval_map_func, augm_map_func


def load_arch_map_func(dataset_format_name, arch_format_name, kwargs=None, pred_mode=False):
    map_func_path = join(tree_base(), 'stereo', 'data', 'map_func', 'arch',
                         '%s_to_%s.py' % (dataset_format_name, arch_format_name))
    try:
        map_func = imp.load_source('map_func', map_func_path).map_func
    except IOError:
        print("Proper map_func not found, none loaded.")
        return None
    map_func_kwargs = {"pred_mode": pred_mode}
    if kwargs is not None:
        map_func_kwargs.update(kwargs)
    print("Loaded arch map_func %s" % map_func_path)
    return lambda features: map_func(features, **map_func_kwargs)


def load_eval_map_func(dataset_format_name, pred_mode=False):
    map_func_path = join(tree_base(), 'stereo', 'data', 'map_func', 'eval', '%s.py' % dataset_format_name)
    try:
        map_func = imp.load_source('map_func', map_func_path).map_func
    except IOError:
        print("Proper map_func not found, none loaded.")
        return None
    map_func_kwargs = {"pred_mode": pred_mode}
    print("Loaded eval map_func %s" % map_func_path)
    return lambda features: map_func(features, **map_func_kwargs)


def load_loss_map_func(dataset_format_name, loss_format_name, consts_path=None, kwargs=None, pred_mode=False):

    map_func_path = join(tree_base(), 'stereo', 'data', 'map_func', 'loss',
                         '%s_to_%s.py' % (dataset_format_name, loss_format_name))
    try:
        map_func = imp.load_source('map_func', map_func_path).map_func
    except IOError:
        print("Proper map_func not found, none loaded.")
        return None

    map_func_kwargs = {"pred_mode": pred_mode}
    if consts_path is not None:
        map_func_kwargs['consts'] = {}
        with my_open(consts_path) as f:
            consts_npz = np.load(f)
            for npz_key in consts_npz.keys():
                map_func_kwargs['consts'][npz_key] = 1. * consts_npz[npz_key]
    if kwargs is not None:
        map_func_kwargs.update(kwargs)

    print("Loaded loss map_func %s" % map_func_path)
    return lambda features: map_func(features, **map_func_kwargs)


def load_augm_map_func(augmentation_name, kwargs):
    if augmentation_name is None:
        return None
    map_func_path = join(tree_base(), 'stereo', 'data', 'map_func', 'augm', '%s.py' % augmentation_name)
    map_func = imp.load_source('map_func', map_func_path).map_func
    print("Loaded augm map_func %s" % map_func_path)
    return lambda features: map_func(features, **kwargs)