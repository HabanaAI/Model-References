import os
import json
import imp

from stereo.common.general_utils import tree_base, get_non_default_args
from stereo.common.s3_utils import my_open


def load_format(fmt_type, fmt, model_params=None):
    if model_params is not None and fmt == 'anonymous':
        if fmt_type == 'arch':
            arch_module = imp.load_source(fmt_type, os.path.join(tree_base(), 'stereo', 'models', 'arch',
                                                                 model_params['arch']['name'] + '.py'))
            return get_non_default_args(arch_module.arch)
        else:
            assert fmt_type == 'loss'
            loss_module = imp.load_source('loss', os.path.join(tree_base(), 'stereo', 'models', 'loss',
                                                               model_params['loss']['name'] + '.py'))
            non_default_args = get_non_default_args(loss_module.loss)
            non_default_args.remove('out')
            return non_default_args
        
    if fmt_type == 'eval':
        return ['clip_name', 'gi', 'center_im', 'inp_ims', 'ground_truth', 'x']
    json_path = os.path.join(tree_base(), 'stereo', 'interfaces', fmt_type + '_formats.json')
    with open(json_path) as f:
        json_ = json.load(f)
    assert fmt in json_.keys()
    return json_[fmt]


def include_original(dec):
    def meta_decorator(f):
        decorated = dec(f)
        decorated._original = f
        return decorated
    return meta_decorator


def implements_format(fmt_type, fmt):

    @include_original
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        non_default_args = get_non_default_args(func)
        if fmt_type == 'loss':
            non_default_args.remove('out')
        assert sorted(load_format(fmt_type, fmt)) == sorted(non_default_args)

        return wrapper
    return decorator


def load_dataset_attributes(dataset):
    if isinstance(dataset, dict):
        dataset_attributes = {}
        dataset_attributes['format_name'] = dataset.get('format_name', 'anonymous')
        dataset_attributes['compressed'] = dataset.get('compressed', True)
        dataset_attributes['s3'] = dataset['s3']
        dataset_attributes['prep'] = dataset.get('prep', None)
        if 'views_names' in dataset.keys():
            dataset_attributes['views_names'] = dataset['views_names']
        with my_open(os.path.join(dataset_attributes['s3'], 'ds_format.json')) as f:
            ds_format = json.load(f)
        return  dataset_attributes, ds_format
    else:
        json_path = os.path.join(tree_base(), 'stereo', 'interfaces', 'dataset_attributes.json')
        with open(json_path) as f:
            json_ = json.load(f)
        assert dataset in json_.keys()
        return json_[dataset], load_format('dataset', json_[dataset]['format_name'])


def get_arch_format(params):
    arch_module = imp.load_source('arch', os.path.join(tree_base(), 'stereo', 'models', 'arch',
                                                   params['arch']['name'] + '.py'))
    try:
        return arch_module.arch_format[1]
    except:
        print ('anonymous arch format')
        return 'anonymous'


def get_loss_format(params):
    loss_module = imp.load_source('loss', os.path.join(tree_base(), 'stereo', 'models', 'loss',
                                                   params['loss']['name'] + '.py'))
    try:
        return loss_module.loss_format[1]
    except:
        print ('anonymous loss format')
        return 'anonymous'
