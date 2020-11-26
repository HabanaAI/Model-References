import os
import inspect
import numpy as np


class Struct:
    def __init__(self, dictionary):
        self.__dict__.update(**dictionary)


def tree_base():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def shared_env_python():
    shared_env = "/mobileye/algo_STEREO3/stereo/venv/latest"
    return os.path.join(shared_env, "bin", "python")


def get_args_names(func):
    args, _, _, _ = inspect.getargspec(func)
    return args


def get_non_default_args(func):
    args, _, _, defaults = inspect.getargspec(func)
    if defaults:
        non_default_args = args[:-len(defaults)]
        return non_default_args
    return args


def makedir_recursive(*args):
    path = os.path.join(*args)
    if not os.path.exists(path):
        os.makedirs(path)


def batch_list(lst, n=1):
    batched_lst = [lst[i:i + n] for i in range(0, len(lst), n)]
    return batched_lst


def flatten_list(lst):
    flat_lst = [item for sublst in lst for item in sublst]
    return flat_lst


def flatten_mepjs_out(out_batched, inp_batched):
    out = []
    for out_, inp_ in zip(out_batched, inp_batched):
        out.extend(out_ if out_ else [None]*len(inp_))
    return out


def crop_symmetric(img, to_shape):
    from_shape = img.shape
    diff_shape = np.array(to_shape) - np.array(from_shape)
    assert np.all(np.mod(diff_shape, 2) == 0)
    half_diff_shape = diff_shape // 2
    half_diff_shape = np.minimum(half_diff_shape, 0)
    return img[-half_diff_shape[0]:from_shape[0] + half_diff_shape[0],
               -half_diff_shape[1]:from_shape[1] + half_diff_shape[1]]


def crop_pad_symmetric(img, to_shape):
    from_shape = img.shape
    diff_shape = np.array(to_shape) - np.array(from_shape)
    assert np.all(np.mod(diff_shape, 2) == 0)
    half_diff_shape = diff_shape // 2
    half_diff_shape = np.maximum(half_diff_shape, 0)
    img_cropped = crop_symmetric(img, to_shape)
    pad_width = ((half_diff_shape[0], half_diff_shape[0]), (half_diff_shape[1], half_diff_shape[1]))
    return np.pad(img_cropped, pad_width, mode='constant')