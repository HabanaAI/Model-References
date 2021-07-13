from TensorFlow.common.horovod_helpers import hvd, horovod_enabled

import glob
import os
from multiprocessing import Pool, cpu_count

import numpy as np
import tensorflow.compat.v1 as tf

from .estimator import HorovodEstimator
from .utis import hvd_info, hvd_info_rank0, hvd_try_init


def _count_per_file(filename):
    c = 0
    for _ in tf.python_io.tf_record_iterator(filename):
        c += 1

    return c


def get_record_num(filenames):
    pool = Pool(cpu_count())
    c_list = pool.map(_count_per_file, filenames)
    total_count = np.sum(np.array(c_list))
    return total_count


def get_filenames(data_dir: str, filename_regexp: str, show_result=True):
    filenames = glob.glob(os.path.join(data_dir, filename_regexp))
    if show_result:
        hvd_info_rank0('find {} files in {}, such as {}'.format(len(filenames), data_dir, filenames[0:5]))
    return filenames


def _idx_a_minus_b(a, b):
    a_splits = a.split('/')
    b_splits = b.split('/')
    for i in range(min(len(a_splits), len(b_splits))):
        if a_splits[i] != b_splits[i]:
            break

    return len('/'.join(a_splits[0:i]))


def show_model():
    prev = None
    for var in tf.global_variables():
        # if var.name.split('/')[-1] in ['beta:0', 'moving_mean:0', 'moving_variance:0']:
        #     continue

        if prev is None:
            print('{} - {}'.format(var.name, var.shape.as_list()))
        else:
            idx = _idx_a_minus_b(var.name, prev.name)
            short_name = var.name[idx:]
            if short_name.startswith('/bn'):
                short_name = '/bn'
            print('{}{} - {}'.format(' ' * idx, short_name, var.shape.as_list()))
        prev = var
