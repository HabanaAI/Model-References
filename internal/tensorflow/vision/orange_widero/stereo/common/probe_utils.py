import gc
import tensorflow as tf

from stereo.common.tfrecord_utils import read_tfrecord, write_tfrecord


def batch_func(func):
    def wrapper(args_lst, **kwargs):
        kwargs.pop('workerId', None)
        out = [None] * len(args_lst)
        for i, args in enumerate(args_lst):
            out[i] = func(args, **kwargs)
            gc.collect()

        return out
    return wrapper


@batch_func
def probe_tfrecord(tfrecord_path, probe_func=None):
    assert probe_func
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.compat.v1.enable_eager_execution(config=session_conf)

    print(tfrecord_path)
    ds_format_path = '/'.join(tfrecord_path.split('/')[:-2]) + '/ds_format.json'
    try:
        example_dicts = read_tfrecord(tfrecord_path, ds_format_path, verbose=True, parse=True, to_numpy=True)
    except:
        print('read_tfrecord() failed')
        return None
    probe_out = [probe_func(example) for example in example_dicts]
    print('Ran probe_func on {n} examples'.format(**{'n': len(example_dicts)}))
    return probe_out


def collect_and_write_tfrecord(tfrecord_to_write, ds_format=None, verbose=False):
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.compat.v1.enable_eager_execution(config=session_conf)

    tfr_path_inp = None
    examples_out = []

    for tfr_inds_inp in tfrecord_to_write['tfrecord_example_inds_inp']:
        if tfr_path_inp != tfr_inds_inp[1]:
            examples_inp = read_tfrecord(tfr_inds_inp[1], ds_format, verbose=verbose)
            tfr_path_inp = tfr_inds_inp[1]
        examples_out.append(examples_inp[tfr_inds_inp[0]])

    write_tfrecord(examples_out, tfrecord_to_write['tfrecord_path_out'], ds_format, verbose=verbose)