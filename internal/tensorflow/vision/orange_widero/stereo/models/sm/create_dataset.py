"""
This code is based on: mepy_algo/appcode/DL/freespace/training/create_dataset.py
"""

import os
from functools import partial
import tensorflow as tf

from stereo.common.s3_utils import my_glob
from stereo.models.map_func_loaders import load_map_funcs
from stereo.interfaces.implements import load_dataset_attributes
from stereo.models.model_utils import cleanup_features, required_features
from stereo.common.tfrecord_utils import parse_example


def split_features_labels(all_features):
    features = {}
    labels = {}
    for k in all_features:
        features[k] = all_features[k]  # TODO: decide on feature vs label
    return features, labels


def ds_spec_from_ds(ds):
    import tensorflow as tf
    ds_shapes = ds.output_shapes
    ds_dtypes = ds.output_types
    if isinstance(ds_shapes, tuple):
        ds_shapes = ds_shapes[0]
        ds_dtypes = ds_dtypes[0]
    # in TF2:
    # return ds.element_spec
    TensorSpec = tf.TensorSpec if hasattr(tf, "TensorSpec") else tf.contrib.framework.TensorSpec
    ds_spec = {k: TensorSpec(shape=ds_shapes[k][1:],
                             dtype=ds_dtypes[k],
                             name=k)
               for k in ds_shapes.keys()}

    for k, v in ds_spec.items():
        print("{:20}\tshape={:17}\tdtype={}".format(v.name, str(v.shape), v.dtype))
    return ds_spec


def create_dataset(conf, section, channel, pipe_mode=False, shuf=False, npc=-1,
                   eval=False, augm=False, alt_dataset=None, run_in_menta=False):

    """ create dataset either from an existing pipe channel (constructed by train_sm.py), in pipe_mode==True, or from
    the s3 path dataset_path + channel (where channel is either 'train' or 'test'), containing tf_record files. """

    if alt_dataset is not None:
        conf['data_params']['datasets'][section] = alt_dataset
    arch_map_func, loss_map_func, eval_map_func, augm_map_func = \
        load_map_funcs(conf, section, load_eval=eval, load_augm=augm)

    dataset_attributes, ds_format = load_dataset_attributes(conf['data_params']['datasets'][section])

    if npc < 0:
        if int(tf.version.VERSION.split('.')[1]) > 12:
            npc = tf.data.experimental.AUTOTUNE
        else:
            npc = 10

    if pipe_mode:
        from sagemaker_tensorflow import PipeModeDataset
        ds = PipeModeDataset(channel=channel, record_format='TFRecord')
    else:
        tfrecord_list = my_glob(os.path.join(dataset_attributes['s3'], channel, '*.tfrecord'))
        if shuf:
            from random import shuffle
            shuffle(tfrecord_list)
        print("Number of TFRecord files: {}".format(len(tfrecord_list)))
        comp_type = "GZIP" if dataset_attributes['compressed'] else None
        ds = tf.data.TFRecordDataset(tfrecord_list, compression_type=comp_type)

    ds = ds.map(partial(parse_example, ds_format=ds_format), num_parallel_calls=npc)

    if arch_map_func is not None:
        ds = ds.map(arch_map_func, num_parallel_calls=npc)
    if loss_map_func is not None:
        ds = ds.map(loss_map_func, num_parallel_calls=npc)
    if augm_map_func is not None:
        ds = ds.map(augm_map_func, num_parallel_calls=npc)

    if eval:
        if eval_map_func is not None:
            ds = ds.map(eval_map_func, num_parallel_calls=npc)

    req_features = required_features(conf['model_params'], eval=eval)
    ds = ds.map(partial(cleanup_features, req_features=req_features), num_parallel_calls=npc)

    if run_in_menta:
        from stereo.data.map_func.map_func_utils import menta_map_func
        ds = ds.map(menta_map_func, num_parallel_calls=npc)
    else:
        ds = ds.map(partial(split_features_labels), num_parallel_calls=npc)

    return ds
