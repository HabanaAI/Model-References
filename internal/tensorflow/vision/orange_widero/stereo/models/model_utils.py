import json
import tensorflow as tf
import numpy as np
import sys
from os.path import join, exists
from os import mkdir
import subprocess

from stereo.interfaces.implements import load_dataset_attributes, get_arch_format, get_loss_format, load_format
from stereo.models.map_func_loaders import load_arch_map_func, load_loss_map_func, load_map_funcs
from stereo.models.mvs_model import get_model
import os

from stereo.common.s3_utils import my_exists, my_open, my_listdir

MODELS_CACHE_DIR = '/mobileye/algo_STEREO3/stereo/models/cache/'


def cache_model(model_s3_path):
    model_name = model_s3_path.split('/')[-2]
    if not exists(join(MODELS_CACHE_DIR, model_name)):
        mkdir(join(MODELS_CACHE_DIR, model_name))
    model_iter = model_s3_path.split('/')[-1]
    if not exists(join(MODELS_CACHE_DIR, model_name, model_iter + '.meta')):
        command = ["aws", "s3", "cp",
                   '/'.join(model_s3_path.split('/')[:-1]),
                   join(MODELS_CACHE_DIR, model_name),
                   "--recursive",
                   "--exclude", "*",
                   "--include", model_iter + "*"
                   ]
        print("caching model at %s" % join(MODELS_CACHE_DIR, model_name, model_iter))
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        _ = p.communicate()


def get_cached_model(model_s3_path):
    model_name = model_s3_path.split('/')[-2]
    model_iter = model_s3_path.split('/')[-1]
    if not exists(join(MODELS_CACHE_DIR, model_name, model_iter + '.meta')):
        cache_model(model_s3_path)
    return join(MODELS_CACHE_DIR, model_name, model_iter)


def tf_type(x):
    str_type = str if sys.version_info.major >= 3 else basestring
    if isinstance(x, str_type):
        return tf.string
    if type(x) is np.ndarray:
        if x.dtype == np.int64:
            return tf.int64
        if x.dtype == np.int32:
            return tf.int32
        if x.dtype == np.uint8:
            return tf.uint8
        if x.dtype == np.float32:
            return tf.float32
        if x.dtype == np.float64:
            return tf.float64
    return None


def to_batch(placeholders_list):
    batch = {}
    for k in placeholders_list[0].keys():
        batch[k] = []
        for placeholders in placeholders_list:
            batch[k].append(np.expand_dims(placeholders[k],0))
        batch[k] = np.concatenate(batch[k], 0)
    return batch


def placeholder_init(np_features):
    """ convert numpy feature dict to TF placeholders """
    placeholders = {}
    with tf.compat.v1.name_scope('placeholders'):
        for key, v in np_features.items():
            s = list(v.shape[:])
            s.insert(0, None)
            placeholders[key] = tf.compat.v1.placeholder(tf_type(v), s, key)
    return placeholders


def get_model_graph_path(json_path, use_cache=False):
    with open(json_path, 'rb') as fp:
        conf = json.load(fp)
    model_name = os.path.splitext(os.path.split(json_path)[1])[0]
    model_s3 = '/'.join(conf['model_base_path'].split('/')[3:])
    checkpoint_dir = '/'.join([model_s3, model_name])
    model_graph_path = 's3://mobileye-habana/mobileye-team-stereo/' + os.path.join(checkpoint_dir, 'model_graph')
    if use_cache:
        model_name = model_graph_path.split('/')[-2]
        if not exists(join(MODELS_CACHE_DIR, model_name)):
            mkdir(join(MODELS_CACHE_DIR, model_name))
        cached_path = join(MODELS_CACHE_DIR, model_name, 'model_graph')
        if not exists(cached_path + '.meta'):
            command = ["aws", "s3", "cp",
                       model_graph_path + '.meta',
                       cached_path + '.meta'
                       ]
            print("caching model_graph meta at %s" % cached_path + '.meta')
            p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            _ = p.communicate()
        model_graph_path = cached_path
    return model_graph_path


def save_model_to_meta(sess, json_path=None, model_graph_path=None, override=False):  # TODO: fix caching
    assert json_path is not None or model_graph_path is not None
    if json_path is not None:
        model_graph_path = get_model_graph_path(json_path)
    if my_exists(model_graph_path + '.meta') and not override:
        print("%s.meta exist! use override flag if that what you meant.. Skipping" % model_graph_path)
    else:
        saver = tf.compat.v1.train.Saver()
        print("Saving meta to %s" % model_graph_path)
        saver.save(sess, model_graph_path, write_state=False)


def load_model_from_meta(json_path=None, model_graph_path=None):  # TODO: fix caching
    """ return out, placeholders by loading the graph from the .meta file and getting the tensors """
    from tensorflow.contrib import image  # Do not remove, needed for TF load for no good reason
    assert json_path is not None or model_graph_path is not None
    if json_path is not None:
        model_graph_path = get_model_graph_path(json_path)
    print("Restoring graph from meta at %s" % model_graph_path + '.meta')

    tf.compat.v1.train.import_meta_graph(model_graph_path + '.meta')

    out_tensors_names = [n.name for n in tf.compat.v1.get_default_graph().as_graph_def().node if n.name.startswith('arch/out')]
    #TODO fix duplications at source
    out_tensors_names = [x for x in out_tensors_names if x not in ['arch/out_0', 'arch/out_1']]
    out = {}
    # Patch for yolo_v0.3
    if "yolo_v0.3" in model_graph_path:
        out_tensors_names = ['arch/out_1']
    for out_tensor_name in out_tensors_names:
        out[out_tensor_name.split('/')[-1]] = tf.compat.v1.get_default_graph().get_tensor_by_name(out_tensor_name+':0')

    feature_names = [v.name for v in tf.compat.v1.get_default_graph().as_graph_def().node if v.name.startswith('placeholders/')]
    placeholders = {}
    for f in feature_names:
        placeholders[f.split('/')[-1]] = tf.compat.v1.get_default_graph().get_tensor_by_name('%s:0' % f)
    return out, placeholders


def load_model_from_mvs_model(json_path, load_loss=False):
    """ set self.out and self.placeholders by initializing an mvs_model """

    with open(json_path, 'rb') as fp:
        model_conf = json.load(fp)
    sector_name = list(model_conf['data_params']['datasets'].keys())[0]
    dataset_attributes, ds_format = load_dataset_attributes(model_conf['data_params']['datasets'][sector_name])

    # load map_funcs
    arch_map_func, loss_map_func, _, _ = load_map_funcs(model_conf,
                                                        sector_name,
                                                        load_arch=True, load_loss=load_loss, load_eval=False,
                                                        load_augm=False, pred_mode=True)

    # create np_features
    np_features = {}
    for k in ds_format.keys():
        if ds_format[k]['dtype'] != 'string':
            np_features[k] = np.zeros(shape=ds_format[k]['shape'], dtype=ds_format[k]['dtype'])

    # possibly apply map_funcs to np_features and run cleanup
    np_features = np_features if arch_map_func is None else arch_map_func(np_features)
    np_features = np_features if loss_map_func is None else loss_map_func(np_features)
    np_features = cleanup_features(np_features, required_features(model_conf['model_params']))

    # build model
    placeholders = placeholder_init(np_features)
    out, _ = get_model(placeholders, model_conf['model_params'], load_loss=load_loss)
    return out, placeholders


def required_features(model_params, eval=False):
    arch_features = load_format('arch', get_arch_format(model_params))
    loss_features = load_format('loss', get_loss_format(model_params))
    eval_features = load_format('eval', '') if eval else []

    summarized_tensors = list(set().union(*[v for k, v in model_params.items() if k.endswith('to_summarize')]))
    summarized_features = [f.split('features/')[-1] for f in summarized_tensors if f.startswith('features/')]
    return arch_features, loss_features, eval_features, summarized_features


def cleanup_features(features, req_features):
    req_features_names = list(set().union(*req_features))
    features = {k: v for k, v in features.items() if k in req_features_names}
    return features


def save_simple_model(json_path, model_dir):

    tf.compat.v1.reset_default_graph()
    load_model_from_mvs_model(json_path, load_loss=True)
    init = tf.compat.v1.global_variables_initializer()
    model_graph_path = os.path.join(model_dir, 'model_graph')
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        save_model_to_meta(sess, model_graph_path=model_graph_path)
    tf.compat.v1.reset_default_graph()


def ckpt_fix_mean_to_sum(meta_graph, orig_ckpt, fixed_path="/tmp/warm_start_fixed.ckpt"):
    with tf.compat.v1.Session() as sess:
        # saver = tf.train.Saver()
        saver = tf.compat.v1.train.import_meta_graph(meta_graph)
        saver.restore(sess, orig_ckpt)
        # find convolution weights
        last_corr_layer = "corr_l2_upucl2"
        corr_weights = [v for v in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES) if last_corr_layer in v.name]
        corr_out_channels = corr_weights[0].shape[-1].value
        weight_scale = (1. / corr_out_channels) ** 0.5
        print("SCALING LAYER {} BY FACTOR {}".format(last_corr_layer, weight_scale))

        corr_weights_updated = []
        for w in corr_weights:
            w_updated = w.assign(w * weight_scale)
            corr_weights_updated.append(w_updated)
        sess.run(corr_weights_updated)
        saver.save(sess, fixed_path)
        print("Saved fixed checkpoint to {}".format(fixed_path))
    tf.compat.v1.reset_default_graph()
    return fixed_path


def keras_warm_start(keras_model, warm_start_conf):
    from stereo.models.sm.sm_utils import get_checkpoint_path, latest_model
    # TODO: support both ckpt and h5
    restore_iter = warm_start_conf.get('restore_iter', -1)
    if restore_iter < 0:
        restore_iter = latest_model(warm_start_conf['ckpt_dir'])
    ckpt_to_initialize_from = get_checkpoint_path(warm_start_conf['ckpt_dir'], restore_iter)
    vars_to_warm_start = str(warm_start_conf.get('vars_to_warm_start', '.*'))
    load_ckpt_to_keras_model(keras_model, ckpt_path=ckpt_to_initialize_from,
                             var_regex=vars_to_warm_start,
                             verify_all_layers=vars_to_warm_start == '.*',
                             model_prefix=warm_start_conf['model_prefix'],
                             mean_to_sum=warm_start_conf.get('mean_to_sum', False))
    return restore_iter


def load_ckpt_to_keras_model(keras_model, ckpt_path, var_regex='.*', verify_all_layers=False, model_prefix=None,
                             mean_to_sum=False):
    """
    Load weights from tensorflow checkpoint (ckpt) to a keras model based on variable names.
    Notes:
    - Does not load (for now) the optimizer variables.
    - Need to make sure it does upload batch norm variables.
    :param keras_model: Keras model
    :param ckpt_path: Path to checkpoint from which to load the model weights
    :param var_regex: A regular expression (string) that captures which variables to load.
            Defaults to '.*', which loads all variables that appear in model layers.
    :param verify_all_layers: If true, the function verifies that all layers have corresponding variables
            in the checkpoint. This flag is useful to restore all weights from an equivalent model.
            It works only with var_regex='.*'.
    :param model_prefix: a prefix of the keras model not used in the checkpoint.
            This prefix is removed when searching for the corresponding variables
    :param mean_to_sum: fix the last correlation layer weights if the checkpoint's architecture used mean
            whereas the keras model architecture uses sum
    """
    from tensorflow.python.pywrap_tensorflow import NewCheckpointReader
    import re
    from collections import OrderedDict

    print("Parsing checkpoint: {}".format(ckpt_path))
    reader = NewCheckpointReader(ckpt_path)
    var_pattern = re.compile(var_regex)
    if model_prefix is None:
        model_prefix = ""

    for l in keras_model.layers:
        weight_map = OrderedDict([(w.name[len(model_prefix):], None)
                                  for w in l.weights])
        if len(weight_map) == 0:
            continue
        print("Layer: {} ({})".format(l.name, l.__class__.__name__))
        for v in reader.get_variable_to_dtype_map().keys():
            if var_pattern.match(v) is None:
                continue
            for w_name in weight_map.keys():
                v_compat = v.replace("ucl2", "_ucl2").replace("__ucl2", "_ucl2")  # backward compatibility
                if v_compat.endswith(w_name[:-2]):  # remove the ":0" suffix
                    if weight_map[w_name] is None:
                        weight_map[w_name] = v
                    else:
                        raise RuntimeError("Multiple tensors in checkpoint with the same name as layer")
        print(os.linesep.join(["\t{} -> {}".format(v, k) for k, v in weight_map.items()]))
        if None in weight_map.values():
            if verify_all_layers:
                raise RuntimeError("Missing weights for layer: {}".format(l.name))
            else:
                print("Layer not loaded")
        else:
            weight_scale = 1.0
            if mean_to_sum:
                if l.name == "corr_l2_upucl2" or l.name == "corr_l2_up_ucl2":
                    corr_out_channels = l.weights[0].shape[-1].value
                    weight_scale = (1. / corr_out_channels) ** 0.5
                    print("SCALING LAYER {} BY FACTOR {}".format(l.name, weight_scale))

            # Load weights
            l.set_weights([reader.get_tensor(w) * weight_scale for w in weight_map.values()])
    print("Finished loading weights to keras model.")


def load_keras_model(keras_json, load_weights=True):
    from stereo.models.menta.menta_utils import get_dlo_custom_objects
    custom_objects = get_dlo_custom_objects()

    with my_open(keras_json, mode='r') as f:
        model_config = json.load(f)
    keras_model = tf.keras.models.model_from_config(config=model_config,
                                                    custom_objects=custom_objects)
    if load_weights:
        checkpoint_path = get_keras_checkpoint(keras_json)
        keras_model.load_weights(checkpoint_path)

    return keras_model


def get_keras_checkpoint(keras_json):
    checkpoint_path = None
    # order is important:
    checkpoint_dirs = [os.path.join(os.path.dirname(keras_json), "checkpoints"),
                       os.path.dirname(keras_json)]
    for d in checkpoint_dirs:
        checkpoints = sorted([f for f in my_listdir(d) if f.endswith(".h5")])
        if len(checkpoints) > 0:
            checkpoint_path = os.path.join(d, checkpoints[-1])
            break
    if checkpoint_path is None:
        raise ValueError("No h5 checkpoint was found")
    print("Loading checkpoint: {}".format(checkpoint_path))
    if checkpoint_path.startswith("s3"):
        import s3fs
        fs = s3fs.S3FileSystem()
        tmp_checkpoint_path = os.path.join("/tmp", os.path.basename(checkpoint_path))
        #TODO: use caching instead
        fs.get(checkpoint_path, tmp_checkpoint_path)
        checkpoint_path = tmp_checkpoint_path
    return checkpoint_path


def pred_keras(keras_model, inputs, extra_tensor_names=None):
    from tensorflow.python.keras import backend as K
    model_inputs = [inputs[k] for k in keras_model.input_names]
    all_tensors = keras_model.outputs
    if extra_tensor_names is not None:
        for tensor_name in extra_tensor_names:
            all_tensors.append(K.get_session().graph.get_tensor_by_name(tensor_name + ':0'))

    out = K.function(keras_model.input, all_tensors)(model_inputs)
    output_names = keras_model.output_names
    if len(output_names) > 1:
        # if arch have mutliple outputs, convert to dict
        arch_out = {name: out[i] for i, name in enumerate(output_names)}
        out = [arch_out] + out[len(output_names):]

    return out
