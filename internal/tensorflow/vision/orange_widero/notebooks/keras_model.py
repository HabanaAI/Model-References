# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: menta_py36_tf1.12
#     language: python
#     name: menta_py36_tf1.12
# ---

import json
import tensorflow as tf
import imp
import numpy as np
import os
from stereo.interfaces.implements import load_dataset_attributes, get_arch_format, get_loss_format, load_format
from stereo.common.general_utils import tree_base
from stereo.models.map_func_loaders import load_map_funcs
from stereo.models.model_utils import cleanup_features, required_features, tf_type

# +
model_name = 'stereo/diet_main_v3.0.v0.0.12_conf_menta'
model_conf_file = '../stereo/models/conf/' + model_name + '.json'

json_path = model_conf_file
with open(json_path, 'rb') as fp:
    model_conf = json.load(fp)
model_params = model_conf['model_params']

# +
sector_name = list(model_conf['data_params']['datasets'].keys())[0]
dataset_attributes, ds_format = load_dataset_attributes(model_conf['data_params']['datasets'][sector_name])


load_loss = False
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

TensorSpec = tf.TensorSpec if hasattr(tf, "TensorSpec") else tf.contrib.framework.TensorSpec
ds_spec = {k: TensorSpec(name=k, shape=v.shape, dtype=tf_type(v)) for k, v in np_features.items()}
ds_spec
# build model
# placeholders = placeholder_init(np_features)
# out, _ = get_model(placeholders, model_conf['model_params'], load_loss=load_loss)
# return out, placeholders

# +
arch_module = imp.load_source('arch', os.path.join(tree_base(), 'stereo', 'models', 'arch',
                                               model_params['arch']['name'] + '.py'))
loss_module = imp.load_source('loss', os.path.join(tree_base(), 'stereo', 'models', 'loss',
                                               model_params['loss']['name'] + '.py'))
arch_func_ = arch_module.arch
loss_func_ = loss_module.loss

arch_arg_names_ = load_format('arch', get_arch_format(model_params), model_params=model_params)
loss_arg_names_ = load_format('loss', get_loss_format(model_params), model_params=model_params)
# -

with tf.compat.v1.name_scope("inputs"):
    arch_inputs = {name: tf.keras.layers.Input(shape=ds_spec[name].shape,
                                               name=name,
                                               dtype=ds_spec[name].dtype)
                   for name in arch_arg_names_}

out = arch_func_(**arch_inputs,
                 **model_params['arch']['kwargs'])
if isinstance(out, dict):
    out_list = [out['out']]
    if out.get('out_conf', None) is not None:
        out_list.append(out['out_conf'])
else:
    out_list = [out]
out = out_list

keras_model = tf.keras.Model(inputs=list(arch_inputs.values()),
                             outputs=out,
                             name=model_params['arch']['name'].split("/")[1])

json_fname = "keras_" + os.path.basename(model_params['arch']['name']) + ".json"

import json
model_json = keras_model.to_json(indent=4)
with open(json_fname, 'w') as f:
    f.write(model_json)

filename= "model.png"
tf.keras.utils.plot_model(keras_model, show_shapes=True, show_layer_names=True, to_file=filename)

from menta.src.utils.layer_utils import get_custom_layers_api, get_keras_layers_api
from stereo.models.layers.correlation_map import Correlation
from stereo.models.layers.addconst import AddConst
custom_objects = get_custom_layers_api()

with open(json_fname, mode='r') as f:
    model_config = json.load(f)
keras_model = tf.keras.models.model_from_config(config=model_config,
                                             custom_objects=custom_objects)  # type: keras.Model

# +
# keras_model.summary()
