import os
import imp
import tensorflow as tf
from abc import ABCMeta, abstractmethod

from stereo.common.general_utils import tree_base
from stereo.interfaces.implements import load_format, get_arch_format, get_loss_format
from stereo.common.s3_utils import my_open

import numpy as np


class BaseModel(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, model_input):
        pass

    @abstractmethod
    def loss(self, out, loss_input):
        pass


class MVSModel(BaseModel):
    """
    Multi-View Stereo model
    """
    def __init__(self, params):
        super(MVSModel, self).__init__()
        self.with_model_scoping = params.get('with_model_scoping', False)
        arch_module = imp.load_source('arch', os.path.join(tree_base(), 'stereo', 'models', 'arch',
                                                       params['arch']['name'] + '.py'))
        loss_module = imp.load_source('loss', os.path.join(tree_base(), 'stereo', 'models', 'loss',
                                                       params['loss']['name'] + '.py'))
        self.arch_func_ = arch_module.arch
        self.loss_func_ = loss_module.loss

        self.arch_arg_names_ = load_format('arch', get_arch_format(params), model_params=params)
        self.loss_arg_names_ = load_format('loss', get_loss_format(params), model_params=params)

        # TODO: remove once possible
        self.consts = {}
        if 'consts' in params.keys():
            try:
                path = params['consts']['path']
                with my_open(path) as f:
                    self.consts = {}
                    consts_npz = np.load(f)
                    for key in consts_npz.keys():
                        self.consts[key] = 1. * consts_npz[key]
            except:
                print("didn't load model consts")
                pass

    def __call__(self, model_input):
        if self.with_model_scoping:
            with tf.compat.v1.name_scope("arch"):
                out = self.arch_func_(**model_input)
        else:
            print('must use with_model_scoping')
            assert False
        return out

    def loss(self, out, loss_input):
        if self.with_model_scoping:
            with tf.compat.v1.name_scope("loss"):
                loss_ = self.loss_func_(out, **loss_input)
                if isinstance(loss_, dict):
                    _ = tf.identity(loss_['loss'], name="loss")
                else:
                    _ = tf.identity(loss_, name="loss")
        else:
            loss_ = self.loss_func_(out, **loss_input)
        return loss_


def get_model(features, model_params, training=False, load_loss=True):

    # add name-scoped + named features to graph
    with tf.compat.v1.name_scope("features"):
        for k, v in features.items():
            tf.identity(v, name=k)

    model = MVSModel(model_params)

    arch_args = {}
    for k in model.arch_arg_names_:
        arch_args[k] = features[k]
    arch_args.update(model_params['arch']['kwargs'])
    arch_args['training'] = training

    if load_loss:
        loss_args = {}
        for k in model.loss_arg_names_:
            loss_args[k] = features[k]
        loss_args.update(model_params['loss']['kwargs'])

        with tf.compat.v1.variable_scope('consts'):
            for const in model.consts.keys():
                init = tf.constant(model.consts[const].astype(np.float32))
                tf.compat.v1.get_variable(name=const, initializer=init, trainable=False)

    with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=tf.compat.v1.AUTO_REUSE):
        out = model(arch_args)
        if load_loss:
            loss = model.loss(out, loss_args)
        else:
            loss = None

    return out, loss
