###############################################################################
# Copyright (C) 2020-2022 Habana Labs, Ltd. an Intel Company
###############################################################################

from functools import reduce
import time
import os
import tensorflow as tf
from tensorflow.python.training.saver import BaseSaverBuilder
from tensorflow.python.eager import context
from tensorflow.python.ops import variables

import threading
import logging
log = logging.getLogger(__name__)

try:
    import horovod.tensorflow as hvd
except ImportError:
    hvd = None


class LightweightCheckpointSaverBuilder(BaseSaverBuilder):
    def __init__(
            self, global_variables, global_variables_remote, global_variables_local=[],
            global_variables_local_cache=[]):
        self._global_variables = global_variables
        self._global_variables_remote = global_variables_remote
        self._global_variables_local = global_variables_local
        self._global_variables_local_cache = global_variables_local_cache

        self._sharded = True if len(self._global_variables_remote) != 0 else False
        self._cached = len(global_variables_local_cache) != 0

        assert len(global_variables_local_cache) == 0 or len(global_variables_local_cache) == len(
            global_variables_local), \
            "local variable cache list has to be either empty (caching disabled) or equal to length of local variables"
        assert len(global_variables_local) + len(global_variables_remote) == len(global_variables)

        if self._cached:
            # creating operation for caching variables
            # caching is executed before starting thread that is supposed to store variables as checkpoint on disk
            self._local_to_cache_op = tf.group(*[var_cached.assign(var) for var, var_cached in
                                                 zip(self._global_variables_local, self._global_variables_local_cache)])

        BaseSaverBuilder.__init__(self)

    @property
    def copy_local_vars_to_cache_op(self):
        return self._local_to_cache_op

    def _AddRestoreOps(self, *args, **kwargs):
        standard_restore_ops = super()._AddRestoreOps(*args, **kwargs)

        if self._cached:
            with tf.control_dependencies([standard_restore_ops]):
                # creating operation for copying back values from cache to real variables
                restore_op = tf.group(*[var.assign(var_cached) for var_cached, var in
                                        zip(self._global_variables_local_cache, self._global_variables_local)])
        else:
            restore_op = standard_restore_ops

        if self._sharded:
            with tf.control_dependencies([*self.zero_variables(self._global_variables_remote),
                                          restore_op]):
                operation = self.allreduce(
                    self._global_variables, name="restore_all")
        else:
            operation = restore_op

        return operation

    @staticmethod
    def zero_variables(variables):
        return [var.assign(tf.zeros(var.get_shape(), var.dtype)) for var in variables]

    @staticmethod
    def allreduce(variables, name):
        hvd.init()

        def _filter_floats(variables):
            for var in variables:
                if var.dtype in [tf.float32, tf.bfloat16]:
                    yield var
        return tf.group(*[var.assign(hvd.allreduce(var, average=False)) for var in _filter_floats(variables)],
                        name=name)


def vars_summary(var_list):
    result_dict = {}
    for var in var_list:
        if var.dtype not in result_dict:
            result_dict[var.dtype] = {"vars": 0, "elements": 0}

        result_dict[var.dtype]["vars"] += 1
        if len(var.shape) > 0:
            result_dict[var.dtype]["elements"] += reduce(lambda a, b: a * b, var.shape)

    return result_dict


def get_device_name_by_type(type):
    from tensorflow.python.client import device_lib

    local_devices = device_lib.list_local_devices()

    for local_dev in local_devices:
        if local_dev.device_type == type:
            return local_dev.name
    raise RuntimeError(f"No device with given type {type} is present!")


class LightweightCheckpointSaver(tf.compat.v1.train.Saver):
    def __init__(self, sharded_checkpoints_enabled=True, log_enabled=True, log_dir="~/", comm_rank=None, comm_size=None,
                 variables_caching=True, *args, **kwargs):
        self._lws_log_enabled = log_enabled
        self._variables_caching = variables_caching

        rank = comm_rank()
        self._lws_out_file = os.path.expanduser(os.path.join(log_dir, f"rank_{rank}_saver_log.txt"))

        var_list = tf.compat.v1.global_variables()

        self._lws_rank = rank
        self._lws_var_list_local = []
        self._lws_var_list_remote = []
        self._lws_var_list_local_cached = []
        self._lws_var_list_all = var_list

        for idx, var in enumerate(var_list):
            # - if sharded checkpoints are disabled then all variables should be handled locally by workers participating
            #   in checkpoint storing/restoring
            # - otherwise (sharding enabled) distribute variables across all workers by rank
            if not sharded_checkpoints_enabled or rank == (
                    idx % comm_size()) or var.dtype not in [
                    tf.float32, tf.bfloat16]:
                self._lws_var_list_local.append(var)
            else:
                self._lws_var_list_remote.append(var)

        kwargs["var_list"] = self._lws_var_list_local

        if self._variables_caching:
            self._lws_var_list_local_cached = self._create_local_variables_cache()

            # replace var_list with cached variables, cached variables will be passed to saver, it means that save and
            # restore will be performed using this copy of variables
            kwargs["var_list"] = self._lws_var_list_local_cached

        self._lws_builder = LightweightCheckpointSaverBuilder(
            self._lws_var_list_all, self._lws_var_list_remote, self._lws_var_list_local, self._lws_var_list_local_cached)

        kwargs["builder"] = self._lws_builder

        self._lws_log(
            f"\nLightweightCheckpointSaver initialized with sharded_checkpoints_enabled={sharded_checkpoints_enabled} for rank {rank}\n")
        self._lws_log(f"Local variables summary: {vars_summary(kwargs['var_list'])}\n")

        tf.compat.v1.train.Saver.__init__(self, *args, **kwargs)

    def _create_local_variables_cache(self):
        with tf.device(get_device_name_by_type("CPU")):
            variables_cache = [tf.Variable(v.initial_value, name=f"{v.name.split(':')[0]}-cache") for v in self._lws_var_list_local]
        return variables_cache

    def pre_save(self, sess):
        if self._variables_caching:
            log.info("Copying model variables to separate instance")
            start_time = time.time()
            sess.run(self._lws_builder.copy_local_vars_to_cache_op)
            log.info("Variables cached (%.3f sec)", time.time()-start_time)

    def save(self, sess, *args, **kwargs):
        with context.graph_mode():
            start_time = time.time()
            log.info(f"Start checkpoint saving in thread {threading.current_thread().name}")
            tf.compat.v1.train.Saver.save(self, sess, *args, **kwargs)
            end_time = time.time()
            self._lws_log_time(start_time, end_time, "Saving")

    def restore(self, sess, *args, **kwargs):
        start_time = time.time()
        tf.compat.v1.train.Saver.restore(self, sess, *args, **kwargs)
        end_time = time.time()
        self._lws_log_time(start_time, end_time, "Restoring")

    def _lws_log(self, msg):
        if self._lws_log_enabled:
            with open(self._lws_out_file, "a") as out_file:
                out_file.write(msg)

    def _lws_log_time(self, start_time, end_time, prefix):
        if self._lws_log_enabled:
            delta = int((end_time - start_time)*1e6)
            msg = "{} time: {}ms\n".format(prefix, delta/1000)
            self._lws_log(msg)
