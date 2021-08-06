###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################
import os
import tensorflow as tf
import logging
from tensorflow.python.client import device_lib

from TensorFlow.common.multinode_helpers import setup_env_for_multinode

log = logging.getLogger("library_loader")

_mandatory_libs = ["graph_writer.so", "habana_device.so"]


def get_lib_path(directory, lib):
    return os.path.abspath(os.path.join(directory, lib + '.' + tf.__version__))


def get_wheel_libs_directory():
    try:
        from habana_frameworks.tensorflow.sysconfig import get_lib_dir
        return get_lib_dir()
    except:
        return None


def _check_modules_directory(directory):
    if not os.path.isdir(directory):
        return False

    for module in _mandatory_libs:
        if not os.path.isfile(get_lib_path(directory, module)):
            return False

    return True


def _get_modules_directory():
    """
    Returns a directory containing Habana modules.
    Directory containing modules is looked up as instructed by the following
    environmental variables, in order, until a location is found with all
    the needed libraries:
        $LD_LIBRARY_PATH
        $BUILD_ROOT_LATEST
        $TF_MODULES_RELEASE_BUILD
        $TF_MODULES_DEBUG_BUILD
    """

    locations = []

    wheel_libs_dir = get_wheel_libs_directory()
    if wheel_libs_dir:
        locations.append(wheel_libs_dir)

    if "LD_LIBRARY_PATH" in os.environ:
        locations += os.environ.get("LD_LIBRARY_PATH").split(":")
    if "BUILD_ROOT_LATEST" in os.environ:
        locations.append(os.path.abspath(os.environ["BUILD_ROOT_LATEST"]))

    if "TF_MODULES_RELEASE_BUILD" in os.environ:
        locations.append(os.path.abspath(os.environ["TF_MODULES_RELEASE_BUILD"]))
    if "TF_MODULES_DEBUG_BUILD" in os.environ:
        locations.append(os.path.abspath(os.environ["TF_MODULES_DEBUG_BUILD"]))

    locations.append('/usr/lib/habanalabs')

    for directory in locations:
        if _check_modules_directory(directory):
            return directory

    log.critical(f"Cannot find at least one of {[lib + '.' + tf.__version__ for lib in _mandatory_libs]}\n"
                 f"Searched following directories: {locations}")
    return None


class _HabanaOps:
    def __init__(self):
        self.ops = None

    def __getattr__(self, op):
        assert self.ops, f"looking for {op}, but habana module seems not to be loaded yet"
        return getattr(self.ops, op)


habana_ops = _HabanaOps()
habana_modules_directory = _get_modules_directory()
import TensorFlow.common.hpu_grads

def load_habana_module():
    setup_env_for_multinode()
    """Load habana libs"""

    # This part injects builtin Habana TF hooks that can be used e.g. to get HW trace for specific batch
    if os.getenv("TF_HOOK_MODE"):
        log.info("TF_HOOK: hooks enabled")
        from TensorFlow.common.hw_profiler_helpers import SynapseLoggerHook, SynapseLoggerHelpers, append_param
        tf_hook = None

        # Setup HW tracing hook
        if os.getenv("TF_HW_PROFILE_RANGE"):
            log.info("TF_HOOK: HW profile enabled")
            # Get range and prepare hook for it.
            profile_range = os.environ["TF_HW_PROFILE_RANGE"].split(":")
            SynapseLoggerHelpers.synapse_logger()
            tf_hook = SynapseLoggerHook(steps_to_log=range(int(profile_range[0]), int(profile_range[1])))

        # elif ....
        # TODO: add any other hook option we would need here...
        # ....

        # Check if any hook was chosen by user.
        if tf_hook is not None:
            # Wrapping TF 1.x
            # Estimator API also uses MonitoredSession inside so it supports it as well:
            if os.environ["TF_HOOK_MODE"] == "v1" or os.environ["TF_HOOK_MODE"] == "all":
                log.info("TF_HOOK: v1 enabled")
                default_session_run = tf.compat.v1.train.MonitoredSession.run

                def hooked_session_run(self, *args, **kwargs):
                    tf_hook.before_run()
                    default_outputs = default_session_run(self, *args, **kwargs)
                    tf_hook.after_run()
                    return default_outputs
                tf.compat.v1.train.MonitoredSession.run = hooked_session_run

            # Wrapping Keras (model.fit, model.fit_generator, both training and training_v1):
            if os.environ["TF_HOOK_MODE"] == "keras" or os.environ["TF_HOOK_MODE"] == "all":
                log.info("TF_HOOK: keras enabled")
                from tensorflow.python.keras.callbacks import Callback
                from tensorflow.python.keras.engine import training, training_v1

                class HBNKerasCallback(Callback):
                    def on_train_batch_begin(self, batch, logs=None):
                        tf_hook.before_run()

                    def on_train_batch_end(self, batch, logs=None):
                        tf_hook.after_run()

                def add_callback(param):
                    param.append(HBNKerasCallback())
                    return param

                training.Model.fit = append_param(training.Model.fit, "callbacks", add_callback)
                training.Model.fit_generator = append_param(training.Model.fit_generator, "callbacks", add_callback)

                training_v1.Model.fit = append_param(training_v1.Model.fit, "callbacks", add_callback)
                training_v1.Model.fit_generator = append_param(training_v1.Model.fit_generator, "callbacks", add_callback)

            # Wrapping CTL (single loop/tf.function call; if it consists of multiple steps, then we
            # profile whole loop/call as we cannot profile specific step inside tf.function):
            if os.environ["TF_HOOK_MODE"] == "ctl" or os.environ["TF_HOOK_MODE"] == "all":
                log.info("TF_HOOK: ctl enabled")
                from TensorFlow.common.training.standard_runnable import StandardTrainable
                default_ctl_train = StandardTrainable.train

                def hooked_ctl_train(self, *args, **kwargs):
                    tf_hook.before_run()
                    default_outputs = default_ctl_train(self, *args, **kwargs)
                    tf_hook.after_run()
                    return default_outputs
                StandardTrainable.train = hooked_ctl_train
    # end of built-in hooks related code

    if not habana_modules_directory:
        raise Exception("Cannot find Habana modules")

    log.info("Loading Habana modules from %s", str(habana_modules_directory))

    for module in _mandatory_libs:
        tf.load_library(get_lib_path(habana_modules_directory, module))

    op_library = tf.load_op_library(get_lib_path(habana_modules_directory, _mandatory_libs[1]))

    setattr(habana_ops, "ops", op_library)
    return device_lib.list_local_devices()
