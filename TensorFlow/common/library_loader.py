# ******************************************************************************
# Copyright (C) 2020 HabanaLabs, Ltd.
# All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
#
# ******************************************************************************

import os
import sys
import tensorflow as tf
import logging
from tensorflow.python.client import device_lib

from TensorFlow.common.hw_profiler_helpers import SynapseLoggerHook, SynapseLoggerHelpers

log = logging.getLogger("library_loader")

_mandatory_libs = ["graph_writer.so", "habana_device.so"]


def _check_modules_directory(directory):
    if not os.path.isdir(directory):
        return False

    for module in _mandatory_libs:
        if not os.path.isfile(os.path.join(directory, module)):
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
    """Load habana libs"""

    ## This part is used to gather single-file HW traces for multi-graphs iterations
    if os.getenv("TF_HW_PROFILE_RANGE"):
        profile_range = os.environ["TF_HW_PROFILE_RANGE"].split(":")

        SynapseLoggerHelpers.synapse_logger()
        logger_hook = SynapseLoggerHook(steps_to_log=range(int(profile_range[0]), int(profile_range[1])))

        default_session_run = tf.compat.v1.train.MonitoredSession.run
        def hooked_session_run(self, *args, **kwargs):
            logger_hook.before_run()
            default_outputs = default_session_run(self, *args, **kwargs)
            logger_hook.after_run()
            return default_outputs
        tf.compat.v1.train.MonitoredSession.run = hooked_session_run
    ## end of HW-trace related code

    if habana_modules_directory == None:
        raise Exception("Cannot find Habana modules")

    log.info("Loading Habana modules from %s", str(habana_modules_directory))
    for module in _mandatory_libs:
        tf.load_library(os.path.abspath(os.path.join(habana_modules_directory, module)))

    op_library = tf.load_op_library(os.path.abspath(os.path.join(habana_modules_directory, _mandatory_libs[1])))
    setattr(habana_ops, "ops", op_library)
    return device_lib.list_local_devices()
