###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################
import os
import tensorflow as tf
from tensorflow.python.client import device_lib
import habana_frameworks.tensorflow as htf

import logging
logging.warning(
    f"The '{__file__}' has been deprecated. Please load habana modules from Python module. "
    "Example: \"import habana_frameworks.tensorflow as htf; htf.load_habana_module()\"")


def get_lib_path(directory, lib):
    return os.path.abspath(os.path.join(directory, lib + '.' + tf.__version__))


def get_wheel_libs_directory():
    return htf.sysconfig.get_lib_dir()


habana_ops = htf.habana_ops
habana_modules_directory = get_wheel_libs_directory()


def load_habana_module():
    """Load habana libs"""
    htf.load_habana_module()
    return device_lib.list_local_devices()
