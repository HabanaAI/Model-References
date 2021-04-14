# ******************************************************************************
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
# ******************************************************************************

import os
import sys
import torch

_mandatory_libs = ["libhabana_pytorch_plugin.so"]


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
        $PYTORCH_MODULES_RELEASE_BUILD
        $PYTORCH_MODULES_DEBUG_BUILD
    """

    locations = []
    if "LD_LIBRARY_PATH" in os.environ:
        locations += os.environ.get("LD_LIBRARY_PATH").split(":")
    if "BUILD_ROOT_LATEST" in os.environ:
        locations.append(os.path.abspath(os.environ["BUILD_ROOT_LATEST"]))

    if "PYTORCH_MODULES_RELEASE_BUILD" in os.environ:
        locations.append(os.path.abspath(os.environ["PYTORCH_MODULES_RELEASE_BUILD"]))
    if "PYTORCH_MODULES_DEBUG_BUILD" in os.environ:
        locations.append(os.path.abspath(os.environ["PYTORCH_MODULES_DEBUG_BUILD"]))

    locations.append('/usr/lib/habanalabs')

    for directory in locations:
        if _check_modules_directory(directory):
            return directory

    return None


habana_modules_directory = _get_modules_directory()


def load_habana_module():
    """Load habana libs"""
    if habana_modules_directory is None:
        raise Exception("Cannot find Habana modules")

    print("Loading Habana modules from %s", str(habana_modules_directory))
    for module in _mandatory_libs:
        torch.ops.load_library(os.path.abspath(os.path.join(
            habana_modules_directory, module)))
        sys.path.insert(0, habana_modules_directory)
