# ******************************************************************************
# Copyright (C) 2020 HabanaLabs, Ltd.
# All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
#
# ******************************************************************************

import sys
import os
import ctypes
import logging

log = logging.getLogger("dynpatch_loader")

dynpatch_name = "dynpatch_prf_remote_call.so"
libtensorflow = "libtensorflow_framework.so.2"


class DynpatchLoadError(Exception):
    pass


def _IsSoLoaded(soname):
    pid = os.getpid()
    for line in open(f"/proc/{pid}/maps").readlines():
        if line.find(soname) >= 0:
            log.info(f"found {line}")
            return True
    return False


if _IsSoLoaded(dynpatch_name):
    log.info(f"{dynpatch_name} is already loaded")
else:
    if _IsSoLoaded(libtensorflow):
        raise DynpatchLoadError(
            f"Found that tensorflow has been loaded before {dynpatch_name}.\n"
            "Application of dynamic patch was unsuccessful."
        )
    log.info("loading {dynpatch_name}")

    ctypes.CDLL(dynpatch_name, os.RTLD_LAZY + ctypes.RTLD_GLOBAL)
    import tensorflow as tf
