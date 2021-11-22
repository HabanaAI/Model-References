###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
import logging
import subprocess
import sys
_log = logging.getLogger(__file__)


def setup_preloading() -> None:
    """
    Setup preloading via env variable
    """
    _log.info("Data-preloading feature has been requested")
    os.environ["HBN_TF_REGISTER_DATASETOPS"] = "1"


def setup_jemalloc() -> None:
    """
    Setup libjemalloc.so.1 (U18 & Amazon linux 2) or libjemalloc.so.2 (U20)
    by exporting LD_PRELOAD env variable.
    """
    _log.info("libjemalloc.so has been requested")
    paths = {"LD_LIBRARY_PATH"}
    env_vals = [os.environ[x] for x in paths if os.environ.get(x) is not None]
    env_vals.extend(["/usr/lib/x86_64-linux-gnu", "/usr/lib64"])
    sep = ":"
    final_path = None
    locations = sep.join(env_vals).split(sep)
    for path in locations:
        if path:
            libpath = f"{path}/libjemalloc.so.1"
            if os.path.isfile(libpath):
                final_path = os.path.realpath(libpath)
    for path in locations:
        if path:
            libpath = f"{path}/libjemalloc.so.2"
            if os.path.isfile(libpath):
                final_path = os.path.realpath(libpath)
    if final_path:
        os.environ["LD_PRELOAD"] = f"{final_path}:{os.environ.get('LD_PRELOAD', '')}"
    else:
        raise FileExistsError("Neither libjemalloc.so.1 nor libjemalloc.so.2 found.")
