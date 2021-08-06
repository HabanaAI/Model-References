###############################################################################
# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
###############################################################################

import os
import json

import tensorflow as tf

from tensorflow.python.platform import tf_logging as logging

SUPPORTED_HLS_TYPE = [
    "HLS1",
    "HLS1-H",
]
HLS1_MODULE_CNT = {
    "HLS1": 8,
    "HLS1-H": 4
}


HCL_CONFIG_PATH_VAR = "HCL_CONFIG_PATH"

HLS_RANK_MODULE_BINDINGS = {
    "HLS1-H": {
        "0": 3,
        "1": 0,
        "2": 2,
        "3": 1
    },
    "HLS1": {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7
    }
}


def setup_env_for_multinode(required=False):
    if not setup_env_for_multinode._setup_done:
        size = comm_size()
        rank = comm_rank()

        hcl_type = os.environ.get("HCL_TYPE", "HLS1")

        if hcl_type in SUPPORTED_HLS_TYPE:
            # All env variables should be set before loading_habana_modules
            if size > 1:
                os.environ["HLS1_MODULE_ID"] = str(HLS_RANK_MODULE_BINDINGS[hcl_type][str(comm_local_rank())])
                os.environ["ID"] = str(comm_local_rank())

            # Make sure every rank logging to different file
            # Only important on the same machine - so pretty much every scenarios
        if size > 1:
            rank_prefix = "rank_{}_".format(rank)
            _set_env_prefix("TF_RANK_PREFIX", rank_prefix, False)
            _set_env_prefix("HBN_TF_GRAPH_PREFIX", rank_prefix, False)
            _set_env_prefix("TF_DUMP_GRAPH_PREFIX", rank_prefix, True)
            _hvd_rank_prefix = rank_prefix
        setup_env_for_multinode._setup_done = True


setup_env_for_multinode._setup_done = False


def comm_size():
    return int(os.environ.get("OMPI_COMM_WORLD_SIZE", 1))


def comm_rank():
    return int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))


def comm_local_size():
    return int(os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE", 1))


def comm_local_rank():
    return int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))


def is_hierarchical():
    return (bool(os.environ.get("HOROVOD_HIERARCHICAL_ALLREDUCE", 0)) or
            bool(os.environ.get("HOROVOD_HCL_HIERARCHICAL_ALLREDUCE", 0))) and \
        comm_size() > comm_local_size()


def _set_env_prefix(env_name, prefix, leave_empty):
        old_prefix = os.environ.get(env_name, "")
        if leave_empty and not old_prefix:
            return
        new_prefix = f"{old_prefix}{prefix}"
        os.environ[env_name] = new_prefix

def get_hw_module_id(rank, hls_type="HLS1"):
    local_rank = str(rank % HLS1_MODULE_CNT[hls_type])
    return HLS_RANK_MODULE_BINDINGS[hls_type][local_rank]

def get_hcl_config():
    hcl_config_path = os.environ.get(HCL_CONFIG_PATH_VAR)
    assert hcl_config_path != None, "{} is not set, but required by Horovod".format(
        HCL_CONFIG_PATH_VAR)
    assert os.path.isfile(hcl_config_path), "{} points to not accessible file: {}".format(
        HCL_CONFIG_PATH_VAR, hcl_config_path)

    with open(hcl_config_path, "r") as hcl_config_file:
        try:
            return json.load(hcl_config_file)
        except json.JSONDecodeError:
            logging.error("{} indicated by {} is not valid json file".format(
                hcl_config_path, HCL_CONFIG_PATH_VAR))
            raise
