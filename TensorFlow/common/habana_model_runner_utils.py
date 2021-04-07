###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
###############################################################################

import os
from pathlib import Path
from typing import Dict, List
import yaml


class HabanaEnvVariables:
    """Utility context manager that sets and unsets chosen ev variables."""

    def __init__(self, env_vars_to_set: Dict[str, str]) -> None:
        """Initialize env variables to set from a dict."""
        self._env_vars_to_set: Dict[str, str] = env_vars_to_set

    def __enter__(self) -> None:
        """Set all env variables."""
        for env_var_name, env_var_value in self._env_vars_to_set.items():
            os.environ[str(env_var_name)] = str(env_var_value)

    def __exit__(self, *args) -> None:
        """Delete all env variables."""
        for env_var_name in self._env_vars_to_set.keys():
            del os.environ[env_var_name]


def print_env_info(command, env_variables: Dict[str, str]) -> None:
    """Print info about env variables and model parameters."""

    print("\n\nRunning with the following env variables\n")
    for key, value in env_variables.items():
        print(f"{key}={value}")

    print("\n\nRunning the following command:\n")
    if isinstance(command, str):
        print(command)
    else:
        for idx, line in enumerate(command):
            if idx != len(command) -1: # avoid printing ' \' on the last line
                print(f"{line} \\")
            else:
                print(f"{line}")


def get_script_path(model: str) -> Path:
    """Return path to script for available models."""
    main_tf_dir = Path(__file__).parent.parent
    model_to_path = {
        "resnet_estimator": main_tf_dir / "computer_vision" / "Resnets" / "imagenet_main.py",
        "resnet_keras": main_tf_dir / "computer_vision" / "Resnets" / "resnet_keras" / "resnet_ctl_imagenet_main.py",
        "bert": main_tf_dir / "nlp" / "bert" / "demo_bert.py",
        "maskrcnn": main_tf_dir / "computer_vision" / "maskrcnn_matterport_demo" / "samples" / "coco" / "coco.py",
        "ssd_resnet34": main_tf_dir / "computer_vision" / "SSD_ResNet34" / "ssd.py",
    }
    assert model in model_to_path, f"model {model} not available, please provide one of {model_to_path.keys()}"
    return model_to_path[model]

# Call this function when you need a path object
def get_canonical_path(name):
    return Path(os.path.expandvars(os.path.expanduser(name))).resolve()

# Call this function when you need a string representing a fully-qualified path
def get_canonical_path_str(name):
    return os.fspath(Path(os.path.expandvars(os.path.expanduser(name))).resolve())

# Returns True if the environment variable 'MULTI_HLS_IPS' (by default) is set to a
# valid comma-separated string of host IP addresses
def is_valid_multi_node_config(env_var='MULTI_HLS_IPS'):
    return os.environ.get(env_var) is not None and os.environ.get(env_var) != ''

# Returns a list of valid host IP names found in the 'MULTI_HLS_IPS' environment
# variable if this is set, else an empty list
def get_multi_node_config_nodes(env_var='MULTI_HLS_IPS', sep=','):
    if is_valid_multi_node_config(env_var):
        return os.environ.get(env_var).split(sep)
    else:
        return []
