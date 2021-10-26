###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
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
        if self._env_vars_to_set:
            for env_var_name, env_var_value in self._env_vars_to_set.items():
                if "LD_PRELOAD" in str(env_var_name) and os.environ.get(str(env_var_name), None):
                    os.environ[str(env_var_name)] = str(env_var_value) + ":" + os.environ.get(str(env_var_name), None)
                else:
                    os.environ[str(env_var_name)] = str(env_var_value)

    def __exit__(self, *args) -> None:
        """Delete all env variables."""
        if self._env_vars_to_set:
            for env_var_name in self._env_vars_to_set.keys():
                del os.environ[env_var_name]


def print_env_info(command, env_variables: Dict[str, str]) -> None:
    """Print info about env variables and model parameters."""

    print("\n\nRunning with the following env variables\n")
    if not env_variables:
        return
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

def is_horovod_hierarchical():
    return os.environ.get('HOROVOD_HIERARCHICAL_ALLREDUCE') == '1'

# Returns a list of valid host IP names found in the 'MULTI_HLS_IPS' environment
# variable if this is set, else an empty list
def get_multi_node_config_nodes(env_var='MULTI_HLS_IPS', sep=','):
    if is_valid_multi_node_config(env_var):
        return os.environ.get(env_var).split(sep)
    else:
        return []
