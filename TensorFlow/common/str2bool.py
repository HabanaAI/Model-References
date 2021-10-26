###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################
import os
import typing

def str2bool(s : str) -> bool:
    assert s is not None
    if s.lower() in ["1", "t", "true", "y", "yes", "enable", "enabled"]:
        return True
    elif s.lower() in ["0", "f", "false", "n", "no", "disable", "disabled"]:
        return False
    else:
        raise ValueError(f"{__name__}('{s}'): String not recognized as a boolean value")

def condition_env_var(env_var_name : str, default : bool) -> bool:
    s = os.environ.get(env_var_name, None)
    if s is None:
        return default
    else:
        return str2bool(s)
