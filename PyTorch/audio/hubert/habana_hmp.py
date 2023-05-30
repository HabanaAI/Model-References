import os
import glob
import lightning
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.plugins import HPUPrecisionPlugin
from typing import Any, Optional

__PTL_VER__ = lightning.__version__


class HPUHMPlugin(HPUPrecisionPlugin):

    def __init__(self, verbosity: bool = False, level: str = "O1", model_name: Optional[str]=None, precision=16):
        currentpath = os.path.dirname(os.path.realpath(__file__))
        dirlist16 = glob.glob(f'{currentpath}/ops_bf16*.txt')
        dirlist32 = glob.glob(f'{currentpath}/ops_fp32*.txt')
        if model_name:
            op_file_16 = [fn for fn in dirlist16 if model_name in fn ]
            op_file_32 = [fn for fn in dirlist32 if model_name in fn ]
            assert(op_file_16 and op_file_32),\
                f"Model '{model_name}' hmp files are not found."

        hmp_params = { "precision": precision,
                       "opt_level": level,
                       "bf16_file_path": op_file_16[0] if model_name else "",
                       "fp32_file_path": op_file_32[0] if model_name else "",
                       "verbose": verbosity
        }
        super(HPUHMPlugin, self).__init__(**hmp_params)
