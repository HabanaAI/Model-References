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
import shutil

# Call this function when you need a path object
def get_canonical_path(name):
    return Path(os.path.expandvars(os.path.expanduser(name))).resolve()

# Call this function when you need a string representing a fully-qualified path
def get_canonical_path_str(name):
    return os.fspath(Path(os.path.expandvars(os.path.expanduser(name))).resolve())

# Create an output directory and optionally clear its existing contents first
def create_output_dir(outdir, clear_flag=False):
    try:
        od_path = get_canonical_path(outdir)
        if clear_flag == True:
            if os.path.isfile(od_path):
                print(f"*** Deleting existing file {str(od_path)}...\n\n")
                os.remove(od_path)
            elif os.path.isdir(od_path):
                print(f"*** Deleting existing directory {str(od_path)}...\n\n")
                shutil.rmtree(od_path)
        os.makedirs(od_path, mode=0o777, exist_ok=True)
        if os.path.isfile(str(od_path) + ".tar.gz"):
            print(f"*** Deleting existing file {str(od_path)}.tar.gz...\n\n")
            os.remove(str(od_path) + ".tar.gz")
        return od_path
    except Exception as exc:
        raise RuntimeError(f"Error in creating directory {outdir} with clear_flag = {clear_flag}") from exc
