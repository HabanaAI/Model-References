# ******************************************************************************
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ******************************************************************************

from setuptools import setup
from torch.utils import cpp_extension
from habana_frameworks.torch.utils.lib_utils import get_include_dir, get_lib_dir
import os
import pybind11

torch_include_dir = get_include_dir()
torch_lib_dir = get_lib_dir()
habana_modules_directory = "/usr/include/habanalabs"
pybind_include_path = pybind11.get_include()

setup(name='hpu_custom_relu',
      ext_modules=[cpp_extension.CppExtension('hpu_custom_relu', ['hpu_custom_relu.cpp'],
            language='c++', extra_compile_args=["-std=c++17"],
            libraries=['habana_pytorch_plugin'],
            library_dirs=[torch_lib_dir])],
      include_dirs=[torch_include_dir,
                    habana_modules_directory,
                    pybind_include_path,
                    ],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

