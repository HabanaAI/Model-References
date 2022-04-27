# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Register flags for experimentla features."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing

from absl import flags    # pylint: disable=g-bad-import-order
import tensorflow as tf   # pylint: disable=g-bad-import-order

from TensorFlow.computer_vision.Resnets.ResNeXt.utils.flags._conventions import help_wrap


def define_experimental(experimental_preloading=True):
  """Register flags for experimental features.

  Args:
    experimental_preloading: Create a flag to specify parallelism of data loading.

  Returns:
    A list of flags for core.py to marks as key flags.
  """

  key_flags = []

  if experimental_preloading:
    flags.DEFINE_bool(
        name="experimental_preloading",
        default=True,
        help=help_wrap("Enables support for 'data.experimental.prefetch_to_device' TensorFlow operator."
                       "Enabled by default - pass --experimental_preloading=False to disable."))

  return key_flags
