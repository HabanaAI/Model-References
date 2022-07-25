# coding=utf-8
# Copyright 2021 The Tensor2Tensor Authors.
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
###############################################################################
# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
###############################################################################
# Changes:
# - updated and removed unnecessary imports

"""Models defined in T2T. Imports here force registration."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

# pylint: disable=unused-import

from TensorFlow.nlp.transformer.layers import modalities  # pylint: disable=g-import-not-at-top
from TensorFlow.nlp.transformer.models import transformer
from TensorFlow.nlp.transformer.utils import contrib
from TensorFlow.nlp.transformer.utils import registry


def model(name):
  return registry.model(name)
