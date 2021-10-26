# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
#
# Two functions taken from tensorflow_estimator/python/estimator/util.py:
#     - parse_iterator_result
#     - parse_input_fn_result - extended to set iterator on HPU
#
###############################################################################
from habana_frameworks.tensorflow import HabanaEstimator
import logging
logging.warning(f"The {__file__} has been deprecated. Please import HabanaEstimator from Python module. "
                "Example: \"from habana_frameworks.tensorflow import HabanaEstimator\"")
