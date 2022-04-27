# Copyright 2018 MLBenchmark Group. All Rights Reserved.
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
"""Convenience function for logging compliance tags to stdout.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import inspect
import json
import logging
import os
import re
import sys
import time
from TensorFlow.common.horovod_helpers import hvd, horovod_enabled

def get_mllog_mlloger(output_dir=None):
    from mlperf_logging import mllog

    str_hvd_rank = str(hvd.rank()) if horovod_enabled() else "0"
    mllogger = mllog.get_mllogger()
    mllogger.propagate = False
    mllog.propagate=False
    if output_dir is None: output_dir='./log'
    filenames = os.path.normpath(output_dir) + "/result_rank_" + str_hvd_rank + ".txt"
    mllog.config(filename=filenames)
    workername = "worker" + str_hvd_rank
    mllog.config(
            default_namespace = workername,
            default_stack_offset = 1,
            default_clear_line = False,
            root_dir = os.path.normpath(
           os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")))

    return mllogger, mllog
