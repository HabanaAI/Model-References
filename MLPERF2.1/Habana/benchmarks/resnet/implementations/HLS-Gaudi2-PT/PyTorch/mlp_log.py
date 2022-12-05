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
###############################################################################
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
###############################################################################
# List of changes:
# - Added pytorch worker ranks using utils

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
import utils

def get_mllog_mlloger(output_dir=None):
    from mlperf_logging import mllog

    if utils.get_rank() is not None:
        str_worker_rank = str(utils.get_rank())
    else:
        str_worker_rank = "0"

    mllogger = mllog.get_mllogger()
    mllogger.propagate = False
    mllog.propagate=False
    if output_dir is None: output_dir='./log'
    filenames = os.path.normpath(output_dir) + "/result_rank_" + str_worker_rank + ".txt"
    mllog.config(filename=filenames)
    workername = "worker" + str_worker_rank
    mllog.config(
            default_namespace = workername,
            default_stack_offset = 1,
            default_clear_line = False,
            root_dir = os.path.normpath(
           os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")))

    return mllogger, mllog
