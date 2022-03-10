#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================================================
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#
# ==============================================================================
###############################################################################
# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
###############################################################################
# Changes:
# - script migration to Tensorflow 2.x version
# - included HPU horovod setup


from TensorFlow.common.horovod_helpers import hvd, hvd_init, hvd_size, hvd_rank, horovod_enabled
from utils import hvd_utils

__all__ = ["_log_hparams"]


def _log_hparams(classname, layername, **kwargs):

    log_msg = "%s: `%s`" % (classname, layername)

    for arg, val in sorted(kwargs.items()):
        log_msg += "\n\t[*] {}: {}".format(arg, val)

    log_msg += "\n"

    if not horovod_enabled() or hvd.rank() == 0:
        print(log_msg)
