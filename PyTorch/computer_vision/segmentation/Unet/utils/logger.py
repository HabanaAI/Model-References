# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import os
import operator
import time
import dllogger as logger
import numpy as np
import torch.cuda.profiler as profiler
from dllogger import JSONStreamBackend, StdOutBackend, Verbosity
from typing import Optional, Any
from lightning_utilities import module_available
if module_available('lightning'):
    from lightning.pytorch import Callback
elif module_available('pytorch_lightning'):
    from pytorch_lightning import Callback

from utils.utils import is_main_process


class LoggingCallback(Callback if os.getenv('framework')=='PTL' else object):
    def __init__(self, log_dir, global_batch_size, mode, warmup, dim, profile, measurement_type, perform_epoch=1):
        logger.init(backends=[JSONStreamBackend(Verbosity.VERBOSE, log_dir), StdOutBackend(Verbosity.VERBOSE)])
        self.warmup_steps = warmup
        self.global_batch_size = global_batch_size
        self.step = 0
        self.dim = dim
        self.mode = mode
        self.profile = profile
        self.timestamps = []
        self.perform_epoch = perform_epoch
        self.measurement_type = measurement_type
        """
        measurement_type: options
        1) 'latency': timestamps will be recorded at the
            beginning of every iteration.
        2) 'throughput': Only two timestamps will be recorded corresponding
            to start and end of test step.
        """

    def do_step(self):
        self.step += 1
        if self.profile and self.step == self.warmup_steps:
            profiler.start()
        if self.step > self.warmup_steps and (self.mode=='train' or self.measurement_type == 'latency'):
            # Record the time at the beginning of every iteration 1) During training,
            # 2) During inference, when measurement_type == 'latency'
            self.timestamps.append(time.time())

    def on_train_batch_start(self, trainer, pl_module:Optional[Any]=None, batch:Optional[int]=0, batch_idx:Optional[int]=0):
        if trainer.current_epoch == self.perform_epoch:
            self.do_step()

    def on_test_batch_start(self, trainer, pl_module:Optional[Any]=None, batch:Optional[int]=0, batch_idx:Optional[int]=0, dataloader_idx:Optional[int]=0):
        if trainer.current_epoch == self.perform_epoch:
            self.do_step()

    def process_performance_stats(self, deltas):
        def _round3(val):
            return round(val, 3)

        throughput_imgps = _round3(self.global_batch_size / np.mean(deltas))
        timestamps_ms = 1000 * deltas
        stats = {
            f"throughput_{self.mode}": throughput_imgps,
            f"latency_{self.mode}_mean": _round3(timestamps_ms.mean()),
        }
        for level in [90, 95, 99]:
            stats.update({f"latency_{self.mode}_{level}": _round3(np.percentile(timestamps_ms, level))})

        return stats

    def _log(self, n_batches=1):

        diffs = list(map(operator.sub, self.timestamps[1:], self.timestamps[:-1]))
        deltas = np.array(diffs)
        if self.measurement_type=='throughput':
            # Adjust the deltas so that the scale of deltas
            # remains same across different measurement_type options
            deltas /= n_batches
        stats = self.process_performance_stats(deltas)
        if is_main_process():
            logger.log(step=(), data=stats)
            logger.flush()
        return stats

    def on_train_end(self, trainer, pl_module:Optional[Any]=None):
        if self.profile:
            profiler.stop()
        if not pl_module.args.benchmark:
            return None
        stats = self._log()
        return stats

    def on_test_start(self, trainer, pl_module:Optional[Any]=None):
        # Record the time at the beginning (if measurement_type == 'throughput')
        if self.measurement_type == 'throughput' and trainer.current_epoch == self.perform_epoch:
            self.timestamps.append(time.time())

    def on_test_end(self, trainer, pl_module:Optional[Any]=None):
        # Record the time at the end (if measurement_type == 'throughput')
        if self.measurement_type == 'throughput' and trainer.current_epoch == self.perform_epoch:
            self.timestamps.append(time.time())
        if trainer.current_epoch == self.perform_epoch and pl_module.args.benchmark:
            self._log(pl_module.args.test_batches)
