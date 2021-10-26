# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
# - added tensorboard logging possibility
# - added performance parameters logging during training
# - added possibility to log data from every horovod worker

import time

import numpy as np
import tensorflow as tf


def get_hooks(params, logger):
    if 'train' in params.exec_mode:
        hooks = []
        if params.use_horovod:
            hooks += [params.hvd.BroadcastGlobalVariablesHook(0)]
        if params.worker_id == 0 or params.log_all_workers:
            if params.benchmark:
                hooks += [ProfilingHook(warmup_steps=params.warmup_steps,
                                        global_batch_size=params.num_workers * params.batch_size,
                                        logger=logger,
                                        mode='train')]
            else:
                hooks += [TrainingHook(params,
                                       logger=logger,
                                       tensor_names=['total_loss_ref:0'])]
                if params.tensorboard_logging:
                    from TensorFlow.common.tb_utils import ExamplesPerSecondEstimatorHook
                    hooks += [ExamplesPerSecondEstimatorHook(params.batch_size,
                                                             params.log_every,
                                                             output_dir=params.log_dir)]
        return hooks

    elif 'predict' == params.exec_mode:
        hooks = []
        if params.worker_id == 0:
            if params.benchmark:
                hooks += [ProfilingHook(warmup_steps=params.warmup_steps,
                                        global_batch_size=params.batch_size,
                                        logger=logger,
                                        mode='test')]
            return hooks


class ProfilingHook(tf.estimator.SessionRunHook):
    def __init__(self, warmup_steps, global_batch_size, logger, mode):
        self._warmup_steps = warmup_steps
        self._global_batch_size = global_batch_size
        self._step = 0
        self._timestamps = []
        self._logger = logger
        self._mode = mode

    def before_run(self, run_context):
        self._step += 1
        if self._step >= self._warmup_steps:
            self._timestamps.append(time.time())

    def end(self, session):
        deltas = np.array([self._timestamps[i + 1] - self._timestamps[i]
                          for i in range(len(self._timestamps) - 1)])
        stats = process_performance_stats(np.array(deltas),
                                          self._global_batch_size,
                                          self._mode)

        self._logger.log(step=(), data={metric: float(
            value) for (metric, value) in stats})
        self._logger.flush()


class TrainingHook(tf.estimator.SessionRunHook):
    def __init__(self, params, logger, tensor_names):
        self._params = params
        self._step = 0
        self._timestamp = time.time()
        self._logger = logger
        self._tensor_names = tensor_names

    def before_run(self, run_context):
        run_args = tf.estimator.SessionRunArgs(
            fetches=self._tensor_names
        )

        return run_args

    def after_run(self,
                  run_context,
                  run_values):
        if self._step % self._params.log_every == 0:
            duration = float(time.time() - self._timestamp) / \
                self._params.log_every
            self._timestamp = time.time()
            data = {}
            for i in range(len(self._tensor_names)):
                data[self._tensor_names[i]] = str(run_values.results[i])
            data["iter duration [ms]"] = 1000 * duration
            data["examples/sec"] = self._params.batch_size / duration
            self._logger.log(
                step=(self._step, self._params.max_steps), data=data)
        self._step += 1

    def end(self, session):
        self._logger.flush()


def process_performance_stats(timestamps, batch_size, mode):
    timestamps_ms = 1000 * timestamps
    latency_ms = timestamps_ms.mean()
    std = timestamps_ms.std()
    n = np.sqrt(len(timestamps_ms))
    throughput_imgps = (1000.0 * batch_size / timestamps_ms).mean()

    stats = [("throughput_{}".format(mode), str(throughput_imgps)),
             ('latency_{}:'.format(mode), str(latency_ms))]
    for ci, lvl in zip(["90%:", "95%:", "99%:"],
                       [1.645, 1.960, 2.576]):
        stats.append(("Latency_{} ".format(mode) + ci,
                     str(latency_ms + lvl * std / n)))
    return stats
