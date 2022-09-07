"""
Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
All Rights Reserved.

Unauthorized copying of this file or any element(s) within it, via any medium
is strictly prohibited.
This file contains Habana Labs, Ltd. proprietary and confidential information
and is subject to the confidentiality and license agreements under which it
was provided.
"""

import tensorflow.compat.v1 as tf
import tensorflow.profiler.experimental as profiler


class ProfilerHook(tf.train.SessionRunHook):
    def __init__(self, steps, log_dir) -> None:
        profile_steps = [int(i) for i in steps.split(',')]
        if len(profile_steps) != 2:
            raise ValueError(
                "Step has to be a pair of numbers, got {} instead".format(steps))
        self._step_count = 0
        self._start_step = profile_steps[0]
        self._end_step = profile_steps[1]
        self._log_dir = log_dir

    def after_run(self, _, __):
        if self._step_count == self._end_step:
            profiler.stop()

    def before_run(self, _):
        self._step_count += 1
        if self._step_count == self._start_step:
            profiler.start(self._log_dir)
