# ******************************************************************************
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
# ******************************************************************************

import logging
import os
import sys
from contextlib import contextmanager

import tensorflow as tf
from tensorflow.python.training import training_util
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import timeline
from tensorflow.python.platform import gfile

@contextmanager
def disable_session_recovery():
    """ Disable session recovery on backend errors.

    MonitoredSession that is used by Estimator hard-codes that AbortedError
    and UnavailableError should not terminate but instead silently restart
    training. This constitutes a broad list of c++ backend errors, including
    OOM, that may cause endless error/restart loop.
    """
    from tensorflow.python.training import training
    module= sys.modules['tensorflow.python.training.monitored_session']
    ignored_error_list_attr = "_PREEMPTION_ERRORS"
    orig_ignored_errors = getattr(module, ignored_error_list_attr)
    setattr(module, ignored_error_list_attr, tuple())
    yield
    setattr(module, ignored_error_list_attr, orig_ignored_errors)


class RangeTFProfilerHook(tf.compat.v1.estimator.SessionRunHook):

    def __init__(self,
               start_iter=None,
               num_iters=None,
               output_dir="."):
        self._start_iter=start_iter
        self._end_iter=start_iter+num_iters
        self._curr_iter=0
        self._output_dir=output_dir
        self._metadata=[]

    def before_run(self, run_context):
        self._curr_iter=self._curr_iter+1
        if self._curr_iter > self._start_iter and self._curr_iter <= self._end_iter:
            return tf.estimator.SessionRunArgs(None, options=config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE))
        else:
            return None

    def after_run(self, run_context, run_values):
        if self._curr_iter > self._start_iter and self._curr_iter <= self._end_iter:
            self._metadata.append(run_values.run_metadata.step_stats)

        if self._curr_iter == self._end_iter:
            self._save(self._curr_iter, self._output_dir)
            run_context.request_stop()

    def _save(self, step, save_path):
        logging.info("Saving timeline for %d into '%s'.", step, save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        traces=self._metadata[0]
        for ds in self._metadata[1:]:
            traces.dev_stats.MergeFrom(ds.dev_stats)

        with gfile.Open("{}/tf_trace.json".format(save_path), "w") as f:
            trace = timeline.Timeline(traces)
            f.write(
                trace.generate_chrome_trace_format(show_dataflow=False, show_memory=False))
