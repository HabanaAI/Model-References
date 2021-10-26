import time
import numpy as np
import tensorflow as tf


class PerfLoggingHook(tf.compat.v1.train.SessionRunHook):

  def __init__(self, batch_size, mode):
    self.batch_size = batch_size
    self.mode = mode
    self.throughput = []
    self.iter_times = []

  def begin(self):
    self.t0 = time.time()

  def before_run(self, run_context):
    run_args = tf.compat.v1.train.SessionRunArgs(
        fetches=[
            tf.compat.v1.train.get_global_step()
        ]
    )
    self.iter_t0 = time.time()
    return run_args

  def after_run(self, run_context, run_values):
    global_step = run_values.results
    iter_time = time.time() - self.iter_t0
    self.iter_times.append(iter_time)
    iter_throughput = self.batch_size / iter_time
    self.throughput.append(iter_throughput)

  def end(self, session):
    total_time = time.time() - self.t0
    avg_throughput = np.mean(self.throughput)
    avg_throughput_first_ommited = np.mean(self.throughput[1:])
    tf.compat.v1.logging.info("***** %s statistics *****", self.mode)
    tf.compat.v1.logging.info("  Time = %0.2f [s]", total_time)
    tf.compat.v1.logging.info("  First iteration time = %0.2f ", self.iter_times[0])
    tf.compat.v1.logging.info("  Total batch size = %d", self.batch_size)
    tf.compat.v1.logging.info("  Avg total throughput = %0.2f [examples/sec]", avg_throughput)
    tf.compat.v1.logging.info("  Avg total (first ommited) = %0.2f [examples/sec]", avg_throughput_first_ommited)
