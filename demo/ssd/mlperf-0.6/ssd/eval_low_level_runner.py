# Copyright 2018 Google. All Rights Reserved.
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
"""Training SSD with low level API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import threading

from absl import flags

import tensorflow as tf

from tensorflow.contrib import tpu
from tensorflow.contrib.tpu.python.tpu import tpu_function
from tensorflow.python.data.util import nest as data_nest
import utils

FLAGS = flags.FLAGS


class EvalLowLevelRunner(object):
  """Run eval via direct session.run calls."""

  def __init__(self, eval_steps):
    tf.logging.info("EvalLowLevelRunner: constructor")
    tf.logging.info("eval_steps: %s", eval_steps)

    self.feature_structure = {}
    self.infeed_queue = []
    self.enqueue_ops = []
    self.dataset_initializer = []
    self.eval_steps = eval_steps
    self.sess = None
    self.eval_op = None
    self.graph = tf.Graph()
    self.outfeed_tensors = []
    self.outfeed_names = []
    self.dequeue_ops = {}
    self.saver = None
    self.tpu_cluster_resolver = None
    with self.graph.as_default():
      self.tpu_init = [tpu.initialize_system()]
      self.tpu_shutdown = tpu.shutdown_system()

  def _get_host(self, host_id):
    if self.tpu_cluster_resolver.get_master() in ("", "local"):
      return "/replica:0/task:0"
    job_name = self.tpu_cluster_resolver.get_job_name() or "tpu_worker"
    return "/job:%s/task:%d" % (job_name, host_id)

  def initialize(self, input_fn, params):
    """Initialize all the things required for evaluation."""
    tf.logging.info("EvalLowLevelRunner: initialize method")

    self.tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    def get_enqueue_ops_fn():
      """Generate the enqueue ops graph function."""

      with tf.device(utils.device_for_host(self._get_host(0))):
        dataset = input_fn(params)
        iterator = dataset.make_initializable_iterator()
        self.dataset_initializer.append(iterator.initializer)

        def enqueue_ops_fn():
          """Enqueue ops function for one host."""
          per_host_sharded_inputs = []
          control_deps = []
          for _ in range(FLAGS.num_shards_per_host):
            with tf.control_dependencies(control_deps):
              features = iterator.get_next()
            self.feature_structure["features"] = features
            flattened_inputs = data_nest.flatten(self.feature_structure)
            control_deps.extend(flattened_inputs)
            per_host_sharded_inputs.append(flattened_inputs)

          infeed = tpu.InfeedQueue(
              number_of_tuple_elements=len(per_host_sharded_inputs[0]))
          self.infeed_queue.append(infeed)
          return infeed.generate_enqueue_ops(
              per_host_sharded_inputs,
              tpu_ordinal_function=utils.tpu_ordinal_fn)

        return enqueue_ops_fn

    with self.graph.as_default():
      self.enqueue_ops.append(
          utils.wrap_computation_in_while_loop(
              get_enqueue_ops_fn(),
              n=self.eval_steps,
              host_name=self._get_host(0)))

    session_config = tf.ConfigProto(
        allow_soft_placement=True, isolate_session_state=True,
        operation_timeout_in_ms=600 * 60 * 1000)  # 10 hours
    cluster_spec = self.tpu_cluster_resolver.cluster_spec()
    if cluster_spec:
      session_config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
    self.sess = tf.Session(
        self.tpu_cluster_resolver.get_master(),
        graph=self.graph,
        config=session_config)
    if FLAGS.mode != "eval_once":
      self.sess.run(self.tpu_init)

  def build_model(self, model_fn, params):
    """Build the TPU model and infeed enqueue ops."""
    tf.logging.info("EvalLowLevelRunner: build_model method")

    def tpu_eval_step():
      """Generate the TPU graph."""
      values = self.infeed_queue[0].generate_dequeue_op(tpu_device=0)
      unflattened_inputs = data_nest.pack_sequence_as(self.feature_structure,
                                                      values)
      features = unflattened_inputs["features"]
      estimator_spec = model_fn(features, None, tf.estimator.ModeKeys.PREDICT,
                                params)
      for k, v in six.iteritems(estimator_spec.predictions):
        self.outfeed_names.append(k)
        self.outfeed_tensors.append(v)

      with tf.device(utils.device_for_tpu_core(self._get_host(0))):
        outfeed_enqueue_ops = tpu.outfeed_enqueue_tuple(self.outfeed_tensors)
      with tf.control_dependencies([outfeed_enqueue_ops]):
        return tf.no_op()

    @tpu_function.on_device_training_loop
    def eval_loop():
      return tpu.repeat(self.eval_steps, tpu_eval_step, [])

    def create_dequeue_ops():
      """Create outfeed dequeue ops."""
      dequeue_ops = []
      tensor_dtypes = []
      tensor_shapes = []
      for v in self.outfeed_tensors:
        dequeue_ops.append([])
        tensor_dtypes.append(v.dtype)
        tensor_shapes.append(v.shape)
      for i in range(FLAGS.num_shards):
        with tf.device(utils.device_for_host(self._get_host(0))):
          outfeed_tensors = tpu.outfeed_dequeue_tuple(
              dtypes=tensor_dtypes, shapes=tensor_shapes, device_ordinal=i)
          for j, item in enumerate(outfeed_tensors):
            dequeue_ops[j].append(item)
      for j in range(len(outfeed_tensors)):
        dequeue_ops[j] = tf.concat(dequeue_ops[j], axis=0)
      return dequeue_ops

    with self.graph.as_default():
      (self.eval_op,) = tpu.shard(
          eval_loop,
          inputs=[],
          num_shards=FLAGS.num_shards,
          outputs_from_all_shards=False,
      )

      for i, dequeue_tenor in enumerate(create_dequeue_ops()):
        self.dequeue_ops[self.outfeed_names[i]] = dequeue_tenor

      self.saver = tf.train.Saver()

  def predict(self, checkpoint_path=None):
    """Run the predict loop on the TPU device."""
    if not checkpoint_path:
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.model_dir)

    self.saver.restore(self.sess, checkpoint_path)
    # Initialize dataset variables
    self.sess.run(self.dataset_initializer)

    if FLAGS.mode == "eval_once":
      self.sess.run(self.tpu_init)

    # Infeed thread.
    def infeed_thread_fn(sess, enqueue_ops):
      sess.run([enqueue_ops])

    infeed_thread = threading.Thread(
        target=infeed_thread_fn, args=(self.sess, self.enqueue_ops))
    infeed_thread.start()

    # Eval thread.
    def eval_thread_fn(sess, eval_op):
      sess.run([eval_op])

    eval_thread = threading.Thread(
        target=eval_thread_fn, args=(self.sess, self.eval_op))
    eval_thread.start()

    for step in range(self.eval_steps):
      tf.logging.info("EvalLowLevelRunner: start eval step:%d", step)
      predictions = self.sess.run(self.dequeue_ops)
      for i in range(FLAGS.eval_batch_size):
        yield {key: value[i] for key, value in six.iteritems(predictions)}

    infeed_thread.join()
    eval_thread.join()
