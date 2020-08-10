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

import math
import threading
import time
from absl import flags
import numpy as np
from six.moves import queue as Queue

import tensorflow as tf

from tensorflow.contrib import tpu
from tensorflow.contrib.tpu.python.tpu import device_assignment as tpu_device_assignment
from tensorflow.contrib.tpu.python.tpu import tpu_feed
from tensorflow.contrib.tpu.python.tpu import tpu_function
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.data.util import nest as data_nest
from tensorflow.python.framework import graph_io
import ssd_constants
import utils

FLAGS = flags.FLAGS

_INITIAL_LOSS = 1e7
_STOP = -1
# for spatial partition
_NUM_CORES_TO_COMPUTATION_SHAPE = {
    1: [1, 1, 1],
    2: [1, 1, 2],
    4: [1, 2, 2],
    8: [2, 2, 2],
    16: [4, 2, 2],
}


class TrainLowLevelRunner(object):
  """Run Train via direct session.run calls."""

  def __init__(self,
               iterations,
               num_cores_per_shard=1,
               input_partition_dims=None):
    tf.logging.info("TrainLowLevelRunner: constructor")

    self.feature_structure = {}
    self.loss = None
    self.infeed_queue = []
    self.enqueue_ops = []
    self.dataset_initializer = []
    self.iterations = iterations
    # TODO(wangtao): change FLAGS.num_shards_per_host to
    # FLAGS.num_cores_per_host after other low level API
    # support spatial partition. FLAGS.num_shards_per_host means number of TPU
    # cores for each host.
    self.replicas_per_worker = FLAGS.num_shards_per_host // num_cores_per_shard
    self.num_hosts = FLAGS.num_shards * num_cores_per_shard // FLAGS.num_shards_per_host
    self.num_shards = FLAGS.num_shards
    self.scaffold_fn = None
    # Having two separate sessions and graphs to make the initialization faster.
    self.input_sess = None
    self.train_sess = None
    self.input_graph = tf.Graph()
    self.train_graph = None
    self.tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    # Disable grappler for better performance.
    self.session_config = tf.ConfigProto(
        allow_soft_placement=True,
        graph_options=tf.GraphOptions(
            rewrite_options=rewriter_config_pb2.RewriterConfig(
                disable_meta_optimizer=True)),
        isolate_session_state=True)
    cluster_spec = self.tpu_cluster_resolver.cluster_spec()
    if cluster_spec:
      self.session_config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
    self.tpu_init = tpu.initialize_system()
    self.tpu_shutdown = tpu.shutdown_system()
    self.init_sess = tf.Session(self.tpu_cluster_resolver.get_master(),
                                config=self.session_config)
    self.queue = Queue.Queue()

    # Init for spatial partitioning.
    self.device_topology = self.init_sess.run(self.tpu_init)
    self.input_partition_dims = input_partition_dims
    self.use_spatial_partition = (
        input_partition_dims is not None and
        int(np.prod(FLAGS.input_partition_dims)) > 1)
    self.num_cores_per_shard = num_cores_per_shard
    if self.use_spatial_partition:
      computation_shape = _NUM_CORES_TO_COMPUTATION_SHAPE[
          self.num_cores_per_shard]
      self.device_assignment = tpu_device_assignment.device_assignment(
          topology=self.device_topology,
          computation_shape=computation_shape,
          num_replicas=self.num_shards)
      tf.logging.info("num_cores_per_shard: %d", self.num_cores_per_shard)
      tf.logging.info("num_hosts: %d", self.num_hosts)
      tf.logging.info("replicas_per_worker: %d", self.replicas_per_worker)
      tf.logging.info("computation_shape: %s", str(computation_shape))
      tf.logging.info("num_shards: %d", self.num_shards)
      tf.logging.info("device_assignment.topology.device_coordinates: %s",
                      str(self.device_assignment.topology.device_coordinates))
      tf.logging.info("device_assignment.core_assignment: %s",
                      str(self.device_assignment.core_assignment))
    else:
      self.device_assignment = None

  def shutdown(self):
    """Shut down TrainLowLevelRunner."""
    tf.logging.info("TrainLowLevelRunner: shutdown")
    self.queue.put(_STOP)
    self.infeed_thread.join()
    self.input_sess.close()
    self.train_sess.close()

  def _get_host(self, host_id):
    if self.tpu_cluster_resolver.get_master() in ("", "local"):
      return "/replica:0/task:0"
    job_name = self.tpu_cluster_resolver.get_job_name() or "tpu_worker"
    return "/job:%s/task:%d" % (job_name, host_id)

  def build_enqueue_ops(self, input_fn, params, host_id):
    """Build enqueue ops."""
    tf.logging.info("TrainLowLevelRunner: build_enqueue_ops")

    def get_enqueue_ops_fn(host_id):
      """Generate the enqueue ops graph function."""

      params["dataset_num_shards"] = self.num_hosts
      params["dataset_index"] = host_id
      with tf.device(utils.device_for_host(self._get_host(host_id))):
        dataset = input_fn(params)
        iterator = dataset.make_initializable_iterator()
        self.dataset_initializer.append(iterator.initializer)

        def enqueue_ops_fn():
          """Enqueue ops function for one host."""
          per_host_sharded_inputs = []
          control_deps = []
          for _ in range(self.replicas_per_worker):
            with tf.control_dependencies(control_deps):
              features, labels = iterator.get_next()
            if self.use_spatial_partition:
              num_elements = []
              for i, d in enumerate(ssd_constants.FEATURE_SIZES):
                num_elements.append(d * d * ssd_constants.NUM_DEFAULTS[i])
              gt_boxes = tf.split(labels[ssd_constants.BOXES], num_elements, 1)
              gt_classes = tf.split(labels[ssd_constants.CLASSES], num_elements,
                                    1)

              def transpose_gt_box(gt_box, i):
                return tf.transpose(
                    tf.reshape(gt_box, [
                        -1, ssd_constants.NUM_DEFAULTS[i],
                        ssd_constants.FEATURE_SIZES[i],
                        ssd_constants.FEATURE_SIZES[i], 4
                    ]), [0, 2, 3, 1, 4])

              def transpose_gt_class(gt_class, i):
                return tf.transpose(
                    tf.reshape(gt_class, [
                        -1, ssd_constants.NUM_DEFAULTS[i],
                        ssd_constants.FEATURE_SIZES[i],
                        ssd_constants.FEATURE_SIZES[i]
                    ]), [0, 2, 3, 1])

              labels[ssd_constants.BOXES] = {
                  i: transpose_gt_box(gt_boxes[i], i)
                  for i in range(len(ssd_constants.NUM_DEFAULTS))
              }
              labels[ssd_constants.CLASSES] = {
                  i: transpose_gt_class(gt_classes[i], i)
                  for i in range(len(ssd_constants.NUM_DEFAULTS))
              }
            self.feature_structure["features"] = features
            self.feature_structure["labels"] = labels
            flattened_inputs = data_nest.flatten(self.feature_structure)
            control_deps.extend(flattened_inputs)
            per_host_sharded_inputs.append(flattened_inputs)

          if self.use_spatial_partition:
            flattened_input_dims = []
            for i in per_host_sharded_inputs[0]:
              if i.shape.ndims >= len(self.input_partition_dims[0]):
                if i.shape.as_list(
                ) == self.feature_structure["features"].shape.as_list():
                  flattened_input_dims.append(self.input_partition_dims[0])
                else:
                  flattened_input_dims.append(
                      FLAGS.input_partition_dims + [1] *
                      (i.shape.ndims - len(self.input_partition_dims[0])))
              else:
                flattened_input_dims.append([1] * i.shape.ndims)
            # pylint: disable=protected-access
            infeed = tpu_feed._PartitionedInfeedQueue(
                number_of_tuple_elements=len(per_host_sharded_inputs[0]),
                host_id=host_id,
                input_partition_dims=flattened_input_dims,
                device_assignment=self.device_assignment)
            self.infeed_queue.append(infeed)
            return infeed.generate_enqueue_ops(per_host_sharded_inputs)

          infeed = tpu.InfeedQueue(
              number_of_tuple_elements=len(per_host_sharded_inputs[0]))
          self.infeed_queue.append(infeed)
          return infeed.generate_enqueue_ops(
              per_host_sharded_inputs,
              tpu_ordinal_function=utils.tpu_ordinal_fn)

        return enqueue_ops_fn

    with self.input_graph.as_default():
      self.enqueue_ops.append(
          utils.wrap_computation_in_while_loop(
              get_enqueue_ops_fn(host_id),
              n=self.iterations,
              host_name=self._get_host(host_id)))

  def initialize(self, input_fn, model_fn, params):
    """Build graph and do initialization for training."""
    tf.logging.info("TrainLowLevelRunner: initialize method")

    for i in range(self.num_hosts):
      self.build_enqueue_ops(input_fn, params, host_id=i)

    def infeed_thread_fn():
      """Build and infeed session.run calls in a background thread."""
      # Build infeed sesssion
      self.input_sess = tf.Session(
          self.tpu_cluster_resolver.get_master(),
          graph=self.input_graph,
          config=self.session_config)
      # Initialize dataset variables
      self.input_sess.run(self.dataset_initializer)
      # Run infeed session.run calls
      while True:
        iterations = self.queue.get(block=True)
        if iterations == _STOP:
          return
        tf.logging.info("Start to infeed %d batches", iterations)
        self.input_sess.run([self.enqueue_ops])

    def tpu_train_step(loss):
      """Generate the TPU graph."""
      del loss
      values = self.infeed_queue[0].generate_dequeue_op(tpu_device=0)
      unflattened_inputs = data_nest.pack_sequence_as(self.feature_structure,
                                                      values)
      features = unflattened_inputs["features"]
      labels = unflattened_inputs["labels"]
      estimator_spec = model_fn(features, labels, tf.estimator.ModeKeys.TRAIN,
                                params)
      loss, train_op = estimator_spec.loss, estimator_spec.train_op
      self.scaffold_fn = estimator_spec.scaffold_fn
      with tf.control_dependencies([train_op]):
        return tf.identity(loss)

    @tpu_function.on_device_training_loop
    def train_loop():
      return tpu.repeat(self.iterations, tpu_train_step, [_INITIAL_LOSS])

    self.train_graph = tf.Graph()
    with self.train_graph.as_default():
      (self.loss,) = tpu.shard(
          train_loop,
          inputs=[],
          num_shards=self.num_shards,
          outputs_from_all_shards=False,
          device_assignment=self.device_assignment,
      )
      if self.scaffold_fn:
        self.scaffold_fn()
      global_initializer = tf.global_variables_initializer()
      local_initializer = tf.local_variables_initializer()
      graph_io.write_graph(
          self.input_graph.as_graph_def(add_shapes=True), FLAGS.model_dir,
          "input_graph.pbtxt")
      graph_io.write_graph(
          self.train_graph.as_graph_def(add_shapes=True), FLAGS.model_dir,
          "graph.pbtxt")
      self.saver = tf.train.Saver()

    # Build tpu train model session and initialize graph
    self.train_sess = tf.Session(
        self.tpu_cluster_resolver.get_master(),
        graph=self.train_graph,
        config=self.session_config)

    self.train_sess.run(global_initializer)
    self.train_sess.run(local_initializer)

    # Complete infeed graph generation and session.run calls
    self.infeed_thread = threading.Thread(target=infeed_thread_fn)
    self.infeed_thread.start()

  def train(self, train_steps, base_step=0, num_threads=2):
    """Run the Train loop on the TPU device."""
    tf.logging.info("TrainLowLevelRunner: train for %d steps in total",
                    train_steps)
    if train_steps % self.iterations != 0:
      tf.logging.warning(
          "train_steps %d is not divisible by iterations_per_loop %d",
          train_steps, self.iterations)
      train_steps = self.iterations * int(
          math.ceil(train_steps / self.iterations))

    def checkpoint_thread_fn(saver, sess):
      saver.save(sess,
                 FLAGS.model_dir + "/model.ckpt-%d" % (cur_step + base_step))

    cur_step = 0
    thread_id = 0
    checkpoint_threads = []
    for i in range(num_threads):
      checkpoint_threads.append(None)

    while cur_step < train_steps:
      start = time.time()
      tf.logging.info("TrainLowLevelRunner: start train step:%d", cur_step)
      self.queue.put(self.iterations)
      cur_step += self.iterations

      loss = self.train_sess.run([self.loss])
      tf.logging.info("TrainLowLevelRunner: sess run loss: %s", loss)

      if checkpoint_threads[thread_id] is not None:
        checkpoint_threads[thread_id].join()
      checkpoint_threads[thread_id] = threading.Thread(
          target=checkpoint_thread_fn, args=(self.saver, self.train_sess))
      checkpoint_threads[thread_id].start()
      thread_id += 1
      if thread_id >= num_threads:
        thread_id = 0

      end = time.time()
      tf.logging.info(
          "TrainLowLevelRunner: step time {} sec {} examples/sec".format(
              end - start,
              self.iterations * FLAGS.train_batch_size / (end - start)))

    for i in range(num_threads):
      if checkpoint_threads[i] is not None:
        checkpoint_threads[i].join()
        checkpoint_threads[i] = None
