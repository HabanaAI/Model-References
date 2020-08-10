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
"""Training and Eval SSD with low level API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import multiprocessing
import os
import threading
import time
from absl import flags
import numpy as np
import six

import tensorflow as tf

from tensorflow.contrib.tpu.python.tpu import device_assignment as tpu_device_assignment
from tensorflow.contrib.tpu.python.tpu import tpu_feed
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.data.util import nest as data_nest
from tensorflow.python.framework import graph_io
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import training_loop
from tensorflow.python.tpu.ops import tpu_ops
from mlp_log import mlp_log
import coco_metric
import ssd_constants
import utils

FLAGS = flags.FLAGS

_INITIAL_LOSS = 1e7
_STOP = -1
_ITEM = 1
# for spatial partition
_NUM_CORES_TO_COMPUTATION_SHAPE = {
    1: [1, 1, 1],
    2: [1, 1, 2],
    4: [1, 2, 2],
    8: [2, 2, 2],
    16: [4, 2, 2],
}


# Decorator function for tpu computation func that was passed to tpu.rewrite()
# if there are embedded train and eval loops in this func, trace tools will
# generate step markers for each iteration.
def on_device_train_and_eval_loops(func):
  # Value for this attribute is from xla.DebugOptions.StepMarkerLocation.
  setattr(func, "step_marker_location", "STEP_MARK_AT_SECOND_LEVEL_WHILE_LOOP")
  return func


def predict_post_processing(q_in, q_out):
  """Run post-processing on CPU for predictions."""
  coco_gt = coco_metric.create_coco(FLAGS.val_json_file, use_cpp_extension=True)

  current_step, predictions = q_in.get()
  while current_step != _STOP and q_out is not None:
    tf.logging.info("Start to predict for step %d.", current_step)
    q_out.put((current_step,
               coco_metric.compute_map(
                   predictions,
                   coco_gt,
                   use_cpp_extension=True,
                   nms_on_tpu=True)))
    current_step, predictions = q_in.get()


class TrainAndEvalLowLevelRunner(object):
  """Run Train and Eval via direct session.run calls.

     Training weights will remain on the device and TPU will
     use the weights on the device to do eval/predict without restoring from
     the checkpoint.
  """

  def __init__(self,
               iterations,
               eval_steps,
               sleep_seconds=120,
               num_multiprocessing_workers=ssd_constants.WORKER_COUNT,
               num_cores_per_shard=1,
               input_partition_dims=None):
    tf.logging.info("TrainAndEvalLowLevelRunner: constructor")

    self.eval_steps = eval_steps
    self.feature_structure = {}
    self.eval_feature_structure = {}
    self.loss = None
    self.infeed_queue = []
    self.eval_infeed_queue = []
    self.enqueue_ops = []
    self.dequeue_ops = []
    self.predictions = {}
    self.eval_enqueue_ops = []
    self.train_eval_compile_op = None
    self.dataset_initializer = []
    self.eval_dataset_initializer = []
    self.iterations = iterations
    # TODO(wangtao): change FLAGS.num_shards_per_host to
    # FLAGS.num_cores_per_host after other low level API
    # support spatial partition. FLAGS.num_shards_per_host means number of TPU
    # cores for each host.
    self.replicas_per_worker = FLAGS.num_shards_per_host // num_cores_per_shard
    self.num_hosts = FLAGS.num_shards * num_cores_per_shard // FLAGS.num_shards_per_host
    self.num_shards = FLAGS.num_shards
    self.scaffold_fn = None
    self.sess = None
    self.input_sess = None
    self.graph = tf.Graph()
    self.input_graph = tf.Graph()
    self.eval_op = None
    self.infeed_thread = None
    self.eval_epochs = []
    self.success_epoch = 1000
    self.log_epochs = {}
    self.params = {}
    self.train_loop = None
    self.tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name or FLAGS.master,
        zone=FLAGS.tpu_zone,
        project=FLAGS.gcp_project)
    # Disable grappler for better performance.
    self.session_config = tf.ConfigProto(
        allow_soft_placement=True,
        graph_options=tf.GraphOptions(
            rewrite_options=rewriter_config_pb2.RewriterConfig(
                disable_meta_optimizer=True)),
        isolate_session_state=True,
        operation_timeout_in_ms=600 * 60 * 1000)  # 10 hours
    cluster_spec = self.tpu_cluster_resolver.cluster_spec()
    if cluster_spec:
      self.session_config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
    self.tpu_init = tpu.initialize_system()
    self.tpu_shutdown = tpu.shutdown_system()
    self.master = self.tpu_cluster_resolver.get_master()
    self.init_sess = tf.Session(self.master, config=self.session_config)
    self.outfeed_tensors = []
    self.outfeed_names = []
    self.run_success = False
    self.log_run_success = False
    self.num_multiprocessing_workers = num_multiprocessing_workers

    # Figure out the steps and epochs to eval for MLPerf.
    self.eval_at_steps = np.cumsum(ssd_constants.EVAL_STEPS).tolist()
    self.eval_iterations = [steps // 20000 - 1 for steps in self.eval_at_steps]
    self.max_train_iterations = int(
        math.ceil(FLAGS.num_epochs * FLAGS.num_examples_per_epoch /
                  (FLAGS.train_batch_size * self.iterations)))
    self.sleep_seconds = sleep_seconds

    tf.logging.info("eval_at_steps: %s", self.eval_at_steps)
    tf.logging.info("eval_iterations: %s", self.eval_iterations)

    # Init for spatial partitioning.
    self.device_topology = self.init_sess.run(self.tpu_init)
    self.input_partition_dims = [input_partition_dims, None]
    self.use_spatial_partition = (
        input_partition_dims is not None and
        int(np.prod(FLAGS.input_partition_dims)) > 1)
    self.use_spatial_partition = input_partition_dims is not None
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
      eval_input_partition_dims = [{
          ssd_constants.BOXES: None,
          ssd_constants.CLASSES: None,
          ssd_constants.IMAGE: input_partition_dims,
          ssd_constants.RAW_SHAPE: None,
          ssd_constants.SOURCE_ID: None,
      }, None]
      if FLAGS.eval_batch_size * eval_steps > FLAGS.eval_samples:
        eval_input_partition_dims[0][ssd_constants.IS_PADDED] = None
      self.eval_input_dims_flattener = utils.InputDimsFlattener(
          eval_input_partition_dims)
    else:
      self.device_assignment = None
      self.eval_input_dims_flattener = None

  def shutdown(self):
    """Shut down TrainLowLevelRunner."""
    tf.logging.info("TrainAndEvalLowLevelRunner: shutdown")
    self.infeed_thread.join()
    self.input_sess.close()
    self.sess.close()

  def _get_host(self, host_id):
    if self.master in ("", "local"):
      return "/replica:0/task:0"
    job_name = self.tpu_cluster_resolver.get_job_name() or "tpu_worker"
    return "/job:%s/task:%d" % (job_name, host_id)

  def build_enqueue_ops(self, input_fn, params, host_id, is_training=True):
    """Build enqueue ops for training."""
    tf.logging.info("TrainAndEvalLowLevelRunner: build_enqueue_ops")

    def get_enqueue_ops_fn(host_id):
      """Generate the enqueue ops graph function."""

      params["dataset_num_shards"] = self.num_hosts
      params["dataset_index"] = host_id
      with tf.device(utils.device_for_host(self._get_host(host_id))):
        dataset = input_fn(params)
        iterator = dataset.make_initializable_iterator()
        if is_training:
          self.dataset_initializer.append(iterator.initializer)
        else:
          self.eval_dataset_initializer.append(iterator.initializer)

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

              # TODO(dehao): This causes 3s overhead in startup, fix it.
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

          infeed = tpu_feed.InfeedQueue(
              number_of_tuple_elements=len(per_host_sharded_inputs[0]))
          self.infeed_queue.append(infeed)
          return infeed.generate_enqueue_ops(
              per_host_sharded_inputs,
              tpu_ordinal_function=utils.tpu_ordinal_fn)

        return enqueue_ops_fn

    def get_eval_enqueue_ops_fn(host_id):
      """Generate the eval enqueue ops graph function."""

      params["dataset_num_shards"] = self.num_hosts
      params["dataset_index"] = host_id
      with tf.device(utils.device_for_host(self._get_host(host_id))):
        dataset = input_fn(params)
        iterator = dataset.make_initializable_iterator()
        self.eval_dataset_initializer.append(iterator.initializer)

        def eval_enqueue_ops_fn():
          """Enqueue ops function for one host."""
          per_host_sharded_inputs = []
          control_deps = []
          for _ in range(self.replicas_per_worker):
            with tf.control_dependencies(control_deps):
              features = iterator.get_next()
            if self.use_spatial_partition:
              self.eval_input_dims_flattener.validate_and_flatten_input_dims(
                  features, None)
            self.eval_feature_structure["features"] = features
            flattened_inputs = data_nest.flatten(self.eval_feature_structure)
            control_deps.extend(flattened_inputs)
            per_host_sharded_inputs.append(flattened_inputs)

          if self.use_spatial_partition:
            flattened_input_dims = (
                self.eval_input_dims_flattener.flattened_input_dims)
            # pylint: disable=protected-access
            infeed = tpu_feed._PartitionedInfeedQueue(
                number_of_tuple_elements=len(per_host_sharded_inputs[0]),
                host_id=host_id,
                input_partition_dims=flattened_input_dims,
                device_assignment=self.device_assignment)
            self.eval_infeed_queue.append(infeed)
            return infeed.generate_enqueue_ops(per_host_sharded_inputs)

          infeed = tpu_feed.InfeedQueue(
              number_of_tuple_elements=len(per_host_sharded_inputs[0]))
          self.eval_infeed_queue.append(infeed)
          return infeed.generate_enqueue_ops(
              per_host_sharded_inputs,
              tpu_ordinal_function=utils.tpu_ordinal_fn)

        return eval_enqueue_ops_fn

    if is_training:
      num_iterations = self.iterations
    else:
      num_iterations = self.eval_steps

    with self.input_graph.as_default():
      enqueue_op = utils.wrap_computation_in_while_loop(
          get_enqueue_ops_fn(host_id)
          if is_training else get_eval_enqueue_ops_fn(host_id),
          n=num_iterations,
          host_name=self._get_host(host_id))
      if is_training:
        self.enqueue_ops.append(enqueue_op)
      else:
        self.eval_enqueue_ops.append(enqueue_op)

  def initialize(self, input_fn, eval_input_fn, model_fn, params):
    """Build graph and do initialization for training."""
    tf.logging.info("TrainAndEvalLowLevelRunner: initialize method")
    mlp_log.mlperf_print("init_start", None)

    self.params = params
    self.build_enqueue_ops(input_fn, params, host_id=0)

    def infeed_thread_fn():
      """Build and infeed session.run calls in a background thread."""
      # Initialize dataset variables
      for i in range(self.max_train_iterations):
        tf.logging.info("TrainAndEvalRunner: start infeed for %d steps",
                        self.iterations)
        self.input_sess.run([self.enqueue_ops])
        if self.params["eval_every_checkpoint"] or i in self.eval_iterations:
          self.input_sess.run(self.eval_dataset_initializer)
          self.input_sess.run([self.eval_enqueue_ops])

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

    def train_loop():
      return training_loop.repeat(self.iterations, tpu_train_step,
                                  [_INITIAL_LOSS])

    # Start the build of the train graph.
    self.train_loop = train_loop

    for i in range(1, self.num_hosts):
      self.build_enqueue_ops(input_fn, params, host_id=i)

    # Init for eval.
    self.initialize_eval(eval_input_fn, model_fn, params)

    with self.graph.as_default():
      if self.scaffold_fn:
        self.scaffold_fn()
      global_initializer = tf.global_variables_initializer()
      local_initializer = tf.local_variables_initializer()
      graph_io.write_graph(
          self.graph.as_graph_def(add_shapes=True), FLAGS.model_dir,
          "graph.pbtxt")

    # Build tpu train model session and initialize graph
    self.sess = tf.Session(
        self.master, graph=self.graph, config=self.session_config)
    self.input_sess = tf.Session(
        self.master, graph=self.input_graph, config=self.session_config)

    self.sess.run(global_initializer)
    self.sess.run(local_initializer)
    self.input_sess.run(self.dataset_initializer)
    self.input_sess.run(self.eval_dataset_initializer)

    # Complete infeed graph generation.
    self.infeed_thread = threading.Thread(target=infeed_thread_fn)
    # Compile.
    self.sess.run([self.train_eval_compile_op])

  def initialize_eval(self, input_fn, model_fn, params):
    """Initialize all the things required for evaluation."""
    tf.logging.info("TrainAndEvalLowLevelRunner: initialize eval method")

    self.build_enqueue_ops(input_fn, params, host_id=0, is_training=False)

    self.build_eval_model(model_fn, params)

    for i in range(1, self.num_hosts):
      self.build_enqueue_ops(input_fn, params, host_id=i, is_training=False)

  def build_eval_model(self, model_fn, params):
    """Build the Eval TPU model and infeed enqueue ops."""
    tf.logging.info("TrainAndEvalLowLevelRunner: build_model method")

    # TODO(wangtao): refactor to extract common logic with tpu_train_step.
    def tpu_eval_step():
      """Generate the TPU graph."""
      values = self.eval_infeed_queue[0].generate_dequeue_op(tpu_device=0)
      unflattened_inputs = data_nest.pack_sequence_as(
          self.eval_feature_structure, values)
      features = unflattened_inputs["features"]
      estimator_spec = model_fn(features, None, tf.estimator.ModeKeys.PREDICT,
                                params)
      for k, v in six.iteritems(estimator_spec.predictions):
        self.outfeed_names.append(k)
        self.outfeed_tensors.append(v)

      with tf.device(utils.device_for_tpu_core(self._get_host(0))):
        outfeed_enqueue_ops = tpu_ops.outfeed_enqueue_tuple(
            self.outfeed_tensors)
      with tf.control_dependencies([outfeed_enqueue_ops]):
        return tf.no_op()

    def eval_loop():
      return training_loop.repeat(self.eval_steps, tpu_eval_step, [])

    def train_eval_step(iteration):
      with tf.control_dependencies(self.train_loop()):
        should_eval = tf.reduce_any(
            tf.equal(tf.constant(self.eval_iterations), iteration))
        should_eval = tf.logical_or(
            should_eval, tf.constant(self.params["eval_every_checkpoint"]))
        ops = tf.cond(should_eval, lambda: eval_loop(), lambda: tf.no_op())  # pylint: disable=unnecessary-lambda
        with tf.control_dependencies([ops]):
          return iteration + 1

    @on_device_train_and_eval_loops
    def train_eval_loop():
      return training_loop.repeat(self.max_train_iterations, train_eval_step,
                                  [0])

    self.eval_epochs = [
        steps * ssd_constants.DEFAULT_BATCH_SIZE / FLAGS.train_batch_size //
        params["steps_per_epoch"] for steps in self.eval_at_steps
    ]

    self.log_epochs = dict(
        zip(self.eval_epochs, [False for _ in self.eval_epochs]))

    self.epoch_count = dict(
        zip(self.eval_epochs,
            [self.eval_epochs[0]] + np.diff(self.eval_epochs).tolist()))

    # TODO(wangtao): refactor to extract common logic
    # with train create_dequeu_ops.
    def create_dequeue_ops(host_id):
      """Create outfeed dequeue ops."""
      dequeue_ops = []
      tensor_dtypes = []
      tensor_shapes = []
      for v in self.outfeed_tensors:
        tensor_dtypes.append(v.dtype)
        tensor_shapes.append(v.shape)
      with tf.device(utils.device_for_host(self._get_host(host_id))):
        for i in range(self.replicas_per_worker):
          if self.use_spatial_partition:
            replica_id = self.device_assignment.lookup_replicas(host_id, 0)[i]
            ordinal = self.device_assignment.tpu_ordinal(
                replica=replica_id, logical_core=0)
          else:
            ordinal = i
          outfeed = tpu_ops.outfeed_dequeue_tuple(
              dtypes=tensor_dtypes,
              shapes=tensor_shapes,
              device_ordinal=ordinal)
          if len(outfeed) == 2:
            # 2 outfeed tensors
            #   is_pad: [batch]
            #   detections: [batch, 200, 7]
            if outfeed[0].shape.ndims == 3:
              detections, is_pad = outfeed
            else:
              is_pad, detections = outfeed
            num_non_pad = tf.shape(is_pad)[0] - tf.reduce_sum(
                tf.cast(is_pad, tf.int32))
            dequeue_ops.append(
                tf.slice(detections, [0, 0, 0], [num_non_pad, -1, -1]))
          else:
            # no padding, only detections are in the outfeed
            dequeue_ops.append(outfeed)
        dequeue_ops = tf.concat(dequeue_ops, axis=0)
      return dequeue_ops

    with self.graph.as_default():
      (
          self.train_eval_compile_op,
          self.train_eval_op,
      ) = tpu.split_compile_and_shard(
          train_eval_loop,
          inputs=[],
          num_shards=self.num_shards,
          outputs_from_all_shards=False,
          device_assignment=self.device_assignment,
      )

      # Get dequeue ops from each hosts.
      for i in range(self.num_hosts):
        self.dequeue_ops.append(create_dequeue_ops(i))

  def predict(self):
    """Run the predict loop on the TPU device."""
    if FLAGS.train_batch_size != FLAGS.eval_batch_size:
      raise RuntimeError(
          "train batch size should be equal to eval batch size for in memory eval."
      )
    ret = []
    for i in range(self.eval_steps):
      tf.logging.info("TrainAndEvalRunner: predict step %d.", i)
      ret.append(self.sess.run(self.dequeue_ops))
    return ret

  def train_and_eval(self, train_steps):
    """Run the Train and Eval loop on the TPU device."""
    output_dir = os.path.join(FLAGS.model_dir, "eval")
    tf.gfile.MakeDirs(output_dir)
    # Summary writer writes out eval metrics.
    summary_writer = tf.summary.FileWriter(output_dir)
    self.run_success = False

    def log_eval_result_fn(results):
      """Log eval results."""
      cur_step, eval_results = results
      if cur_step == _STOP:
        return
      epoch = cur_step // self.params["steps_per_epoch"]
      with tf.Graph().as_default():
        summaries = []
        for metric in eval_results:
          summaries.append(
              tf.Summary.Value(tag=metric, simple_value=eval_results[metric]))
          tf_summary = tf.Summary(value=list(summaries))
          summary_writer.add_summary(tf_summary, cur_step)
        mlp_log.mlperf_print(
            "eval_accuracy",
            eval_results["COCO/AP"],
            metadata={"epoch_num": epoch + 1})
        mlp_log.mlperf_print(
            "eval_stop", None, metadata={"epoch_num": epoch + 1})

        if epoch in self.epoch_count:
          epoch_count = self.epoch_count[epoch]
        else:
          epoch_count = 1

        mlp_log.mlperf_print(
            "block_stop",
            None,
            metadata={
                "first_epoch_num": epoch - epoch_count + 1,
                "epoch_count": epoch_count
            })

        self.log_epochs[epoch] = True
        if eval_results["COCO/AP"] >= ssd_constants.EVAL_TARGET:
          self.run_success = True
          if epoch < self.success_epoch:
            self.success_epoch = epoch
        log_run_final = self.run_success
        for epoch in self.log_epochs:
          if epoch < self.success_epoch and not self.log_epochs[epoch]:
            log_run_final = False
            break
        # Log run_final when all the previous eval results are logged.
        if log_run_final and not self.log_run_success:
          mlp_log.mlperf_print("run_stop", None, metadata={"status": "success"})
          self.log_run_success = True

    tf.logging.info("TrainAndEvalLowLevelRunner: train for %d steps in total",
                    train_steps)
    if train_steps % self.iterations != 0:
      tf.logging.warning(
          "train_steps %d is not divisible by iterations_per_loop %d",
          train_steps, self.iterations)
      train_steps = self.iterations * int(
          math.ceil(train_steps / self.iterations))

    # Start train and eval op on the background.
    def train_eval_thread_fn(sess, train_eval_op):
      sess.run([train_eval_op])

    train_eval_thread = threading.Thread(
        target=train_eval_thread_fn, args=(self.sess, self.train_eval_op))
    train_eval_thread.start()

    # pylint: disable=line-too-long
    q_in = multiprocessing.Queue(maxsize=ssd_constants.QUEUE_SIZE)
    q_out = multiprocessing.Queue(maxsize=ssd_constants.QUEUE_SIZE)
    processes = [
        multiprocessing.Process(target=predict_post_processing, args=(q_in, q_out))
        for _ in range(self.num_multiprocessing_workers)
    ]
    # pylint: enable=line-too-long

    time.sleep(self.sleep_seconds)
    mlp_log.mlperf_print("init_stop", None)
    mlp_log.mlperf_print("run_start", None)

    for p in processes:
      p.start()
    self.infeed_thread.start()

    def log_eval_results_fn():
      result = q_out.get()
      cur_step, _ = result
      while cur_step != _STOP:
        log_eval_result_fn(result)
        result = q_out.get()
        cur_step, _ = result

    log_eval_result_thread = threading.Thread(target=log_eval_results_fn)
    log_eval_result_thread.start()

    cur_step = 0
    current_epoch = 0
    # Train and eval loop.
    while cur_step < train_steps:
      if self.run_success:
        break
      tf.logging.info("TrainAndEvalLowLevelRunner: start train step:%d",
                      cur_step)
      cur_step += self.iterations
      current_epoch = cur_step // self.params["steps_per_epoch"]
      if self.run_success:
        break
      if self.params[
          "eval_every_checkpoint"] or current_epoch in self.eval_epochs:
        if current_epoch in self.epoch_count:
          epoch_count = self.epoch_count[current_epoch]
        else:
          epoch_count = 1
        mlp_log.mlperf_print(
            "block_start",
            None,
            metadata={
                "first_epoch_num": current_epoch - epoch_count + 1,
                "epoch_count": epoch_count
            })
        mlp_log.mlperf_print(
            "eval_start", None, metadata={"epoch_num": current_epoch + 1})
        # Run predict on device.
        start = time.time()
        predictions = list(self.predict())
        end = time.time()
        tf.logging.info("TrainAndEvalRunner: step {} step time {} sec".format(
            cur_step, end - start))
        # Run predict post processing.
        q_in.put((cur_step, predictions))

    train_eval_thread.join()
    # Turn off predict thread.
    for _ in processes:
      q_in.put((_STOP, None))

    for p in processes:
      p.join(timeout=self.sleep_seconds)

    q_out.put((_STOP, None))
    log_eval_result_thread.join()

    # Clear out all the queues to avoid deadlock.
    while not q_out.empty():
      log_eval_result_fn(q_out.get())
    while not q_in.empty():
      q_in.get()

    summary_writer.close()
    if not self.run_success:
      mlp_log.mlperf_print("run_stop", None, metadata={"status": "abort"})
