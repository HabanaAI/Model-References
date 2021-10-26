# Copyright 2020 Google Research. All Rights Reserved.
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
"""The main training script."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os

import numpy as np
import tensorflow.compat.v1 as tf
from absl import flags
from absl import logging

import dataloader
import det_model_fn
import hparams_config
import utils
from horovod_estimator import HorovodEstimator, hvd_try_init, hvd_info_rank0, hvd, horovod_enabled
from TensorFlow.common.horovod_helpers import hvd_init

from TensorFlow.common.tb_utils import write_hparams_v1
from TensorFlow.common import library_loader


def choose_default_dir(dirs):
    for dir in dirs:
        if os.path.exists(dir):
            return dir
    return None

# Datasets & backbones init
dataset_default_dirs = ['/data/tensorflow_datasets/coco2017/tfrecord_tf1',
                        '/software/data/tf/coco2017/tf_records']
DEFAULT_DATASET_PATH = choose_default_dir(dataset_default_dirs)
DEFAULT_TRAINING_FILE_PATTERN = DEFAULT_DATASET_PATH + "/train-*"
DEFAULT_VAL_FILE_PATTERN = DEFAULT_DATASET_PATH + "/val-*"

backbone_ckpt_default_dirs = ['/data/tensorflow_datasets/efficientdet/backbones/efficientnet-b0',
                              '/software/data/tf/data/efficientdet/backbones/efficientnet-b0']
BACKBONE_CKPT_DEFAULT_DIR = choose_default_dir(backbone_ckpt_default_dirs)

# Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    'tpu', default=None,
    help='The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
    'url.')
flags.DEFINE_string(
    'gcp_project', default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
flags.DEFINE_string(
    'tpu_zone', default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

# Model specific paramenters
flags.DEFINE_string(
    'eval_master', default='',
    help='GRPC URL of the eval master. Set to an appropriate value when running'
    ' on CPU/GPU')
flags.DEFINE_bool('use_tpu', False, 'Use TPUs rather than CPUs/GPUs')
flags.DEFINE_bool('use_amp', False, 'Use AMP')
flags.DEFINE_bool('use_xla', False, 'Use XLA')
flags.DEFINE_bool('use_fake_data', False, 'Use fake input.')
flags.DEFINE_string('model_dir', 'train_model_dir', 'Location of model_dir')
flags.DEFINE_string('backbone_ckpt', BACKBONE_CKPT_DEFAULT_DIR,
                    'Location of the ResNet50 checkpoint to use for model '
                    'initialization.')
flags.DEFINE_string('ckpt', None,
                    'Start training from this EfficientDet checkpoint.')

flags.DEFINE_string('hparams', 'num_classes=91,use_bfloat16=false',
                    'Comma separated k=v pairs of hyperparameters.')
flags.DEFINE_integer(
    'num_cores', default=8, help='Number of TPU cores for training')
flags.DEFINE_bool('use_spatial_partition', False, 'Use spatial partition.')
flags.DEFINE_integer(
    'num_cores_per_replica', default=8, help='Number of TPU cores per'
    'replica when using spatial partition.')
flags.DEFINE_multi_integer(
    'input_partition_dims', [1, 4, 2, 1],
    'A list that describes the partition dims for all the tensors.')
flags.DEFINE_integer('train_batch_size', 8, 'training batch size')
flags.DEFINE_integer('eval_batch_size', 1, 'evaluation batch size')
flags.DEFINE_integer('eval_samples', 5000, 'The number of samples for '
                     'evaluation.')
flags.DEFINE_integer(
    'iterations_per_loop', 200, 'Number of iterations per TPU training loop')
flags.DEFINE_string(
    'training_file_pattern', DEFAULT_TRAINING_FILE_PATTERN,
    'Glob for training data files (e.g., COCO train - minival set)')
flags.DEFINE_string(
    'validation_file_pattern', DEFAULT_VAL_FILE_PATTERN,
    'Glob for evaluation tfrecords (e.g., COCO val2017 set)')
flags.DEFINE_string(
    'val_json_file',
    None,
    'COCO validation JSON containing golden bounding boxes.')
flags.DEFINE_string('testdev_dir', None,
                    'COCO testdev dir. If true, ignorer val_json_file.')
flags.DEFINE_integer('num_examples_per_epoch', 120000,
                     'Number of examples in one epoch')
flags.DEFINE_integer('num_epochs', 300, 'Number of epochs for training')
flags.DEFINE_string('mode', 'train',
                    'Mode to run: train or eval (default: train)')
flags.DEFINE_string('model_name', 'efficientdet-d0',
                    'Model name: retinanet or efficientdet')
flags.DEFINE_bool('eval_after_training', False, 'Run one eval after the '
                  'training finishes.')

flags.DEFINE_integer('log_every_n_steps', 100, 'Number of iterations after which '
                     'training parameters are logged.')
flags.DEFINE_integer('cp_every_n_steps', 600, 'Number of iterations after which '
                     'checkpoint is saved.')
flags.DEFINE_integer('keep_checkpoint_max', 5, 'The maximum number of recent checkpoint files to keep.')

# For Eval mode
flags.DEFINE_integer('min_eval_interval', 180,
                     'Minimum seconds between evaluations.')
flags.DEFINE_integer(
    'eval_timeout', None,
    'Maximum seconds between checkpoints before evaluation terminates.')

flags.DEFINE_bool('sbs_test', False, 'Config topology run for sbs testing.')
flags.DEFINE_bool('deterministic', False, 'Deterministic input data.')
flags.DEFINE_bool('no_hpu', False, 'Do not load Habana modules = train on CPU/GPU')
flags.DEFINE_bool('dump_all_ranks', False, 'Dump tfevent file for each rank')
flags.DEFINE_integer('use_horovod', None, 'Use Horovod for distributed training, <num_workers_per_hls> parameter')

FLAGS = flags.FLAGS


def set_env(use_amp, use_fast_math=False):
    os.environ['CUDA_CACHE_DISABLE'] = '0'
    os.environ['HOROVOD_GPU_ALLREDUCE'] = 'NCCL'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_GPU_THREAD_COUNT'] = '1' if not horovod_enabled() else str(hvd.size())
    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
    os.environ['TF_ADJUST_HUE_FUSED'] = '1'
    os.environ['TF_ADJUST_SATURATION_FUSED'] = '1'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'
    os.environ['TF_DISABLE_NVTX_RANGES'] = '1'

    if use_amp:
        hvd_info_rank0("AMP is activated")
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE'] = '1'
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION_LOSS_SCALING'] = '1'

    if use_fast_math:
        hvd_info_rank0("use_fast_math is activated")
        os.environ['TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32'] = '1'
        os.environ['TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32'] = '1'
        os.environ['TF_ENABLE_CUDNN_RNN_TENSOR_OP_MATH_FP32'] = '1'

def set_deterministic():
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.reset_default_graph()
    SEED=0
    os.environ['PYTHONHASHSEED']=str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    np.random.seed(SEED)
    tf.set_random_seed(SEED)


def get_session_config(use_xla):
    config = tf.ConfigProto()

    config.allow_soft_placement = True
    config.log_device_placement = False
    config.gpu_options.allow_growth = True

    if horovod_enabled():
        config.gpu_options.visible_device_list = str(hvd.local_rank())

    if use_xla:
        hvd_info_rank0("XLA is activated - Experimental Feature")
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    config.gpu_options.force_gpu_compatible = True  # Force pinned memory

    config.intra_op_parallelism_threads = 1  # Avoid pool of Eigen threads
    if FLAGS.deterministic:
      config.inter_op_parallelism_threads = 1
    elif horovod_enabled():
        config.inter_op_parallelism_threads = max(2, (multiprocessing.cpu_count() // hvd.size()) - 2)
    else:
        config.inter_op_parallelism_threads = 4

    return config

def main(argv):
  del argv  # Unused.

  # if given an efficentdet ckpt don't use default backbone ckpt
  if FLAGS.backbone_ckpt == BACKBONE_CKPT_DEFAULT_DIR and FLAGS.ckpt is not None:
      print("Using ckpt flag: {}, ignoring default backbone_ckpt: {}".format(FLAGS.ckpt, FLAGS.backbone_ckpt))
      FLAGS.backbone_ckpt = None

  if FLAGS.use_horovod is not None:
      if FLAGS.dump_all_ranks:
          FLAGS.model_dir += "/worker_" + str(hvd.rank())
      if not 'HOROVOD_CYCLE_TIME' in os.environ:
          os.environ['HOROVOD_CYCLE_TIME'] = '0.5'
      if not 'HABANA_NCCL_COMM_API' in os.environ:
          os.environ['HABANA_NCCL_COMM_API'] = '0'
      hvd_init()

  if not FLAGS.no_hpu:
      from habana_frameworks.tensorflow import load_habana_module
      load_habana_module()

      if FLAGS.use_horovod:
          assert(horovod_enabled())

  set_env(use_amp=FLAGS.use_amp)

  # deterministic setting
  if FLAGS.sbs_test or FLAGS.deterministic:
    set_deterministic()

  # Check data path
  if FLAGS.mode in ('train',
                    'train_and_eval') and FLAGS.training_file_pattern is None:
    raise RuntimeError('You must specify --training_file_pattern for training.')
  if FLAGS.mode in ('eval', 'train_and_eval'):
    if FLAGS.validation_file_pattern is None:
      raise RuntimeError('You must specify --validation_file_pattern '
                         'for evaluation.')
    if not FLAGS.val_json_file and not FLAGS.testdev_dir:
      raise RuntimeError(
          'You must specify --val_json_file or --testdev for evaluation.')

  # Parse and override hparams
  config = hparams_config.get_detection_config(FLAGS.model_name)
  config.override(FLAGS.hparams)

  # The following is for spatial partitioning. `features` has one tensor while
  # `labels` had 4 + (`max_level` - `min_level` + 1) * 2 tensors. The input
  # partition is performed on `features` and all partitionable tensors of
  # `labels`, see the partition logic below.
  # In the TPUEstimator context, the meaning of `shard` and `replica` is the
  # same; follwing the API, here has mixed use of both.
  if FLAGS.use_spatial_partition:
    # Checks input_partition_dims agrees with num_cores_per_replica.
    if FLAGS.num_cores_per_replica != np.prod(FLAGS.input_partition_dims):
      raise RuntimeError('--num_cores_per_replica must be a product of array'
                         'elements in --input_partition_dims.')

    labels_partition_dims = {
        'mean_num_positives': None,
        'source_ids': None,
        'groundtruth_data': None,
        'image_scales': None,
    }
    # The Input Partition Logic: We partition only the partition-able tensors.
    # Spatial partition requires that the to-be-partitioned tensors must have a
    # dimension that is a multiple of `partition_dims`. Depending on the
    # `partition_dims` and the `image_size` and the `max_level` in config, some
    # high-level anchor labels (i.e., `cls_targets` and `box_targets`) cannot
    # be partitioned. For example, when `partition_dims` is [1, 4, 2, 1], image
    # size is 1536, `max_level` is 9, `cls_targets_8` has a shape of
    # [batch_size, 6, 6, 9], which cannot be partitioned (6 % 4 != 0). In this
    # case, the level-8 and level-9 target tensors are not partition-able, and
    # the highest partition-able level is 7.
    image_size = config.get('image_size')
    for level in range(config.get('min_level'), config.get('max_level') + 1):

      def _can_partition(spatial_dim):
        partitionable_index = np.where(
            spatial_dim % np.array(FLAGS.input_partition_dims) == 0)
        return len(partitionable_index[0]) == len(FLAGS.input_partition_dims)

      spatial_dim = image_size // (2 ** level)
      if _can_partition(spatial_dim):
        labels_partition_dims[
            'box_targets_%d' % level] = FLAGS.input_partition_dims
        labels_partition_dims[
            'cls_targets_%d' % level] = FLAGS.input_partition_dims
      else:
        labels_partition_dims['box_targets_%d' % level] = None
        labels_partition_dims['cls_targets_%d' % level] = None
    num_cores_per_replica = FLAGS.num_cores_per_replica
    input_partition_dims = [
        FLAGS.input_partition_dims, labels_partition_dims]
    num_shards = FLAGS.num_cores // num_cores_per_replica
  else:
    num_cores_per_replica = None
    input_partition_dims = None
    num_shards = FLAGS.num_cores
    if horovod_enabled():
        num_shards = hvd.size()
    else:
        num_shards = 1

  params = dict(
      config.as_dict(),
      model_name=FLAGS.model_name,
      num_epochs=FLAGS.num_epochs,
      iterations_per_loop=FLAGS.iterations_per_loop,
      model_dir=FLAGS.model_dir,
      num_shards=num_shards,
      num_examples_per_epoch=FLAGS.num_examples_per_epoch,
      use_tpu=FLAGS.use_tpu,
      backbone_ckpt=FLAGS.backbone_ckpt,
      ckpt=FLAGS.ckpt,
      val_json_file=FLAGS.val_json_file,
      testdev_dir=FLAGS.testdev_dir,
      mode=FLAGS.mode,
      dropout_rate=None
  )

  # disabling input data scaling/flip manipulations.
  if FLAGS.sbs_test:
    sbs_params = dict(
      input_rand_hflip=False,
      train_scale_min=1,
      train_scale_max=1,
      dropout_rate=0.0
    )
    params.update(sbs_params)

  tf_random_seed = 0 if FLAGS.deterministic else None
  run_config = tf.estimator.RunConfig(
      session_config=get_session_config(use_xla=FLAGS.use_xla),
      save_checkpoints_steps=FLAGS.cp_every_n_steps,
      tf_random_seed=tf_random_seed,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max)
  write_hparams_v1(FLAGS.model_dir,
    {'batch_size': FLAGS.train_batch_size, **FLAGS.flag_values_dict()})

  model_fn_instance = det_model_fn.get_model_fn(FLAGS.model_name)

  # TPU Estimator
  logging.info(params)
  if FLAGS.mode == 'train':
    params['batch_size'] = FLAGS.train_batch_size
    params['every_n_iter'] = FLAGS.log_every_n_steps
    params['dump_all_ranks'] = FLAGS.dump_all_ranks
    train_estimator = HorovodEstimator(
        model_fn=model_fn_instance,
        model_dir=FLAGS.model_dir,
        config=run_config,
        params=params)

    # for deterministic input, we pass to dataloader False for not manipulating input data
    is_training = not FLAGS.deterministic
    use_fake_data = FLAGS.use_fake_data or FLAGS.deterministic

    input_fn = dataloader.InputReader(FLAGS.training_file_pattern,
                                      is_training=is_training,
                                      params=params,
                                      use_fake_data=use_fake_data,
                                      is_deterministic=FLAGS.deterministic)
    max_steps = int((FLAGS.num_epochs * FLAGS.num_examples_per_epoch) / (FLAGS.train_batch_size * num_shards)) + 1

    # for sbs test, train under sbs callbacks
    if FLAGS.sbs_test:
      from TensorFlow.common.debug import dump_callback
      SBS_TEST_CONFIG = os.path.join(os.environ['TF_TESTS_ROOT'], "tests/tf_training_tests/side_by_side/topologies/efficientdet/dump_config.json")
      with dump_callback(SBS_TEST_CONFIG):
        train_estimator.train(input_fn=input_fn, max_steps=max_steps)
    else:
      train_estimator.train(input_fn=input_fn, max_steps=max_steps)

  elif FLAGS.mode == 'eval':
    params['every_n_iter'] = FLAGS.log_every_n_steps
    config_proto = tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=False)
    if FLAGS.use_xla and not FLAGS.use_tpu:
      config_proto.graph_options.optimizer_options.global_jit_level = (
        tf.OptimizerOptions.ON_1)

    tpu_config = tf.estimator.tpu.TPUConfig(
      FLAGS.iterations_per_loop,
      num_shards=num_shards,
      num_cores_per_replica=num_cores_per_replica,
      input_partition_dims=input_partition_dims,
      per_host_input_for_training=tf.estimator.tpu.InputPipelineConfig
        .PER_HOST_V2)

    run_config = tf.estimator.tpu.RunConfig(
      cluster=None,
      evaluation_master=FLAGS.eval_master,
      model_dir=FLAGS.model_dir,
      log_step_count_steps=FLAGS.iterations_per_loop,
      session_config=config_proto,
      tpu_config=tpu_config,
    )

    # Eval only runs on CPU or GPU host with batch_size = 1.
    # Override the default options: disable randomization in the input pipeline
    # and don't run on the TPU.
    # Also, disable use_bfloat16 for eval on CPU/GPU.
    eval_params = dict(
        params,
        use_tpu=False,
        input_rand_hflip=False,
        is_training_bn=False,
        use_bfloat16=False,
    )

    eval_estimator = tf.estimator.tpu.TPUEstimator(
        model_fn=model_fn_instance,
        use_tpu=False,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        config=run_config,
        params=eval_params)

    def terminate_eval():
      logging.info('Terminating eval after %d seconds of no checkpoints',
                   FLAGS.eval_timeout)
      return True

    # Run evaluation when there's a new checkpoint
    for ckpt in tf.train.checkpoints_iterator(
        FLAGS.model_dir,
        min_interval_secs=FLAGS.min_eval_interval,
        timeout=FLAGS.eval_timeout,
        timeout_fn=terminate_eval):

      logging.info('Starting to evaluate.')
      try:
        eval_results = eval_estimator.evaluate(
            input_fn=dataloader.InputReader(FLAGS.validation_file_pattern,
                                            is_training=False),
            steps=FLAGS.eval_samples//FLAGS.eval_batch_size)
        logging.info('Eval results: %s', eval_results)

        # Terminate eval job when final checkpoint is reached.
        try:
          current_step = int(os.path.basename(ckpt).split('-')[1])
        except IndexError:
          logging.info('%s has no global step info: stop!', ckpt)
          break

        utils.archive_ckpt(eval_results, eval_results['AP'], ckpt)
        total_step = int((FLAGS.num_epochs * FLAGS.num_examples_per_epoch) /
                         FLAGS.train_batch_size)
        if current_step >= total_step:
          logging.info('Evaluation finished after training step %d',
                       current_step)
          break

      except tf.errors.NotFoundError:
        # Since the coordinator is on a different job than the TPU worker,
        # sometimes the TPU worker does not finish initializing until long after
        # the CPU job tells it to start evaluating. In this case, the checkpoint
        # file could have been deleted already.
        logging.info('Checkpoint %s no longer exists, skipping checkpoint',
                     ckpt)

  elif FLAGS.mode == 'train_and_eval':
    for cycle in range(FLAGS.num_epochs):
      logging.info('Starting training cycle, epoch: %d.', cycle)
      train_estimator = tf.estimator.tpu.TPUEstimator(
          model_fn=model_fn_instance,
          use_tpu=FLAGS.use_tpu,
          train_batch_size=FLAGS.train_batch_size,
          config=run_config,
          params=params)
      train_estimator.train(
          input_fn=dataloader.InputReader(FLAGS.training_file_pattern,
                                          is_training=True,
                                          use_fake_data=FLAGS.use_fake_data),
          steps=int(FLAGS.num_examples_per_epoch / FLAGS.train_batch_size))

      logging.info('Starting evaluation cycle, epoch: %d.', cycle)
      # Run evaluation after every epoch.
      eval_params = dict(
          params,
          use_tpu=False,
          input_rand_hflip=False,
          is_training_bn=False,
      )

      eval_estimator = tf.estimator.tpu.TPUEstimator(
          model_fn=model_fn_instance,
          use_tpu=False,
          train_batch_size=FLAGS.train_batch_size,
          eval_batch_size=FLAGS.eval_batch_size,
          config=run_config,
          params=eval_params)
      eval_results = eval_estimator.evaluate(
          input_fn=dataloader.InputReader(FLAGS.validation_file_pattern,
                                          is_training=False),
          steps=FLAGS.eval_samples//FLAGS.eval_batch_size)
      logging.info('Evaluation results: %s', eval_results)
      ckpt = tf.train.latest_checkpoint(FLAGS.model_dir)
      utils.archive_ckpt(eval_results, eval_results['AP'], ckpt)
    pass

  else:
    logging.info('Mode not found.')


if __name__ == '__main__':
  tf.disable_v2_behavior()
  # TODO SW-36772: remove the below RUN_TPC_FUSER related lines
  if not 'RUN_TPC_FUSER' in os.environ:
    os.environ['RUN_TPC_FUSER'] = 'false'
  tf.compat.v1.enable_resource_variables()
  tf.app.run(main)
