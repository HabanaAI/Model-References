#!/usr/bin/env python3

# coding=utf-8
# Copyright 2021 The Tensor2Tensor Authors.
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
# Copyright (C) 2021-2022 Habana Labs, Ltd. an Intel Company
###############################################################################
# Changes:
# - renamed t2t_trainer.py to trainer.py
# - updated imports
# - changed default ExportSavedModelApiVersion to V2
# - removed unused flags
# - removed TPU related code
# - added no_checkpoints, deterministic_dataset, save_summary_steps, use_horovod,
#   use_hpu, use_bf16, bf16_config_path flags
# - removed mtf mode handling
# - added support for horovod
# - added disable_v2_behavior and enable_resource_variables calls
# - removed mlperf log
# - removed call to tf.logging.set_verbosity
# - added support for running on GPU through horovod
# - disabled dynamic shapes by default
# - added support for recipe cache
# - added support for fast inference on HPU
# - changed the default value of the log_step_count_steps flag
# - added line tf.get_logger().propagate = False
# - added profile_steps flag

"""Train and evaluate."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os
import sys
import shutil
from TensorFlow.nlp.transformer import models  # pylint: disable=unused-import
from TensorFlow.nlp.transformer.utils import problems as problems_lib  # pylint: disable=unused-import
from TensorFlow.nlp.transformer.data_generators import problem  # pylint: disable=unused-import

from TensorFlow.nlp.transformer.utils import contrib
from TensorFlow.nlp.transformer.utils import decoding
from TensorFlow.nlp.transformer.utils import flags as t2t_flags  # pylint: disable=unused-import
from TensorFlow.nlp.transformer.utils import hparams_lib
from TensorFlow.nlp.transformer.utils import registry
from TensorFlow.nlp.transformer.utils import trainer_lib
from TensorFlow.nlp.transformer.utils import usr_dir
from TensorFlow.nlp.transformer.utils.mpi import MPI_barrier, MPI_is_distributed, MPI_world_rank
import tensorflow.compat.v1 as tf

from TensorFlow.common.debug import dump_callback

tf.get_logger().propagate = False

flags = tf.flags
FLAGS = flags.FLAGS

# See utils/flags.py for additional command-line flags.
flags.DEFINE_string("t2t_usr_dir", None,
                    "Path to a Python module that will be imported. The "
                    "__init__.py file should include the necessary imports. "
                    "The imported files should contain registrations, "
                    "e.g. @registry.register_model calls, that will then be "
                    "available to the t2t-trainer.")
flags.DEFINE_integer("random_seed", None, "Random seed.")
flags.DEFINE_integer("tpu_num_shards", 8, "Number of tpu shards.")
flags.DEFINE_string("tpu_job_name", None,
                    "TPU job name. TPUEstimator can auto-infer this but if the "
                    "configuration is esoteric it should be provided here.")
flags.DEFINE_integer("iterations_per_loop", 100,
                     "Number of iterations in a TPU training loop.")
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU.")
flags.DEFINE_bool("use_tpu_estimator", False, "Whether to use TPUEstimator. "
                  "This is always enabled when use_tpu is True.")
flags.DEFINE_integer("export_saved_model_api_version", 2,
                     "ExportSavedModelApiVersion, 1 V1 or 2 (V2, default). "
                     "Default V2 uses model_fn_inference_on_tpu for rewrite."
                     "Flag use_guarantee_const is only enabled in V2.")
flags.DEFINE_bool("use_guarantee_const_getter", False,
                  "Whether to use GuaranteeConst Ops to mark all weights as "
                  "constant. It may improve TPU inference performance and "
                  "reduce HBM arguments usage. Only available when "
                  "export_saved_model_api_version=2 and use_tpu=True.")
flags.DEFINE_bool("xla_compile", False,
                  "Whether to use XLA to compile model_fn.")
flags.DEFINE_integer("xla_jit_level", -1,
                     "GlobalJitLevel to use while compiling the full graph.")
flags.DEFINE_integer("tpu_infeed_sleep_secs", None,
                     "How long to sleep the infeed thread.")
flags.DEFINE_bool("generate_data", False, "Generate data before training?")
flags.DEFINE_string("tmp_dir", "/tmp/t2t_datagen",
                    "Temporary storage directory, used if --generate_data.")
flags.DEFINE_bool("profile", False, "Profile performance?")
flags.DEFINE_string("profile_steps", None, "When to start and stop profiling")
flags.DEFINE_integer("inter_op_parallelism_threads", 0,
                     "Number of inter_op_parallelism_threads to use for CPU. "
                     "See TensorFlow config.proto for details.")
flags.DEFINE_integer("intra_op_parallelism_threads", 0,
                     "Number of intra_op_parallelism_threads to use for CPU. "
                     "See TensorFlow config.proto for details.")
# TODO(lukaszkaiser): resolve memory and variable assign issues and set to True.
flags.DEFINE_bool(
    "optionally_use_dist_strat", False,
    "Whether to use TensorFlow DistributionStrategy instead of explicitly "
    "replicating the model. DistributionStrategy is used only if the "
    "model replication configuration is supported by the DistributionStrategy.")
# To maintain compatibility with some internal libs, we guard against these flag
# definitions possibly erroring. Apologies for the ugliness.
try:
  flags.DEFINE_string("master", "", "Address of TensorFlow master.")
  flags.DEFINE_string("output_dir", "", "Base output directory for run.")
  flags.DEFINE_string("schedule", "continuous_train_and_eval",
                      "Method of Experiment to run.")
  flags.DEFINE_integer("eval_steps", 100,
                       "Number of steps in evaluation. By default, eval will "
                       "stop after eval_steps or when it runs through the eval "
                       "dataset once in full, whichever comes first, so this "
                       "can be a very large number.")
except:  # pylint: disable=bare-except
  pass

flags.DEFINE_string("std_server_protocol", "grpc",
                    "Protocol for tf.train.Server.")

# Hyperparameter tuning on Cloud ML Engine
# Pass an --hparams_range to enable
flags.DEFINE_string("autotune_objective", None,
                    "TensorBoard metric name to optimize.")
flags.DEFINE_bool("autotune_maximize", True,
                  "Whether to maximize (vs. minimize) autotune_objective.")
flags.DEFINE_integer("autotune_max_trials", 10,
                     "Maximum number of tuning experiments to run.")
flags.DEFINE_integer("autotune_parallel_trials", 1,
                     "How many trials to run in parallel (will spin up this "
                     "many jobs.")
# Note than in open-source TensorFlow, the dash gets converted to an underscore,
# so access is FLAGS.job_dir.
flags.DEFINE_string("job-dir", None,
                    "DO NOT USE. Exists only for Cloud ML Engine to pass in "
                    "during hyperparameter tuning. Overrides --output_dir.")
flags.DEFINE_integer("log_step_count_steps", 50,
                     "Number of local steps after which progress is printed "
                     "out")
flags.DEFINE_bool("gpu_automatic_mixed_precision", False,
                  "Whether to employ GPU automatic mixed precision training "
                  "(via graph rewrite and dynamic loss scaling).")

flags.DEFINE_bool("no_checkpoints", False, "If True checkpoints will not be saved")
flags.DEFINE_bool("deterministic_dataset", False, "If True dataset will be deterministic")
flags.DEFINE_integer("save_summary_steps", 100, "How often to save summaries to TensorBoard")
flags.DEFINE_bool("use_horovod", False, "Use Horovod for training")
flags.DEFINE_bool("use_hpu", False, "Use HPU for training")
flags.DEFINE_bool("use_bf16", False, "Use automatic bfloat16 conversion (HPU only)")

default_bf16_config_path = os.path.normpath(
    os.path.join(os.path.realpath(__file__), '..',
                 'bf16_config', 'transformer.json'))
flags.DEFINE_string("bf16_config_path", default_bf16_config_path,
                    "Path to custom mixed precision config (in JSON format).")

flags.DEFINE_string('recipe_cache',
        default='/tmp/transformer_recipe_cache/',
        help='Path to recipe cache directory. Set to empty to disable recipe cache. Externally set \'TF_RECIPE_CACHE_PATH\' will override this setting.'
    )

def set_hparams_from_args(args):
  """Set hparams overrides from unparsed args list."""
  if not args:
    return

  hp_prefix = "--hp_"
  tf.logging.info("Found unparsed command-line arguments. Checking if any "
                  "start with %s and interpreting those as hparams "
                  "settings.", hp_prefix)

  pairs = []
  i = 0
  while i < len(args):
    arg = args[i]
    if arg.startswith(hp_prefix):
      pairs.append((arg[len(hp_prefix):], args[i+1]))
      i += 2
    else:
      tf.logging.warn("Found unknown flag: %s", arg)
      i += 1

  as_hparams = ",".join(["%s=%s" % (key, val) for key, val in pairs])
  if FLAGS.hparams:
    as_hparams = "," + as_hparams
  FLAGS.hparams += as_hparams


def create_hparams():
  """Create hparams."""
  hparams_path = os.path.join(FLAGS.output_dir, "hparams.json")
  print(FLAGS.hparams)
  return trainer_lib.create_hparams(FLAGS.hparams_set, FLAGS.hparams,
                                    hparams_path=hparams_path)


def create_experiment_fn():
  return trainer_lib.create_experiment_fn(
      model_name=FLAGS.model,
      problem_name=FLAGS.problem,
      data_dir=os.path.expanduser(FLAGS.data_dir),
      train_steps=FLAGS.train_steps,
      eval_steps=FLAGS.eval_steps,
      min_eval_frequency=FLAGS.local_eval_frequency,
      schedule=FLAGS.schedule,
      eval_throttle_seconds=FLAGS.eval_throttle_seconds,
      export=FLAGS.export_saved_model,
      decode_hparams=decoding.decode_hparams(FLAGS.decode_hparams),
      use_tfdbg=FLAGS.tfdbg,
      use_dbgprofile=FLAGS.dbgprofile,
      eval_early_stopping_steps=FLAGS.eval_early_stopping_steps,
      eval_early_stopping_metric=FLAGS.eval_early_stopping_metric,
      eval_early_stopping_metric_delta=FLAGS.eval_early_stopping_metric_delta,
      eval_early_stopping_metric_minimize=FLAGS
      .eval_early_stopping_metric_minimize,
      eval_timeout_mins=FLAGS.eval_timeout_mins,
      eval_use_test_set=FLAGS.eval_use_test_set,
      use_tpu=FLAGS.use_tpu,
      use_tpu_estimator=FLAGS.use_tpu_estimator,
      use_xla=FLAGS.xla_compile,
      export_saved_model_api_version=FLAGS.export_saved_model_api_version,
      use_guarantee_const_getter=FLAGS.use_guarantee_const_getter,
      warm_start_from=FLAGS.warm_start_from,
      decode_from_file=FLAGS.decode_from_file,
      decode_to_file=FLAGS.decode_to_file,
      decode_reference=FLAGS.decode_reference,
      std_server_protocol=FLAGS.std_server_protocol,
      use_horovod=FLAGS.use_horovod,
      use_hpu=FLAGS.use_hpu)


def create_run_config(hp, output_dir=None):
  """Create a run config.

  Args:
    hp: model hyperparameters
    output_dir: model's output directory, defaults to output_dir flag.

  Returns:
    a run config
  """
  save_ckpt_steps = max(FLAGS.iterations_per_loop, FLAGS.local_eval_frequency)
  save_ckpt_secs = FLAGS.save_checkpoints_secs or None
  if save_ckpt_secs:
    save_ckpt_steps = None
  assert FLAGS.output_dir
  tpu_config_extra_kwargs = {}
  if FLAGS.tpu_job_name is not None:
    tpu_config_extra_kwargs["tpu_job_name"] = FLAGS.tpu_job_name

  model_dir = output_dir or os.path.expanduser(FLAGS.output_dir)
  if FLAGS.use_horovod and model_dir:
    model_dir = os.path.join(model_dir, f'worker_{hp.hvd_worker_id}')

  save_checkpoints = save_ckpt_steps
  if FLAGS.no_checkpoints or (FLAGS.use_horovod and hp.hvd_worker_id != 0):
    save_checkpoints = None

  # the various custom getters we have written do not play well together yet.
  # TODO(noam): ask rsepassi for help here.
  daisy_chain_variables = (
      hp.daisy_chain_variables and
      hp.activation_dtype == "float32" and
      hp.weight_dtype == "float32")
  return trainer_lib.create_run_config(
      model_name=FLAGS.model,
      model_dir=model_dir,
      master=FLAGS.master,
      iterations_per_loop=FLAGS.iterations_per_loop,
      num_shards=FLAGS.tpu_num_shards,
      log_device_placement=FLAGS.log_device_placement,
      save_checkpoints_steps=save_checkpoints,
      save_checkpoints_secs=save_ckpt_secs,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours,
      num_gpus=FLAGS.worker_gpu,
      gpu_order=FLAGS.gpu_order,
      num_async_replicas=FLAGS.worker_replicas,
      gpu_mem_fraction=FLAGS.worker_gpu_memory_fraction,
      enable_graph_rewriter=FLAGS.enable_graph_rewriter,
      use_tpu=FLAGS.use_tpu,
      use_tpu_estimator=FLAGS.use_tpu_estimator,
      xla_jit_level=FLAGS.xla_jit_level,
      schedule=FLAGS.schedule,
      no_data_parallelism=hp.no_data_parallelism,
      optionally_use_dist_strat=FLAGS.optionally_use_dist_strat,
      daisy_chain_variables=daisy_chain_variables,
      ps_replicas=FLAGS.ps_replicas,
      ps_job=FLAGS.ps_job,
      ps_gpu=FLAGS.ps_gpu,
      sync=FLAGS.sync,
      worker_id=FLAGS.worker_id,
      worker_job=FLAGS.worker_job,
      random_seed=FLAGS.random_seed,
      tpu_infeed_sleep_secs=FLAGS.tpu_infeed_sleep_secs,
      inter_op_parallelism_threads=FLAGS.inter_op_parallelism_threads,
      log_step_count_steps=FLAGS.log_step_count_steps,
      intra_op_parallelism_threads=FLAGS.intra_op_parallelism_threads,
      save_summary_steps=FLAGS.save_summary_steps,
      use_hpu=FLAGS.use_hpu)


def generate_data():
  # Generate data if requested.
  data_dir = os.path.expanduser(FLAGS.data_dir)
  tmp_dir = os.path.expanduser(FLAGS.tmp_dir)
  tf.gfile.MakeDirs(data_dir)
  tf.gfile.MakeDirs(tmp_dir)

  problem_name = FLAGS.problem
  tf.logging.info("Generating data for %s" % problem_name)
  registry.problem(problem_name).generate_data(data_dir, tmp_dir)


@contextlib.contextmanager
def profile_context():
  if FLAGS.profile:
    with contrib.tfprof().ProfileContext(
        "t2tprof", trace_steps=range(100), dump_steps=range(100)) as pctx:
      opts = tf.profiler.ProfileOptionBuilder.time_and_memory()
      pctx.add_auto_profiling("op", opts, range(100))
      yield
  else:
    yield


def maybe_log_registry_and_exit():
  if FLAGS.registry_help:
    tf.logging.info(registry.help_string())
    sys.exit(0)


def is_chief():
  schedules = ["train", "train_and_evaluate", "continuous_train_and_eval"]
  return FLAGS.worker_id == 0 and FLAGS.schedule in schedules


def save_metadata(hparams):
  """Saves FLAGS and hparams to output_dir."""
  output_dir = os.path.expanduser(FLAGS.output_dir)
  if not tf.gfile.Exists(output_dir):
    tf.gfile.MakeDirs(output_dir)

  # Save FLAGS in txt file
  if hasattr(FLAGS, "flags_into_string"):
    flags_str = FLAGS.flags_into_string()
    t2t_flags_str = "\n".join([
        "--%s=%s" % (f.name, f.value)
        for f in FLAGS.flags_by_module_dict()["TensorFlow.nlp.transformer.utils.flags"]
    ])
  else:
    flags_dict = FLAGS.__dict__["__flags"]
    flags_str = "\n".join(
        ["--%s=%s" % (name, str(f)) for (name, f) in flags_dict.items()])
    t2t_flags_str = None

  flags_txt = os.path.join(output_dir, "flags.txt")
  with tf.gfile.Open(flags_txt, "w") as f:
    f.write(flags_str)

  if t2t_flags_str:
    t2t_flags_txt = os.path.join(output_dir, "flags_t2t.txt")
    with tf.gfile.Open(t2t_flags_txt, "w") as f:
      f.write(t2t_flags_str)

  # Save hparams as hparams.json
  new_hparams = hparams_lib.copy_hparams(hparams)
  # Modality class is not JSON serializable so remove.
  new_hparams.del_hparam("modality")

  hparams_fname = os.path.join(output_dir, "hparams.json")
  with tf.gfile.Open(hparams_fname, "w") as f:
    f.write(new_hparams.to_json(indent=0, sort_keys=True))


def execute_schedule(exp):
  if not hasattr(exp, FLAGS.schedule):
    raise ValueError(
        "Experiment has no method %s, from --schedule" % FLAGS.schedule)
  schedule = FLAGS.schedule
  if schedule == 'continuous_train_and_eval' and \
      FLAGS.use_horovod and exp._hparams.hvd_worker_id != 0:
    schedule = 'train'
  with profile_context():
    getattr(exp, schedule)()


def run_std_server():
  exp = trainer_lib.T2TExperiment(*([None] * 5))
  exp.run_std_server()

def prepare_recipe_cache():
  # Handle recipe cache. Skip if externally set or empty.
  recipe_cache = FLAGS.recipe_cache
  if 'TF_RECIPE_CACHE_PATH' not in os.environ.keys() and recipe_cache:
    os.environ['TF_RECIPE_CACHE_PATH'] = recipe_cache

  if not MPI_is_distributed() or MPI_world_rank() == 0:
  # Clear previous recipe cache.
    if os.path.exists(recipe_cache) and os.path.isdir(recipe_cache):
      shutil.rmtree(recipe_cache)
  # Other ranks should wait for recipe cache to be removed.
  MPI_barrier()

def init_multinode():
  if FLAGS.use_horovod:
    if FLAGS.use_hpu:
      import horovod.tensorflow as hvd
      hvd.init()
      assert hvd.is_initialized()
    else:
      import horovod.tensorflow as hvd
      hvd.init()
      assert hvd.size() > 1
      os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())
    return hvd
  return None

def main(argv):
  tf.disable_v2_behavior()
  tf.enable_resource_variables()

  hvd = init_multinode()

  if FLAGS.use_hpu:
    if FLAGS.recipe_cache:
      prepare_recipe_cache()
    if FLAGS.use_bf16:
      os.environ['TF_BF16_CONVERSION'] = FLAGS.bf16_config_path
    dyn_shapes_flag = 'TF_ENABLE_DYNAMIC_SHAPES'
    if dyn_shapes_flag not in os.environ:
        os.environ[dyn_shapes_flag] = 'false'
    os.environ["TF_CLUSTER_VARIABLES"] = "1"

    from habana_frameworks.tensorflow import load_habana_module  # noqa
    load_habana_module()

  usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)

  # If we just have to print the registry, do that and exit early.
  maybe_log_registry_and_exit()

  # Create HParams.
  if argv:
    set_hparams_from_args(argv[1:])
  if FLAGS.schedule != "run_std_server":
    hparams = create_hparams()
  if FLAGS.gpu_automatic_mixed_precision:
    setattr(hparams, "gpu_automatic_mixed_precision", True)
  if FLAGS.deterministic_dataset:
    hparams.add_hparam("deterministic_dataset", True)

  hparams.add_hparam("use_horovod", FLAGS.use_horovod)
  hparams.add_hparam("use_hpu", FLAGS.use_hpu)
  hparams.add_hparam("profile_steps", FLAGS.profile_steps)
  if FLAGS.use_horovod:
    hparams.add_hparam("hvd_worker_id", hvd.rank())
    hparams.add_hparam("hvd_size", hvd.size())

  if FLAGS.schedule == "run_std_server":
    run_std_server()
  trainer_lib.set_random_seed(FLAGS.random_seed)

  if FLAGS.generate_data:
    generate_data()

  exp_fn = create_experiment_fn()
  exp = exp_fn(create_run_config(hparams), hparams)
  if is_chief():
    save_metadata(hparams)

  with dump_callback():
    execute_schedule(exp)

if __name__ == "__main__":
  tf.app.run()
