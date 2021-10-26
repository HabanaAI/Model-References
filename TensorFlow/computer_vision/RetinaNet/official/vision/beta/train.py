# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
# This file has been modified by HabanaLabs

# Lint as: python3
"""TensorFlow Model Garden Vision training driver."""

from absl import app
from absl import flags
import gin

# pylint: disable=unused-import
from official.common import registry_imports
# pylint: enable=unused-import
from official.common import distribute_utils
from official.common import flags as tfm_flags
from official.core import task_factory
from official.core import train_lib
from official.core import train_utils
from official.modeling import performance

FLAGS = flags.FLAGS


def main(_):
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_params)
  params = train_utils.parse_configuration(FLAGS)

  if params.runtime.num_hpus > 0:
    import os
    #TODO: remove when SW-49334 is fixed [SW-49404]
    os.environ["TF_DISABLE_EAGER_TO_FUNC_REWRITER"] = "1"
    from habana_frameworks.tensorflow import load_habana_module
    load_habana_module()

  if params.task.train_data.deterministic or params.task.validation_data.deterministic:
    import os
    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    import numpy
    numpy.random.seed(0)
    import tensorflow as tf
    tf.random.set_seed(0)
    tf.compat.v1.set_random_seed(0)
    import random
    random.seed(0)

  model_dir = FLAGS.model_dir
  if 'train' in FLAGS.mode:
    # Pure eval modes do not output yaml files. Otherwise continuous eval job
    # may race against the train job for writing the same file.
    train_utils.serialize_config(params, model_dir)

  # Sets mixed_precision policy. Using 'mixed_float16' or 'mixed_bfloat16'
  # can have significant impact on model speeds by utilizing float16 in case of
  # GPUs, and bfloat16 in the case of TPUs. loss_scale takes effect only when
  # dtype is float16
  if params.runtime.mixed_precision_dtype:
    performance.set_mixed_precision_policy(params.runtime.mixed_precision_dtype)
  distribution_strategy = distribute_utils.get_distribution_strategy(
      distribution_strategy=params.runtime.distribution_strategy,
      all_reduce_alg=params.runtime.all_reduce_alg,
      num_gpus=params.runtime.num_gpus,
      num_hpus=params.runtime.num_hpus,
      tpu_address=params.runtime.tpu)
  with distribution_strategy.scope():
    task = task_factory.get_task(params.task, logging_dir=model_dir)

  train_lib.run_experiment(
      distribution_strategy=distribution_strategy,
      task=task,
      mode=FLAGS.mode,
      params=params,
      model_dir=model_dir)

  train_utils.save_gin_config(FLAGS.mode, model_dir)

if __name__ == '__main__':
  tfm_flags.define_flags()
  app.run(main)
