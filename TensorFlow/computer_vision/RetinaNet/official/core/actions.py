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

"""Provides TFM orbit actions and associated helper functions/classes."""

import os
from typing import List

import gin
import orbit
import tensorflow as tf

from official.core import base_trainer
from official.core import config_definitions
from official.modeling import optimization


class EMACheckpointing:
  """Eval action to save checkpoint with average weights when EMA is used.

  This action swaps the weights of the model with the average weights, then it
  saves the checkpoint under export_dir/ema_checkpoints. Checkpointing is
  expensive for large models, so doing this action in eval is more efficient
  than training.
  """

  def __init__(self, export_dir: str, optimizer: tf.keras.optimizers.Optimizer,
               checkpoint: tf.train.Checkpoint, max_to_keep: int = 1):
    """Initializes the instance.

    Args:
      export_dir: `str` for the export directory of the EMA average weights.
      optimizer: `tf.keras.optimizers.Optimizer` optimizer instance used for
        training. This will be used to swap the model weights with the average
        weigths.
      checkpoint: `tf.train.Checkpoint` instance.
      max_to_keep: `int` for max checkpoints to keep in ema_checkpoints subdir.
    """
    if not isinstance(optimizer, optimization.ExponentialMovingAverage):
      raise ValueError('Optimizer has to be instance of'
                       'optimization.ExponentialMovingAverage for'
                       'EMACheckpointing action')

    export_dir = os.path.join(export_dir, 'ema_checkpoints')
    tf.io.gfile.makedirs(
        os.path.dirname(export_dir))
    self._optimizer = optimizer
    self._checkpoint = checkpoint
    self._checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=export_dir,
        max_to_keep=max_to_keep,
        checkpoint_name='average_weights')

  def __call__(self, output: orbit.runner.Output):
    """Swaps model weights, and saves the checkpoint.

    Args:
      output: The train or eval output to test.
    """
    self._optimizer.swap_weights()
    self._checkpoint_manager.save(checkpoint_number=self._optimizer.iterations)
    self._optimizer.swap_weights()


@gin.configurable
def get_eval_actions(
    params: config_definitions.ExperimentConfig,
    trainer: base_trainer.Trainer,
    model_dir: str) -> List[orbit.Action]:
  """Gets eval actions for TFM trainer."""
  eval_actions = []
  # Adds ema checkpointing action to save the average weights under
  # ema_checkpoints subdir.
  if isinstance(trainer.optimizer, optimization.ExponentialMovingAverage):
    eval_actions.append(
        EMACheckpointing(
            export_dir=model_dir,
            optimizer=trainer.optimizer,
            checkpoint=trainer.checkpoint,
            max_to_keep=params.trainer.max_to_keep))

  return eval_actions
