# Lint as: python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Definitions for high level configuration groups.."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from typing import Any, List, Mapping, Optional

import dataclasses

from modeling.hyperparams import base_config


CallbacksConfig = base_config.CallbacksConfig
TensorboardConfig = base_config.TensorboardConfig
RuntimeConfig = base_config.RuntimeConfig


@dataclasses.dataclass
class ExportConfig(base_config.Config):
  """Configuration for exports.

  Attributes:
    checkpoint: the path to the checkpoint to export.
    destination: the path to where the checkpoint should be exported.

  """
  checkpoint: str = None
  destination: str = None


@dataclasses.dataclass
class MetricsConfig(base_config.Config):
  """Configuration for Metrics.

  Attributes:
    accuracy: Whether or not to track accuracy as a Callback. Defaults to None.
    top_5: Whether or not to track top_5_accuracy as a Callback. Defaults to
      None.

  """
  accuracy: bool = None
  top_5: bool = None


@dataclasses.dataclass
class TimeHistoryConfig(base_config.Config):
  """Configuration for the TimeHistory callback.

  Attributes:
    log_steps: Interval of steps between logging of batch level stats.

  """
  log_steps: int = None


@dataclasses.dataclass
class TrainConfig(base_config.Config):
  """Configuration for training.

  Attributes:
    resume_checkpoint: Whether or not to enable load checkpoint loading.
      Defaults to None.
    epochs: The number of training epochs to run. Defaults to None.
    steps: The number of steps to run per epoch. If None, then this will be
      inferred based on the number of images and batch size. Defaults to None.
    callbacks: An instance of CallbacksConfig.
    metrics: An instance of MetricsConfig.
    tensorboard: An instance of TensorboardConfig.

  """
  resume_checkpoint: bool = None
  epochs: int = None
  steps: int = None
  callbacks: CallbacksConfig = CallbacksConfig()
  metrics: MetricsConfig = None
  tensorboard: TensorboardConfig = TensorboardConfig()
  time_history: TimeHistoryConfig = TimeHistoryConfig()


@dataclasses.dataclass
class EvalConfig(base_config.Config):
  """Configuration for evaluation.

  Attributes:
    epochs_between_evals: The number of train epochs to run between evaluations.
      Defaults to None.
    steps: The number of eval steps to run during evaluation. If None, this will
      be inferred based on the number of images and batch size. Defaults to
      None.
    skip_eval: Whether or not to skip evaluation.

  """
  epochs_between_evals: int = None
  steps: int = None
  skip_eval: bool = False


@dataclasses.dataclass
class LossConfig(base_config.Config):
  """Configuration for Loss.

  Attributes:
    name: The name of the loss. Defaults to None.
    loss_scale: The type of loss scale
    label_smoothing: Whether or not to apply label smoothing to the loss. This
      only applies to 'categorical_cross_entropy'.

  """
  name: str = None
  loss_scale: str = None
  label_smoothing: float = None


@dataclasses.dataclass
class OptimizerConfig(base_config.Config):
  """Configuration for Optimizers.

  Attributes:
    name: The name of the optimizer. Defaults to None.
    decay: Decay or rho, discounting factor for gradient. Defaults to None.
    epsilon: Small value used to avoid 0 denominator. Defaults to None.
    momentum: Plain momentum constant. Defaults to None.
    nesterov: Whether or not to apply Nesterov momentum. Defaults to None.
    moving_average_decay: The amount of decay to apply. If 0 or None, then
      exponential moving average is not used. Defaults to None.
    lookahead: Whether or not to apply the lookahead optimizer. Defaults to
      None.
    beta_1: The exponential decay rate for the 1st moment estimates. Used in
      the Adam optimizers. Defaults to None.
    beta_2: The exponential decay rate for the 2nd moment estimates. Used in
      the Adam optimizers. Defaults to None.
    epsilon: Small value used to avoid 0 denominator. Defaults to 1e-7.

  """
  name: str = None
  decay: float = None
  epsilon: float = None
  momentum: float = None
  nesterov: bool = None
  moving_average_decay: Optional[float] = None
  lookahead: Optional[bool] = None
  beta_1: float = None
  beta_2: float = None
  epsilon: float = None


@dataclasses.dataclass
class LearningRateConfig(base_config.Config):
  """Configuration for learning rates.

  Attributes:
    name: The name of the learning rate. Defaults to None.
    initial_lr: The initial learning rate. Defaults to None.
    decay_epochs: The number of decay epochs. Defaults to None.
    decay_rate: The rate of decay. Defaults to None.
    warmup_epochs: The number of warmup epochs. Defaults to None.
    batch_lr_multiplier: The multiplier to apply to the base learning rate,
      if necessary. Defaults to None.
    examples_per_epoch: the number of examples in a single epoch.
      Defaults to None.
    boundaries: boundaries used in piecewise constant decay with warmup.
    multipliers: multipliers used in piecewise constant decay with warmup.
    scale_by_batch_size: Scale the learning rate by a fraction of the batch
      size. Set to 0 for no scaling (default).

  """
  name: str = None
  initial_lr: float = None
  decay_epochs: float = None
  decay_rate: float = None
  warmup_epochs: int = None
  examples_per_epoch: int = None
  boundaries: List[int] = None
  multipliers: List[float] = None
  scale_by_batch_size: float = 0.


@dataclasses.dataclass
class ModelConfig(base_config.Config):
  """Configuration for Models.

  Attributes:
    name: The name of the model. Defaults to None.
    model_params: The parameters used to create the model. Defaults to None.
    num_classes: The number of classes in the model. Defaults to None.
    loss: A `LossConfig` instance. Defaults to None.
    optimizer: An `OptimizerConfig` instance. Defaults to None.

  """
  name: str = None
  model_params: Mapping[str, Any] = None
  num_classes: int = None
  loss: LossConfig = None
  optimizer: OptimizerConfig = None


@dataclasses.dataclass
class ExperimentConfig(base_config.Config):
  """Base configuration for an image classification experiment.

  Attributes:
    model_dir: The directory to use when running an experiment.
    mode: e.g. 'train_and_eval', 'export'
    runtime: A `RuntimeConfig` instance.
    train: A `TrainConfig` instance.
    evaluation: An `EvalConfig` instance.
    model: A `ModelConfig` instance.
    export: An `ExportConfig` instance.

  """
  model_dir: str = None
  model_name: str = None
  mode: str = None
  runtime: RuntimeConfig = None
  train_dataset: Any = None
  validation_dataset: Any = None
  train: TrainConfig = None
  evaluation: EvalConfig = None
  model: ModelConfig = None
  export: ExportConfig = None
