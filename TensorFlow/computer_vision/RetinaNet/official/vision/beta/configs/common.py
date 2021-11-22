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

# Lint as: python3
"""Common configurations."""

from typing import Optional
# Import libraries

import dataclasses

from official.core import config_definitions as cfg
from official.modeling import hyperparams


@dataclasses.dataclass
class RandAugment(hyperparams.Config):
  """Configuration for RandAugment."""
  num_layers: int = 2
  magnitude: float = 10
  cutout_const: float = 40
  translate_const: float = 10
  prob_to_apply: Optional[float] = None


@dataclasses.dataclass
class AutoAugment(hyperparams.Config):
  """Configuration for AutoAugment."""
  augmentation_name: str = 'v0'
  cutout_const: float = 100
  translate_const: float = 250


@dataclasses.dataclass
class Augmentation(hyperparams.OneOfConfig):
  """Configuration for input data augmentation.

  Attributes:
    type: 'str', type of augmentation be used, one of the fields below.
    randaug: RandAugment config.
    autoaug: AutoAugment config.
  """
  type: Optional[str] = None
  randaug: RandAugment = RandAugment()
  autoaug: AutoAugment = AutoAugment()


@dataclasses.dataclass
class NormActivation(hyperparams.Config):
  activation: str = 'relu'
  use_sync_bn: bool = True
  norm_momentum: float = 0.99
  norm_epsilon: float = 0.001


@dataclasses.dataclass
class PseudoLabelDataConfig(cfg.DataConfig):
  """Psuedo Label input config for training."""
  input_path: str = ''
  data_ratio: float = 1.0  # Per-batch ratio of pseudo-labeled to labeled data.
  is_training: bool = True
  dtype: str = 'float32'
  shuffle_buffer_size: int = 10000
  cycle_length: int = 10
  aug_rand_hflip: bool = True
  aug_type: Optional[
      Augmentation] = None  # Choose from AutoAugment and RandAugment.
  file_type: str = 'tfrecord'

  # Keep for backward compatibility.
  aug_policy: Optional[str] = None  # None, 'autoaug', or 'randaug'.
  randaug_magnitude: Optional[int] = 10
