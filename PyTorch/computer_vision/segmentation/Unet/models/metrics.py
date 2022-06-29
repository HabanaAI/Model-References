# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import torch


# from pytorch_lightning.metrics.functional import stat_scores
# from pytorch_lightning.metrics.metric import Metric

# pl.metrics is deprecated since pl 1.3
# asked to change to torchmetrics
# stat_scores implementation is different in recent version.
# older version was copied here from pl 1.0 implementation as it is simple function
# and has no other dependency

# Metric was copied to pl_metric.py from older version implementation (pytorch-ligthning 1.0.4)
# implementation in pytorch-ligthning 1.4.0, torch,etric is different and cause problems

# from torchmetrics import Metric
# from pytorch_lightning.metrics.metric import Metric
from .pl_metric import Metric

class Dice(Metric):
    def __init__(self, nclass):
        super().__init__(dist_sync_on_step=True)
        self.add_state("n_updates", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("dice", default=torch.zeros((nclass,)), dist_reduce_fx="sum")

    def update(self, pred, target):
        self.n_updates += 1
        self.dice += self.compute_stats(pred, target)

    def compute(self):
        return 100 * self.dice / self.n_updates

    @staticmethod
    def compute_stats(pred, target):
        num_classes = pred.shape[1]
        scores = torch.zeros(num_classes - 1, device=pred.device, dtype=torch.float32)
        for i in range(1, num_classes):
            if (target != i).all():
                # no foreground class
                _, _pred = torch.max(pred, 1)
                scores[i - 1] += 1 if (_pred != i).all() else 0
                continue
            _tp, _fp, _tn, _fn, _ = stat_scores(pred=pred, target=target, class_index=i)
            denom = (2 * _tp + _fp + _fn).to(torch.float)
            score_cls = (2 * _tp).to(torch.float) / denom if torch.is_nonzero(denom) else 0.0
            scores[i - 1] += score_cls
        return scores


def stat_scores(
        pred: torch.Tensor,
        target: torch.Tensor,
        class_index: int, argmax_dim: int = 1,
) :
    """
    Calculates the number of true positive, false positive, true negative
    and false negative for a specific class

    Args:
        pred: prediction tensor
        target: target tensor
        class_index: class to calculate over
        argmax_dim: if pred is a tensor of probabilities, this indicates the
            axis the argmax transformation will be applied over

    Return:
        True Positive, False Positive, True Negative, False Negative, Support

    Example:

        >>> x = torch.tensor([1, 2, 3])
        >>> y = torch.tensor([0, 2, 3])
        >>> tp, fp, tn, fn, sup = stat_scores(x, y, class_index=1)
        >>> tp, fp, tn, fn, sup
        (tensor(0), tensor(1), tensor(2), tensor(0), tensor(0))

    """
    if pred.ndim == target.ndim + 1:
        pred = to_categorical(pred, argmax_dim=argmax_dim)

    tp = ((pred == class_index) * (target == class_index)).to(torch.long).sum()
    fp = ((pred == class_index) * (target != class_index)).to(torch.long).sum()
    tn = ((pred != class_index) * (target != class_index)).to(torch.long).sum()
    fn = ((pred != class_index) * (target == class_index)).to(torch.long).sum()
    sup = (target == class_index).to(torch.long).sum()

    return tp, fp, tn, fn, sup


def to_categorical(
        tensor: torch.Tensor,
        argmax_dim: int = 1
) -> torch.Tensor:
    """
    Converts a tensor of probabilities to a dense label tensor

    Args:
        tensor: probabilities to get the categorical label [N, d1, d2, ...]
        argmax_dim: dimension to apply

    Return:
        A tensor with categorical labels [N, d2, ...]

    Example:

        >>> x = torch.tensor([[0.2, 0.5], [0.9, 0.1]])
        >>> to_categorical(x)
        tensor([1, 0])

    """
    return torch.argmax(tensor, dim=argmax_dim)