###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F


class Dice:
    def __init__(self,
                 to_onehot_y: bool = True,
                 to_onehot_x: bool = False,
                 use_softmax: bool = True,
                 use_argmax: bool = False,
                 include_background: bool = False):
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.to_onehot_x = to_onehot_x
        self.use_softmax = use_softmax
        self.use_argmax = use_argmax
        self.smooth_nr = 1e-6
        self.smooth_dr = 1e-6

    def __call__(self, prediction, target):
        channel_axis = 1
        reduce_axis = list(range(2, len(prediction.shape)))
        num_pred_ch = prediction.shape[channel_axis]

        if self.use_softmax:
            prediction = torch.nn.functional.softmax(prediction, dim=channel_axis)
        elif self.use_argmax:
            prediction = torch.argmax(prediction, dim=channel_axis)

        if self.to_onehot_y:
            target = to_one_hot(target, channel_axis)

        if self.to_onehot_x:
            prediction = to_one_hot(prediction, channel_axis)

        if not self.include_background:
            assert num_pred_ch > 1, \
                f"To exclude background the prediction needs more than one channel. Got {num_pred_ch}."
            target = target[:, 1:]
            prediction = prediction[:, 1:]

        assert (target.shape == prediction.shape), \
            f"Target and prediction shape do not match. Target: ({target.shape}), prediction: ({prediction.shape})."

        intersection = torch.sum(target * prediction, dim=reduce_axis)
        target_sum = torch.sum(target, dim=reduce_axis)
        prediction_sum = torch.sum(prediction, dim=reduce_axis)

        return (2.0 * intersection + self.smooth_nr) / (target_sum + prediction_sum + self.smooth_dr)


def to_one_hot(array, channel_axis):
    if len(array.shape) >= 5:
        array = torch.squeeze(array, dim=channel_axis)
    array = F.one_hot(array.long(), num_classes=3)
    array = array.permute(0, 4, 1, 2, 3).float()
    return array


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.channel_axis = 1

    def forward(self, prediction, target):
        one_hot_labels = F.one_hot(torch.squeeze(target, dim=self.channel_axis).long(), 3)
        one_hot_labels = one_hot_labels.permute(0, 4, 1, 2, 3)
        log_probs = F.log_softmax(prediction, self.channel_axis)
        return torch.mean(-1 * torch.sum((one_hot_labels * log_probs), dim=self.channel_axis))


class DiceCELoss(nn.Module):
    def __init__(self, to_onehot_y, use_softmax, include_background):
        super(DiceCELoss, self).__init__()
        self.dice = Dice(to_onehot_y=to_onehot_y, use_softmax=use_softmax,
                         include_background=include_background)
        self.cross_entropy = CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        cross_entropy = self.cross_entropy(y_pred, y_true)
        dice = torch.mean(1.0 - self.dice(y_pred, y_true))
        return (dice + cross_entropy) / 2


class DiceScore:
    def __init__(self, to_onehot_y: bool = True, use_argmax: bool = True,
                 include_background: bool = False):
        self.dice = Dice(to_onehot_y=to_onehot_y, to_onehot_x=True, use_softmax=False,
                         use_argmax=use_argmax, include_background=include_background)

    def __call__(self, y_pred, y_true):
        return torch.mean(self.dice(y_pred, y_true), dim=0)
