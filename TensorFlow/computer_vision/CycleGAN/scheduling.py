###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import math
from functools import partial
from horovod.tensorflow.keras.callbacks import LearningRateScheduleCallback


def polynomial(cstep, tstep, pow=2, increasing=False):
    rate = ((cstep + 1) / (tstep + 1) ) ** pow
    if increasing:
        return rate
    return (1 - rate)


class MultiOptimizerLR(LearningRateScheduleCallback):
    """A callback adjusting learning rate after each epoch"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _adjust_learning_rate(self, epoch):
        for k, optimizer in self.model.optimizer.items():
            old_lr = self.backend.get_value(optimizer.lr)
            new_lr = self.initial_lr[k] * self.multiplier(epoch)
            self.backend.set_value(optimizer.lr, new_lr)

            if hasattr(self.model.optimizer, 'momentum') and self.momentum_correction:
                # See the paper cited above for more information about momentum correction.
                if self.restore_momentum is None:
                    self.restore_momentum = {}
                self.restore_momentum[k] = self.backend.get_value(
                    optimizer.momentum)
                self.backend.set_value(optimizer.momentum,
                                       self.restore_momentum[k] * new_lr / old_lr)

    def on_train_begin(self, logs=None):
        if self.initial_lr is None:
            self.initial_lr = dict(
                (k, self.backend.get_value(optimizer.lr)) for k, optimizer in self.model.optimizer.items())
        if not self.staircase and not self.steps_per_epoch:
            self.steps_per_epoch = self._autodetect_steps_per_epoch()

    def _restore_momentum_if_needed(self):
        if self.restore_momentum:
            for k, optimizer in self.model.optimizer.items():
                self.backend.set_value(
                    optimizer.momentum, self.restore_momentum[k])
            self.restore_momentum = None

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            # Log current learning rate.
            for k, optimizer in self.model.optimizer.items():
                logs[f'lr_{k}'] = self.backend.get_value(optimizer.lr)


class LinearDecay():
    def __init__(self, epochs, decay=100, clif=0):
        self._epochs = epochs
        self._decay = decay
        self._clif = clif

    # return multiplier for initial lr
    def __call__(self, epoch):
        if epoch > self._clif:
            adjusted = epoch - self._clif
            return (1 - 1 / (self._epochs - self._decay) * (adjusted - self._decay))
        return 1.0


class CosineDecay():
    def __init__(self, epochs, clif=0):
        assert epochs > 0, "--cosine_decay_delay should be set to smaller value than --epochs"
        self._epochs = epochs
        self._clif = clif

    def __call__(self, epoch):
        if epoch > self._clif:
            adjusted_epoch = epoch - self._clif
            return 0.5 * (math.cos(2 * math.acos(0) / self._epochs * adjusted_epoch) + 1)
        return 1.0


class Warmup():
    def __init__(self, schedule, warmup_fn=partial(polynomial, pow=2, increasing=True), warmup_epochs=5):
        assert warmup_epochs > 0
        self._warmup_epochs = warmup_epochs
        self._warmup_fn = warmup_fn
        self._schedule = schedule

    def __call__(self,epoch):
        if epoch < self._warmup_epochs:
            return self._warmup_fn(epoch, self._warmup_epochs)

        return self._schedule(epoch)
