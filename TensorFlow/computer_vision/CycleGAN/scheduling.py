###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import math

from horovod.tensorflow.keras.callbacks import LearningRateScheduleCallback


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
    def __init__(self, steps, decay=100, clif=0):
        self._steps = steps
        self._decay = decay
        self._clif = clif

    # return multiplier for initial lr
    def __call__(self, step):
        if step > self._clif:
            adjusted_step = step - self._clif
            return (1 - 1 / (self._steps - self._decay) * (adjusted_step - self._decay))
        return 1.0


class CosineDecay():
    def __init__(self, steps, clif=0):
        assert steps > 0, "--cosine_decay_delay should be set to smaller value than --epochs"
        self._steps = steps
        self._clif = clif

    def __call__(self, step):
        if step > self._clif:
            adjusted_step = step - self._clif
            return 0.5 * (math.cos(2 * math.acos(0) / self._steps * adjusted_step) + 1)
        return 1.0
