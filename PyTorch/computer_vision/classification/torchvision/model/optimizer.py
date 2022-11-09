# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company

import torch
from torch.optim.lr_scheduler import _LRScheduler

class PolynomialDecayWithWarmup(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step

    Args
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """

    def __init__(self, optimizer, batch_size, steps_per_epoch, train_steps, initial_learning_rate=9.0, warmup_epochs=3,
                 end_learning_rate=0.0001, power=2.0, lars_decay_epochs=36, opt_name=None):
        self.last_step = 0
        self.steps_per_epoch = steps_per_epoch
        self.train_steps = train_steps
        self.initial_learning_rate = initial_learning_rate
        self.warmup_epochs = warmup_epochs
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.warmup_steps = warmup_epochs * (steps_per_epoch - 1)
        self.decay_steps = lars_decay_epochs * (steps_per_epoch - 1) - self.warmup_steps + 1
        self.opt_name = opt_name.lower()

        super().__init__(optimizer)

    def get_lr(self):
        warmup_steps = self.warmup_steps
        warmup_rate = (
            self.initial_learning_rate * self.last_step / warmup_steps)

        poly_steps = self.last_step - warmup_steps
        poly_steps = poly_steps if poly_steps > 1 else 1

        poly_rate = (self.initial_learning_rate - self.end_learning_rate) * \
            ((1 - poly_steps / self.decay_steps) **
             (self.power)) + self.end_learning_rate

        decay_rate = warmup_rate if self.last_step <= warmup_steps else poly_rate
        return decay_rate

    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        if self.last_step <= self.decay_steps + self.warmup_steps:
            decay_lrs = [self.get_lr()]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr