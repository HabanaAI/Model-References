"""models.py

Implemented model: Densenet121 ('densenet121')
"""
###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################
# Changes:
# -- Removed get_lr_func() method
# -- Added step_decay_with_warmup and StepLearningRateScheduleWithWarmup()

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.python.keras import backend
from tensorflow.python.framework import ops
from config import config
from .adamw import AdamW
from .optimizer import convert_to_accum_optimizer
from .optimizer import convert_to_lookahead_optimizer


def step_decay_with_warmup(global_step, initial_lr, warmup_steps, decay_schedule):
    if global_step < warmup_steps:
      return initial_lr * float(global_step) / warmup_steps
    else:
      decay_multiplier = [mulitplier for (decay_step, mulitplier) in decay_schedule.items()
                          if decay_step <= global_step][-1]
      return initial_lr * decay_multiplier


class StepLearningRateScheduleWithWarmup(tf.keras.callbacks.Callback):
  def __init__(self, initial_lr, initial_global_step, warmup_steps, decay_schedule, verbose=0):
      super(StepLearningRateScheduleWithWarmup, self).__init__()

      self.warmup_steps = warmup_steps
      self.initial_lr = initial_lr
      self.global_step = initial_global_step
      self.verbose = verbose
      self.decay_schedule = decay_schedule
      self.learning_rates = []
      self.verbose = verbose

  def on_train_batch_end(self, batch, logs=None):
      learning_rate = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
      self.learning_rates.append((self.global_step, learning_rate))
      self.global_step = self.global_step + 1

  def on_train_batch_begin(self, batch, logs=None):
      learning_rate = step_decay_with_warmup(global_step = self.global_step,
                                             initial_lr = self.initial_lr,
                                             warmup_steps = self.warmup_steps,
                                             decay_schedule = self.decay_schedule)
      tf.keras.backend.set_value(self.model.optimizer.learning_rate, learning_rate)
      if self.verbose > 0 and (self.global_step + 1) % 100 == 0:
        print('Step %5d: learning rate: %.5f' % (self.global_step + 1, learning_rate))


def get_optimizer(model_name, optim_name, initial_lr, epsilon=1e-2):
    """get_optimizer

    Note:
    1. Learning rate decay is implemented as a callback in model.fit(),
       so I do not specify 'decay' in the optimizers here.
    2. Refer to the following for information about 'epsilon' in Adam:
       https://github.com/tensorflow/tensorflow/blob/v1.14.0/tensorflow/python/keras/optimizer_v2/adam.py#L93
    """
    if optim_name == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=initial_lr, momentum=0.9, nesterov=True)
    elif optim_name == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=initial_lr, epsilon=epsilon)
    elif optim_name == 'rmsprop':
        return tf.keras.optimizers.RMSprop(learning_rate=initial_lr, epsilon=epsilon,
                                           rho=0.9)
    else:
        # implementation of 'AdamW' is removed temporarily
        raise ValueError