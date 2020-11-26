# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: stereo_dlo
#     language: python
#     name: stereo_dlo
# ---

# +
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.compat.v1.enable_eager_execution()
# -

# global_step = tf.train.get_or_create_global_step()
global_step = np.arange(0,1800000)

import stereo.models.lr_schedules as schedules
def get_learning_rate(global_step, learning_rate, lr_schedule):
    if lr_schedule is not None:
        learning_rate = getattr(schedules, lr_schedule['name'])(global_step, learning_rate, **lr_schedule['kwargs'])
    return learning_rate


# +
learning_rate = 1e-4
lr_schedule = {
  "name": "exponential_decay",
  "kwargs": {
    "decay_steps": 100000,
    "decay_rate": 0.8,
    "staircase": True}
}

lr = get_learning_rate(global_step, learning_rate, lr_schedule)
plt.plot(lr)
plt.grid()
# -

learning_rate = 2e-4
lr_schedule = {
  "name": "cosine_decay_restarts",
  "kwargs": {
      "first_decay_steps": 50000,
      "t_mul": 1.5,
      "m_mul": 0.85,
      "alpha": 0.001}
}
lr = get_learning_rate(global_step, learning_rate, lr_schedule)
plt.plot(lr)
plt.grid()

