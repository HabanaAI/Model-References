# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 2
#     language: python
#     name: python2
# ---

import tensorflow as tf
import imp

arch = "/homes/galt/gitlab/stereo/stereo/models/arch/sdsv3_v0.4.0_stereo_single_phase_corr_fc_u_sm.py"

arch_func_ = imp.load_source('arch', arch).arch

inputs = [tf.keras.Input(batch_size=1, dtype=tf.float32, shape=(3,310,720)), 
          tf.keras.Input(batch_size=1, dtype=tf.float32, shape=(2, 3)), 
          tf.keras.Input(batch_size=1, dtype=tf.float32, shape=(2,)), 
          tf.keras.Input(batch_size=1, dtype=tf.float32, shape=(2,))]
out = arch_func_(inputs[0], inputs[1], inputs[2], inputs[3], 
                 None, None, keras=True)

model = tf.keras.Model(inputs=inputs, outputs=out)

model.summary()

model.run_eagerly = False
model.to_json()

tf.keras.utils.plot_model(model)
