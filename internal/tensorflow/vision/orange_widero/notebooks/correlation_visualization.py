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

# +
import numpy as np
import matplotlib.pyplot as plt
# #%matplotlib notebook
from stereo.data.vis_utils import view_pptk, depth_visualization, blend_depth
from stereo.data.lidar_utils import interp_lidar
from stereo.models.pred import StereoPredictorCLI

import tensorflow as tf
# -

pred = StereoPredictorCLI(['../stereo/models/conf/lidar+phot_v2.11.5_sds_big_corr_cntr_25_batch.json'])
im_cntr, im_srnd, im_lidar, im_mask, T_cntr_srnd, focal, origin, out = pred.next_batch(); plt.show()
im_lidar_interp, im_lidar_interp_mask = interp_lidar(im_lidar[0, :, :])

[op.values() for op in pred.sess.graph.get_operations()][0:]

layer = pred.pred_layer(im_cntr, im_srnd, T_cntr_srnd, "unet_phase_a/down_convs/corr/concat:0")
layer.shape

fig, ax = plt.subplots(2, 1, figsize=(8, 8))
ax[0].imshow(im_cntr[0, :, :], origin='lower')
ax[1].imshow(layer[0, :, :, 80:125].argmax(axis=2), origin='lower')
#for i in range (126, 172):
#    print(i)
#    plt.imshow(layer[0, :, :, i], origin='lower')
#    plt.show()

for i in range (80, 125):
    print(i)
    plt.imshow(layer[0, :, :, i], origin='lower')
    plt.show()

plt.imshow(layer[0, :, :, 3], origin='lower')
plt.show()

# %matplotlib notebook
plt.imshow(im_lidar[0, :, :], origin='lower')


