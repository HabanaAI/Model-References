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

pred = StereoPredictorCLI(['../stereo/models/conf/lidar+phot_v2.10.24_sds_big_corr_fc_diet3_reg.json'])
im_cntr, im_srnd, im_lidar, im_mask, T_cntr_srnd, focal, origin, out = pred.next_batch(); plt.show()
im_lidar_interp, im_lidar_interp_mask = interp_lidar(im_lidar[0, :, :])
#from stereo.models.conf_utils import best_model
#print(best_model(pred.args.json_path))

ll = [op.values() for op in pred.sess.graph.get_operations()]
for t in ll:
    if len(t) and t[0].name[-6:] == "Relu:0":
        print(t[0])
    if len(t) and t[0].name[-8:] == "kernel:0":
        print(t[0])


layer = pred.pred_layer(im_cntr, im_srnd, T_cntr_srnd, "unet_phase_a/down_convs/conv_4/conv2d/Relu:0")
for i in range(layer.shape[3]):
    plt.imshow(layer[0, :, :, i], origin='lower')
    plt.show()

layer = pred.pred_layer(im_cntr, im_srnd, T_cntr_srnd, "conv2d_48/kernel:0")
layer.shape

for i in range(20):
    plt.imshow(layer[0, 0, i, 0] * (pred.pred_layer(im_cntr, im_srnd, T_cntr_srnd, "argmax_phase_a/conv2d/Relu:0")[0, :, :, i]), origin='lower')
    plt.show()

plt.imshow(np.abs(pred.pred_layer(im_cntr, im_srnd, T_cntr_srnd, "conv2d_46/kernel:0")).sum(axis=1).sum(axis=0))
plt.show()

from scipy.stats import binned_statistic
hist, bins, _ = binned_statistic(np.abs(pred.pred_layer(im_cntr, im_srnd, T_cntr_srnd, "conv2d_19/kernel:0")).sum(axis=2).sum(axis=1).sum(axis=0), np.abs(pred.pred_layer(im_cntr, im_srnd, T_cntr_srnd, "conv2d_29/kernel:0")).sum(axis=2).sum(axis=1).sum(axis=0), statistic='count', bins=np.arange(0, 200, 1))
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.plot(bins[:-1], hist)

print((np.abs(pred.pred_layer(im_cntr, im_srnd, T_cntr_srnd, "conv2d_29/kernel:0")).sum(axis=2).sum(axis=1).sum(axis=0)<75).sum())
