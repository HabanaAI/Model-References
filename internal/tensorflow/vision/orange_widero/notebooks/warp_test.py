# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernel_info:
#     name: pycharm-db2981c2
#   kernelspec:
#     display_name: Python 2
#     language: python
#     name: python2
# ---

# + inputHidden=false outputHidden=false
import matplotlib.pyplot as plt
from stereo.data.dataset_queue import DatasetQueue as DSQ
from stereo.data.dataset_queue import get_dataset_queue_batch
from stereo.common.visualization.vis_utils import depth_visualization

# + inputHidden=false outputHidden=false
camera_cntr = 'main'
camera_srnd = ['frontCornerLeft', 'frontCornerRight']
level = -2
scale = 0.5
frame_step = 30
crop_rect = [-720, 720, -310, 310]
calib_base = '/mobileye/algo_DL/ofers/data/stereo/surround_calibs'
mest_itrks_dir = '/mobileye/algo_DL/ofers/data/stereo/dataset_v3/itrks/'
gtem_itrks_dir = '/mobileye/algo_DL/ofers/data/stereo/dataset_v3/gtem/main/'
min_pair_dist = 0.5

pls_name = 'Clips_Front_224899_11.pls'

# + inputHidden=false outputHidden=false
dsq = DSQ(pls_name, camera_cntr, camera_srnd, level, scale, frame_step, crop_rect, calib_base, lidar_scope=20,
          mest_itrks_dir=mest_itrks_dir, gtem_itrks_dir=gtem_itrks_dir, min_pair_dist=min_pair_dist)

# +
dsq.curr_gfi = 650 # 596

im_cntr1, im_srnd1, im_lidar1, im_mask1, \
im_cntr2, im_srnd2, im_lidar2, im_mask2, \
T_cntr_srnd, RT12, focal, origin = get_dataset_queue_batch(batch_sz=1, dsq=dsq)
# -

"""
plt.figure(figsize=(16, 5)); plt.imshow(im_cntr1[0, :, :], origin='lower', cmap='gray'); plt.show()
plt.figure(figsize=(16, 5)); plt.imshow(im_mask1[0, :, :], origin='lower', cmap='gray'); plt.show()
plt.figure(figsize=(16, 5)); plt.imshow(depth_visualization(im_lidar1[0, :, :]), origin='lower', cmap='gray'); plt.show()
plt.figure(figsize=(16, 5)); plt.imshow(blend_depth(im_cntr1[0, :, :], im_lidar1[0, :, :]), origin='lower'); plt.show()
plt.figure(figsize=(16, 5)); plt.imshow(im_srnd1[0, 0, :, :], origin='lower', cmap='gray'); plt.show()
plt.figure(figsize=(16, 5)); plt.imshow(im_srnd1[0, 1, :, :], origin='lower', cmap='gray'); plt.show()
"""
pass

"""
plt.figure(figsize=(16, 5)); plt.imshow(im_cntr2[0, :, :], origin='lower', cmap='gray'); plt.show()
plt.figure(figsize=(16, 5)); plt.imshow(im_mask2[0, :, :], origin='lower', cmap='gray'); plt.show()
plt.figure(figsize=(16, 5)); plt.imshow(depth_visualization(im_lidar2[0, :, :]), origin='lower'); plt.show()
plt.figure(figsize=(16, 5)); plt.imshow(blend_depth(im_cntr2[0, :, :], im_lidar2[0, :, :]), origin='lower'); plt.show()
plt.figure(figsize=(16, 5)); plt.imshow(im_srnd2[0, 0, :, :], origin='lower', cmap='gray'); plt.show()
plt.figure(figsize=(16, 5)); plt.imshow(im_srnd2[0, 1, :, :], origin='lower', cmap='gray'); plt.show()
"""
pass

# +
import numpy as np
from scipy.interpolate import griddata
from scipy.signal import convolve2d

def interp_lidar(im_lidar, mask_kernel_sz=7):   
    x, y = np.meshgrid(range(im_lidar.shape[1]), range(im_lidar.shape[0]))
    lidar_y, lidar_x = np.where(im_lidar > 0)
    lidar_Z = im_lidar[im_lidar > 0]
    im_lidar_interp = griddata((lidar_y, lidar_x), lidar_Z**-1, (y, x), method='linear')**-1
    im_lidar_interp_mask = convolve2d((im_lidar>0)*1.0, np.ones((3, 3)), 'same') > 0
    return im_lidar_interp, im_lidar_interp_mask

im_lidar1_interp, im_lidar1_interp_mask = interp_lidar(im_lidar1[0, :, :])
im_lidar2_interp, im_lidar2_interp_mask = interp_lidar(im_lidar2[0, :, :])

plt.figure(figsize=(16, 5)); plt.imshow(depth_visualization(im_lidar2[0, :, :]), origin='lower'); plt.show()
plt.figure(figsize=(16, 5)); plt.imshow(depth_visualization(im_lidar2_interp[:, :]), origin='lower'); plt.show()
plt.figure(figsize=(16, 5)); plt.imshow(im_lidar2_interp_mask, origin='lower'); plt.show()
# -

import tensorflow as tf
from stereo.models.loss_utils import warp
from matplotlib import animation
# %matplotlib notebook

# +
num_srnd = im_srnd1.shape[1]
im_sz = im_cntr1.shape[1:3]
left, right = -origin[0, 0], im_sz[1] - origin[0, 0]
bottom, top = -origin[0, 1], im_sz[0] - origin[0, 1]
x, y = np.meshgrid(np.arange(left, right), np.arange(bottom, top))
x_ = tf.Variable((x[np.newaxis, :, :, np.newaxis]).astype('float32'))
y_ = tf.Variable((y[np.newaxis, :, :, np.newaxis]).astype('float32'))
focal_ = tf.Variable(focal.astype('float32'))
origin_ = tf.Variable(origin.astype('float32'))

RT21 = tf.linalg.inv(RT12)
I1 = tf.Variable(im_cntr1[:, :, :, np.newaxis].astype('float32'))
I2 = tf.Variable(im_cntr2[:, :, :, np.newaxis].astype('float32'))
I10 = tf.Variable(im_srnd1[0, 0, :, :][np.newaxis, :, :, np.newaxis].astype('float32'))
I11 = tf.Variable(im_srnd1[0, 1, :, :][np.newaxis, :, :, np.newaxis].astype('float32'))
Z1_inv = tf.Variable(im_lidar1_interp[np.newaxis, :, :, np.newaxis].astype('float32')**-1)
Z2_inv = tf.Variable(im_lidar2_interp[np.newaxis, :, :, np.newaxis].astype('float32')**-1)

# +
I1_warped_to_I2_ = warp(I1, Z2_inv, focal_, origin_, RT21, x_, y_)
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
I1_warped_to_I2 = sess.run([I1_warped_to_I2_])[0]

fig = plt.figure(figsize=(9,4))
ax1=fig.add_subplot(111)
ims = []
for i,im in enumerate([im_lidar2_interp_mask*I1_warped_to_I2[0, :, :, 0], 
                       im_lidar2_interp_mask*im_cntr2[0, :, :]]):
    ims.append([ax1.imshow(im, cmap='gray', origin='lower', animated=True)])

ani = animation.ArtistAnimation(fig, ims, interval=600, blit=True)

# +
T_cntr_srnd0_ = tf.Variable(T_cntr_srnd.astype('float32')[:, 0, :])
I10_warped_to_I1_ = warp(I10, Z1_inv, focal_, origin_, T_cntr_srnd0_, x_, y_)
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
I10_warped_to_I1 = sess.run([I10_warped_to_I1_])[0]

fig = plt.figure(figsize=(9,4))
ax1=fig.add_subplot(111)
ims = []
for i,im in enumerate([im_lidar1_interp_mask*I10_warped_to_I1[0, :, :, 0], 
                       im_lidar1_interp_mask*im_cntr1[0, :, :]]):
    ims.append([ax1.imshow(im, cmap='gray', origin='lower', animated=True)])

ani = animation.ArtistAnimation(fig, ims, interval=600, blit=True)

# +
T_cntr_srnd1_ = tf.Variable(T_cntr_srnd.astype('float32')[:, 1, :])
I11_warped_to_I1_ = warp(I11, Z1_inv, focal_, origin_, T_cntr_srnd1_, x_, y_)
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
I11_warped_to_I1 = sess.run([I11_warped_to_I1_])[0]

fig = plt.figure(figsize=(9,4))
ax1=fig.add_subplot(111)
ims = []
for i,im in enumerate([im_lidar1_interp_mask*I11_warped_to_I1[0, :, :, 0], 
                       im_lidar1_interp_mask*im_cntr1[0, :, :]]):
    ims.append([ax1.imshow(im, cmap='gray', origin='lower', animated=True)])

ani = animation.ArtistAnimation(fig, ims, interval=600, blit=True)
