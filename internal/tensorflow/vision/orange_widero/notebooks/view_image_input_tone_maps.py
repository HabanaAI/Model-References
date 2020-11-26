# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 2
#     language: python
#     name: python2
# ---

# +
import numpy as np
import matplotlib.pyplot as plt

from mepy_algo.appcode.ground_truth.utils.clip_utils import get_fof_from_clip
from stereo.data.frame_utils import get_img, rcc_interp
from stereo.data.view_image_input import ViewImageInput
# -

clip_name, gi = '19-03-06_13-13-29_Alfred_Front_0061', 138539

clip = get_fof_from_clip(clip_name, compact=True)
vii = ViewImageInput(clip, camera_from='main', camera_to='main', level=-2, undistort=True, 
                     crop_rect=[-720, 720, -310, 310], car_body_mask=None)

# +
meim_ltm = vii.get_frame(gi, tone_map='ltm')[0]
plt.figure(figsize=[20,20])
plt.imshow(meim_ltm.im, origin='lower', cmap='gray')
print(meim_ltm.im.shape)

meim_gtm = vii.get_frame(gi, tone_map='gtm')[0]
plt.figure(figsize=[20,20])
plt.imshow(meim_gtm.im, origin='lower', cmap='gray')
print(meim_gtm.im.shape)

meim_red = vii.get_frame(gi, tone_map='red')[0]
plt.figure(figsize=[20,20])
plt.imshow(meim_red.im, origin='lower', cmap='gray')
print(meim_red.im.shape)

meim_clear = vii.get_frame(gi, tone_map='clear')[0]
plt.figure(figsize=[20,20])
plt.imshow(meim_clear.im, origin='lower', cmap='gray')
print(meim_clear.im.shape)
