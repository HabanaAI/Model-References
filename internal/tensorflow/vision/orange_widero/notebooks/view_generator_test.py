# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# %cd /homes/jeffm/work/gitlab/stereo

# +
# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib
matplotlib.use('nbagg')
# %matplotlib notebook
from matplotlib import pyplot as plt
import matplotlib.animation

from stereo.common.vis_utils import warp_source2target_by_Z as warp
from stereo.data.lidar_utils import interp_lidar

from stereo.data.view_generator import ViewGenerator
from stereo.data.predump_v2 import PreDumpIndexV2
from stereo.data.predump_utils import PreDumpIndex

from devkit.clip import MeClip
# -

cntr_cams = ['main', 'frontCornerLeft', 'frontCornerRight', 'rearCornerLeft', 'rearCornerRight', 'rear']
cntr_views_names = {}
for cam in cntr_cams:
    cntr_views_names[cam] = cam+'_to_'+cam

# predump = PreDumpIndex('/mobileye/algo_STEREO3/old_stereo/data/data_eng/')
predump = PreDumpIndexV2('/mobileye/algo_STEREO3/stereo/data/predump.v2')

clip_name = '19-01-10_11-52-50_Front_0121'
gi = 2686969
meclip = MeClip(clip_name)
gfis = np.arange(meclip.first_gfi(), meclip.last_gfi())
t0_grabs = np.array([meclip.get_grab_by_gfi(gfi=gfi, camera_name='main') for gfi in gfis])

# clip_name = '19-01-29_15-14-41_Alfred_Front_0071'
clip_name = '18-08-06_11-31-50_Alfred_Front_0006'
meclip = MeClip(clip_name)
gfis = np.arange(meclip.first_gfi(), meclip.last_gfi())
t0_grabs = np.array([meclip.get_grab_by_gfi(gfi=gfi, camera_name='main') for gfi in gfis])

vg = ViewGenerator(clip_name, cntr_views_names.values(), predump=predump, mode='train')

gi = t0_grabs[200]
views = vg.get_gi_views(gi)

cntr_cam = cntr_cams[0]
cntr_view = cntr_views_names[cntr_cam]
im_lidar = 1*views[cntr_view]['lidars']['medium']
im_lidar[im_lidar == 0.] = np.nan
fig = plt.figure()
ax = fig.add_subplot(111, title=cntr_cam)
plt.tight_layout()
ax.imshow(views[cntr_view]['image'], origin='lower', cmap='gray')
ax.imshow(1./im_lidar, origin='lower', alpha=1.)
# ax.imshow(views[cntr_view]['masks']['car_body'], origin='lower', alpha=0.5)

fig = plt.figure()
ax = fig.add_subplot(111, title=cntr_cam)
plt.tight_layout()
ax.imshow(views[cntr_view]['image'], origin='lower', cmap='gray')
