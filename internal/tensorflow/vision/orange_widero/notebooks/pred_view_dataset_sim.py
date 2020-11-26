# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 2
#     language: python
#     name: python2
# ---

import numpy as np
from matplotlib import pyplot as plt
from stereo.data.view_generator.simulator_view_generator import SimulatorViewGenerator
from stereo.common.visualization.vis_utils import warp_source2target_by_Z as warp
from matplotlib.animation import ArtistAnimation


import matplotlib
# %matplotlib notebook
import matplotlib.animation

# +
clip_path, gi = "/mobileye/algo_STEREO3/stereo/data/simulator/Clips_v5.1/Alfred/Town10HD_1597146728764465", 750
# clip_path, gi = "/mobileye/algo_STEREO3/guygo/temp/Alfred", 97
# clip_path, gi = "/mobileye/algo_STEREO3/guygo/temp/Test/Town10HD_1596441194552786", 218
# view1 = "main_to_main"
# view1 = 'rear_to_rear'
# view1 = 'rearCornerRight_to_rearCornerRight'
view1 = "frontCornerLeft_to_frontCornerLeft"
# view1 = "frontCornerRight_to_frontCornerRight"
# view1 = "rearCornerLeft_to_rearCornerLeft"

# view2 = "rearCornerLeft_to_rearCornerLeft"
# view2 = "frontCornerLeft_to_frontCornerLeft"
# view2 = "frontCornerRight_to_frontCornerRight"
view2 = "parking_left_to_frontCornerLeft"
# view2 = "parking_right_to_frontCornerRight"
# view2 = "frontCornerRight_to_main"
# view2 = 'rearCornerRight_to_rear'
# view2 = 'parking_rear_to_rear'
# view2 = 'rearCornerRight_to_rearCornerRight'
# view2 = "parking_left_to_rearCornerLeft"


svg = SimulatorViewGenerator(clip_path=clip_path, view_names=[view1, view2]) 
views = svg.get_gi_views(gi=gi, mask=True)

# +
Z = views[view1]['sim_depth']
RT_view1_to_view2 = np.matmul(np.linalg.inv(views[view2]['RT_view_to_main']), views[view1]['RT_view_to_main'])
warped_view2_to_view1 = warp(views[view2]['image'], views[view2]['origin'], views[view2]['focal'],
                            RT_view1_to_view2, Z, origin_target=list(views[view1]['origin']), focal_target=views[view1]['focal'])
# %matplotlib notebook
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
imgs = []
imgs.append([ax.imshow(views[view1]['image'], origin='lower', cmap='gray')])
imgs.append([ax.imshow(warped_view2_to_view1, origin='lower', cmap='gray')])

a = ArtistAnimation(fig, imgs, interval=500)
# -
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(211)
ax.imshow(views[view1]['image'], origin='lower', cmap='gray')
ax2 = fig.add_subplot(212)
ax2.imshow(views[view2]['image'], origin='lower', cmap='gray')


