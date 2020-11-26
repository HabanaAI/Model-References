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

# + pycharm={"is_executing": false}
import numpy as np
import matplotlib.pyplot as plt
from stereo.data.view_dataset import ViewDataset
from stereo.data.predump_utils import PreDumpIndex

# + pycharm={"is_executing": false}
predump = PreDumpIndex('/mobileye/algo_RP_8/jeff/TrajectoryNet/itrk', rebuild_index=False, save_index=False)

# + pycharm={"is_executing": false}
clip_name = predump.valid_front_clips_with_lidar[1000]

# + pycharm={"is_executing": false}
vds = ViewDataset(clip_name, view_names=['main_to_main'], predump=predump, mode='train')

# + pycharm={"is_executing": false}
gi = vds.gtem_main['data']['gis'][400]
views = vds.get_gi_views(gi)
# -

vds.gtem['main_to_main']['path']

plt.figure()
plt.imshow(views['main_to_main']['image'], origin='lower', alpha=1.0)
plt.imshow(views['main_to_main']['instance_masks']['vd_masks_scaled_compressed'], origin='lower', alpha=0.5)
plt.show()


