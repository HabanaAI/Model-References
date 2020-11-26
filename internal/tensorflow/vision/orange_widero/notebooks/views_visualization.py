# ---
# jupyter:
#   jupytext:
#     formats: py:light
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
import os
import json
import numpy as np

from stereo.prediction.vidar.pred_utils import pred_views
from stereo.data.view_generator.view_generator import ViewGenerator
from stereo.prediction.vidar.stereo_predictor import StereoPredictor
from stereo.common.visualization.vis_utils import view_pptk3d

# %load_ext autoreload
# %autoreload 2
# -

model_confs = {'main': 'fc_u_main_sm_v0.0.3.2.json',
               'frontCornerLeft': 'fc_u_fcl_sm_v0.0.1.1.json',
               'frontCornerRight': 'fc_u_fcr_sm_v0.0.1.1.json',
               'rearCornerLeft': 'fc_u_rcl_sm_v0.0.1.1.json',
               'rearCornerRight': 'fc_u_rcr_sm_v0.0.1.1.json'
              }
conf_dir = '../stereo/models/conf'

views_names = []
predictor = {}
for k in model_confs.keys():
    json_path = os.path.join(conf_dir, model_confs[k])
    with open(json_path, 'rb') as fp:
        conf = json.load(fp)
    predictor[k] = StereoPredictor(json_path, sector_name=k)
    views_names.extend(predictor[k].views_names)

clip_name = '19-06-04_12-01-21_Alfred_Front_0083'
view_gen= ViewGenerator(clip_name, views_names, mode='pred', no_labels=True)

gi = 342185  # you can run "ishow [clip_name]" in the linux shell to play the clip and pick a gi (grab index)

views = view_gen.get_gi_views(gi)

pcd = np.zeros((0, 3))
grayscale = np.zeros((0,))
for section in model_confs.keys():
    pcd_, grayscale_, _, _, _ = pred_views(views, predictor[section], section)
    pcd = np.r_[pcd, pcd_]
    grayscale = np.r_[grayscale, grayscale_]

# ## Visualization code
# Assume we have point cloud (pcd) with grayscales, and a list of views names.<br>
# The following code gives you a colored visualization tool to explore intersection of views

v3d = view_pptk3d(pcd, grayscale, fix_coords_type=1)

v3d.register_views(clip_name, views_names, pcd, view_gen)

# +
# Add colored layer of points that apear in main_to_main but not in frontCornerRight_to_main

white_list_1 = ['main_to_main']
black_list_1 = ['frontCornerRight_to_main']
v3d.add_views_layer(white_list_1, black_list_1, color='red')

# +
# Add colored layer of points that apear in frontCornerLeft_to_frontCornerLeft but not 
# in parking_front_to_frontCornerLeft
# Name it - 'no parking'
# Print a summary of the current layers

white_list_2 = ['frontCornerLeft_to_frontCornerLeft']
black_list_2 = ['parking_front_to_frontCornerLeft']
v3d.add_views_layer(white_list_2, black_list_2, name="no parking", color='green', verbose=True)

# Now you can switch between layers with '[' and ']' keys

# +
# Add layer of points viewed only from the main camera

v3d.add_non_stereo_layer('main', 'yellow', verbose=True)
# -

# Clear all layers
v3d.clear_views_layers()
v3d.print_layers()

v3d.add_views_layer(white_list_1, black_list_1, color='magenta')
v3d.add_views_layer(white_list_2, black_list_2, name="no parking", color='green', verbose=True)

v3d.join_layers(['layer_1', 'no parking'])

v3d.print_layers()
