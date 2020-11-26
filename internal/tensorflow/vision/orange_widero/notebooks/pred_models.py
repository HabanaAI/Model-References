# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: latest-venv
#     language: python
#     name: latest-venv
# ---

# +
import open3d

from stereo.data.dataset_utils import ViewDatasetIndex
from stereo.prediction.vidar.pred_utils import pred_views
from stereo.common.visualization.vis_utils import view_pptk3d
from stereo.data.view_generator.view_generator import ViewGenerator
from stereo.prediction.vidar.stereo_predictor import StereoPredictor
from stereo.common.general_utils import tree_base

from os.path import join
import numpy as np

# +
confs = {'main': 'stereo/diet_main_sm_reg_v0.0.2.2.json',
         'rear': 'stereo/diet_main_rear_v3.0.v0.0.3.json',
         'frontCornerLeft': 'stereo/diet_combined_corners_v3.0.v0.0.2.json',
         'frontCornerRight': 'stereo/diet_combined_corners_v3.0.v0.0.2.json',
         'rearCornerLeft': 'stereo/diet_combined_corners_v3.0.v0.0.2.json',
         'rearCornerRight': 'stereo/diet_combined_corners_v3.0.v0.0.2.json',
         }

restore_iter = {'main': -1,
                'rear': -1,
                'frontCornerLeft': 100000,
                'frontCornerRight': 100000,
                'rearCornerLeft': 100000,
                'rearCornerRight': 100000}

filter_outliers_thresh = {'main': 0.008,
                          'rear': 0.008,
                          'frontCornerLeft': 0.003,
                          'frontCornerRight': 0.003,
                          'rearCornerLeft': 0.003,
                          'rearCornerRight': 0.003}
# -

predictors = {}
for k in confs.keys():
    predictors[k] = StereoPredictor(join(tree_base(), 'stereo', 'models', 'conf', confs[k]),
                                    sector_name=k, restore_iter=restore_iter[k])

view_names = []
for k in confs.keys():
    view_names.extend(predictors[k].views_names)

# clip_name, gi = '19-05-19_16-45-49_Alfred_Front_0096', 346285
frame = '19-05-14_16-27-44_Alfred_Front_0072@gi@0151551'

if 'frame' in locals():
    if frame.startswith('Town'):
        dataset_dir = '/mobileye/algo_STEREO3/stereo/data/simulator/v3/main'
    else:
        dataset_dir = '/mobileye/algo_STEREO3/stereo/data/view_dataset.v3.1'
    vdsi = ViewDatasetIndex(dataset_dir)
    views = vdsi.read_views(view_names, frame)
else:
    if 'vds_clip_name' not in locals() or vds_clip_name != clip_name:
        vds = ViewGenerator(clip_name, view_names=pred.views_names, predump=None, mode='pred',
                            no_labels=True, etc_dir=None)
        vds_clip_name = clip_name

    if 'views_gi' not in locals() or views_gi != gi:
        views = vds.get_gi_views(gi=gi)
        views_gi = gi

# +
pcds = []
grayscales = []
for k in confs.keys():
    pcd_, grayscale_, _, _, _ = pred_views(views, predictors[k], k, run_filter=True,
                                           filter_outliers_thresh=filter_outliers_thresh[k], azimuthal_sections=False)
    pcds.append(pcd_)
    grayscales.append(grayscale_)

view_pptk3d(np.concatenate(pcds), np.concatenate(grayscales), fix_coords_type=1, save_to_tmp=False)
# -
