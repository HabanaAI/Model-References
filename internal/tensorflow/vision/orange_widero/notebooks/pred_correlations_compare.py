# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: stereo_py27
#     language: python
#     name: stereo_py27
# ---

# +
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from stereo.prediction.vidar.pred_utils import pred_views

from stereo.common.visualization.vis_utils import view_pptk3d
from stereo.data.dataset_utils import ViewDatasetIndex
from stereo.data.view_generator.view_generator import ViewGenerator
from stereo.prediction.vidar.stereo_predictor import StereoPredictor

import numpy as np
import matplotlib.pyplot as plt

# +
model_1 = {'model_name': 'diet_main_sky_v1.3.2.2',
           'restore_iter': -1,
           'section_name': 'main',
           'checkpoint_path': None,
           'from_meta': True}
model_2 = {'model_name': 'diet_main_sm_reg_v1.2.2.2',
           'restore_iter': -1, #580000,
           'section_name': 'main',
           'checkpoint_path': None,
           'from_meta': True}

models = [model_1, model_2]


extra_tensor_names = [['arch/corr_scores_l2/add_1'],
                      ['arch/corr_scores_l2/add_1']]

predictors = []
for model in models:
    model_conf_file = '../stereo/models/conf/stereo/' + model['model_name'] + '.json'
    predictors.append(StereoPredictor(model_conf_file, 
                                      sector_name=model['section_name'], 
                                      restore_iter=model['restore_iter'],
                                      checkpoint_path=model['checkpoint_path'],
                                      from_meta=model['from_meta']))

# -

# frame = 'Town02_1591133552@gi@0047822'
# frame = '19-05-14_16-27-44_Alfred_Front_0072@gi@0151551'
# frame = '19-05-14_11-44-47_Alfred_Front_0081@gi@155179'
# frame = '19-09-11_15-17-08_Alfred_Front_0025@gi@324539'
# frame = '19-01-15_09-56-27_Front_0028@gi@2222819'
# frame = '19-09-22_16-51-54_Alfred_Front_0003@gi@007577'
# frame = '19-09-23_09-22-55_Alfred_Front_0079@gi@167073'
# frame = '19-09-23_16-24-24_Alfred_Front_0104@gi@1077259'
frame = '19-03-19_15-34-05_Alfred_Front_0049@gi@633107'
# frame = '18-08-29_10-11-49_Alfred_Front_0100@gi@248351'
# frame = '19-03-19_15-03-31_Alfred_Front_0014@gi@567807'
# frame = '19-09-23_16-25-19_Alfred_Front_0105@gi@'+str(1079843+4*8)

if 'frame' in locals():
    if frame.startswith('Town'):
        dataset_dir = '/mobileye/algo_STEREO3/stereo/data/simulator/v4/main'
    else:
        dataset_dir = '/mobileye/algo_STEREO3/stereo/data/view_dataset.v3.1'
    view_names = ['main_to_main', 'frontCornerLeft_to_main', 'frontCornerRight_to_main', 'parking_front_to_main']
    vdsi = ViewDatasetIndex(dataset_dir)
    views = vdsi.read_views(view_names, frame)
else:
    if 'vds_clip_name' not in locals() or vds_clip_name != clip_name:
        vds = ViewGenerator(clip_name, view_names=predictors[0].views_names, predump=None, mode='pred', 
                            etc_dir=None, try_creating_lidar=True)
        vds_clip_name = clip_name

    if 'views_gi' not in locals() or views_gi != gi:
        views = vds.get_gi_views(gi=gi)
        views_gi = gi

outs = []
corrs = []
for pred, extra_tensor_names_ in zip(predictors, extra_tensor_names):
    out, extra_tensors = pred.pred(extra_tensor_names_, views=views)
    if isinstance(out, dict):
        out = out['out']
    corr_scores_l2_max = extra_tensors[-1]
    outs.append(out)
    corrs.append(corr_scores_l2_max)

# +
# # %matplotlib nbagg

plt.figure()
plt.imshow(views['parking_front_to_main']['image'], origin='lower', cmap='gray')
plt.figure()
plt.imshow(views['frontCornerLeft_to_main']['image'], origin='lower', cmap='gray')
plt.figure()
plt.imshow(views['frontCornerRight_to_main']['image'], origin='lower', cmap='gray')
plt.figure()
plt.imshow(views['main_to_main']['image'], origin='lower', cmap='gray')


for out, corr_scores_l2_max in zip(outs, corrs):
    plt.figure()
    plt.imshow(np.squeeze(out), origin='lower')

    plt.figure()
    plt.imshow(-np.squeeze(np.argmax(corr_scores_l2_max[:,:,:,:], axis=3)), origin='lower')
# -
from stereo.common.visualization.vis_utils import points3d_from_image
cam = 'main'
try:
    has_lidar = 'short' in views[cam + '_to_' + cam]['lidars'].keys()
except:
    has_lidar = False
if has_lidar:
    lidar_pcd, lidar_grayscale = points3d_from_image(views[cam + '_to_' + cam]['lidars']['short'], 
                                                     views[cam + '_to_' + cam]['image']/255.,
                                                     views[cam + '_to_' + cam]['origin'],
                                                     [views[cam + '_to_' + cam]['focal'],
                                                     views[cam + '_to_' + cam]['focal']])
else:
    lidar_pcd, lidar_grayscale = np.zeros((0,3)), np.zeros((0,))

for pred in predictors:
    pcds, grayscales, depth_images, images, _ = pred_views(views, pred, cam)
    # view_pptk3d(pcds, grayscales, fix_coords_type=1, save_to_tmp=False)
    view_pptk3d(np.concatenate([pcds, lidar_pcd], 0), 
                np.concatenate([grayscales, np.zeros_like(lidar_grayscale)]), 
                fix_coords_type=1, save_to_tmp=False)

# +
# import tensorflow as tf

# # Print the model graph

# for n in tf.get_default_graph().as_graph_def().node:
#     print(n.name)
# -


