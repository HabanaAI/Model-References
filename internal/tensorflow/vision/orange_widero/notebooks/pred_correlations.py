# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: stereo_train_py36
#     language: python
#     name: stereo_train_py36
# ---

# +
import open3d
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from stereo.common.general_utils import crop_pad_symmetric
from stereo.common.visualization.vis_utils import view_pptk3d
from stereo.data.dataset_utils import ViewDatasetIndex
# from stereo.data.view_generator.view_generator import ViewGenerator
from stereo.prediction.vidar.stereo_predictor import StereoPredictor
from stereo.prediction.vidar.pred_utils import pred_views

import numpy as np
import matplotlib.pyplot as plt

# +
restore_iter = -1
# model_name = 'vidar_main_v0.2.3_conf_dlo_add'
# model_name = 'vidar_main_v0.3.0_conf_dlo_add'
# model_name = 'vidar_main_v0.4.0_conf_dlo_add'
model_name = 'vidar_main_v0.5.0_conf_dlo_taper'
extra_tensor_names = []
section_name = 'main'
# section_name = 'rear'
# section_name = 'rearCornerLeft'
# section_name = 'rearCornerRight'
# section_name = 'frontCornerLeft'
# section_name = 'frontCornerRight'

model_conf_file = '../stereo/models/conf/stereo/' + model_name + '.json'
pred = StereoPredictor(model_conf_file, sector_name=section_name, restore_iter=restore_iter)
# -

# clip_name, gi = '18-10-22_11-20-37_Diego_Front_0053', 109802
# frame = 'Town02_1587403676@gi@0006522'
# frame = '19-03-19_15-34-05_Alfred_Front_0049@632779'
frame = '19-09-23_15-18-38_Alfred_Front_0030@935857'
# frame = '19-05-14_17-06-39_Alfred_Front_0102@234923'
# frame = '19-03-19_15-04-26_Alfred_Front_0015@568667'

# +
# frame = vdsi.test_frames_list[15]
# print(frame)
# print(frame in vdsi.train_frames_list)
# print(frame in vdsi.test_frames_list)
# -

if 'frame' in locals():
    if frame.startswith('Town'):
        dataset_dir = '/mobileye/algo_STEREO3/stereo/data/simulator/v3/main'
    else:
        dataset_dir = '/mobileye/algo_STEREO3/stereo/data/view_dataset.v3.1'
    view_names = ['main_to_main', 'frontCornerLeft_to_main', 'frontCornerRight_to_main', 'parking_front_to_main']
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
out, extra_tensors = pred.pred(views=views, extra_tensor_names=extra_tensor_names)
taper_Z = pred.model_conf['model_params']['loss']['kwargs'].get('taper_Z', 0.0)
inv_Z = crop_pad_symmetric(np.squeeze((out['out']**-1 - taper_Z)**-1), (310, 720))

if 'main' in section_name:
    plt.figure()
    plt.imshow(views['parking_front_to_main']['image'], origin='lower', cmap='gray')
    plt.figure()
    plt.imshow(views['frontCornerLeft_to_main']['image'], origin='lower', cmap='gray')
    plt.figure()
    plt.imshow(views['frontCornerRight_to_main']['image'], origin='lower', cmap='gray')
    plt.figure()
    plt.imshow(views['main_to_main']['image'], origin='lower', cmap='gray')
elif section_name == 'rear':
    plt.figure()
    plt.imshow(views['parking_rear_to_rear']['image'], origin='lower', cmap='gray')
    plt.figure()
    plt.imshow(views['rearCornerLeft_to_rear']['image'], origin='lower', cmap='gray')
    plt.figure()
    plt.imshow(views['rearCornerRight_to_rear']['image'], origin='lower', cmap='gray')
    plt.figure()
    plt.imshow(views['rear_to_rear']['image'], origin='lower', cmap='gray')
elif section_name == 'rearCornerRight':
    plt.figure()
    plt.imshow(views['parking_rear_to_rearCornerRight']['image'], origin='lower', cmap='gray')
    plt.figure()
    plt.imshow(views['parking_right_to_rearCornerRight']['image'], origin='lower', cmap='gray')
    plt.figure()
    plt.imshow(views['rearCornerRight_to_rearCornerRight']['image'], origin='lower', cmap='gray')
elif section_name == 'frontCornerRight':
    plt.figure()
    plt.imshow(views['parking_front_to_frontCornerRight']['image'], origin='lower', cmap='gray')
    plt.figure()
    plt.imshow(views['parking_right_to_frontCornerRight']['image'], origin='lower', cmap='gray')
    plt.figure()
    plt.imshow(views['frontCornerRight_to_frontCornerRight']['image'], origin='lower', cmap='gray')
elif section_name == 'frontCornerLeft':
    plt.figure()
    plt.imshow(views['parking_front_to_frontCornerLeft']['image'], origin='lower', cmap='gray')
    plt.figure()
    plt.imshow(views['parking_left_to_frontCornerLeft']['image'], origin='lower', cmap='gray')
    plt.figure()
    plt.imshow(views['frontCornerLeft_to_frontCornerLeft']['image'], origin='lower', cmap='gray')
elif section_name == 'rearCornerLeft':
    plt.figure()
    plt.imshow(views['parking_rear_to_rearCornerLeft']['image'], origin='lower', cmap='gray')
    plt.figure()
    plt.imshow(views['parking_rear_to_rearCornerLeft']['image'], origin='lower', cmap='gray')
    plt.figure()
    plt.imshow(views['rearCornerLeft_to_rearCornerLeft']['image'], origin='lower', cmap='gray')
    
plt.figure()
plt.imshow(inv_Z, origin='lower', clim=[1/60.0, 1/5.0], cmap='jet')

if extra_tensors:
    corr_scores_l2_max = extra_tensors[0]
    plt.figure()
    plt.imshow(-np.squeeze(np.argmax(corr_scores_l2_max[:,:,:,:], axis=3)), origin='lower')
# -
pcds, grayscales, depth_images, images, _ = pred_views(views, pred, section_name)
view_pptk3d(pcds, grayscales, fix_coords_type=1, save_to_tmp=False)


