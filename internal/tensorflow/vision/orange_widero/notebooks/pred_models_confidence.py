# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: stereo_latest
#     language: python
#     name: stereo_latest
# ---

# +
import open3d

from stereo.data.dataset_utils import ViewDatasetIndex
from stereo.prediction.vidar.pred_utils import pred_views
from stereo.common.visualization.vis_utils import view_pptk3d
from stereo.data.view_generator import ViewGenerator
from stereo.prediction.vidar.stereo_predictor import StereoPredictor
from stereo.common.general_utils import tree_base

from os.path import join
import numpy as np

# +
confs = {'main': 'stereo/diet_main_v3.0.v0.0.12_conf.json',
         'rear': 'stereo/diet_main_rear_v3.0.v0.0.3_conf.json',
         'frontCornerLeft': 'stereo/diet_combined_corners_v3.0.v0.0.2_conf.json',
         'frontCornerRight': 'stereo/diet_combined_corners_v3.0.v0.0.2_conf.json',
         'rearCornerLeft': 'stereo/diet_combined_corners_v3.0.v0.0.2_conf.json',
         'rearCornerRight': 'stereo/diet_combined_corners_v3.0.v0.0.2_conf.json',
         }

restore_iter = {'main': 400000,
                'rear': 400000,
                'frontCornerLeft': 400000,
                'frontCornerRight': 400000,
                'rearCornerLeft': 400000,
                'rearCornerRight': 400000}

# filter_outliers_thresh = {'main': 0.008,
#                           'rear': 0.008,
#                           'frontCornerLeft': 0.003,
#                           'frontCornerRight': 0.003,
#                           'rearCornerLeft': 0.003,
#                           'rearCornerRight': 0.003}
# -

predictors = {}
for k in confs.keys():
    predictors[k] = StereoPredictor(join(tree_base(), 'stereo', 'models', 'conf', confs[k]),
                                    sector_name=k, restore_iter=restore_iter[k], from_meta=False)

view_names = []
for k in confs.keys():
    view_names.extend(predictors[k].views_names)

# clip_name, gi = '19-05-19_16-45-49_Alfred_Front_0096', 346285
# frame = '19-05-14_16-27-44_Alfred_Front_0072@gi@0151551'
frame = '19-05-14_10-58-28_Alfred_Front_0025@gi@0054449'

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
inv_errors = []
rel_errors = []
depth_imgs = []
grayscale_imgs = []
error_maps = []

for k in confs.keys():
    pcd_, pcd_attr_, depth_image, image, extra_tensors = pred_views(views, predictors[k], k,
                                                                    with_conf=True,
                                                                    run_filter=False,
                                                                    filter_outliers_thresh=-1,
                                                                    dist_and_height=True,
                                                                    azimuthal_sections=False)
    grayscale_ = pcd_attr_[:, 0]
    inv_error_ = pcd_attr_[:, 1]
    rel_error_ = pcd_attr_[:, 2]
    
    pcds.append(pcd_)
    grayscales.append(grayscale_)
    inv_errors.append(inv_error_)
    rel_errors.append(rel_error_)
    depth_imgs.append(depth_image)
    grayscale_imgs.append(image)
    error_maps.append(extra_tensors[0])

v = view_pptk3d(np.concatenate(pcds), np.concatenate(grayscales), fix_coords_type=1, save_to_tmp=False)
v.show_confidence(np.concatenate(rel_errors), threshold=0.025)

# +
import matplotlib.pyplot as plt

def plot_images(img_cols, row_names=None, div_factor=80.0):
    for i in range(len(img_cols)):
        # if one of the elements in img_cols is a list, convert to dict
        if not isinstance(img_cols[i], dict) and isinstance(img_cols[i], list):
            img_cols[i] = {'imgs': img_cols[i]}

    nrows=len(img_cols[0]['imgs'])  # assume all img lists have the same length
    ncols=len(img_cols)
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(720*ncols/div_factor, 310*nrows/div_factor),
                             squeeze=False)
    fig.subplots_adjust(hspace=0.08, wspace=0.01)
    
    for row in range(axes.shape[0]):
        if row_names:
            axes[row, 0].set_ylabel(row_names[row], fontsize=11)

        for col in range(axes.shape[1]):
            cmap = img_cols[col].get('cmap', None)
            origin = img_cols[col].get('origin', 'lower')
            scale = img_cols[col].get('cm_scale', (None, None))
            img = np.squeeze(img_cols[col]['imgs'][row])
            axes[row, col].imshow(img, origin=origin, cmap=cmap, vmin=scale[0], vmax=scale[1])
#             axes[row, col].axis('off')
            axes[row, col].set_yticks([])
#             axes[row, col].set_xticks([])
            axes[row, col].axis('tight')            
    fig.savefig('confidence_views.png', bbox_inches = 'tight')

view_imgs =      {'imgs': grayscale_imgs, 'cmap': 'gray'}
inv_depth_imgs = {'imgs': [d**-1 for d in depth_imgs]}
rel_error_imgs = {'imgs': [e.squeeze()*d for e, d in zip(error_maps, depth_imgs)], 'cmap': 'jet', 'cm_scale': [.0, .7]}
img_cols = [view_imgs, inv_depth_imgs, rel_error_imgs]

plot_images(img_cols, row_names=confs.keys())

# +
# pcds = []
# grayscales = []
# for k in confs.keys():
#     pcd_, grayscale_, _, _, _ = pred_views(views, predictors[k], k, run_filter=True,
#                                            filter_outliers_thresh=filter_outliers_thresh[k], azimuthal_sections=False)
#     pcds.append(pcd_)
#     grayscales.append(grayscale_)

# view_pptk3d(np.concatenate(pcds), np.concatenate(grayscales), fix_coords_type=1, save_to_tmp=False)
