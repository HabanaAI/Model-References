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
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from stereo.prediction.vidar.pred_utils import pred_views
import open3d as o3d

from stereo.common.visualization.vis_utils import view_pptk3d
from stereo.data.dataset_utils import ViewDatasetIndex
from stereo.data.view_generator import ViewGenerator
from stereo.prediction.vidar.stereo_predictor import StereoPredictor

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# +
from_meta = False
restore_iter = 400000

section_name = 'main'
models = [
    {'model_name': 'diet_main_v3.0.v0.0.12_conf',
     'restore_iter': restore_iter,
     'section_name': section_name,
     'checkpoint_path': None,
     'from_meta': from_meta}
]

# # section_name = 'rearCornerLeft'
# # section_name = 'rearCornerRight'
# # section_name = 'frontCornerLeft'
# section_name = 'frontCornerRight'
# models = [
#     {'model_name': 'diet_combined_corners_v3.0.v0.0.2_conf',
#      'restore_iter': restore_iter,
#      'section_name': section_name,
#      'checkpoint_path': None,
#      'from_meta': from_meta}
# ]

# section_name = 'rear'
# models = [
#     {'model_name': 'diet_main_rear_v3.0.v0.0.3_conf',
#      'restore_iter': restore_iter,
#      'section_name': section_name,
#      'checkpoint_path': None,
#      'from_meta': from_meta}
# ]
# -

predictors = []
for i, model in enumerate(models):
    print("{} | Loading model: {}".format(i, model['model_name']))
    model_conf_file = '../stereo/models/conf/stereo/' + model['model_name'] + '.json'
    predictors.append(StereoPredictor(model_conf_file, 
                                      sector_name=model['section_name'], 
                                      restore_iter=model['restore_iter'],
                                      checkpoint_path=model['checkpoint_path'],
                                      from_meta=model['from_meta']))

# clip_name, gi = '18-10-22_11-20-37_Diego_Front_0053', 109802
# frame = '19-05-14_13-23-59_Alfred_Front_0015@gi@0032969'
frame = '19-05-14_11-20-42_Alfred_Front_0052@gi@0102837'
# frame = '19-05-14_11-03-06_Alfred_Front_0032@gi@0065745'
# frame = '18-12-26_11-57-36_Alfred_Front_0073@gi@0261453'
# frame = '19-05-14_15-40-30_Alfred_Front_0020@gi@0048435'
# frame = '19-05-14_15-40-30_Alfred_Front_0020@gi@0048435'
# frame = '19-05-14_13-38-48_Alfred_Front_0033@gi@0065133'

if 'frame' in locals():
    if frame.startswith('Town'):
        dataset_dir = '/mobileye/algo_STEREO3/stereo/data/simulator/v3/main'
    else:
        dataset_dir = '/mobileye/algo_STEREO3/stereo/data/view_dataset.v3.1'
    view_names = predictors[0].views_names
    vdsi = ViewDatasetIndex(dataset_dir)
    views = vdsi.read_views(view_names, frame)
else:
    if 'vds_clip_name' not in locals() or vds_clip_name != clip_name:
        vds = ViewGenerator(clip_name, view_names=predictors[0].views_names, predump=None, mode='pred', 
                            no_labels=True, etc_dir=None)
        vds_clip_name = clip_name

    if 'views_gi' not in locals() or views_gi != gi:
        views = vds.get_gi_views(gi=gi)
        views_gi = gi

# +
extra_tensor_names = ['arch/corr_scores_l2/Maximum_1'] if len(predictors[0].views_names) == 4 else ['arch/corr_scores_l2/Maximum']
extra_tensor_names.append('arch/out_conf')

outs = []
confs = []
corrs = []
for pred in predictors:
    out, extra_tensors = pred.pred(views=views,
                                   extra_tensor_names=extra_tensor_names)
    if isinstance(out, dict):
        out = out['out']
    corr_scores_l2_max = extra_tensors[0]
    error_map = extra_tensors[-1]
    outs.append(out)
    confs.append(error_map)
    corrs.append(corr_scores_l2_max)


# +
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
            img = np.squeeze(img_cols[col]['imgs'][row])
            axes[row, col].imshow(img, origin=origin, cmap=cmap)
#             axes[row, col].axis('off')
            axes[row, col].set_yticks([])
#             axes[row, col].set_xticks([])
            axes[row, col].axis('tight')            
         
view_imgs =      {'imgs': [views[k]['image'] for k in views.keys()],
                  'cmap': 'gray'}
inv_depth_imgs = {'imgs': outs}
corr_imgs =      {'imgs': [-np.argmax(corr_scores_l2_max[:,:,:,:], axis=3) for corr_scores_l2_max in corrs]}
inv_error_imgs = {'imgs': confs, 'cmap': 'jet'}
error_imgs =     {'imgs': [c*(o**-2) for c, o in zip(confs, outs)], 'cmap': 'jet'}
rel_error_imgs =     {'imgs': [c*(o**-1) for c, o in zip(confs, outs)], 'cmap': 'jet'}

img_cols = [inv_depth_imgs, rel_error_imgs, inv_error_imgs, error_imgs, corr_imgs]

plot_images([view_imgs], row_names=list(views.keys()), div_factor=90.0)
plot_images(img_cols, row_names=[m['model_name'] for m in models])
# -

from stereo.prediction.vidar.pred_utils import filter_outliers, distacnce_and_height_filter
from stereo.common.visualization.vis_utils import points3d_from_image

import matplotlib.colors as mcolors
import matplotlib.cm as cm
colors = mcolors.BASE_COLORS.keys()[::-1]
colors.remove('k')
colors.remove('w')
cmap = mcolors.ListedColormap(colors)
rgb_colors = [mcolors.to_rgb(c) for c in colors]

viewers = []
for i in range(len(models)):
#     pcd, pcd_color, depth_image, image, _ = pred_views(views, pred, 'main', with_conf=True)
    pcd, pcd_attr, depth_image, image, _ = pred_views(views, predictors[i], section_name, with_conf=True,
                                                   filter_outliers_thresh=-1, dist_and_height=True)
       
    grayscale = pcd_attr[:, 0]
    inv_error = pcd_attr[:, 1]
    rel_error = pcd_attr[:, 2]
    print("model:{}, mask color: {}".format(models[i]['model_name'], colors[i]))
    
    viewers.append(view_pptk3d(pcd, grayscale, fix_coords_type=1, save_to_tmp=False))
    viewers[-1].show_confidence(rel_error, threshold=0.04, mask_color=rgb_colors[i])

# +
# # !pkill -f pptk/viewer/viewer

# +
# # !ps aux | grep pptk/viewer/viewer
