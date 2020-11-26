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
#     display_name: PyCharm (stereo)
#     language: python
#     name: pycharm-db2981c2
# ---

# + pycharm={"is_executing": false}
from stereo.common.visualization.vis_utils import points3d_from_image, view_pptk3d

import numpy as np
import matplotlib.pyplot as plt
from stereo.data.view_dataset import ViewDataset
from stereo.models.pred import StereoPredictor

# + pycharm={"is_executing": false}
clip_name = '14-04-24_GG-EY_289_Russia_0660'
pred = StereoPredictor('../stereo/models/conf/lidar+phot_v0.5.8_sds_big_mono.json')
vds = ViewDataset(clip_name, ['mono'], predump=None, mode='pred', no_labels=True, mono=True)

# + pycharm={"name": "#%% \n", "is_executing": false}
# Taken from arbitrary Alfred clip. Hopefully has little effect
T_cntr_srnd = np.array([[ 0.57438004, -0.20033877,  0.67524585],
                        [-0.54800802, -0.19331741,  0.65412092]])[np.newaxis, :, :]

def view_to_imgs(view):
    im_cntr = view['image'][np.newaxis, :, :]
    im_srnd = np.stack((im_cntr*0, im_cntr*0), axis=1)
    im_cntr = (im_cntr.astype('float32') / 255.0)
    im_cntr[np.isnan(im_cntr)] = 0.0
    im_srnd[np.isnan(im_srnd)] = 0.0    
    return im_cntr, im_srnd    


# + pycharm={"name": "#%% \n", "is_executing": false}
# Example where sidewalk is visible
gfi = 500        
gi = vds.clip.get_grab_by_gfi(camera_name='main', gfi=gfi)
views = vds.get_gi_views(gi)
im_cntr, im_srnd = view_to_imgs(views['mono'])
out = pred.pred(im_cntr, im_srnd, T_cntr_srnd)
plt.imshow(im_cntr[0, :, :], origin='lower'); plt.show()
plt.imshow(out[0, :, :, 0], origin='lower'); plt.show()

# + pycharm={"name": "#%%\n", "is_executing": false}
# Example with speed bump. I can't tell if it's being noticed by the net. Odd artifact on the right
gfi = 117        
gi = vds.clip.get_grab_by_gfi(camera_name='main', gfi=gfi)
views = vds.get_gi_views(gi)
im_cntr, im_srnd = view_to_imgs(views['mono'])
out = pred.pred(im_cntr, im_srnd, T_cntr_srnd)
plt.imshow(im_cntr[0, :, :], origin='lower'); plt.show()
plt.imshow(out[0, :, :, 0], origin='lower'); plt.show()

# + pycharm={"name": "#%%\n", "is_executing": false}
# Another speed bump
gfi = 221        
gi = vds.clip.get_grab_by_gfi(camera_name='main', gfi=gfi)
views = vds.get_gi_views(gi)
im_cntr, im_srnd = view_to_imgs(views['mono'])
out = pred.pred(im_cntr, im_srnd, T_cntr_srnd)
plt.imshow(im_cntr[0, :, :], origin='lower'); plt.show()
plt.imshow(out[0, :, :, 0], origin='lower'); plt.show()

# + pycharm={"name": "#%%\n", "is_executing": false}
# PPTK visualization
pcd, grayscale = points3d_from_image(out[0,:,:,0]**-1, im_cntr[0,:,:], views['mono']['origin'], [732, 732])
view_pptk3d(pcd, grayscale)

