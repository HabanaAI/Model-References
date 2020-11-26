# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: stereo_deps_py2_v0.5.0
#     language: python
#     name: stereo_deps_py2_v0.5.0
# ---

# +
import numpy as np
import matplotlib
matplotlib.use('nbagg')
# %matplotlib notebook
from matplotlib import pyplot as plt

from stereo.common.gt_packages_wrapper import apply_rt

from stereo.data.view_generator.view_generator import ViewGenerator
from stereo.data.lidar_utils import z_buffer

# -

cntr_cams = ['main', 'frontCornerLeft', 'frontCornerRight', 'rearCornerLeft', 'rearCornerRight', 'rear']
cntr_views_names = {}
for cam in cntr_cams:
    cntr_views_names[cam] = cam+'_to_'+cam


# init ViewGenerator + all Lidars
clip_name = '20-04-06_14-40-41_Leo_Front_0014'
vg = ViewGenerator(clip_name, cntr_views_names.values(), mode='pred')
devices = ['RoboSenseFront', 'RoboSenseRight', 'RoboSenseLeft', 'RoboSenseRear', 'Velodyne']
lidars = {}
for device in devices:
    vg.clip.init_lidar(device=device)
    lidars[device] = vg.clip.lidar_data

# read ~one roll of all lidars, and transform to main cs
gi = 121477
frange = np.arange(-2,3)
cam = 'main'
P = np.zeros((0, 3))
reflectivity = np.zeros((0, ))
ptss =  np.zeros((0, ))
for gii in gi+frange:
    gii_ts = vg.clip.ts_map.ts_by_gi(gii, cam='main') * 1e-6
    for device in devices:
        # RoboSenseRear has bad calibration, discard for now
        if device == 'RoboSenseRear' :
            continue
        gii_data = lidars[device].get_lidar_data(gii)
        if gii_data is not None:
            l2c = lidars[device].get_l2c(cam=cam)
            P_ = apply_rt(gii_data[['X', 'Y', 'Z']].values, l2c)
            P = np.concatenate((P, P_[:, :]))
            reflectivity_ = gii_data[['intensity']].values.squeeze()/255.
            reflectivity = np.concatenate((reflectivity, reflectivity_))
            ptss_ = gii_data['delta']* 1e-6 + gii_ts
            ptss = np.concatenate((ptss, ptss_))

views = vg.get_gi_views(gi)

# pick a camera
cntr_cam = cntr_cams[4]
cntr_view = cntr_views_names[cntr_cam]
im_sz = views[cntr_view]['image'].shape
origin = views[cntr_view]['origin']
focal = views[cntr_view]['focal']

# compute each lidar point approximate 'pixel (image line) timestamp'
let = vg.clip.get_line_exposure_time(cam=cntr_cam)
gi_ts = vg.clip.ts_map.ts_by_gi(gi, cam='main') * 1e-6
top = vg.clip.get_image_top(cam=cntr_cam)
c2c = vg.clip.get_transformation_matrix('main', cntr_cam)
P_cam = apply_rt(P, c2c)
Z = P_cam[:,2]
pos = Z > 0
P_cam = P_cam[pos,:]
Z = Z[pos]
p_lm2 = vg.clip.focal_length(cntr_cam, level=-2)*(P_cam[:,:2]/np.c_[Z, Z])
p = focal*(P_cam[:,:2]/np.c_[Z, Z])
xs, ys = vg.clip.unrectify(p_lm2[:, 0], p_lm2[:, 1], cam=cntr_cam)
tss1 = (top - ys.astype(np.float64)) * let + gi_ts

# set thresh of diff between lidar point timestamp and 'pixel timestamp'
lm2_lines_diff_thresh = 500
ind = np.abs(tss1 - ptss[pos]) < let*lm2_lines_diff_thresh
img_lidar = z_buffer(p[ind], Z[ind], im_sz, origin, 1)
img_lidar[img_lidar == 0] = np.nan

fig = plt.figure()
ax = fig.add_subplot(111, title=cntr_cam)
plt.tight_layout()
ax.imshow(views[cntr_view]['image'], origin='lower', cmap='gray')
clim_min = np.nanquantile(1./img_lidar, 0.05)
clim_max = np.nanquantile(1./img_lidar, 0.95)
ax.imshow(1./img_lidar, origin='lower', alpha=1., clim=(clim_min, clim_max))


# +
# display all lidar points in pptk
# view_pptk3d(P, reflectivity, fix_coords_type=1, save_to_tmp=True)
# -


