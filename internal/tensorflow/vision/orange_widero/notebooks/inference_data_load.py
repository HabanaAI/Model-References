# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# +
import s3fs
import os
import open3d as o3d # Required to prevent import errors

from stereo.common.s3_utils import my_read, my_list_dir
from stereo.common.visualization.vis_utils import view_pptk3d, points3d_from_image
from stereo.data.dataset_utils import ViewDatasetIndex
from stereo.interfaces.implements import load_dataset_attributes
# -

s3 = s3fs.S3FileSystem()
base_path = "s3://mobileye-team-stereo/inference/"
model_name = "diet_main_v3.0.v0.0.12_conf"
restore_iter = "520000"
ds_name, sector = "main.v3.1.3", 'main'
clips_path = os.path.join(base_path, model_name, restore_iter, ds_name)
print(clips_path)
clip_names = my_list_dir(clips_path)
clip_name = clip_names[0]

clip_path = os.path.join(clips_path, clip_name)
clip_gis = my_list_dir(clip_path)
gi_pkl = clip_gis[0]
gi = int(gi_pkl[:-4])
gi_path = os.path.join(clip_path, gi_pkl)
gi_data = my_read(gi_path)
depth_image = gi_data['out'][..., 0] ** -1

center_view = "%s_to_%s" % (sector, sector)
vdsi = ViewDatasetIndex("/mobileye/algo_STEREO3/stereo/data/view_dataset.v3.1/")
views = vdsi.read_views([center_view], "%s@gi@%s" %(clip_name, gi))
grayscale_image = views[center_view]['image'] / 255.
pcd, grayscale = points3d_from_image(depth_image, grayscale_image,
                                        views[center_view]['origin'],
                                        [views[center_view]['focal'],
                                         views[center_view]['focal']])

view_pptk3d(pcd, pcd_attr, fix_coords_type=1)


