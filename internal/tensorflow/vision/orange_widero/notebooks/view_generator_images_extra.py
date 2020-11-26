# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: stereo_deps_py2_v0.1.0
#     language: python
#     name: stereo_deps_py2_v0.1.0
# ---

# +
import matplotlib.pyplot as plt

from stereo.data.view_generator.view_generator import ViewGenerator
# -

clip_name, gi = '19-03-06_13-13-29_Alfred_Front_0061', 138539

vg = ViewGenerator(clip_name, view_names=['main_to_main'], mode='pred', 
                   no_labels=True, require_lidar=False)

views = vg.get_gi_views(gi)
view = views['main_to_main']

# +
plt.figure(figsize=[15,15])
plt.imshow(view['image'], origin='lower', cmap='gray')
print(view['image'].shape)

plt.figure(figsize=[15,15])
plt.imshow(view['images_extra']['ltm'], origin='lower', cmap='gray')
print(view['images_extra']['ltm'].shape)

plt.figure(figsize=[15,15])
plt.imshow(view['images_extra']['red'], origin='lower', cmap='gray')
print(view['images_extra']['red'].shape)
# -


