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

from stereo.common.visualization.vis_utils import view_pptk3d
from stereo.data.sest.top_view_generator import pred_vidar_gi, get_predictors_and_views_name, get_vg
from stereo.prediction.vidar.pred_utils import read_conf
from os.path import join
from os import listdir
import numpy as np
# %load_ext autoreload
# %autoreload 2
# %matplotlib notebook
import matplotlib.pyplot as plt

# +
base_path = "/mobileye/algo_STEREO3/stereo/data/sest/vidar_preds/"
vidar_version = 'vidar.0.3.2_conf'
pred = True
vehicle = "Bella"

# clip_name, gi, pred = "19-09-22_17-13-12_Alfred_Front_0026", 52453, False
# clip_name, gi, pred = "19-10-03_09-24-55_Alfred_Front_0091", 178847, False
# clip_name, gi, pred, is_simulaotr = "19-05-14_16-12-00_Alfred_Front_0053", 116233, False, False
# clip_name, gi = "Town07_159393886122", 289
# clip_name, gi = "Town10HD_1595940849527624", 105
# clip_name, gi = "Town10HD_1596541035591698", 99
# clip_name, gi = "Town02_1596541527617066", 97
# clip_name, gi = "Town03_1596541529854127", 212
# clip_name, gi = "Town03_159654384023745", 189
# clip_name, gi = "Town03_1596544936183638", 201
# clip_name, gi = "Town10HD_1596548270294279", 82
# clip_name, gi = "Town10HD_1596551233721129", 289
# clip_name, gi = "Town05_1596614356955212", 336
# clip_name, gi = "Town04_159662602016717", 404
# clip_name, gi, vehicle = "Town02_1596627076318206", 137, "CitroenC3"
# clip_name, gi = "Town05_159663103223107", 341
clip_name, gi, pred, is_simulaotr = "Town01_1597228961701451", 512, False, True

clip_path = join("/mobileye/algo_STEREO3/stereo/data/simulator/Clips_v5", vehicle, clip_name)

# +
if pred:
    if 'predictors' not in globals():
        predictors, views_names = get_predictors_and_views_name(read_conf(vidar_version))
    vg, _, _ = get_vg(clip_path, views_names)
    pcds, pcd_attr, _, _, _ = pred_vidar_gi(clip_path, gi, vidar_version, vg, predictors, views_names)

else:
    npz_path = join(base_path, vidar_version, clip_name, "%07d.npz" % gi)
    vidar_pred = np.load(npz_path)
    vidar_pred = np.expand_dims(vidar_pred['arr_0'], 0)[0]
    pcds, pcd_attr = vidar_pred['pcds'], vidar_pred['pcd_attr']
# -

pcds = np.concatenate(pcds.values())
pcd_attr = np.concatenate(pcd_attr.values())
grayscale, confidence = pcd_attr[..., 0], pcd_attr[..., 2]

thresh = 0.03
pcds = pcds[confidence < thresh]
grayscale = grayscale[confidence < thresh]

view_pptk3d(pcds, grayscale, fix_coords_type=1)

# +
sectors = {'frontCornerLeft': ['parking_left', 'parking_front'], 'main': ['parking_front', 'frontCornerLeft', 'frontCornerRight'], 'frontCornerRight': ['parking_right', 'parking_front'], 'rearCornerLeft': ['parking_left', 'parking_rear'], 'rear':['rearCornerRight','rearCornerLeft', 'parking_rear'], 'rearCornerRight':['parking_right', 'parking_rear']}
views = []

for s in sectors:
    views += ["%s_to_%s" %(c, s) for c in sectors[s] + [s]]
if is_simulaotr:     
    from stereo.data.view_generator.simulator_view_generator import SimulatorViewGenerator

    vg = SimulatorViewGenerator(clip_path, views)
else:
    from stereo.data.view_generator.view_generator import ViewGenerator
    vg = ViewGenerator(clip_name, views, mode='pred')
# -

gi_views = vg.get_gi_views(gi)

img = gi_views['main_to_main']['image']
fig = plt.figure()
ax = plt.subplot(111)
ax.imshow(img, origin='lower', cmap='gray')


