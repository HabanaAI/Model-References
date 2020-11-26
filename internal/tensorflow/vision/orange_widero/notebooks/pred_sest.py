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

# + code_folding=[]
import os
from tqdm import tqdm
import numpy as np

# %matplotlib notebook
# %load_ext autoreload
# %autoreload 2

from stereo.prediction.sest.sest_predictor import SestPredictor
from stereo.data.clip_utils import get_combined_images
from stereo.common.visualization.video import VideoManager
from stereo.common.visualization.vis_utils import text_to_image
from devkit.clip import MeClip

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# + code_folding=[]
# %%capture --no-stdout
models = {
    #"peds_vehicles" : "top_view_v2_peds_weighted_vehicles.json",
    #"peds_vehicles_no_warm" : "top_view_v2_peds_weighted_vehicles_no_warm.json",
    #"top_view_v2_3_frames_corr_features" : "top_view_v2_3_frames_corr_features.json",
    "top_view_v4_curved_v0.1": "top_view_v4_curved_v0.1.json"
}

conf_dir = "/homes/guygo/gitlab/stereo/stereo/models/conf/top_view"

predictors = {}
for model in models:
    model_path = os.path.join(conf_dir, models[model])
    predictors[model] = SestPredictor(model_path, restore_iter=240000)

# +
base_path = "/mobileye/algo_STEREO3/stereo/data/sest/labels/v2.2/"

# ENUM of categories
UNKNOWN = 0
AGENT = 1
VEHICLES = 2
PEDS = 3
ROAD = 4
GROUND = 5 # Non-Road Ground
OBJECT = 6 # Non-Ground object points


def get_labels(clip_name, gis):
    labels = []
    for gi in gis:
        label = np.load(os.path.join(base_path, clip_name, "%07d.npy" % gi))
        labels.append(label)
    return labels


# -

gi_start = 0
# clip_name, num_of_gis = "19-05-14_17-10-21_Alfred_Front_0106", 20
# clip_name, num_of_gis = "19-10-03_09-11-56_Alfred_Front_0076", 20
# clip_name, num_of_gis = "19-10-03_08-31-11_Alfred_Front_0031", 50
# clip_name, num_of_gis = "19-10-03_09-04-32_Alfred_Front_0068", 100
# clip_name, num_of_gis = "19-09-22_16-53-45_Alfred_Front_0005", 100
# clip_name, num_of_gis = "19-10-03_09-20-17_Alfred_Front_0086", 50
# clip_name, num_of_gis = "18-08-22_10-39-18_Alfred_Front_0013", 50
# clip_name, num_of_gis = "19-09-22_17-13-12_Alfred_Front_0026", 70
# clip_name, num_of_gis = "19-09-22_16-53-45_Alfred_Front_0005", 70
clip_name, num_of_gis, gi_start = "19-10-03_09-24-55_Alfred_Front_0091", 80, 200
# clip_name, num_of_gis = "19-09-22_17-12-17_Alfred_Front_0025", 80

gis_files = os.listdir(os.path.join(base_path, clip_name))
gis = [int(gi[:-4]) for gi in gis_files]
gis.sort()
if num_of_gis:
    gis = gis[gi_start: gi_start + num_of_gis]
print("clip gis from %s to %s" % (gis[0], gis[-1]))

labels = get_labels(clip_name, gis)

predictions = {p:[predictors[p].pred(clip_name=clip_name, gi=gi) for gi in gis] for p in tqdm(predictors)}

combined_imgs = get_combined_images(clip_name, gis, load_from='/homes/guygo/temp')

# +
thresh_dict = {
    PEDS: 0.1,
    VEHICLES: 0.5,
    ROAD: 0.8,
    GROUND : 0.5,
    OBJECT: 0.5
}

category_to_channel = {
    VEHICLES: 0,
    PEDS: 1,
    ROAD : 2,
    GROUND: 3,
    OBJECT: 4
}

def get_pred_imgs(predictor_name, category):
    pred_imgs = []
    preds = predictions[predictor_name]
    h,w = out_shape = preds[0][0][:, :, 0].shape
    for i in range(len(gis)):
        gi_pred = np.zeros_like(out_shape)
        gi_pred = preds[i][0][:,:, category_to_channel[category]]
        alpha = np.expand_dims(gi_pred > thresh_dict[category], -1)
        rgb = 1 - np.repeat(gi_pred, 3).reshape(h, w, 3)
        pred_imgs.append(np.c_[rgb, alpha])
    return pred_imgs


# +
gis_imgs = []
for i,gi in enumerate(gis):
    gis_imgs.append(text_to_image("%s:%s" % (i, gi), (60,20)))

category = PEDS
v = VideoManager(figsize=(12,8), table_size=(3,3), interval=300)
v.add_images([labels, get_pred_imgs("top_view_v4_curved_v0.1", category)], cols_from=2, cmaps=['hot', 'gray'], alphas=[1.0, None], origin='lower')
v.add_images(combined_imgs, cols_to=2, rows_to=2, cmaps='gray')
v.add_images(gis_imgs, cols_to=1, rows_from=2, cmaps='gray')
v.play()


# +
def get_prediction_of_gis(clip_name, gis, predictor, label_example=None):
    outs = []
    for gi in gis:
        net_out = predictor.pred(clip_name=clip_name, gi=gi)
        max_categories_pred = np.argmax(net_out[0], axis=2)
        max_categories_pred_mapped = map_categories_to_labels(max_categories_pred)
        if label_example is not None:
            max_categories_pred_mapped[label_example == AGENT] = AGENT
        outs.append(max_categories_pred_mapped)
    return outs

map_colors = [VEHICLES, PEDS, ROAD, OBJECT]
def map_categories_to_labels(pred):
    mapped = np.zeros_like(pred)
    for i, color in enumerate(map_colors):
        mapped[pred == i] = color
    return mapped


def add_pred_video(sub_plot_base, title, imgs_to_play, clim=[0,6], cmap='viridis', interval=100):
    ax = fig.add_subplot(sub_plot_base)
    ax.title.set_text(title)
    imgs = []
    for img in imgs_to_play:
        imgs.append([ax.imshow(img, origin='lower', clim=clim, cmap=cmap)])
    return animation.ArtistAnimation(fig, imgs,  blit=True, interval=interval)


# +
fig = plt.figure(figsize=(12,8))
rows, cols = 1, len(predictions) + 1
videos = []

for i, p in enumerate(predictions):
    ax = plt.subplot(rows, cols, i+1)
    videos.append(show_gt_vs_pred(p, ax, category, thresh=0.1, interval=interval))

ax = plt.subplot(rows, cols, len(predictions)+1)
gis_imgs = []
for gi in gis:
    img = Image.new('RGB', (40, 20))
    d = ImageDraw.Draw(img)
    d.text((5, 5),str(gi), fill=(255, 0, 0))
    gis_imgs.append([ax.imshow(img)])
animation.ArtistAnimation(fig, gis_imgs,  blit=True, interval=interval)
# -

outs_array = {}
for p in tqdm(predictors):
    outs_array[p] = (get_prediction_of_gis(clip_name, gis, predictors[p], labels[0]))

# +
view_labels = True
interval = 500

fig = plt.figure()
animations = []
sub_plot_base = 100 + 10 * (len(outs_array) + view_labels) + 1
for i, outs_description in enumerate(outs_array):
    animations.append(add_pred_video(sub_plot_base + i, outs_description, outs_array[outs_description], interval=interval))
if view_labels:
    animations.append(add_pred_video(sub_plot_base + len(outs_array), 'labels', labels, interval=interval))
    
fig.tight_layout()
# -

