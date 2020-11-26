# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: latest-venv
#     language: python
#     name: latest-venv
# ---

# +
from random import shuffle, seed
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import stereo
from stereo.common.general_utils import batch_list, flatten_mepjs_out
from stereo.common.probe_utils import batch_func
from stereo.data.dataset_utils import ViewDatasetIndex
from stereo.common.visualization.vis_utils import points3d_from_image

from me_pjs import PythonJobServer, SchedulerType
# -

restore_mepjs = True
save_blacklist = False
dataset_dir = '/mobileye/algo_STEREO3/stereo/data/view_dataset.v2.3'
mepjs_dir = "/mobileye/algo_STEREO3/ofers/mepjs/probe_view_dataset.v2.3_close_objects"
inp_blacklist = ["19-01-17_15-41-29_Front_0015@gi@0853407"]
out_blacklist = '/mobileye/algo_STEREO3/stereo/data/lists/non_close_object_frames_in_view_dataset.v2.3.lst'

vdsi = ViewDatasetIndex(dataset_dir)


@batch_func
def count_close_points_in_frame(frame, max_dist=4.0, min_height=-1.2):

    view = vdsi.read_views(['main_to_main'], frame)['main_to_main']
    lidar = view['lidars']['short']
    pcd, _ = points3d_from_image(lidar, None, view['origin'], view['focal'])
    num_close = np.sum((pcd[:, 1] > min_height) * (pcd[:, 2] < max_dist))
    num_total = np.sum((pcd[:, 1] > min_height))
    
    return num_close, num_total


frames = vdsi.test_frames_list + vdsi.train_frames_list
frames = list(set(frames) - set(inp_blacklist))

# +
frames_batched = batch_list(frames, 3000)

if not restore_mepjs:
    pjs = PythonJobServer(task_scheduler_type=SchedulerType.MEJS, folder=mepjs_dir,
                          full_copy_modules=[stereo], max_rerun=0, timeout_minutes=7)
    out_batched = pjs.run(count_close_points_in_frame, frames_batched)
    out = flatten_mepjs_out(out_batched, frames_batched)
else:
    pjs = PythonJobServer.load(mepjs_dir)
    out_batched = pjs.wait()
    out = flatten_mepjs_out(out_batched, frames_batched)

# +
out = flatten_mepjs_out(out_batched, frames_batched)
valid_frames, num_close, num_total = [], [], []

for i, (out_, frame) in enumerate(zip(out, frames)):
    if out_:
        num_close.append(out_[0])
        num_total.append(out_[1])
        valid_frames.append(frame)

num_close = np.array(num_close)
num_total = np.array(num_total)
# -

validity = (num_total > 2000)*(num_close > 50)
valid_frames = [valid_frames[i] for i in np.where(validity)[0]]
valid_frames_scores = [float(num_close[i])/num_total[i] for i in np.where(validity)[0]]
valid_frames_sorted = zip(valid_frames, valid_frames_scores)
valid_frames_sorted.sort(key=lambda x:x[1], reverse=True)

# +
unique_clips_sorted = set([])
unique_clip_frames_sorted = []

for frame in valid_frames_sorted:
    clip = frame[0].split('@')[0]
    if not clip in unique_clips_sorted:
        unique_clips_sorted.add(clip)
        unique_clip_frames_sorted.append(frame)
        
print(len(unique_clip_frames_sorted))
print(len(valid_frames_sorted))

seed(1234)
frames_to_whitelist = valid_frames_sorted
shuffle(frames_to_whitelist)
# -

for frame_ in frames_to_whitelist[500:600]:
    print(frame_)
    frame = frame_[0]
    views = vdsi.read_views(['main_to_main'], frame)
    fig = plt.figure(figsize=[10, 5])
    ax = fig.add_subplot(111)
    lidar = views['main_to_main']['lidars']['short'].copy()
    lidar[lidar == 0.0] = np.nan
    ax.imshow(views['main_to_main']['image'], origin='lower', cmap='gray')
    ax.imshow(lidar**-1, origin='lower', cmap='jet', alpha=0.75, clim=[15.0**-1, 3.0**-1])
    plt.show()

frames_black_list = list(set(frames) - set([it[0] for it in frames_to_whitelist]))
len(frames_black_list)

if save_blacklist:
    with open(out_blacklist, 'w') as f:
        for frame in frames_black_list:
            f.write("%s\n" % frame)
