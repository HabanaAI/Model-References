# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 2
#     language: python
#     name: python2
# ---

# + code_folding=[0]
# Wide paragraph in notebook:
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
# -

# Boost tfrecords with deeplab segmentation masks, mainly for adding sky-labeled pixels to lidar data (as very far points).
#
# Load deeplab masks that were run on ViewDataSetIndex_v3.1, saved in `/mobileye/algo_STEREO3/stereo/data/sky/view_dataset.v3.1/deeplab`, check 
#
#
# Relevant paths and settings are defined in the `2.3 tfr's paths` cell

# ### Imports

# +
# %%capture --no-stdout
# imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from stereo.data.dataset_utils import my_file2path, ViewDatasetIndex
from stereo.common.s3_utils import my_glob
from stereo.common.tfrecord_utils import read_tfrecord
from stereo.common.probe_utils import probe_tfrecord, collect_and_write_tfrecord
import os
import time
import shutil
from scipy.ndimage import binary_erosion
from stereo.data.sky_seg_utils import vis_seg
from stereo.common.general_utils import batch_list, flatten_mepjs_out
from me_pjs import PythonJobServer, SchedulerType
import stereo
from stereo.data.clip_utils import clip_name_to_clip_name_by_cam
import json
# from ground_truth.common_avm.four_on_four import get_clip
# from ground_truth.utils.clip_utils import get_fof_from_clip # used by lidar dumping
# from stereo.data import clip_utils as cu
# from stereo.prediction.vidar.stereo_predictor import StereoPredictor
# from stereo.data.view_generator.view_generator import ViewGenerator
# from stereo.data.lidar_utils import LidarProcessor
# from ground_truth.utils.clip_utils import get_fof_from_clip
# from stereo.data import sky_seg_utils as su

# import argparse
# import sys

# from PIL import Image

# %load_ext autoreload
# %autoreload 2
# %matplotlib notebook
# %matplotlib notebook
plt.style.use('dark_background')
# -

# ### funcs for stats from tfr's

# `frame_inf = (clip_name, view_name, gi_str)`

# #### Deeplab paths

# + code_folding=[]
# from file2path import markdist, isdist, listdir # used `markdist` to have `isdist=True`, `listdir` shows all distributed files
# markdist(DEEPLAB_PATH)
# hex_path_list = os.listdir(DEEPLAB_PATH)
# isdist(DEEPLAB_PATH)
# deeplab_list = listdir(DEEPLAB_PATH)
# my_file2path
# print(len(deeplab_list))
# print(deeplab_list[0])
# -

# #### Function for finding deeplab file from `frame_name`

# +
DEEPLAB_PATH = '/mobileye/algo_STEREO3/stereo/data/sky/view_dataset.v3.1/deeplab'
def frame_to_deeplab(frame_inf):
    deeplab_file_name = '_'.join(frame_inf) + '.npz'
    deeplab_file_path = DEEPLAB_PATH + '/' + my_file2path(deeplab_file_name)
    return deeplab_file_path
    
frame_inf_xmp = ('19-08-08_14-31-22_OCT_Front_0141','frontCornerLeft_to_frontCornerLeft','0398375')
frame_to_deeplab(frame_inf_xmp)
# -

# #### tfr's paths

# +
# tf_dataset_path_inp = '/mobileye/algo_STEREO3/stereo/data/v3.1.3/main' # real path, for main
tf_dataset_path_inp = '/mobileye/algo_STEREO3/stereo/data/v3.1/fcl' # real path, for corners
# tf_dataset_path_inp = '/mobileye/algo_STEREO3/stereo/data/sky/v3.1.3/dbg/tfrs'  # path for debugging
ds_format_path = tf_dataset_path_inp + '/ds_format.json' 
# tf_dataset_path_out = '/mobileye/algo_STEREO3/stereo/data/sky/v3.1.3/dbg/main' # for debugging
tf_dataset_path_out = '/mobileye/algo_STEREO3/stereo/data/v3.1.3_deeplab/fcl' # output path for boosted ds

tfrecord_paths_tst = my_glob(tf_dataset_path_inp + '/test/*')
tfrecord_paths_trn = my_glob(tf_dataset_path_inp + '/train/*')
restore_mepjs_probe = False # False for running, True for uploading previous run
restore_mepjs_write = False # False for running, True for uploading previous run

mepjs_dir_probe = '/mobileye/algo_STEREO3/stereo/data/sky/view_dataset.v3.1/mepjs/probe' # for the real data
# mepjs_dir_probe = '/mobileye/algo_STEREO3/stereo/data/sky/v3.1.3/dbg/mepjs/probe' # for debugging
mepjs_dir_write = '/mobileye/algo_STEREO3/stereo/data/sky/view_dataset.v3.1/mepjs/write' # for the real data
# mepjs_dir_write = '/mobileye/algo_STEREO3/stereo/data/sky/v3.1.3/dbg/mepjs/write' # for debugging




tfrecord_paths_tst.sort()
tfrecord_paths_trn.sort()

tfrecord_paths = tfrecord_paths_tst + tfrecord_paths_trn
tfrecord_sets = (['test'] * len(tfrecord_paths_tst)) + (['train'] * len(tfrecord_paths_trn))

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.compat.v1.enable_eager_execution(config=session_conf)
# -

print(len(tfrecord_paths))
print(len(tfrecord_paths)*197*6./1e6)

# ####  tfr:example for debugging

# Load some tfr:example for debugging:
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.compat.v1.enable_eager_execution(config=session_conf)

examples = read_tfrecord([tfrecord_paths[0]], ds_format_path,parse=True,
                        to_numpy=True, verbose=True)


# #### funcs: stats+cond for a tfr:example

# +
# For a tfr:example, load its deeplab and calculate stats+cond for sky_points

def data_from_example(example):
    im_lidar_inv = example['im_lidar_inv']
    im_lidar_inv[im_lidar_inv==0] = np.nan
    im_lidar = (im_lidar_inv[:, :, 0])**-1
    im = example['I_cntr'][:,:,0] * 255.
    gi = example['gi']
    gi_str = str(gi).zfill(7)
    clip_name = str(example['clip_name'])
    print("clip_name: ", clip_name)
    clip_name_front = clip_name_to_clip_name_by_cam(clip_name, cam='main')
    cntr_cam_name = str(example['cntr_cam_name'])
    view_name = cntr_cam_name + "_to_" + cntr_cam_name
    frame_inf = (clip_name_front, view_name, gi_str)
    return im, im_lidar, frame_inf


def sky_stats_from_data(im, im_lidar, seg_mask):
    seg_sky = (seg_mask==10)
    seg_sky = binary_erosion(seg_sky, iterations=5)
    im_size = seg_mask.size
    n_sky = seg_sky.sum()
    sky_ratio = n_sky / im_size
    n_lidar = (~np.isnan(im_lidar)).sum()
    lidar_and_sky = ( (~np.isnan(im_lidar) & (seg_sky)) ).sum()
    lidar_sky_ratio = lidar_and_sky* 1. / n_lidar
    sky_ratio = n_sky *1.0 / im_size
    return sky_ratio, lidar_sky_ratio

def sky_stats_in_example(example,min_sky_ratio=0.05, max_lidar_sky=1.5e-3):
    image, im_lidar, frame_inf = data_from_example(example)
    deeplab_file_path = frame_to_deeplab(frame_inf)
    try:
        npfile = np.load(deeplab_file_path)
    except:
        print("Couldn't load %s" % deeplab_file_path)
        sky_ratio = 0
        lidar_sky_ratio = 0
        good_for_sky_points = False
        return sky_ratio, lidar_sky_ratio, good_for_sky_points
    print("Loaded seg mask %s" % deeplab_file_path)
    seg_mask = npfile['seg_mask']
    sky_ratio, lidar_sky_ratio = sky_stats_from_data(image, im_lidar, seg_mask)
#     vis_seg(image, seg_mask) # visualize when debugging
    good_for_sky_points = ( (sky_ratio>min_sky_ratio) and (lidar_sky_ratio<max_lidar_sky) )
    return sky_ratio, lidar_sky_ratio, good_for_sky_points


# -

sky_stats_in_example(examples[0])

# ### probe: run over ftr's for stats

# +
# Run over tfrecords and examples. Maintain a list of lists of (per tfr per example:
# frame_name = (clip_name, view_name, gi_str)
# and path to deeplab on frame.
# For each example, load deeplab and collect stats sky_ratio & P(sky|lidar)

t1 = time.time()

tfrecord_paths_batched = batch_list(tfrecord_paths, 5)
# tfrecord_paths_batched = batch_list(tfrecord_paths[:120], 5) # small batch for debugging
if not restore_mepjs_probe:
    pjs = PythonJobServer(task_scheduler_type=SchedulerType.NET_BATCH, folder=mepjs_dir_probe,
                          full_copy_modules=[stereo], max_rerun=0, timeout_minutes=30,
                          memory_gb=12, poll=True)
    out_batched = pjs.run(probe_tfrecord, tfrecord_paths_batched,
                          probe_func=sky_stats_in_example)
else:
    pjs = PythonJobServer.load(mepjs_dir_probe)
    out_batched = pjs.wait()
out = flatten_mepjs_out(out_batched, tfrecord_paths_batched)

t2 = time.time()
print("Took %d sec for %d batches of %d tfr's" % ((t2-t1), len(tfrecord_paths_batched),
                                                  len(tfrecord_paths_batched[0]) )   )
# -

424/ 60.

# +
# Flattened output.

flat_out = [item for sublist in out for item in sublist]

flat_out_subs = []
for i1, sublist in enumerate(out):
    for i2,_ in enumerate(sublist):
        flat_out_subs.append([i2,i1]) # subscripts/inds of (example, tfr) ; same convention as in `probe_tf_dataset.v3.1.3_close_objects`

sky_ratio_vec = np.array([c[0] for c in flat_out])
lidar_sky_vec = np.array([c[1] for c in flat_out])
# -

print(np.sum([c[2] for c in flat_out]))
print(len(flat_out))


# +
# np.savez('./tfr_sky_stats.npz', flat_out=flat_out, flat_out_subs=flat_out_subs)

# + [markdown] heading_collapsed=true
# #### debug non-loading files

# + hidden=true
# debug non-loading files
file_path1 = '/mobileye/algo_STEREO3/stereo/data/sky/view_dataset.v3.1/deeplab/d10/' \
    + '19-09-23_15-19-33_Alfred_Front_0031_main_to_main_936551.npz'
file_name1 = os.path.split(file_path1)[1]
clip_name1, _, _, _, gi_str = os.path.splitext(file_name1)[0].rsplit('_',4)
print(clip_name1)
print(file_path1)
print(file_name1)
print(gi_str)

# + hidden=true
DEEPLAB_PATH

# + hidden=true
my_file2path(DEEPLAB_PATH + '/' + file_name1)

# + hidden=true
[fn for fn in os.listdir(DEEPLAB_PATH  +'/d10')
 if '19-09-23_15-19-33_Alfred_Front_0031' in fn ]

# + hidden=true
from file2path import listdir, file2path
deeplab_list = listdir(DEEPLAB_PATH)
len(deeplab_list) /1e6

# + hidden=true
ind_list = [i for i,path1 in enumerate(deeplab_list) if 
            ( (clip_name1 in path1) and (gi_str in path1) )]
print([ind_list])
print('\n'.join([deeplab_list[i] for i in ind_list]) )

# + hidden=true
my_file2path('19-09-23_15-19-33_Alfred_Front_0031_main_to_main_0936551.npz')

# + hidden=true
dataset_dir = '/mobileye/algo_STEREO3/stereo/data/view_dataset.v3.1'
vdsi = ViewDatasetIndex(dataset_dir)
frames = vdsi.test_frames_list + vdsi.train_frames_list
frames = list(set(frames))

# + hidden=true
print(type(frames))
print(len(frames) /1e6 * 6.)
# -

# ### Show stats and samples

# + [markdown] heading_collapsed=true
# #### histograms of heuristics

# + hidden=true
# Plot histograms
# %matplotlib notebook
fig = plt.figure(figsize=(8,3))
plt.subplot(1,2,1)
plt.hist(sky_ratio_vec[sky_ratio_vec>0],bins=1050, label='sky_ratio')
# plt.xlabel('x')
plt.xlabel('sky_ratio')
plt.grid(alpha=0.2)
# plt.xlim([0,])
plt.subplot(1,2,2)
cond_show  = (~np.isnan(lidar_sky_vec)) & (lidar_sky_vec != 0)
plt.hist(lidar_sky_vec[cond_show],bins=5050, color='r', label='lidar_sky')
# plt.xlabel('x')
plt.xlabel('P(sky|lidar)')
plt.grid(alpha=0.2)
plt.xlim([-1e-4, 3e-3])

plt.tight_layout()
plt.show()

# + [markdown] heading_collapsed=true
# #### Show samples

# + hidden=true
# Find samples close to threshold
# t_lidar_sky = 0.003 # has misses, mostly near borders (btwn sky and non-sky)
t_lidar_sky = 0.0002 # shows samples close to this threshold value
t_lidar_sky = 1.5e-3 # determined with Ofer
m_sky_ratio = 0.05 # accept samples with sky_above this minimal value
inds_w_sky = np.nonzero(sky_ratio_vec > m_sky_ratio)[0]
dist_ls_thr = np.sqrt( (lidar_sky_vec[inds_w_sky] - t_lidar_sky)**2  ) # "distnace" from threshold value
ind_min_dist_ls = np.argsort(dist_ls_thr) # samples with minimal distance from threshold value
inds_samp = inds_w_sky[ind_min_dist_ls[:24]]

# + hidden=true code_folding=[]
# Show image
from matplotlib.backends.backend_pdf import PdfPages
# pdf = PdfPages('/mobileye/algo_STEREO3/stereo/data/sky/v3.1.3/examples/'+('lidar_sky_threshold_%.1e' % t_lidar_sky ) + '.pdf')
# %matplotlib notebook
# fig = plt.figure(figsize=(15,16)) # wider screen (office)
# fig = plt.figure(figsize=(11,16)) # narrower screen (laptop)

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.compat.v1.enable_eager_execution(config=session_conf)
fig = plt.figure(figsize=(17,20)) # even wider screen
for i,ind in enumerate(inds_samp):
    ind_ex, ind_tfr = flat_out_subs[ind]
    examples = read_tfrecord(tfrecord_paths[ind_tfr], ds_format_path,parse=True,
                    to_numpy=True, verbose=True)
    im, im_lidar, frame_inf = data_from_example(examples[ind_ex])
    deeplab_file_path = frame_to_deeplab(frame_inf)
    npfile = np.load(deeplab_file_path)
    seg_mask = npfile['seg_mask']
    seg_sky = binary_erosion( (seg_mask==10), iterations=5)
    plt.subplot(8,3,i+1)
    plt.imshow(im, origin='lower', cmap='gray')
#     plt.imshow( seg_sky, alpha=0.2, origin='lower', cmap='tab20b_r')
    plt.imshow( seg_sky, alpha=0.2, origin='lower', cmap='brg_r') #  alpha=0.4 is good for pdfs,  alpha=0.2 is good for jupyter
    plt.imshow(im_lidar, origin='lower')
    sky_ratio, lidar_sky_ratio = sky_stats_from_data(im, im_lidar, seg_mask)
    plt.title("sky_ratio = %.2f, P(sky|lidar)= %.4f" % (sky_ratio, lidar_sky_ratio) , fontsize=8)

plt.tight_layout()
plt.show()

# pdf.savefig(fig)
# pdf.close()

# + hidden=true
type(examples[0])
# -

# ### build tfrecords

# #### "good" subset - condition holds

print(len(sky_ratio_vec))

# +
# Set thresholds for the subset
min_sky_ratio = 0.05
max_lidar_sky_ratio = 1.5e-3

sky_cond = np.nonzero( (sky_ratio_vec>min_sky_ratio) * (lidar_sky_vec< max_lidar_sky_ratio) )[0]


sky_subs = np.array(flat_out_subs)[sky_cond] # (n_examples, 2)
# inds_hold = []
# for i,params in enumerate(flat_out):
 

print('sky_subs contains %d examples out of %s' % (len(sky_subs), len(flat_out)) )    
# -

# #### split train-test

# +
# Split close_inds to train and test sets

sky_subs_tst = [subs for subs in sky_subs if tfrecord_sets[subs[1]] == 'test']
sky_subs_trn = [subs for subs in sky_subs if tfrecord_sets[subs[1]] == 'train']

# If want 1/10 of the whole data:
n_tst = len(sky_subs_tst)
n_trn = len(sky_subs_trn)
sky_subs_tst = [sky_subs_tst[i] for i in np.random.choice(n_tst, size= np.round(n_tst/10).astype(int), replace=False)]
sky_subs_trn = [sky_subs_trn[i] for i in np.random.choice(n_trn, size= np.round(n_trn/10).astype(int), replace=False)]

print('sky_subs_trn contains %d examples' % len(sky_subs_trn))
print('sky_subs_tst contains %d examples' % len(sky_subs_tst))


# -

# #### funcs: update example & ds_format

# + code_folding=[]
# Functions
def ds_format_update(ds_format, ds_format_add, write_path=None, overwrite_path=False):
    if isinstance(ds_format, str):
        if (not write_path) and overwrite_path:
            write_path = ds_format
        with my_open(ds_format, 'rb') as fp:
            ds_format = json.load(fp)
    
    ds_format_out = ds_format.copy()
    ds_format_out.update(ds_format_add)

    if write_path:
        try:
            with open(write_path, 'w') as f:
                json.dump(ds_format_out, f)
        except:
            print("Cannot write json to path.")

    return ds_format_out

def update_example(example, example_extra):
    example_out = example.copy()
    example_out.update(example_extra)
    return example_out

def frame_info_example(example):
    """
    Information of the frame corresponding to example.
    """
    gi = example['gi']
    gi_str = str(gi).zfill(7)
    clip_name = str(example['clip_name'])
    clip_name_front = clip_name_to_clip_name_by_cam(clip_name, cam='main')
    cntr_cam_name = str(example['cntr_cam_name'])
    view_name = cntr_cam_name + "_to_" + cntr_cam_name
    frame_info = (clip_name_front, view_name, gi_str)
    return frame_info

def add_deeplab_to_example(example):
    """
    Add deeplab segmentation image to example
    """
    t0 = time.time()
    frame_inf = frame_info_example(example)
    t1=time.time()
    deeplab_file_path = frame_to_deeplab(frame_inf)
    
    t2 = time.time()
    npfile = np.load(deeplab_file_path)
    t3 = time.time()
    seg_mask = npfile['seg_mask']
    seg_mask = np.expand_dims(seg_mask,2)
    example_out = update_example(example, {'deeplab': seg_mask})
    t4 = time.time()
    print("Times in `add_deeplab_to_example1` map func:\nt1=%.2f, t2=%.2f, t3=%.2f, t4=%.2f." % ((t1-t0), (t2-t0), (t3-t0), (t4-t0)))
    return example_out


# -

# verify that this "map_func" works
example1 = examples[1]
example2 = add_deeplab_to_example(example1)
# vis_seg(example2['I_cntr'].squeeze(), example2['deeplab'].squeeze())

example2['clip_name'].astype('str')

# update ds_format
import json
with open(ds_format_path, 'rb') as f:
    ds_format = json.load(f)
dsf_extra =  {u'deeplab': {u'dtype': u'uint8', u'shape':[310, 720, 1]}}
ds_format2 = ds_format_update(ds_format, dsf_extra)

80* 60 / 1000.

# #### build tfrecords

tf_dataset_path_out

with open(tf_dataset_path_out + '/ds_format.json', 'w') as f:
        json.dump(ds_format2, f, sort_keys=True, indent=None, separators=(', ',': '))

# +
# Build tfrecords_to_write tuples for the test and train sets separately

examples_per_tfrecord = 200

sky_subs_tst_batched = batch_list(sky_subs_tst, examples_per_tfrecord)
sky_subs_trn_batched = batch_list(sky_subs_trn, examples_per_tfrecord)

tfrecords_to_write_tst = []
tfrecords_to_write_trn = []

prompt = raw_input('Remove %s ? (y/n): ' % tf_dataset_path_out)

if prompt == 'y':
    if os.path.exists(tf_dataset_path_out):
        shutil.rmtree(tf_dataset_path_out)
    os.mkdir(tf_dataset_path_out)
    os.mkdir(tf_dataset_path_out + '/test')
    os.mkdir(tf_dataset_path_out + '/train')
#     shutil.copyfile(ds_format_path, tf_dataset_path_out + '/ds_format.json')
    with open(tf_dataset_path_out + '/ds_format.json', 'w') as f:
        json.dump(ds_format2, f)
    
    for i in range(len(sky_subs_tst_batched)):
        curr_batch = [(subs[0], tfrecord_paths[subs[1]]) for subs in sky_subs_tst_batched[i]]
        curr_batch = {'tfrecord_example_inds_inp': curr_batch, 
                      'tfrecord_path_out': tf_dataset_path_out + '/test/%07d.tfrecord' % (i*examples_per_tfrecord)}
        tfrecords_to_write_tst.append(curr_batch)

    for i in range(len(sky_subs_trn_batched)):
        curr_batch = [(subs[0], tfrecord_paths[subs[1]]) for subs in sky_subs_trn_batched[i]]
        curr_batch = {'tfrecord_example_inds_inp': curr_batch, 
                      'tfrecord_path_out': tf_dataset_path_out + '/train/%07d.tfrecord' % (i*examples_per_tfrecord)}
        tfrecords_to_write_trn.append(curr_batch)    
        
    tfrecords_to_write = tfrecords_to_write_tst + tfrecords_to_write_trn
else:
    print('Aborting')
# -
# ### collect_and_write_tfrecord

tfrecord_paths_batched = batch_list(tfrecord_paths, 5)

len(tfrecords_to_write)

t1 = time.time()
if not restore_mepjs_write:
    pjs = PythonJobServer(task_scheduler_type=SchedulerType.NET_BATCH, folder=mepjs_dir_write, poll=True,
                          full_copy_modules=[stereo], max_rerun=0, timeout_minutes=120, memory_gb=14)
    out = pjs.run(collect_and_write_tfrecord, tfrecords_to_write, ds_format=ds_format, verbose=True, ds_format_out = ds_format2, map_func=add_deeplab_to_example)    
else:
    pjs = PythonJobServer.load(mepjs_dir_write)
    out = pjs.wait()
t2 = time.time()
print("Written to %s" % tf_dataset_path_out)
print("Took %s sec" % (t2-t1))
# Took 80 minutes for 1/10 of the dataset

2000 / 60.

3922 /60.

# #### Check saved tfr's

tfr_saved_path = '/mobileye/algo_STEREO3/stereo/data/v3.1.3_deeplab/main/test/0000000.tfrecord'
t1 = time.time()
tfr_saved = read_tfrecord(tfr_saved_path, ds_format2, parse=True, to_numpy=True, verbose=True)
t2 = time.time()
print("Took %.2f sec" % (t2-t1))

example_saved = tfr_saved[0]
example_saved.keys()

# %matplotlib notebook
im_saved = example_saved['I_cntr'].squeeze()
deeplab_saved = example_saved['deeplab'].squeeze()
vis_seg(im_saved, deeplab_saved)

# + [markdown] heading_collapsed=true
# ### Sandbox

# + hidden=true
from stereo.common import tfrecord_utils as tfu

# + hidden=true
example1 = examples[1]
# (example1).keys()

# + hidden=true
import json
with open(ds_format_path) as f:
    ds_format = json.load(f)

# + hidden=true
(ds_format).keys()

# + hidden=true
ds_format['I_cntr']

# + hidden=true
example1.keys()

# + hidden=true
tfu.parse_example(example1, ds_format, to_numpy=True)

# + hidden=true
# Check options of read_tfrecord
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.compat.v1.enable_eager_execution(config=session_conf)

tf_path = '/mobileye/algo_STEREO3/stereo/data/sky/v3.1.3/dbg/main/train/0000000.tfrecord'
tfr1 = read_tfrecord(tf_path, ds_format, parse=True, to_numpy=True, verbose=True)

# + hidden=true
len(tfr1)

# + hidden=true
tfr1[0]['origin']

# + hidden=true
tfr1[0].keys()

# + hidden=true
print(type(tfr1[0][u'I_cntr']))
print((tfr1[0][u'I_cntr']).shape)

# + hidden=true
ds_format

# + hidden=true
ds_format_add = {u'deeplab': {u'dtype': u'uint8', u'shape':[310, 720]}}
ds_format_add

# + hidden=true
ds_format2 = ds_format.copy()
ds_format2.update(ds_format_add)
ds_format2.keys()

# + hidden=true
ds_format_path2 = '/mobileye/algo_STEREO3/stereo/data/sky/v3.1.3/dbg/main/ds_format2.json'

# + hidden=true
with open(ds_format_path2, 'w') as f:
    json.dump(ds_format2, f)

# + hidden=true
tfu.ds_format_update(ds_format, ds_format_add, ds_format_path2)

# + hidden=true
(example1).keys()

# + hidden=true
example2 = example1.copy()
example2.update({"deeplab": np.ones((310, 720))})

# + hidden=true
example2.keys()

# + hidden=true
example3 = tfu.update_example(example1,{"deeplab": np.ones((310, 720))})

# + hidden=true
example3.keys()

# + hidden=true
del examples
del example1, example2, example3

# + hidden=true
ds_format2
