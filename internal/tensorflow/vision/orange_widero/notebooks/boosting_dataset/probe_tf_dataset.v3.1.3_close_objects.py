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
import os
import shutil
import gc
import json
import numpy as np
import stereo
import matplotlib.pyplot as plt
import tensorflow as tf

from stereo.common.s3_utils import my_glob
from stereo.common.visualization.vis_utils import points3d_from_image
from stereo.common.general_utils import batch_list, flatten_mepjs_out
from stereo.common.probe_utils import probe_tfrecord, collect_and_write_tfrecord
from stereo.common.tfrecord_utils import read_tfrecord

from me_pjs import PythonJobServer, SchedulerType

# +
# tf_dataset_path_inp = 's3://mobileye-team-stereo/data/tf_datasets/v3.1.3/main'
# from nebula.cloud_tools.iam_tools.load_creds_to_env import set_aws_env_variables_from_credentials_file
# set_aws_env_variables_from_credentials_file()
# -

restore_mepjs_probe = True
restore_mepjs_write = True
dry_run = True
tf_dataset_path_inp = '/mobileye/algo_STEREO3/stereo/data/v3.1.3/main' 
tf_dataset_path_out = '/mobileye/algo_STEREO3/stereo/data/v3.1.3_close_objects/main_'
mepjs_dir_probe = "/mobileye/algo_STEREO3/ofers/mepjs/probe_tf_dataset.v3.1.3_close_objects_probe"
mepjs_dir_write = "/mobileye/algo_STEREO3/ofers/mepjs/probe_tf_dataset.v3.1.3_close_objects_write"

ds_format_path = tf_dataset_path_inp + '/ds_format.json'


def count_close_points_in_example(example, max_dist=4.0, min_height=-1.2):
    
    lidar = (1e-12 + example['im_lidar_short_inv'][:, :, 0])**-1
    lidar[lidar > 1e11] = 0.0
    pcd, _ = points3d_from_image(lidar, None, example['origin'], example['focal'])
    num_close = np.sum((pcd[:, 1] > min_height) * (pcd[:, 2] < max_dist))
    num_total = np.sum((pcd[:, 1] > min_height))    
    return num_close, num_total


# +
tfrecord_paths_tst = my_glob(tf_dataset_path_inp + '/test/*')
tfrecord_paths_trn = my_glob(tf_dataset_path_inp + '/train/*')

tfrecord_paths_tst.sort()
tfrecord_paths_trn.sort()

tfrecord_paths = tfrecord_paths_tst + tfrecord_paths_trn
tfrecord_sets = (['test'] * len(tfrecord_paths_tst)) + (['train'] * len(tfrecord_paths_trn))
# -

tfrecord_paths_batched = batch_list(tfrecord_paths, 5)
if not restore_mepjs_probe:
    pjs = PythonJobServer(task_scheduler_type=SchedulerType.NET_BATCH, folder=mepjs_dir_probe,
                          full_copy_modules=[stereo], max_rerun=0, timeout_minutes=30, memory_gb=8)
    out_batched = pjs.run(probe_tfrecord, tfrecord_paths_batched, probe_func=count_close_points_in_example)
else:
    pjs = PythonJobServer.load(mepjs_dir_probe)
    out_batched = pjs.wait()
out = flatten_mepjs_out(out_batched, tfrecord_paths_batched)

out = flatten_mepjs_out(out_batched, tfrecord_paths_batched)

# +
# Find from within the valid outputs those samples for which the "close object" condition holds

inds, num_close, num_total = [], [], []

for tfr_ind, tfr_out in enumerate(out):
    if not tfr_out:
        continue
    for example_ind, example_out in enumerate(tfr_out):
        if not example_out:
            continue
        num_close.append(example_out[0])
        num_total.append(example_out[1])
        inds.append((example_ind, tfr_ind))

num_close = np.array(num_close)
num_total = np.array(num_total)

close_examples_cond = (num_total > 2000)*(num_close > 50)
close_inds = [inds[i] for i in np.where(close_examples_cond)[0]]

print('close_inds contains %d examples' % len(close_inds))

# +
# Show the first 15 examples from the close_inds set

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.compat.v1.enable_eager_execution(config=session_conf)
curr_tfr_ind = None

for (example_ind, tfr_ind) in close_inds[800:850]:
    
    if tfr_ind != curr_tfr_ind:
        tfrecord_path = tfrecord_paths[tfr_ind]        
        examples = read_tfrecord(tfrecord_paths[tfr_ind], ds_format_path, verbose=True, parse=True, to_numpy=True)
        curr_tfr_ind = tfr_ind
        
    example = examples[example_ind]
    
    print(example['clip_name'], example['gi'])
    fig = plt.figure(figsize=[10, 5])
    ax = fig.add_subplot(111)
    image = example['I_cntr'][:, :, 0]
    lidar_inv = example['im_lidar_short_inv'][:, :, 0]
    lidar_inv[lidar_inv == 0.0] = np.nan
    ax.imshow(image, origin='lower', cmap='gray')
    ax.imshow(lidar_inv, origin='lower', cmap='jet', alpha=0.75, clim=[15.0**-1, 3.0**-1])
    plt.show()

# +
# Split close_inds to train and test sets

close_inds_tst = [ci for ci in close_inds if tfrecord_sets[ci[1]] == 'test']
print('close_inds_tst contains %d examples' % len(close_inds_tst))
close_inds_trn = [ci for ci in close_inds if tfrecord_sets[ci[1]] == 'train']
print('close_inds_trn contains %d examples' % len(close_inds_trn))

# +
# Build tfrecords_to_write tuples for the test and train sets separately

examples_per_tfrecord = 200

close_inds_tst_batched = batch_list(close_inds_tst, examples_per_tfrecord)
close_inds_trn_batched = batch_list(close_inds_trn, examples_per_tfrecord)

tfrecords_to_write_tst = []
tfrecords_to_write_trn = []

if not dry_run:
    prompt = raw_input('Remove %s ? (y/n): ' % tf_dataset_path_out)
    if prompt == 'y':
        shutil.rmtree(tf_dataset_path_out)
        os.mkdir(tf_dataset_path_out)
        os.mkdir(tf_dataset_path_out + '/test')
        os.mkdir(tf_dataset_path_out + '/train')
        shutil.copyfile(ds_format_path, tf_dataset_path_out + '/ds_format.json')
    else:
        print('Aborting')
        assert(False)

for i in range(len(close_inds_tst_batched)):
    curr_batch = [(inds[0], tfrecord_paths[inds[1]]) for inds in close_inds_tst_batched[i]]
    curr_batch = {'tfrecord_example_inds_inp': curr_batch, 
                  'tfrecord_path_out': tf_dataset_path_out + '/test/%07d.tfrecord' % (i*examples_per_tfrecord)}
    tfrecords_to_write_tst.append(curr_batch)

for i in range(len(close_inds_trn_batched)):
    curr_batch = [(inds[0], tfrecord_paths[inds[1]]) for inds in close_inds_trn_batched[i]]
    curr_batch = {'tfrecord_example_inds_inp': curr_batch, 
                  'tfrecord_path_out': tf_dataset_path_out + '/train/%07d.tfrecord' % (i*examples_per_tfrecord)}
    tfrecords_to_write_trn.append(curr_batch)    

tfrecords_to_write = tfrecords_to_write_tst + tfrecords_to_write_trn

# -

tfrecord_paths_batched = batch_list(tfrecord_paths, 5)
if not restore_mepjs_write:
    pjs = PythonJobServer(task_scheduler_type=SchedulerType.NET_BATCH, folder=mepjs_dir_write,
                          full_copy_modules=[stereo], max_rerun=0, timeout_minutes=20, memory_gb=8)
    out = pjs.run(collect_and_write_tfrecord, tfrecords_to_write, ds_format=ds_format_path, verbose=True)    
else:
    pjs = PythonJobServer.load(mepjs_dir_write)
    out = pjs.wait()
