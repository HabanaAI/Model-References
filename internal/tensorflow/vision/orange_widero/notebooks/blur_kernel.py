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
#     display_name: Python 2
#     language: python
#     name: python2
# ---

import sys
if sys.path[1] is not '/homes/nadavs/work/stereo.git':
    sys.path.insert(1, '/homes/nadavs/work/stereo.git')
if '/mobileye/algo_STEREO2/stereo/code.freeze' not in sys.path[2]:
    sys.path.insert(2, '/mobileye/algo_STEREO2/stereo/code.freeze')
import boto3
import json
import imp
import tensorflow as tf
from os.path import join
import numpy as np
import matplotlib
matplotlib.use('nbagg')
# %matplotlib notebook
from matplotlib import pyplot as plt
from stereo.data.view_generator.view_generator import ViewGenerator


def tf_type(x):
    if isinstance(x, basestring):
        return tf.string
    if type(x) is np.ndarray:
        if x.dtype == np.int64:
            return tf.int64
        if x.dtype == np.int32:
            return tf.int32
        if x.dtype == np.uint8:
            return tf.uint8
        if x.dtype == np.float32:
            return tf.float32
        if x.dtype == np.float64:
            return tf.float64
    return None


# +
model_conf_file = '/homes/nadavs/work/stereo.git/stereo/models/conf/blur_kernel_sm_v0.0.json'
exec_dir = '/homes/nadavs/work/stereo.git/stereo'
with open(model_conf_file, 'rb') as fp:
    model_conf = json.load(fp)
model_name = model_conf_file.split('/')[-1].split('.json')[0]

tf.compat.v1.reset_default_graph()
arch_func = imp.load_source('arch', join(exec_dir, 'models', 'arch', model_conf['model_params']['arch']['name'] + '.py')).arch
prep_func = imp.load_source('prep', join(exec_dir,  'data', 'prep', model_conf['prep']['name'] + '.py')).prep_frame

views_names = prep_func(dataSetIndex=None, frame=None, view_names=None, fail_missing_lidar=None, inferenceOnly=True, views=None, return_views_names=True)
clip_name = '18-12-26_11-11-17_Alfred_Front_0018'
gi = 160177
vds = ViewGenerator(clip_name, view_names=views_names, predump=None, mode='pred', no_labels=True, etc_dir=None)

views = vds.get_gi_views(gi=gi)
placeholders = prep_func(dataSetIndex=None, frame=None, view_names=None, fail_missing_lidar=None, inferenceOnly=True, views=views)[0]

placeholders_ = {}
for k, v in placeholders.items():
    s = list(v.shape[:])
    s.insert(0,None)
    placeholders_[k] = tf.compat.v1.placeholder(tf_type(v), s, k)

def model():
    arch_arg_names = model_conf['model_params']['arch']['arg_names']
    arch_args = {}
    for k in arch_arg_names:
        arch_args[k] = placeholders_[k]
    arch_args.update(model_conf['model_params']['arch']['kwargs'])
    return arch_func(**arch_args)
with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=tf.compat.v1.AUTO_REUSE):
    out = model()
sess = tf.compat.v1.Session()
saver = tf.compat.v1.train.Saver()

s3 = boto3.resource('s3')
sm_bucket = s3.Bucket('mobileye-team-stereo')
model_s3 = '/'.join(model_conf['model_base_path'].split('/')[3:])
try:
    model_s3 = '/'.join([model_s3, user, user+'_'+model_name])
    latest_ckpt = max([int(obj.key.split('ckpt-')[1].split('.meta')[0]) for obj in sm_bucket.objects.filter(Prefix=model_s3) if 'ckpt' in obj.key and 'meta' in obj.key])
except:
    model_s3 = '/'.join(model_conf['model_base_path'].split('/')[3:])
    model_s3 = '/'.join([model_s3, model_name])
    latest_ckpt = max([int(obj.key.split('ckpt-')[1].split('.meta')[0]) for obj in sm_bucket.objects.filter(Prefix=model_s3) if 'ckpt' in obj.key and 'meta' in obj.key])

ckpt_keys = [obj.key for obj in  sm_bucket.objects.filter(Prefix=model_s3) if 'ckpt' in obj.key and ('%d' % latest_ckpt) in obj.key]
for key in ckpt_keys:
    if '.meta' in key:
        ckpt = 's3://mobileye-team-stereo/%s' % key.split('.meta')[0]
saver.restore(sess, ckpt)

# +
kernels_names = ['main_to_frontCornerLeft',
                'main_to_frontCornerRight',
                'main_to_parking_front',
                'frontCornerLeft_to_parking_front',
                'frontCornerLeft_to_parking_left',
                'frontCornerRight_to_parking_front',
                'frontCornerRight_to_parking_right',
                'rearCornerLeft_to_parking_rear',
                'rearCornerLeft_to_parking_left',
                'rearCornerRight_to_parking_rear',
                'rearCornerRight_to_parking_right',
                'rear_to_rearCornerLeft',
                'rear_to_rearCornerRight',
                'rear_to_parking_rear']

kernels = {}
for kernel_name in kernels_names:
    kernel = tf.compat.v1.get_default_graph().get_tensor_by_name(kernel_name+'/kernel:0')
    kernels[kernel_name] = np.squeeze(sess.run([kernel]))
# -

fig = plt.figure(figsize=(9,9))
for i,kernel_name in enumerate(kernels_names):
    ax = fig.add_subplot(4,4,i+1)
    ax.set_title(kernel_name, fontdict={'fontsize': 7})
    ax.imshow(kernels[kernel_name]/np.sum(kernels[kernel_name]), cmap='jet')
# plt.tight_layout()

blur_kernels = {}
for kernel_name in kernels_names:
    kernel = np.expand_dims(np.expand_dims(kernels[kernel_name]/np.sum(kernels[kernel_name]),2),3)
    
    blur_kernels['_to_'.join(kernel_name.split('_to_')[::-1])] = kernel
np.savez('/mobileye/algo_RP_NVME/stereo/data/v2.2/blur_kernels.npz', **blur_kernels)


from s3fs.core import S3FileSystem
s3_file = S3FileSystem()
bucket_name = 'mobileye-team-stereo'
file_path = 'data/tf_datasets/v2.2/blur_kernels.npz'
data = np.load(s3_file.open('{}/{}'.format(bucket_name, file_path)))


