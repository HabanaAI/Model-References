import os
import json
import tensorflow as tf
from glob import glob
import numpy as np
import boto3

def latest_model(checkpoint_dir):
    try:
        list_of_files = glob(os.path.join(checkpoint_dir, "*ckpt*"))  # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        restore_iter = np.int32(latest_file.split('/')[-1].split('_')[1].split('.')[0])
    except:
        s3 = boto3.resource('s3')
        sm_bucket = s3.Bucket('mobileye-habana/mobileye-team-stereo')
        restore_iter = max(
            [int(obj.key.split('ckpt-')[1].split('.meta')[0]) for obj in sm_bucket.objects.filter(Prefix=checkpoint_dir) if
             'ckpt' in obj.key and 'meta' in obj.key])
    return restore_iter


def best_model(json_path, alpha=0.05):
    with open(json_path, 'rb') as fp:
        conf = json.load(fp)

    model_name = os.path.splitext(os.path.split(json_path)[1])[0]
    model_dir = os.path.join(conf['output_base_dir'], model_name)

    test_events_dir = os.path.join(model_dir, 'tensorboard/test/*')

    if len(glob(test_events_dir)) == 1:
        test_events_file = glob(test_events_dir)[0]

    maf = False
    mmaf = False
    ii = int(conf['init']['global_step'])
    for e in tf.compat.v1.train.summary_iterator(test_events_file):
        ii += conf['test_every']
        for v in e.summary.value:
            if "loss_lidar" in v.tag:
                if not maf:
                    ma = v.simple_value
                    maf = True
                else:
                    ma = (1 - alpha) * ma + alpha * v.simple_value
        if ii % conf['save_every'] < conf['test_every'] and maf and len(glob(
                os.path.join(model_dir, 'checkpoints/sess_%06d.ckpt*' % (ii - (ii % conf['save_every']))))) > 0:
            if not mmaf:
                mma = ma
                checkpoint = ii - (ii % conf['save_every'])
                mmaf = True
            else:
                if mma > ma:
                    mma = ma
                    checkpoint = ii - (ii % conf['save_every'])

    return checkpoint