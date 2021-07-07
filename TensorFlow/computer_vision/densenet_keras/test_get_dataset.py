"""test_get_dataset.py

This test script could be used to verify either the 'train' or
'validation' dataset, by visualizing data augmented images on
TensorBoard.

Examples:
$ cd ${HOME}/project/keras_imagenet
$ python3 test_get_dataset.py train
$ tensorboard --logdir logs/train
"""


import os
import shutil
import argparse

import tensorflow as tf

from utils.dataset import get_dataset


DATASET_DIR = os.path.join(os.environ['HOME'], 'data/ILSVRC2012/tfrecords')


parser = argparse.ArgumentParser()
parser.add_argument('subset', type=str, choices=['train', 'validation'])
args = parser.parse_args()

log_dir = os.path.join('logs', args.subset)
shutil.rmtree(log_dir, ignore_errors=True)  # clear prior log data

dataset = get_dataset(DATASET_DIR, args.subset, batch_size=64)
iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
batch_xs, batch_ys = iterator.get_next()
mean_rgb = tf.reduce_mean(input_tensor=batch_xs, axis=[0, 1, 2])

# convert normalized image back: [-1, 1] -> [0, 1]
batch_imgs = tf.multiply(batch_xs, 0.5)
batch_imgs = tf.add(batch_imgs, 0.5)

summary_op = tf.compat.v1.summary.image('image_batch', batch_imgs, max_outputs=64)

with tf.compat.v1.Session() as sess:
    writer = tf.compat.v1.summary.FileWriter(log_dir, sess.graph)
    sess.run(iterator.initializer)
    rgb = sess.run(mean_rgb)
    print('Mean RGB (-1.0~1.0):', rgb)

    summary = sess.run(summary_op)
    writer.add_summary(summary)
    writer.close()
