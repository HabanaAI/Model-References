#!/usr/bin/python

import os
import numpy as np
import json
import argparse
import tensorflow as tf
from tensorflow.python.framework.errors_impl import OutOfRangeError
import random

from stereo.models.sm.sm_utils import get_checkpoint_path
from stereo.models.sm.create_dataset import create_dataset
from stereo.models.sm.sm_setup import sm_setup
from stereo.models.model_utils import load_model_from_meta
from stereo.models.mvs_model import get_model
from stereo.common.s3_utils import my_save


def main(args):

    args.debug = args.debug in ['True', 'true', True]
    args.save = args.save in ['True', 'true', True]

    if not args.local:
        sm_setup()

    model_name = os.path.splitext(os.path.split(args.json_path)[1])[0]

    with open(args.json_path, 'rb') as fp:
        conf = json.load(fp)

    if args.loss is not None:
        conf["model_params"]["loss"]["name"] = args.loss

    random.seed(12345)

    num_gpus = int(os.getenv('SM_NUM_GPUS', 1))
    num_cpus = int(os.getenv('SM_NUM_CPUS', 1))

    print ("Using %d gpus and %d cpus" % (num_gpus, num_cpus))

    batch_size = 12 if not args.local else 2

    tf.compat.v1.reset_default_graph()

    params = conf['model_params']
    params['local'] = False

    pipe_mode = not args.local
    prefetch = tf.data.experimental.AUTOTUNE
    ds_test = create_dataset(conf, args.cam, channel='test', pipe_mode=pipe_mode, shuf=False, eval=True, alt_dataset=args.dataset)

    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.prefetch(buffer_size=prefetch)

    if int(args.num_instances) > 0 and not pipe_mode:
        print ("Sharding dataset - %d" % (int(args.idx)))
        ds_test = ds_test.shard(int(args.num_instances), int(args.idx))

    dataset_features = tf.compat.v1.data.make_one_shot_iterator(ds_test).get_next()[0]
    if args.load_graph_from_code:
        out, _ = get_model(dataset_features, params)
    else:
        out, placeholders = load_model_from_meta(json_path=args.json_path)
    session_conf = tf.compat.v1.ConfigProto(device_count={"CPU": num_cpus},
                                  inter_op_parallelism_threads=0,
                                  intra_op_parallelism_threads=0)
    sess = tf.compat.v1.Session(config=session_conf)
    saver = tf.compat.v1.train.Saver()
    # figure out model name and setup checkpoint and tensorboard output dirs
    model_s3 = '/'.join(conf['model_base_path'].split('/')[3:])
    checkpoint_dir = '/'.join([model_s3, model_name])
    restore_iter = int(args.restore_iter)
    checkpoint_path = get_checkpoint_path(checkpoint_dir, restore_iter)
    print ("Restoring weights from %s" % (checkpoint_path))
    saver.restore(sess, checkpoint_path)

    if args.load_graph_from_code:
        eval_list = [dataset_features['clip_name'], dataset_features['gi'], out, dataset_features['center_im'],
                     dataset_features['inp_ims'], dataset_features['ground_truth'], dataset_features['x']]
    else:
        eval_list = [out]
    if not args.save:
        if params.get('with_model_scoping', False):
            loss = tf.compat.v1.get_default_graph().get_tensor_by_name("loss/batch_loss:0")
        else:
            loss = tf.compat.v1.get_default_graph().get_tensor_by_name("batch_loss:0")
        eval_list.append(loss)
    from stereo.evaluation.probe_factory import probeFactory

    factory = probeFactory(probe_name=args.probe_name)
    probe = factory.get_probe_obj(args.json_path, args.out_dir, args.restore_iter, args.idx, int(args.num_instances), args.cam,
                                  args.loss, args.dataset, args.blacklist, args.suffix, args.debug, batch_size, args.local)
    counter = 0
    while True:
        if args.load_graph_from_code:
            out_lst = sess.run(eval_list)
        else:
            try:
                np_features_list = sess.run(list(dataset_features.values()))
            except OutOfRangeError:
                print("Reached end of dataset")
                break
            np_features = dict(zip(dataset_features.keys(), np_features_list))
            feed_dict = {}
            for k, v in placeholders.items():
                feed_dict[v] = np_features[k]
            if args.save:
                eval_out, = sess.run(eval_list, feed_dict=feed_dict)
                out_lst = [eval_out, np_features['clip_name'], np_features['gi']]
            else:
                eval_out, eval_loss = sess.run(eval_list, feed_dict=feed_dict)
                out_lst = [eval_loss, np_features['clip_name'], np_features['gi'], eval_out, np_features['center_im'],
                           np_features['inp_ims'], np_features['ground_truth'], np_features['x']]
        probe.update(out_lst)
        if (args.debug and counter >= 5) or args.local and counter >= 2:
            print("breaking after %s iterations on dataset" % counter)
            break
        counter += 1
    probe.summarize()


if __name__ == '__main__':

    print ("Inside evaluate script")

    parser = argparse.ArgumentParser(description='Train a stereo DNN model for a given configuration.')
    parser.add_argument('--json_path', help='path to JSON file defining the datasets, loss, '
                                          'architecture, hyper-parameters and gpu setup.')
    parser.add_argument('-o', '--out_dir', help='output directory for eval results')
    parser.add_argument('-i', '--restore_iter', help='number iterations of relevant checkpoint. If -1, then most recent',
                        default=-1)
    parser.add_argument('--idx', help='which eval instance/part of the dataset to work on', default=-1)
    parser.add_argument('--num_instances', help='how many total instances are running', default=-1)
    parser.add_argument('--cam', help='inference cam', default='main')
    parser.add_argument('--loss', help='which loss to use', default=None)
    parser.add_argument('--dataset', help='which dataset to use', default=None)
    parser.add_argument('--blacklist', help='list of clips to exclude', default=None)
    parser.add_argument('--suffix', help='suffix to add to saved eval names', default='')
    parser.add_argument('--debug', help='debug mode does only five iterations', default=False)
    parser.add_argument('--local', help='local mode', default=False)
    parser.add_argument('--save', help='save inference output', default=False)
    parser.add_argument('--load_graph_from_code', help='load_graph_from_code', default=False)
    parser.add_argument('--probe_name', help='which probe child class to use', default='testAnalyzer')
    parser.add_argument('--model_dir', default=None)
    args = parser.parse_args()
    main(args)