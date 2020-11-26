# general imports
import argparse
import json
import random
import time
import shutil
import s3fs
from functools import partial
import imp
from os.path import join, abspath, dirname
import tensorflow as tf

from demo.library_loader import load_habana_module
import sys
tree_base = abspath(join(dirname(__file__)))
if all([tree_base != p for p in sys.path]):
    sys.path.append(tree_base)
# stereo imports
from stereo.common.general_utils import tree_base
from stereo.interfaces.implements import load_format
from stereo.common.s3_utils import my_open, my_glob
from stereo.common.tfrecord_utils import parse_example
from stereo.models.arch_utils import im_padding
from stereo.models.sm.train_estimator import get_log_tensors, create_summary_hooks
from sbs.estimator_sbs.estimator_sbs import EstimatorSBS
import numpy as np
import sys
import os
# +
#map_func related functions
def crop_pad_images_and_fix_origin(features, crop_h, crop_w, pad_h, pad_w, image_names):
    paddings = (pad_h, pad_w, (0, 0))
    for image_name in image_names:
        image = features[image_name]
        shape = image.shape
        cropped = image[crop_h[0]:shape[0]-crop_h[1], crop_w[0]:shape[1]-crop_w[1], :]
        padded = tf.pad(tensor=cropped, paddings=paddings)
        features[image_name] = padded
    features['origin'] = features['origin'] + np.array([pad_h[0]-crop_h[0], pad_w[0]-crop_w[0]], dtype='float32')
    return features


def origin_l2_focal_l2(features, unet_levels=6):
    corr_down_levels = 2
    padding = im_padding(features['I_cntr'], unet_levels)
    features['origin_l2'] = (features['origin'] + [padding[1][0], padding[0][0]]) / (2 ** corr_down_levels)
    features['origin_l2'] = tf.cast(features['origin_l2'], tf.float32)
    features['focal_l2'] = features['focal'] / (2 ** corr_down_levels)
    features['focal_l2'] = tf.cast(features['focal_l2'], tf.float32)
    return features


def xy(features, im_name):
    im_sz = features[im_name].shape.as_list()[:2]
    x, y = tf.meshgrid(tf.range(im_sz[1], dtype=tf.float32), tf.range(im_sz[0], dtype=tf.float32))
    features['x'] = tf.expand_dims(x - features['origin'][0], 2)
    features['y'] = tf.expand_dims(y - features['origin'][1], 2)
    return features


def add_blur_kernels(features, blur_kernels, blur_kernels_names):
    blur_kernels_list = []
    for blur_kernel_name in blur_kernels_names:
        blur_kernels_list.append(blur_kernels[blur_kernel_name][:, :, :, 0])
    features['blur_kernels'] = np.concatenate(blur_kernels_list, 2).astype(np.float32)
    return features


def map_func(features, blur_kernels, blur_kernels_names):
    image_names = ['I_cntr', 'I_srnd_0', 'I_srnd_1', 'I_srnd_2', 
                   'im_lidar_inv', 'im_lidar_short_inv', 'im_mask', 'photo_loss_im_mask']
    features = crop_pad_images_and_fix_origin(features, (0, 0), (8, 8), (5, 5), (0, 0), image_names)
    features = origin_l2_focal_l2(features, unet_levels=6)
    features = xy(features, im_name='I_cntr')
    features = add_blur_kernels(features, blur_kernels, blur_kernels_names)

    # add a dummy deeplab image
    features['deeplab'] = tf.ones_like(features['I_cntr'], dtype='uint8') * 255

    return features


# -

# estimator input_fn
def input_fn(tt, batch_sz):
    npc = tf.data.experimental.AUTOTUNE
    dataset_path = '_cache/data/tf_datasets/v3.1.3/main.full'
    with my_open(join(dataset_path, 'ds_format.json')) as f:
        ds_format = json.load(f)
    tfrecord_list = my_glob(join(dataset_path, tt, '*.tfrecord'))
    blur_kernels_path = "_cache/data/tf_datasets/v2.2/blur_kernels.npz"
    blur_kernels = {}
    with my_open(blur_kernels_path,'br') as f:
        blur_kernels_npz = np.load(f)
        for npz_key in blur_kernels_npz.keys():
            blur_kernels[npz_key] = 1. * blur_kernels_npz[npz_key]     
    blur_kernels_names = ["frontCornerLeft_to_main", "frontCornerRight_to_main", "parking_front_to_main"]

    ds = tf.data.TFRecordDataset(tfrecord_list, compression_type="GZIP")
    ds = ds.map(partial(parse_example, ds_format=ds_format), num_parallel_calls=npc)
    ds = ds.map(lambda features: map_func(features, blur_kernels, blur_kernels_names), num_parallel_calls=npc)
    ds = ds.map(lambda features: (features, {}), num_parallel_calls=npc)
    ds = ds.batch(batch_sz, drop_remainder=True)
    ds = ds.repeat(-1)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


# +
#estimator model_fn related functions
def init_arch(features):
    with tf.compat.v1.name_scope("features"):
        for k, v in features.items():
            tf.identity(v, name=k)
    arch_name = 'vidar_v0.2.2_conf_dlo_add'
    arch_module = imp.load_source('arch', join(tree_base(), 'stereo', 'models', 'arch', 'stereo', arch_name + '.py'))
    arch_func = arch_module.arch
    arch_format = load_format(*arch_module.arch_format)
    arch_kwargs = { 
        "min_Z": 1.0,
        "max_Z": 1000.0,
        "regularization": "l2",
        "reg_constant": 1e-4,
        "corr_steps_kwargs": {
          "num_steps": 48,
          "min_Z": 2.0,
          "max_Z": 500.0,
          "min_delta_Z": 0.1
        }
    }
    arch_inputs = {}
    for k in arch_format:
        arch_inputs[k] = features[k]
    arch_inputs.update(arch_kwargs)
    with tf.compat.v1.name_scope("arch"):
        out_dict = arch_func(**arch_inputs)
    return out_dict


def init_loss(out_dict, features):
    loss_name = 'lidar+short+phot+conf+deeplab_v2.1.0.0'
    loss_module = imp.load_source('loss', join(tree_base(), 'stereo', 'models', 'loss', 'stereo', loss_name + '.py'))
    loss_func = loss_module.loss
    loss_format = load_format(*loss_module.loss_format)
    loss_kwargs = {
        "measure": "ssim",
        "alpha_phot": 1.0,
        "alpha_lidar": 10.0,
        "alpha_lidar_short": 0.25,
        "alpha_lidar_sky": 1.0,
        "alpha_conf": 10.0,
        "alpha_conf_short": 10.0
    }

    loss_inputs = {}
    for k in loss_format:
        loss_inputs[k] = features[k]
    loss_inputs.update(loss_kwargs)
    loss_inputs['out'] = out_dict
    with tf.compat.v1.name_scope("loss"):
        loss = loss_func(**loss_inputs)
    return loss


def init_model(features):
    out_dict = init_arch(features)
    loss = init_loss(out_dict, features)
    return loss


def model_fn(features, labels, mode, params):
    assert mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.TRAIN]
    save_dir = params['model_dir']
    loss = init_model(features)
    log_tensors = get_log_tensors(params['conf']['model_params'].get('scalars_to_log', []), mode)
    if mode == tf.estimator.ModeKeys.EVAL:
        summary_hooks = create_summary_hooks(params, save_dir, mode)
        evaluation_hooks = [tf.estimator.LoggingTensorHook(tensors=log_tensors, every_n_iter=1)]
        evaluation_hooks.extend(summary_hooks)
        estimator_spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, evaluation_hooks=evaluation_hooks)
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.compat.v1.train.get_or_create_global_step()
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        learning_rate = params['learning_rate']
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        with tf.control_dependencies(update_ops):
            train_op = opt.minimize(loss, global_step=global_step)
        export_exel()
        log_tensors['global_step'] = global_step
        summary_hooks = create_summary_hooks(params, save_dir, mode)
        training_hooks = [tf.estimator.LoggingTensorHook(tensors=log_tensors, every_n_iter=1)]
        training_hooks.extend(summary_hooks)

        scaffold = tf.compat.v1.train.Scaffold()
        estimator_spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,
                                                    training_chief_hooks=training_hooks, scaffold=scaffold)
    return estimator_spec


def export_exel(file='sergei_all.csv'):
    table = []
    from _collections import OrderedDict
    import pandas as pd

    ops = [op for op in
           tf.compat.v1.get_default_graph().get_operations()]  # if op.type in ['Conv2D','Mul','RealDiv','MatrixInverse','ImageProjectiveTransformV2','DepthwiseConv2dNative']]

    for op in ops:
        entry = OrderedDict()
        entry.update({'name': op.name, 'type': op.type})
        for e, i in enumerate(op.inputs):
            if len(i.shape) > 0:
                entry.update(
                    {
                        f'input_{e}': i.name,
                        f'input_{e}_shape': i.shape.as_list()
                    }
                )
        for e, o in enumerate(op.outputs):
            if len(o.shape) > 0:
                entry.update(
                    {
                        f'output_{e}': o.name,
                        f'output_{e}_shape': o.shape.as_list()
                    }
                )
        table.append(entry)

    df = pd.DataFrame(table)
    df.to_csv(file)
# -

def create_estimator(estimator_params, model_dir, num_gpus=1):

    estimator_params['model_dir'] = model_dir
    
    distribution = None
    if int(num_gpus) > 1:
        distribution = tf.distribute.MirroredStrategy()

    config = tf.estimator.RunConfig(
        model_dir=model_dir,
        save_summary_steps=None,  # None b/c we save summaries using hooks. In tf.estimator.train_and_evaluate
                                  # evaluation is triggered every time a checkpoint is saved.
                                  # See github issue: https://github.com/tensorflow/tensorflow/issues/17650
        save_checkpoints_steps=estimator_params['eval_steps'],
        #session_config=tf.compat.v1.ConfigProto(log_device_placement=True),
        keep_checkpoint_max=10,
        log_step_count_steps=5,
        train_distribute=distribution,
        eval_distribute=None)

    if 'warm_start' in estimator_params.keys():
        ckpt_to_initialize_from = estimator_params['warm_start']
        warm_start_from = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=ckpt_to_initialize_from)
    else:
        warm_start_from = None
    return tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, config=config,
                                  params=estimator_params ,warm_start_from=warm_start_from)


def get_estimator_params(eval_steps, eval_iters, learning_rate, checkpoint):
    estimator_params = {'conf':{'model_params': {
                                    'images_to_summarize': [
                                          "arch/corr_scores_argmax",
                                          "features/I_cntr",
                                          "features/I_srnd_0",
                                          "features/I_srnd_1",
                                          "features/I_srnd_2",
                                          "features/im_mask",
                                          "features/deeplab",
                                          "features/im_lidar_inv",
                                          "features/im_lidar_short_inv",
                                          "loss/I_warp_cntr_0",
                                          "loss/I_warp_cntr_1",
                                          "loss/I_warp_cntr_2",
                                          "loss/real_error",
                                          "loss/real_error_short",
                                          "arch/out",
                                          "arch/out_conf"
                                        ],
                                    'scalars_to_summarize': [
                                          "loss/loss_phot",
                                          "loss/loss_lidar",
                                          "loss/loss_lidar_short",
                                          "loss/loss_lidar_sky",
                                          "loss/loss_reg",
                                          "loss/loss_conf",
                                          "loss/loss_conf_short",
                                          "loss/loss"
                                        ],
                                    'scalars_to_log': [
                                          "loss/loss_phot",
                                          "loss/loss_lidar",
                                          "loss/loss_conf",
                                          "loss/loss"
                                        ],
                                    'reverse_image_summaries': True
                                },
                        },
                        'eval_steps': eval_steps,
                        'eval_iters': eval_iters,
                        'learning_rate': learning_rate,
                        'warm_start': checkpoint
                            #'_cache/mobileye-habana/mobileye-team-stereo/models/vidar_main_v0.2.3_conf_dlo_add/model.ckpt-2000000'
        #'s3://mobileye-habana/mobileye-team-stereo/models/vidar_main_v0.2.3_conf_dlo_add/model.ckpt-2000000'
                       }
    return estimator_params


if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    parser = argparse.ArgumentParser(description='Run the Vidar Stereo workload',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--warm-start', '-w', type=str, help = 'path to the checkpoint directory',
                        default='refs/2000000/checkpoint/model.ckpt-2000000')
    parser.add_argument('--compare-sbs', '-c', action='store_true', help='enable sbs comparison tool')
    parser.add_argument('--dump-json', '-j', type=str, help = 'path to the sbs json config', default='dump_halo.json')
    parser.add_argument('--model-dir', '-m', type=str, help='folder path to save the checkpoints and the tensorboard logs. this folder will be deleted first, then created.',
                        default='tmp/model_dir')
    parser.add_argument('--batch-sz', '-b', type=int, help='batch size', default=2)
    parser.add_argument('--eval-steps', '-e', default=5000, type=int, help='amount of training steps per evaluation and checkpoint')
    parser.add_argument('--eval-iters', '-i', default=500, type=int, help='amount of evaluation iterations')
    parser.add_argument('--max-steps', '-s', default=100000, type=int, help='max steps to train in total')
    parser.add_argument('--num-gpus', '-g', default=1, type=int, help='number of gpus')
    parser.add_argument('--device', '-d', default='HPU', help='run on device: HPU, CPU, GPU')
    parser.add_argument('--ref-dir', '-r', type=str, default=None, help='path to reference dump directory to compare to this run')
    args = parser.parse_args()

    learning_rate = 1e-4

    # remove the model dir to prevent estimator from loading last checkpoint from there
    if os.path.isdir(args.model_dir):
        shutil.rmtree(args.model_dir)
    if args.device == 'HPU':
        load_habana_module()
    if args.compare_sbs:
        estimator_sbs = EstimatorSBS(args.dump_json)
    estimator_params = get_estimator_params(args.eval_steps, args.eval_iters, learning_rate, args.warm_start)
    estimator = create_estimator(estimator_params, args.model_dir, num_gpus=args.num_gpus)
    hooks = None
    if args.compare_sbs:
        hooks = estimator_sbs.get_hooks()
    train_spec = tf.estimator.TrainSpec(lambda: input_fn(tt='train', batch_sz=args.batch_sz), max_steps=args.max_steps, hooks=hooks)
    eval_spec = tf.estimator.EvalSpec(lambda: input_fn(tt='test', batch_sz=args.batch_sz), steps=args.eval_iters, throttle_secs=0)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    sys.exit(estimator_sbs.get_last_run_result())


