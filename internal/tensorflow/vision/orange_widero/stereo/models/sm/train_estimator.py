import argparse
import os
from os.path import isdir, join
from os import listdir
import re

import numpy as np

from stereo.models.mvs_model import get_model
from stereo.models.sm.sm_setup import sm_setup
from stereo.models.sm.sm_utils import get_checkpoint_path, get_tensors_by_regex
from stereo.models.sm.create_dataset import create_dataset
from stereo.models.model_utils import save_simple_model, ckpt_fix_mean_to_sum
from stereo.common.general_utils import tree_base

import json
import tensorflow as tf


def get_optimizer(optimizer_name, learning_rate):
    if optimizer_name == "AdamOptimizer":
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    elif optimizer_name == "GradientDescentOptimizer":
        opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)
    else:
        assert False
    return opt


def get_learning_rate(global_step, learning_rate=1e-4, lr_schedule=None, name='learning_rate'):
    import stereo.models.lr_schedules as schedules
    if lr_schedule is not None:
        learning_rate = getattr(schedules, lr_schedule['name'])(global_step, learning_rate, **lr_schedule['kwargs'])
    learning_rate = tf.identity(learning_rate, name=name)
    return learning_rate


def model_fn(features, labels, mode, params, config=None):

    assert mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.TRAIN]
    model_params = params['conf']['model_params']

    out, loss = get_model(features, model_params, training=(mode == tf.estimator.ModeKeys.TRAIN))
    loss_ = loss if not isinstance(loss, dict) else loss['loss']
    save_dir = os.getenv('SM_HP_MODEL_DIR') or config.model_dir or params['model_dir']  # TODO: cleanup

    log_tensors = get_log_tensors(model_params.get('scalars_to_log', []), mode)

    if mode == tf.estimator.ModeKeys.EVAL:
        summary_hooks = create_summary_hooks(params, save_dir, mode)
        evaluation_hooks = [tf.estimator.LoggingTensorHook(tensors=log_tensors, every_n_iter=1)]
        evaluation_hooks.extend(summary_hooks)
        estimator_spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss_, evaluation_hooks=evaluation_hooks)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.compat.v1.train.get_or_create_global_step()

        # remove freeze_layers from trainable variables
        train_vars = get_train_vars(model_params.get('freeze_layers', None), model_params.get('train_vars_regex', None))
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)

        # setup the optimizer
        if 'optimizers' in model_params.keys():
            optimizers = []
            all_train_vars = []
            for optimizer_name, optimizer_conf in model_params['optimizers'].items():
                learning_rate = get_learning_rate(global_step, optimizer_conf['learning_rate'],
                                                  optimizer_conf.get('lr_schedule', None),
                                                  name='learning_rate_%s' % optimizer_name)
                opt = get_optimizer(optimizer_conf['optimizer_type'], learning_rate)
                pattern = re.compile(optimizer_conf['vars_regex'])
                opt_train_vars = [v for v in train_vars if pattern.match(v.name)]
                optimizers.append({'optimizer': opt, 'train_vars': opt_train_vars,
                                   'loss': optimizer_conf.get('loss', 'loss')})
                all_train_vars.extend(opt_train_vars)
            assert sorted(list(set([v.name for v in all_train_vars]))) == sorted(list(set([v.name for v in train_vars])))

            with tf.control_dependencies(update_ops):
                opt_global_step = global_step
                train_ops = []
                for optimizer in optimizers:
                    optimizer_loss = loss if not isinstance(loss, dict) else loss[optimizer['loss']]
                    grads = tf.gradients(ys=optimizer_loss, xs=optimizer['train_vars'])
                    train_ops.append(optimizer['optimizer'].apply_gradients(zip(grads, optimizer['train_vars']),
                                                                            global_step=opt_global_step))
                    opt_global_step = None
                train_op = tf.group(*train_ops)
        else:
            learning_rate = get_learning_rate(global_step, model_params['learning_rate'], model_params.get('lr_schedule', None))
            log_tensors['learning_rate'] = learning_rate
            opt = get_optimizer(model_params['train_op'], learning_rate)
            with tf.control_dependencies(update_ops):
                train_op = opt.minimize(loss_, global_step=global_step, var_list=train_vars)

        log_tensors['global_step'] = global_step
        summary_hooks = create_summary_hooks(params, save_dir, mode)
        training_hooks = [tf.estimator.LoggingTensorHook(tensors=log_tensors, every_n_iter=1)]
        training_hooks.extend(summary_hooks)

        scaffold = tf.compat.v1.train.Scaffold()

        if save_dir.startswith("s3"):
            from stereo.models.sm.sm_utils import SyncCheckpointListener
            # this is needed in train_and_evaluate since it doesn't get saving_listeners argument.
            saver_hook = tf.estimator.CheckpointSaverHook(
                os.path.join(config.model_dir),  # TODO: cleanup
                save_secs=config.save_checkpoints_secs,
                save_steps=config.save_checkpoints_steps,
                saver=scaffold.saver,
                listeners=[SyncCheckpointListener(src=config.model_dir, dst=save_dir)])  # TODO: cleanup
            training_hooks.append(saver_hook)
        estimator_spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss_, train_op=train_op,
                                                    training_chief_hooks=training_hooks, scaffold=scaffold)
    return estimator_spec


def create_summary_hooks(estimator_params, save_dir, mode):

    model_params = estimator_params['conf']['model_params']

    # add tensorboard summaries and merge them
    summary_scalars = []
    summary_images = []
    summary_strings = []
    summary_histograms = []

    if model_params['scalars_to_summarize']:
        if mode == tf.estimator.ModeKeys.TRAIN:
            model_params['scalars_to_summarize'].append("learning_rate")
        scalars_to_summarize = set()
        for s in model_params['scalars_to_summarize']:
            scalars_to_summarize.update(get_tensors_by_regex(s))
        for s in scalars_to_summarize:
            if s == 'loss':
                name = 'total_loss'
            else:
                name = s
            summary_scalars.append(tf.compat.v1.summary.scalar(name, tf.compat.v1.get_default_graph().get_tensor_by_name(s + ':0')))

    if model_params['images_to_summarize']:
        images_to_summarize = set()
        for s in model_params['images_to_summarize']:
            images_to_summarize.update(get_tensors_by_regex(s))
        for s in images_to_summarize:
            im = tf.compat.v1.get_default_graph().get_tensor_by_name(s + ':0')
            im = im if im.shape.dims is None or len(im.shape.as_list()) == 4 else tf.expand_dims(im, -1)
            im = tf.reverse(im, axis=[1]) if model_params.get('reverse_image_summaries', False) else im
            summary_images.append(tf.compat.v1.summary.image(s, im, max_outputs=1))

    if model_params.get('strings_to_summarize', False):
        for s in model_params['strings_to_summarize']:
            str_ = tf.compat.v1.get_default_graph().get_tensor_by_name(s + ':0')[0, ...]
            summary_strings.append(tf.compat.v1.summary.text(s, str_))

    if model_params.get('histograms_to_summarize', False):
        for s in model_params['histograms_to_summarize']:
            hist = tf.compat.v1.get_default_graph().get_tensor_by_name(s + ':0')
            summary_histograms.append(tf.compat.v1.summary.histogram(s, hist))

    merged_scalar_summaries = None if len(summary_scalars) == 0 else tf.compat.v1.summary.merge(summary_scalars)
    summary_nonscalars = summary_images + summary_strings + summary_histograms
    merged_nonscalar_summaries = None if len(summary_nonscalars) == 0 else tf.compat.v1.summary.merge(summary_nonscalars)

    summary_hooks = []
    if mode == tf.estimator.ModeKeys.TRAIN:
        output_dir = os.path.join(save_dir, 'train')
        scalar_save_step = 5
        nonscalar_save_step = estimator_params['eval_steps']
    elif mode == tf.estimator.ModeKeys.EVAL:
        output_dir = os.path.join(save_dir, 'test')
        scalar_save_step = 1
        nonscalar_save_step = estimator_params['eval_iters']

    if merged_scalar_summaries is not None:
        summary_hooks.append(tf.estimator.SummarySaverHook(
            save_steps=scalar_save_step,
            output_dir=output_dir,
            summary_op=merged_scalar_summaries))

    if merged_nonscalar_summaries is not None:
        summary_hooks.append(tf.estimator.SummarySaverHook(
            save_steps=nonscalar_save_step,
            output_dir=output_dir,
            summary_op=merged_nonscalar_summaries))

    return summary_hooks


def get_train_vars(freeze_layers=None, train_vars_regex=None):

    assert freeze_layers is None or train_vars_regex is None

    all_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
    if freeze_layers is None and train_vars_regex is None:
        return all_vars

    if train_vars_regex is not None:
        pattern = re.compile(train_vars_regex)
        train_vars = [v for v in all_vars if pattern.match(v.name)]
        freeze_vars = set(all_vars).difference(train_vars)

    elif freeze_layers is not None:
        def is_trainable(v_name):
            return sum([v_name.startswith(l) for l in freeze_layers]) == 0
        train_vars = [v for v in all_vars if is_trainable(v.name)]
        freeze_vars = set(all_vars).difference(train_vars)

    print("Frozen variables:")
    print(os.linesep.join([str(v) for v in freeze_vars]))
    print("Trainable variables:")
    print(os.linesep.join([str(v) for v in train_vars]))

    return train_vars


def get_log_tensors(scalars_to_log, mode):
    tensors_to_log = set()
    for s in scalars_to_log:
        tensors_to_log.update(get_tensors_by_regex(s))
    log_tensors = {'{}:{}'.format(mode, s): tf.compat.v1.get_default_graph().get_tensor_by_name(s + ':0')
                   for s in tensors_to_log}
    return log_tensors


def create_estimator(estimator_params, model_dir, local):

    estimator_params['conf']['model_params']['model_dir'] = model_dir
    if model_dir.startswith("s3"):
        model_dir = os.getenv('SM_MODEL_DIR')  # /opt/ml/model. https://github.com/aws/sagemaker-containers#sm-model-dir
        print("local_model_dir={}".format(model_dir))

    distribution = None
    if not local:
        num_gpus = os.getenv('SM_NUM_GPUS', 1)
        if int(num_gpus) > 1:
            distribution = tf.distribute.MirroredStrategy()

    config = tf.estimator.RunConfig(
        model_dir=model_dir,
        save_summary_steps=None,  # None b/c we save summaries using hooks. In tf.estimator.train_and_evaluate
                                  # evaluation is triggered every time a checkpoint is saved.
                                  # See github issue: https://github.com/tensorflow/tensorflow/issues/17650
        save_checkpoints_steps=estimator_params['eval_steps'],
        session_config=None,
        keep_checkpoint_max=10,
        log_step_count_steps=5,
        train_distribute=distribution,
        eval_distribute=None)

    # possibly set warm start checkpoint
    if not estimator_params['conf']['model_params']['init'].get('warm_start', False):
        warm_start_from = None
    else:
        warm_start_conf = estimator_params['conf']['model_params']['init']['warm_start']
        restore_iter = warm_start_conf.get('restore_iter', -1)
        model_s3 = '/'.join(estimator_params['conf']['model_base_path'].split('/')[3:])
        ckpt_dir = '/'.join([model_s3, warm_start_conf['model_name']])
        ckpt_to_initialize_from = get_checkpoint_path(ckpt_dir, restore_iter)
        if warm_start_conf.get('mean_to_sum', False):
            ckpt_to_initialize_from = ckpt_fix_mean_to_sum(os.path.join(estimator_params['conf']['model_params']['model_dir'],
                                                                        "model_graph.meta"),
                                                           ckpt_to_initialize_from)
        vars_to_warm_start = str(warm_start_conf.get('vars_to_warm_start', '.*'))
        warm_start_from = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=ckpt_to_initialize_from,
                                                         vars_to_warm_start=vars_to_warm_start)
    return tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, config=config,
                                  params=estimator_params, warm_start_from=warm_start_from)


def input_fn(channel, conf, pipe_mode, batch_sz, shuffle, epochs):

    ds_list = []
    ds_weights = []
    weighted_sampling_after_batching = conf['data_params'].get('weighted_sampling_after_batching', False)

    for section in conf['data_params']['datasets'].keys():
        channel_k = '%s_%s' % (channel, section) if pipe_mode else channel
        augm = (channel == 'train') and ('augmentation' in conf['data_params'].keys())
        ds = create_dataset(conf, section, channel=channel_k, pipe_mode=pipe_mode, augm=augm)
        if shuffle:
            ds = ds.shuffle(buffer_size=100 * 10, reshuffle_each_iteration=True,
                            seed=conf['data_params'].get('random_seed', None))
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        ds = ds.repeat(epochs)
        if weighted_sampling_after_batching:
            ds = ds.batch(batch_sz, drop_remainder=True)
        ds_list.append(ds)
        if conf['data_params'].get('weights', False):
            if channel in conf['data_params']['weights'].keys():
                w = conf['data_params']['weights'][channel][section]
            else:
                w = conf['data_params']['weights'][section]
            ds_weights.append(w)
    if ds_weights:
        ds_weights = np.array(ds_weights)
        ds_weights = (ds_weights / np.sum(ds_weights)).tolist()
        print("Using dataset weight factor array: ", ds_weights)
    else:  # when no weights in the conf, uniform sampling of ds_list performed
        ds_weights = None

    ds = tf.data.experimental.sample_from_datasets(ds_list, ds_weights)
    if not weighted_sampling_after_batching:
        ds = ds.batch(batch_sz, drop_remainder=True)

    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def load_json(json_path):
    json_name = json_path.split('/')[-1]
    json_conf_dir = json_path.split('/')[-2]
    conf_dirs = listdir(join(tree_base(), 'stereo', 'models', 'conf'))
    conf_dirs.remove(json_conf_dir)
    for conf_dir in conf_dirs:
        if isdir(join(tree_base(), 'stereo', 'models', 'conf', conf_dir)):
            conf_dir_jsons = listdir(join(tree_base(), 'stereo', 'models', 'conf', conf_dir))
            assert json_name not in conf_dir_jsons

    with open(json_path, 'rb') as f:
        print("Reading configuration JSON: {}".format(json_path))
        conf = json.load(f)
    return conf


def main(args):

    if not args.local:
        sm_setup()

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG if args.verbose else tf.compat.v1.logging.INFO)

    args.json_path = os.path.join(tree_base(), args.json_path)
    conf = load_json(args.json_path)
    estimator_params = {'eval_steps': args.eval_steps, 'eval_iters': args.eval_iters, 'conf': conf}

    save_simple_model(args.json_path, args.model_dir)
    estimator = create_estimator(estimator_params, args.model_dir, args.local)

    def train_input_fn():
        return input_fn(channel='train', conf=conf, pipe_mode=args.pipe_mode, batch_sz=args.batch_size,
                        shuffle=not args.local, epochs=args.num_epochs)

    def eval_input_fn():
        return input_fn(channel='test', conf=conf, pipe_mode=args.pipe_mode, batch_sz=args.batch_size,
                        shuffle=not args.local, epochs=1)

    train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=args.max_steps if args.max_steps > 0 else None)
    eval_spec = tf.estimator.EvalSpec(eval_input_fn, steps=args.eval_iters, throttle_secs=0)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Stereo DNN main script')
    parser.add_argument('-j', '--json_path', required=True, help="path to config file")
    parser.add_argument('-l', '--local', action='store_true', help="set this to run locally on the CPU")
    parser.add_argument('-n', '--name', required=True, help="name of the run")
    parser.add_argument('-m', '--model_dir', required=True, action='store', help="directory for checkpoints, events")
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('-e', '--eval_steps', type=int, default=5000)
    parser.add_argument('-i', '--eval_iters', type=int, default=500)   
    parser.add_argument('-x', '--max_steps', type=int, default=1000000)
    parser.add_argument('-u', '--num_epochs', type=int, default=-1)
    parser.add_argument('-p', '--pipe_mode', action='store', type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('-v', '--verbose', action='store', type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('-r', '--requirements', help='requirements file')
    args_ = parser.parse_args()
    main(args_)
