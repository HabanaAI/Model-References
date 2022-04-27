#!/usr/bin/env python3

# Copyright 2018 Google. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################
# Changes:
# - Ported to TF 2.2
# - Cleaned up imports
# - Added support for HPU
# - Replaced tf.flags with argparse
# - Added several flags
# - Set TF_BF16_CONVERSION flag to 0
# - Added default dataset paths
# - Added SSDTrainingHook
# - Removed timeout_mins from next_checkpoint function
# - Added multi-node training on Horovod
# - Removed support for TPU
# - Added inference mode
# - Added support for using static datasets
# - Added support for saving TF summary
# - Removed eval results summary from coco_eval function
# - Removed train_and_eval mode
# - Formatted with autopep8
# - Added absolute paths in imports
# - replaced tf.Estimator with bare SessionRun calls to decrease host overhead
"""Training script for SSD.
"""

from TensorFlow.computer_vision.SSD_ResNet34.argparser import SSDArgParser
from TensorFlow.computer_vision.SSD_ResNet34 import model
from TensorFlow.computer_vision.SSD_ResNet34 import constants
from TensorFlow.computer_vision.SSD_ResNet34 import dataloader
from TensorFlow.computer_vision.SSD_ResNet34 import static_dataloader
from TensorFlow.computer_vision.SSD_ResNet34 import coco_metric
from TensorFlow.common.tb_utils import (
    write_hparams_v1, TBSummary, ExamplesPerSecondEstimatorHook, TensorBoardHook)

import datetime
import math
from mpi4py import MPI
import numpy as np
import os
import platform
import shutil
import tensorflow.compat.v1 as tf
import tensorflow.profiler.experimental as profiler
from tensorflow_estimator.python.estimator import util as estimator_util
from tensorflow.python.training.summary_io import SummaryWriterCache
import time

tf.compat.v1.enable_resource_variables()


class SSDTrainingHook:
    def __init__(self, steps_per_epoch, global_batch_size):
        self.steps_per_epoch = steps_per_epoch
        self.global_batch_size = global_batch_size
        self.cur_epoch = 0
        self.epoch_start_time = time.perf_counter()

    def _format_duration(self, seconds):
        return str(datetime.timedelta(seconds=round(seconds)))

    def start_epoch(self):
        self.epoch_start_time = time.perf_counter()

    def stop_epoch(self):
        epoch_duration = time.perf_counter() - self.epoch_start_time
        ips = self.global_batch_size * self.steps_per_epoch / epoch_duration
        self.cur_epoch += 1
        tf.logging.info("Epoch {} finished. Duration: {} -> IPS: {:.2f}".format(
            self.cur_epoch, self._format_duration(epoch_duration), ips))


def next_checkpoint(model_dir):
    """Yields successive checkpoints from model_dir."""
    last_ckpt = None
    last_step = 0
    while True:
        # Get all the checkpoint from the model dir.
        ckpt_path = tf.train.get_checkpoint_state(model_dir)
        all_model_checkpoint_paths = ckpt_path.all_model_checkpoint_paths

        ckpt_step = np.inf
        next_ckpt = None
        # Find the next checkpoint to eval based on last_step.
        for ckpt in all_model_checkpoint_paths:
            step = int(os.path.basename(ckpt).split('-')[1])
            if step > last_step and step < ckpt_step:
                ckpt_step = step
                next_ckpt = ckpt

        if next_ckpt is not None:
            last_step = ckpt_step
            last_ckpt = next_ckpt

        yield last_ckpt


def construct_run_config(global_batch_size, model_dir, distributed_optimizer, num_shards, shard_index):
    """Construct the run config."""

    if ARGS.static:
        training_file_pattern = os.path.join(
            ARGS.data_dir, f'worker_{shard_index}', 'train-*')
    else:
        training_file_pattern = os.path.join(ARGS.data_dir, 'train*')

    params = dict(
        dtype=ARGS.dtype,
        lr_warmup_epoch=ARGS.lr_warmup_epoch,
        first_lr_drop_epoch=40.0 * (1.0 + ARGS.k * 0.1),
        second_lr_drop_epoch=50.0 * (1.0 + ARGS.k * 0.1),
        weight_decay=ARGS.weight_decay,
        base_learning_rate=ARGS.base_lr,
        visualize_dataloader=ARGS.vis_dataloader,
        use_cocoeval_cc=ARGS.use_cocoeval_cc,
        resnet_checkpoint=ARGS.resnet_checkpoint,
        training_file_pattern=training_file_pattern,
        val_file_pattern=os.path.join(ARGS.data_dir, 'val*'),
        val_json_file=os.path.join(
            ARGS.data_dir, 'raw-data', 'annotations', 'instances_val2017.json'),
        mode=ARGS.mode,
        model_dir=model_dir,
        steps_per_epoch=math.ceil(
            ARGS.num_examples_per_epoch / global_batch_size),
        num_examples_per_epoch=math.ceil(
            ARGS.num_examples_per_epoch / global_batch_size) * global_batch_size,
        eval_samples=ARGS.eval_samples,
        distributed_optimizer=distributed_optimizer,
        batch_size=ARGS.batch_size,
        global_batch_size=global_batch_size,
        num_shards=num_shards,
        shard_index=shard_index,
        distributed_bn=ARGS.distributed_bn and ARGS.use_horovod,
        num_parallel_calls=ARGS.num_parallel_calls,
        threadpool_size=ARGS.threadpool_size,
        save_summary_steps=ARGS.save_summary_steps
    )

    return tf.compat.v1.estimator.RunConfig(
        model_dir=model_dir,
        save_checkpoints_steps=params['steps_per_epoch'] *
        ARGS.save_checkpoints_epochs,
        save_summary_steps=ARGS.save_summary_steps,
        keep_checkpoint_max=ARGS.keep_ckpt_max,
        log_step_count_steps=ARGS.log_step_count_steps), params


def coco_eval(predictions,
              current_step,
              coco_gt,
              eval_model_dir,
              use_cpp_extension=True):
    """Call the coco library to get the eval metrics."""
    tf.logging.info('Eval for step %d started' % current_step)
    eval_results = coco_metric.compute_map(
        predictions,
        coco_gt,
        use_cpp_extension=use_cpp_extension)

    tf.logging.info('Eval results: %s' % eval_results)
    print("COCO/AP value for step {} = {}".format(current_step,
                                                  eval_results['COCO/AP']))
    with TBSummary(eval_model_dir) as summary_writer:
        summary_writer.add_scalar(
            'accuracy', eval_results['COCO/AP'], current_step)
        for tag, value in eval_results.items():
            summary_writer.add_scalar(tag, value, current_step)


def save_summaries(writer, summaries):
    for step in summaries:
        for key, value in summaries[step].items():
            new_summary = tf.compat.v1.Summary(value=[
                tf.compat.v1.Summary.Value(
                    tag=key, simple_value=value)
            ])
            writer.add_summary(new_summary, step)


def save_ckpt(saver, session, global_step, path):
    path += "/model.ckpt"
    tf.logging.info(
        "Saving checkpoint for step {} to {}".format(global_step, path))
    saver.save(sess=session, global_step=global_step, save_path=path)


if __name__ == '__main__':
    parser = SSDArgParser(is_demo=False)
    ARGS = parser.parse_args()
    affinity = os.sched_getaffinity(0)
    print("Host: {}, CPU count = {}, affinity = {}".format(
        platform.node(), os.cpu_count(), affinity))
    if len(affinity) > ARGS.max_cpus:
        new_affinity = set(list(affinity)[:ARGS.max_cpus])
        print("Restricting CPUs to {} because max_cpus={}".format(
            new_affinity, ARGS.max_cpus))
        os.sched_setaffinity(0, new_affinity)

    num_shards = 1
    shard_index = 0

    os.environ["TF_BF16_CONVERSION"] = "0"
    # necessary to run TF built with MKL support
    os.environ["TF_DISABLE_MKL"] = "1"

    if ARGS.recipe_cache and 'TF_RECIPE_CACHE_PATH' not in os.environ:
        os.environ['TF_RECIPE_CACHE_PATH'] = ARGS.recipe_cache

        if os.path.isdir(ARGS.recipe_cache) and \
                (not ARGS.use_horovod or MPI.COMM_WORLD.Get_rank() == 0):
            shutil.rmtree(ARGS.recipe_cache)

    if ARGS.inference:
        ARGS.mode = 'inference'

    if not ARGS.no_hpu:
        if 'TF_DONT_CLUSTER' not in os.environ:
            # In order to split clusters (decreases overhead)
            os.environ['TF_DONT_CLUSTER'] = 'add_1'

        from habana_frameworks.tensorflow import load_habana_module
        load_habana_module()

    if ARGS.use_horovod:
        import horovod.tensorflow as hvd
        hvd.init()
        assert(hvd.is_initialized())
        if ARGS.no_hpu:
            # Horovod on GPU
            os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())

        if ARGS.recipe_cache:
            MPI.COMM_WORLD.Barrier()

    tf.disable_eager_execution()
    tf.logging.set_verbosity(tf.logging.INFO)

    global_batch_size = ARGS.batch_size
    distributed_optimizer = None
    model_dir = ARGS.model_dir

    if ARGS.use_horovod:
        global_batch_size = ARGS.batch_size * hvd.size()
        num_shards = hvd.size()
        shard_index = hvd.rank()
        distributed_optimizer = hvd.DistributedOptimizer

        if hvd.rank() > 0:
            model_dir = os.path.join(
                ARGS.model_dir, 'worker_' + str(hvd.rank()))

    run_config, params = construct_run_config(
        global_batch_size, model_dir, distributed_optimizer, num_shards, shard_index)

    tf.logging.info('steps_per_epoch: %s' % params['steps_per_epoch'])

    if ARGS.mode == 'train':
        write_hparams_v1(params['model_dir'],
                         {**params, 'precision': params['dtype']})

        tf.logging.info(params)

        if ARGS.static:
            input_reader = static_dataloader.SSDStaticInputReader(
                params['training_file_pattern'])
        else:
            input_reader = dataloader.SSDInputReader(
                params['training_file_pattern'],
                is_training=True,
                use_fake_data=ARGS.use_fake_data)

        input_fn = input_reader(params=params)
        input_iterator = tf.compat.v1.data.make_initializable_iterator(
            input_fn)
        features, labels, input_hooks = estimator_util.parse_input_fn_result(
            input_iterator)
        iterator_get_next = features.get_next()

        tf.logging.info('Initializing session')
        sess = tf.compat.v1.Session()

        tf.logging.info('Initializing input iterator')
        sess.run(input_iterator.initializer)

        tf.logging.info(
            "Initializing model. Inputs: {}".format(iterator_get_next))
        model_fn = model.ssd_model_fn(
            iterator_get_next[0], iterator_get_next[1], tf.estimator.ModeKeys.TRAIN, params)
        train_op = model_fn.train_op

        last_ckpt = tf.train.latest_checkpoint(params['model_dir'])
        if last_ckpt:
            tf.logging.info("Loading weights from {}".format(last_ckpt))
            tf.train.init_from_checkpoint(last_ckpt, assignment_map={'/': '/'})
        else:
            tf.logging.info("Loading ResNet34 weights from {}".format(
                params['resnet_checkpoint']))
            tf.train.init_from_checkpoint(params['resnet_checkpoint'], {
                '/': 'resnet%s/' % constants.RESNET_DEPTH,
            })

        sess.run(tf.global_variables_initializer())

        global_step = sess.run(tf.compat.v1.train.get_global_step())
        saver = tf.compat.v1.train.Saver(
            max_to_keep = ARGS.keep_ckpt_max,
            pad_step_number = True,
            save_relative_paths = True
            )
        writer = SummaryWriterCache.get(params['model_dir'])

        if ARGS.use_horovod:
            tf.logging.info('Broadcasting Variables')
            hook = hvd.BroadcastGlobalVariablesHook(0)
            hook.begin()
            hook.after_create_session(sess, None)

        if shard_index == 0:
            save_ckpt(saver, sess, global_step, params['model_dir'])

        tf.logging.info('Run training')

        steps_per_epoch = params['steps_per_epoch']
        epochs = ARGS.epochs
        if ARGS.steps > 0:
            steps_per_epoch = ARGS.steps
            epochs = 1

        profile_steps = [int(i) for i in ARGS.profile.split(',')] + global_step
        epoch_logger = SSDTrainingHook(steps_per_epoch, global_batch_size)

        for epoch in range(1, int(epochs+1)):
            summaries = dict()
            epoch_logger.start_epoch()
            for step in range(1, steps_per_epoch+1):
                global_step += 1

                if shard_index == 0 and global_step == profile_steps[0]:
                    profiler.start(params['model_dir'])

                start_time = time.perf_counter_ns()
                with profiler.Trace('train', step_num=global_step, _r=1):
                    loss, _ = sess.run([model_fn.loss, train_op])
                duration_ns = time.perf_counter_ns() - start_time

                if global_step % ARGS.log_step_count_steps == 0:
                    tf.logging.info("Step {} : loss = {:.3f} , time = {:.1f} ms".format(
                        global_step, loss, duration_ns/1e6))
                if global_step % ARGS.save_summary_steps == 0:
                    summaries[global_step] = {
                        'loss': loss, 'global_step/sec': 1e9/duration_ns}
                if shard_index == 0 and global_step == profile_steps[1]:
                    profiler.stop()
            epoch_logger.stop_epoch()

            save_summaries(writer, summaries)
            if epoch % ARGS.save_checkpoints_epochs == 0 and shard_index == 0:
                save_ckpt(saver, sess, global_step, params['model_dir'])

        if shard_index == 0:
            save_ckpt(saver, sess, global_step, params['model_dir'])

        sess.close()

    elif ARGS.mode == 'eval':
        eval_model_dir = os.path.join(params['model_dir'], 'eval')
        write_hparams_v1(eval_model_dir,
                         {**params, 'precision': params['dtype']})

        coco_gt = coco_metric.create_coco(
            params['val_json_file'], use_cpp_extension=params['use_cocoeval_cc'])

        run_config = run_config.replace(model_dir=eval_model_dir)
        eval_estimator = tf.estimator.Estimator(
            model_fn=model.ssd_model_fn,
            model_dir=eval_model_dir,
            config=run_config,
            params=params)

        last_ckpt = None
        for ckpt in next_checkpoint(params['model_dir']):
            if last_ckpt == ckpt:
                break
            last_ckpt = ckpt

            current_step = int(os.path.basename(ckpt).split('-')[1])
            tf.logging.info('Starting to evaluate step: %s' % current_step)

            try:
                predictions = list(
                    eval_estimator.predict(
                        checkpoint_path=ckpt,
                        input_fn=dataloader.SSDInputReader(
                            params['val_file_pattern'],
                            is_training=False,
                            use_fake_data=ARGS.use_fake_data)))
                coco_eval(predictions, current_step,
                          coco_gt, eval_model_dir,
                          ARGS.use_cocoeval_cc)
            except tf.errors.NotFoundError:
                tf.logging.info(
                    'Checkpoint %s no longer exists, skipping checkpoint' % ckpt)

    elif ARGS.mode == 'inference':
        import cv2

        def input_fn():
            dataset = tf.data.Dataset.range(1)
            image = cv2.imread(ARGS.inference)
            image = image.astype(np.uint8)
            width = image.shape[0]
            height = image.shape[1]
            image = tf.image.resize_images(
                image, size=(constants.IMAGE_SIZE, constants.IMAGE_SIZE))
            image /= 255.

            if params['dtype'] == 'bf16':
                image = tf.cast(image, dtype=tf.bfloat16)

            image = tf.reshape(
                image, [1, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3])
            dataset = dataset.take(1)
            dataset = dataset.map(lambda x: {constants.IMAGE: image,
                                             constants.BOXES: np.zeros((1, 200, 4), dtype=np.float32),
                                             constants.CLASSES: np.zeros((1, 200, 1), dtype=np.float32),
                                             constants.SOURCE_ID: [0],
                                             constants.RAW_SHAPE: [[width, height, 3]]})
            return dataset

        inference_estimator = tf.estimator.Estimator(
            model_fn=model.ssd_model_fn,
            model_dir=params['model_dir'],
            config=run_config,
            params=params)

        ckpt = tf.train.latest_checkpoint(params['model_dir'])
        tf.logging.info('Using checkpoint %s' % ckpt)

        out = list(inference_estimator.predict(
            checkpoint_path=ckpt,
            input_fn=input_fn))

        example = out[0]
        htot, wtot, _ = example[constants.RAW_SHAPE]
        pred_box = example['pred_box']
        pred_scores = example['pred_scores']
        indices = example['indices']
        predictions = []

        loc, label, prob = coco_metric.decode_single(
            pred_box, pred_scores, indices, constants.OVERLAP_CRITERIA,
            10, constants.MAX_NUM_EVAL_BOXES)

        out_img = cv2.imread(ARGS.inference)

        print("Predictions:")
        for loc_, label_, prob_ in zip(loc, label, prob):
            cl = constants.CLASS_INV_MAP[label_]
            color = (((cl >> 4) & 3) * 85, ((cl >> 2) & 3) * 85, (cl & 3) * 85)
            color_neg = (255-((cl >> 4) & 3) * 85, 255 -
                         ((cl >> 2) & 3) * 85, 255 - (cl & 3) * 85)
            top_left = (int(loc_[1]*wtot), int(loc_[0]*htot))
            bottom_right = (int(loc_[3]*wtot), int(loc_[2]*htot))
            text_pos = (int(loc_[1]*wtot+10), int(loc_[0]*htot+20))
            text_end = (int(loc_[1]*wtot+100), int(loc_[0]*htot+30))
            cv2.rectangle(out_img, top_left, bottom_right, color, 2)
            cv2.rectangle(out_img, top_left, text_end, color, -1)

            text = constants.CLASS_LABEL[label_]

            out_img = cv2.putText(
                out_img, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_neg, 1, cv2.LINE_AA)

            print("{:1.2f}  ({:3}) {:16}  {} {}".format(
                prob_, cl, text, top_left, bottom_right))

        cv2.imwrite("out.png", out_img)
        print("Output image saved to out.png.")
