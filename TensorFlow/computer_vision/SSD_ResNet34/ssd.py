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
# - Set TF_ENABLE_BF16_CONVERSION flag to 0
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

"""Training script for SSD.
"""

from TensorFlow.computer_vision.SSD_ResNet34 import model
from TensorFlow.computer_vision.SSD_ResNet34 import constants
from TensorFlow.computer_vision.SSD_ResNet34 import dataloader
from TensorFlow.computer_vision.SSD_ResNet34 import static_dataloader
from TensorFlow.computer_vision.SSD_ResNet34 import coco_metric
from TensorFlow.common.tb_utils import (
    write_hparams_v1, TBSummary, ExamplesPerSecondEstimatorHook)

import argparse
import math
import multiprocessing
import os
import sys
import threading
from absl import app
import numpy as np
import tensorflow.compat.v1 as tf
import time
import datetime

tf.compat.v1.enable_resource_variables()

DEFAULT_DATASET_PATH = "/software/data/tf/coco2017/ssd_tf_records"
DEFAULT_RN34_CKPT_PATH = "/software/data/tf/ssd_r34-mlperf/mlperf_artifact"

DATASET_PATH_LOCAL = "/data/coco2017/ssd_tf_records"
RN34_CKPT_PATH_LOCAL = "/data/ssd_r34-mlperf/mlperf_artifact"

DATASET_PATH_IGK = "/igk-datasets/coco2017/ssd_tf_records"
RN34_CKPT_PATH_IGK = "/igk-software/data/tf/ssd_r34-mlperf/mlperf_artifact"

if os.path.isdir(DATASET_PATH_LOCAL):
    DEFAULT_DATASET_PATH = DATASET_PATH_LOCAL
elif os.path.isdir(DATASET_PATH_IGK):
    DEFAULT_DATASET_PATH = DATASET_PATH_IGK

if os.path.isdir(RN34_CKPT_PATH_LOCAL):
    DEFAULT_RN34_CKPT_PATH = RN34_CKPT_PATH_LOCAL
elif os.path.isdir(RN34_CKPT_PATH_IGK):
    DEFAULT_RN34_CKPT_PATH = RN34_CKPT_PATH_IGK

DEFAULT_TRAINING_FILE_PATTERN = DEFAULT_DATASET_PATH+"/train"
DEFAULT_VAL_FILE_PATTERN = DEFAULT_DATASET_PATH+"/val"
DEFAULT_VAL_JSON_PATH = DEFAULT_DATASET_PATH + \
    "/raw-data/annotations/instances_val2017.json"


class SSDTrainingHook(tf.estimator.SessionRunHook):
    def __init__(self, total_steps, params):
        self.total_steps = total_steps
        self.steps_per_epoch = params['steps_per_epoch']
        self.global_batch_size = params['global_batch_size']
        self.cur_step = 0
        self.cur_step_in_epoch = 0
        self.cur_epoch = 0

    def _format_duration(self, seconds):
        return str(datetime.timedelta(seconds=round(seconds)))

    def before_run(self, run_context):
        cur_time = time.perf_counter()
        if self.cur_step == 0:
            self.training_start_time = cur_time
        if self.cur_step_in_epoch == 0:
            self.epoch_start_time = cur_time

    def after_run(self, run_context, run_values):
        self.cur_step += 1
        self.cur_step_in_epoch += 1
        cur_time = time.perf_counter()

        if self.cur_step_in_epoch == self.steps_per_epoch or self.cur_step == self.total_steps:
            epoch_duration = cur_time - self.epoch_start_time
            ips = self.global_batch_size * self.cur_step_in_epoch / epoch_duration
            self.cur_epoch += 1
            self.cur_step_in_epoch = 0
            print("Epoch {} finished. Duration: {} IPS: {:.2f}".format(
                self.cur_epoch, self._format_duration(epoch_duration), ips))

        if self.cur_step == self.total_steps:
            train_time = cur_time - self.training_start_time
            print("Training finished. Total time: {}".format(
                self._format_duration(train_time)))


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
        val_json_file=ARGS.val_json_file,
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run SSD training', formatter_class=argparse.MetavarTypeHelpFormatter)

    parser.add_argument('-d', '--dtype', metavar='bf16/fp32',
                        help='Data type: fp32 or bf16', type=str, choices=['fp32', 'bf16'], default='bf16')
    parser.add_argument('-b', '--batch_size', metavar='N',
                        default=128, help='Batch size (default: 128)', type=int)
    parser.add_argument('-e', '--epochs', metavar='N', default=50,
                        help='Number of epochs for training (default: 50)', type=float)
    parser.add_argument('--mode', metavar='train/eval', help='\'train\' or \'eval\'',
                        type=str, choices=['train', 'eval'], default='train')
    parser.add_argument('--inference', metavar='IMAGE', type=str,
                        help='path to image for inference (if set then mode is ignored)')
    parser.add_argument(
        '--no_hpu', help='Do not load Habana modules = train on CPU/GPU', action='store_true')
    parser.add_argument(
        '--use_horovod', help='Use Horovod for distributed training', action='store_true')
    parser.add_argument(
        '--distributed_bn', help='Use distributed batch norm', action='store_true')
    parser.add_argument('--vis_dataloader',
                        help='Visualize dataloader', action='store_true')
    parser.add_argument(
        '--profiling', help='turn on profiling (generate json every 20 steps)', action='store_true')
    parser.add_argument('--use_cocoeval_cc',
                        help='use_cocoeval_cc', action='store_true')
    parser.add_argument(
        '-f', '--use_fake_data', help='Use fake data to reduce the input preprocessing overhead (for unit tests)', action='store_true')
    parser.add_argument('--lr_warmup_epoch', default=5.0, metavar='N',
                        help='numer of epochs for learning rate warmup (default: 5)', type=float)
    parser.add_argument('--base_lr', default=3e-3,
                        metavar='BASE_LR', help='base learning rate (default: 3e-3)', type=float)
    parser.add_argument('--weight_decay', default=5e-4,
                        metavar='WD', help='L2 wight decay (default: 5e-4)', type=float)
    parser.add_argument('--k', default=0, help='k is an integer defining at which epochs the learning rate decays: '
                        '[40, 50] * (1 + k/10) (default: k=0)', type=int)
    parser.add_argument('--model_dir', metavar='<dir>', default='/tmp/ssd',
                        help='Location of model_dir (default: /tmp/ssd)', type=str)
    parser.add_argument('--resnet_checkpoint', metavar='<PATH>', default=DEFAULT_RN34_CKPT_PATH,
                        help='Location of the ResNet ckpt to use for model '
                        'init. (default: '+DEFAULT_RN34_CKPT_PATH+')', type=str)
    parser.add_argument('--training_file_pattern', metavar='<PATH>', default=DEFAULT_TRAINING_FILE_PATTERN,
                        help='Prefix for training data files (default: '+DEFAULT_TRAINING_FILE_PATTERN+')', type=str)
    parser.add_argument('--val_file_pattern', metavar='<PATH>', default=DEFAULT_VAL_FILE_PATTERN,
                        help='Prefix for evaluation tfrecords (default: '+DEFAULT_VAL_FILE_PATTERN+')', type=str)
    parser.add_argument('--val_json_file', metavar='<PATH>', default=DEFAULT_VAL_JSON_PATH,
                        help='COCO validation JSON containing golden bounding boxes. (default: '+DEFAULT_VAL_JSON_PATH+')', type=str)
    parser.add_argument('--eval_samples', default=5000, metavar='N',
                        help='number of samples for evaluation.', type=int)
    parser.add_argument('--num_examples_per_epoch', default=117266,
                        metavar='N', help='Number of examples in one epoch (default: 117266)', type=int)
    parser.add_argument('-s', '--steps', default=0,
                        help='Number of training steps (epochs and num_examples_per_epoch are ignored when set)', type=int)
    parser.add_argument('-v', '--log_step_count_steps', default=1, metavar='STEPS',
                        help='How often print global_step/sec and loss', type=int)
    parser.add_argument('-c', '--save_checkpoints_epochs', default=5.0,
                        metavar='EPOCHS', help='How often save checkpoints (default: 5)', type=float)
    parser.add_argument('--keep_ckpt_max', default=20, metavar='N',
                        help='Maximum number of checkpoints to keep. (default: 20)', type=int)
    parser.add_argument('--save_summary_steps', default=1,
                        help='How often save summary', type=int)
    parser.add_argument('--static', default=False, action='store_true',
                        help='Enables use of static dataloader')

    ARGS = parser.parse_args()

    if ARGS.mode == 'train' and ARGS.training_file_pattern is None:
        raise RuntimeError(
            'You must specify --training_file_pattern for training.')
    if ARGS.mode == 'eval':
        if ARGS.val_file_pattern is None:
            raise RuntimeError('You must specify --val_file_pattern '
                               'for evaluation.')
        if ARGS.val_json_file is None:
            raise RuntimeError(
                'You must specify --val_json_file for evaluation.')

    num_shards = 1
    shard_index = 0

    os.environ["TF_ENABLE_BF16_CONVERSION"] = "0"

    if ARGS.inference:
        ARGS.mode = 'inference'

    if not ARGS.no_hpu:
        if ARGS.use_horovod:
            from TensorFlow.common.horovod_helpers import hvd, hvd_init, horovod_enabled
            hvd_init()

        from TensorFlow.common.library_loader import load_habana_module
        log_info_devices = load_habana_module()
        print(f"Devices:\n {log_info_devices}")
        if ARGS.use_horovod:
            assert(horovod_enabled())
    elif ARGS.use_horovod:
        # Horovod on GPU
        import horovod.tensorflow as hvd
        hvd.init()
        assert(hvd.is_initialized())
        os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())

    tf.disable_eager_execution()
    tf.logging.set_verbosity(tf.logging.INFO)

    train_hooks = []
    global_batch_size = ARGS.batch_size
    distributed_optimizer = None
    model_dir = ARGS.model_dir

    if ARGS.use_horovod:
        global_batch_size = ARGS.batch_size * hvd.size()
        num_shards = hvd.size()
        shard_index = hvd.rank()
        train_hooks.append(hvd.BroadcastGlobalVariablesHook(0))
        distributed_optimizer = hvd.DistributedOptimizer

        if hvd.rank() > 0:
            model_dir = os.path.join(
                ARGS.model_dir, 'worker_' + str(hvd.rank()))

    run_config, params = construct_run_config(
        global_batch_size, model_dir, distributed_optimizer, num_shards, shard_index)

    tf.logging.info('steps_per_epoch: %s' % params['steps_per_epoch'])

    if ARGS.mode == 'train':
        write_hparams_v1(params['model_dir'], params)

        if ARGS.steps == 0:
            steps = int(ARGS.epochs * params['steps_per_epoch'])
        else:
            steps = ARGS.steps

        train_hooks.append(SSDTrainingHook(steps, params))
        train_hooks.append(ExamplesPerSecondEstimatorHook(
            params['batch_size'], params['save_summary_steps'],
            output_dir=params['model_dir']))

        tf.logging.info('Starting training cycle for %d steps.' % steps)

        train_estimator = tf.estimator.Estimator(
            model_fn=model.ssd_model_fn,
            model_dir=params['model_dir'],
            config=run_config,
            params=params)

        tf.logging.info(params)
        if ARGS.profiling:
            train_hooks.append(tf.estimator.ProfilerHook(
                save_steps=20, output_dir=ARGS.model_dir))

        if ARGS.static:
            input_reader = static_dataloader.SSDStaticInputReader(
                ARGS.training_file_pattern)
        else:
            input_reader = dataloader.SSDInputReader(
                ARGS.training_file_pattern+"*",
                is_training=True,
                use_fake_data=ARGS.use_fake_data)
        train_estimator.train(
            input_fn=input_reader,
            steps=steps,
            hooks=train_hooks)

    elif ARGS.mode == 'eval':
        eval_model_dir = os.path.join(params['model_dir'], 'eval')
        write_hparams_v1(eval_model_dir, params)

        coco_gt = coco_metric.create_coco(
            ARGS.val_json_file, use_cpp_extension=params['use_cocoeval_cc'])

        run_config = run_config.replace(model_dir=eval_model_dir)
        eval_estimator = tf.estimator.Estimator(
            model_fn=model.ssd_model_fn,
            model_dir=eval_model_dir,
            config=run_config,
            params=params)

        threads = []
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
                            ARGS.val_file_pattern+"*",
                            is_training=False,
                            use_fake_data=ARGS.use_fake_data)))

                t = threading.Thread(
                    target=coco_eval,
                    args=(predictions, current_step,
                          coco_gt, eval_model_dir,
                          ARGS.use_cocoeval_cc))
                threads.append(t)
                t.start()

            except tf.errors.NotFoundError:
                tf.logging.info(
                    'Checkpoint %s no longer exists, skipping checkpoint' % ckpt)

        for t in threads:
            t.join()
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
