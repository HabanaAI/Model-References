# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################
# Changes:
# - logic from main.py was copied to main() function
# - included HPU horovod helpers
# - removed GPU specific mixed_precision handling
# - added synthetic data option for deterministic training
# - added tensorboard logging and performance measurements logs
# - in a training mode return loss from train_step as a numpy object to transfer the data to host

import os
from time import time
from collections import namedtuple

import numpy as np
from PIL import Image
import tensorflow as tf

from model.unet import Unet
from runtime.arguments import parse_args
from runtime.losses import partial_losses
from runtime.parse_results import process_performance_stats
from runtime.setup import get_logger, set_flags, prepare_model_dir
from data_loading.data_loader import Dataset
from TensorFlow.common.debug import dump_callback
from TensorFlow.common.horovod_helpers import hvd, hvd_init, horovod_enabled, hvd_size, hvd_rank
from TensorFlow.common.tb_utils import write_hparams_v2


def train(params, model, dataset, logger, tb_logger=None):
    np.random.seed(params.seed)
    tf.random.set_seed(params.seed)

    num_workers = hvd_size() if horovod_enabled() else 1
    worker_id = hvd_rank() if horovod_enabled() else 0
    max_steps = params.max_steps // num_workers

    optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)

    ce_loss = tf.keras.metrics.Mean(name='ce_loss')
    f1_loss = tf.keras.metrics.Mean(name='dice_loss')
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    if params.resume_training and params.model_dir:
        checkpoint.restore(tf.train.latest_checkpoint(params.model_dir))

    if tb_logger is not None:
        write_hparams_v2(tb_logger.train_writer, vars(params))

    @tf.function
    def train_step(features, labels, warmup_batch=False):
        with tf.GradientTape() as tape:
            output_map = model(features)
            crossentropy_loss, dice_loss = partial_losses(output_map, labels)
            added_losses = tf.add(crossentropy_loss, dice_loss, name="total_loss_ref")
            loss = added_losses + params.weight_decay * tf.add_n(
                [tf.nn.l2_loss(v) for v in model.trainable_variables
                 if 'batch_normalization' not in v.name])

        if horovod_enabled():
            tape = hvd.DistributedGradientTape(tape)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Note: broadcast should be done after the first gradient step to ensure optimizer
        # initialization.
        if horovod_enabled() and warmup_batch:
            hvd.broadcast_variables(model.variables, root_rank=0)
            hvd.broadcast_variables(optimizer.variables(), root_rank=0)

        ce_loss(crossentropy_loss)
        f1_loss(dice_loss)
        return loss

    if params.benchmark:
        assert max_steps * num_workers > params.warmup_steps, \
        "max_steps value has to be greater than warmup_steps"
        timestamps = []
        for iteration, (images, labels) in enumerate(dataset.train_fn(drop_remainder=True)):
            loss = train_step(images, labels, warmup_batch=iteration == 0).numpy()
            if iteration > params.warmup_steps:
                timestamps.append(time())

            if iteration >= max_steps * num_workers:
                break

        if worker_id == 0:
            deltas = np.array([timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)])
            stats = process_performance_stats(deltas, num_workers * params.batch_size, mode="train")
            logger.log(step=(), data=stats)
    else:
        timestamp = time()
        dataset_fn = dataset.synth_fn if params.synth_data else dataset.train_fn
        for iteration, (images, labels) in enumerate(dataset_fn()):
            # assign returned loss as a numpy object to transfer the data to host
            loss = train_step(images, labels, warmup_batch=iteration == 0).numpy()
            if worker_id == 0 or params.log_all_workers:
                if iteration % params.log_every == 0:
                    duration = float(time() - timestamp) / params.log_every
                    timestamp = time()
                    data = {
                        "train_ce_loss": float(ce_loss.result()),
                        "train_dice_loss": float(f1_loss.result()),
                        "train_total_loss": float(f1_loss.result() + ce_loss.result()),
                        "iter duration [ms]": 1000 * duration,
                        "IPS": params.batch_size / duration
                    }
                    logger.log(step=(iteration, max_steps), data=data)

                    if tb_logger is not None:
                        with tb_logger.train_writer.as_default():
                            for name, value in data.items():
                                tf.summary.scalar(name, value, step=iteration)
                            # for consistency
                            tf.summary.scalar("loss", data["train_total_loss"], step=iteration)
                            tf.summary.scalar("examples/sec", data["IPS"], step=iteration)
                            tf.summary.scalar("global_step/sec", 1. / duration, step=iteration)

                if (params.evaluate_every > 0) and (iteration % params.evaluate_every == 0):
                    evaluate(params, model, dataset, logger, tb_logger,
                             restore_checkpoint=False)

                f1_loss.reset_states()
                ce_loss.reset_states()

            if iteration >= max_steps:
                break

        if not params.disable_ckpt_saving and worker_id == 0:
            checkpoint.save(file_prefix=os.path.join(params.model_dir, "checkpoint"))

    logger.flush()


def evaluate(params, model, dataset, logger, tb_logger=None, restore_checkpoint=True):
    if params.fold is None:
        print("No fold specified for evaluation. Please use --fold [int] to select a fold.")
    ce_loss = tf.keras.metrics.Mean(name='ce_loss')
    f1_loss = tf.keras.metrics.Mean(name='dice_loss')
    checkpoint = tf.train.Checkpoint(model=model)
    if params.model_dir and restore_checkpoint:
        checkpoint.restore(tf.train.latest_checkpoint(params.model_dir)).expect_partial()

    def validation_step(features, labels):
        output_map = model(features, training=False)
        crossentropy_loss, dice_loss = partial_losses(output_map, labels)
        ce_loss(crossentropy_loss)
        f1_loss(dice_loss)

    for iteration, (images, labels) in enumerate(dataset.eval_fn(count=1)):
        validation_step(images, labels)
        if iteration >= dataset.eval_size // params.batch_size:
            break

    data = {}
    if dataset.eval_size > 0:
        data = {
            "eval_ce_loss": float(ce_loss.result()),
            "eval_dice_loss": float(f1_loss.result()),
            "eval_total_loss": float(f1_loss.result() + ce_loss.result()),
            "eval_dice_score": 1.0 - float(f1_loss.result()),
            "loss": float(f1_loss.result() + ce_loss.result()),
            "accuracy": 1.0 - float(f1_loss.result()),  # for consistency
        }
        logger.log(step=(), data=data)
        if tb_logger is not None:
            with tb_logger.eval_writer.as_default():
                for name, value in data.items():
                    tf.summary.scalar(name, value, step=iteration)

    logger.flush()


def predict(params, model, dataset, logger):
    checkpoint = tf.train.Checkpoint(model=model)
    if params.model_dir:
        checkpoint.restore(tf.train.latest_checkpoint(params.model_dir)).expect_partial()

    @tf.function
    def prediction_step(features):
        return tf.nn.softmax(model(features, training=False), axis=-1)

    if params.benchmark:
        assert params.max_steps > params.warmup_steps, \
            "max_steps value has to be greater than warmup_steps"
        timestamps = []
        for iteration, images in enumerate(dataset.test_fn(count=None, drop_remainder=True)):
            prediction_step(images)
            if iteration > params.warmup_steps:
                timestamps.append(time())
            if iteration >= params.max_steps:
                break

        deltas = np.array([timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)])
        stats = process_performance_stats(deltas, params.batch_size, mode="test")
        logger.log(step=(), data=stats)
    else:
        predictions = np.concatenate([prediction_step(images).numpy()
                                      for images in dataset.test_fn(count=1)], axis=0)
        binary_masks = [np.argmax(p, axis=-1).astype(np.uint8) * 255 for p in predictions]
        multipage_tif = [Image.fromarray(mask).resize(size=(512, 512), resample=Image.BILINEAR)
                         for mask in binary_masks]

        output_dir = os.path.join(params.model_dir, 'predictions')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        multipage_tif[0].save(os.path.join(output_dir, 'test-masks.tif'),
                              compression="tiff_deflate",
                              save_all=True,
                              append_images=multipage_tif[1:])

        print("Predictions saved at {}".format(output_dir))
    logger.flush()

def main():
    """
    Starting point of the application
    """
    params = parse_args(description="UNet-medical")
    if params.use_horovod:
        hvd_init()
    set_flags(params)

    model_dir = prepare_model_dir(params)
    params.model_dir = model_dir
    logger = get_logger(params)

    tb_logger = None
    if params.tensorboard_logging:
        log_dir = params.log_dir
        if horovod_enabled() and params.log_all_workers:
            log_dir = os.path.join(log_dir, f'worker_{hvd_rank()}')
        tb_logger = namedtuple('TBSummaryWriters', 'train_writer eval_writer')(
            tf.summary.create_file_writer(log_dir),
            tf.summary.create_file_writer(os.path.join(log_dir, 'eval')))

    model = Unet()

    dataset = Dataset(data_dir=params.data_dir,
                      batch_size=params.batch_size,
                      fold=params.fold,
                      augment=params.augment,
                      hpu_id=hvd_rank() if horovod_enabled() else 0,
                      num_hpus=hvd_size() if horovod_enabled() else 1,
                      seed=params.seed)

    if 'train' in params.exec_mode:
        with dump_callback(params.dump_config):
            train(params, model, dataset, logger, tb_logger)

    if 'evaluate' in params.exec_mode:
        evaluate(params, model, dataset, logger, tb_logger)

    if 'predict' in params.exec_mode:
        predict(params, model, dataset, logger)


if __name__ == '__main__':
    main()
