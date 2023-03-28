# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
###############################################################################
# Changes:
# - added support for handling cpu/gpu/hpu devices
# - added support for handling both float16 and bfloat16 mixed precision
# - wrapped horovod import in a try-catch block so that the user is not required to install this library
#   when the model is being run on a single card
# - added cast to output of the model for numerical stability according to www.tensorflow.org/guide/mixed_precision
# - added tensorboard logging for hyper parameters and benchmark data
# - added support for changeable frequency of logging during training
# - removed usage of deprecated numpy.asscalar function

import logging
import os
import time

import dllogger
import numpy as np
import tensorflow as tf
from data.outbrain.features import DISPLAY_ID_COLUMN
from tensorflow.keras import backend as K
from trainer.utils.schedulers import get_schedule

from TensorFlow.common.tb_utils import write_hparams_v2

from habana_frameworks.tensorflow import backward_compatible_optimizers


try:
    import horovod.tensorflow as hvd
except ImportError:
    hvd = None


def train(args, model, config):
    logger = logging.getLogger('tensorflow')

    train_dataset = config['train_dataset']
    eval_dataset = config['eval_dataset']
    steps = int(config['steps_per_epoch'])
    schedule = get_schedule(
        args=args,
        steps_per_epoch=steps
    )
    writer = tf.summary.create_file_writer(os.path.join(args.model_dir, 'event_files'))
    hparams = {**vars(args), 'precision': args.dtype, 'batch_size': args.global_batch_size}
    write_hparams_v2(writer, hparams)

    deep_optimizer = backward_compatible_optimizers.RMSprop(
        learning_rate=args.deep_learning_rate,
        rho=0.5
    )

    wide_optimizer = backward_compatible_optimizers.Ftrl(
        learning_rate=args.linear_learning_rate
    )

    compiled_loss = tf.keras.losses.BinaryCrossentropy()
    eval_loss = tf.keras.metrics.Mean()

    metrics = [
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.AUC()
    ]

    current_step_var = tf.Variable(0, trainable=False, dtype=tf.int64)
    display_id_counter = tf.Variable(0., trainable=False, dtype=tf.float64)
    streaming_map = tf.Variable(0., name='STREAMING_MAP', trainable=False, dtype=tf.float64)

    checkpoint = tf.train.Checkpoint(
        deep_optimizer=deep_optimizer,
        wide_optimizer=wide_optimizer,
        model=model,
        current_step=current_step_var
    )
    manager = tf.train.CheckpointManager(
        checkpoint=checkpoint,
        directory=os.path.join(args.model_dir, 'checkpoint'),
        max_to_keep=1
    )

    if args.use_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            logger.warning(f'Model restored from checkpoint {args.model_dir}')
            if args.benchmark:
                current_step_var.assign(0)
        else:
            logger.warning(f'Failed to restore model from checkpoint {args.model_dir}')

    if args.dtype == "fp16":
        deep_optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
            deep_optimizer,
            loss_scale='dynamic'
        )
        wide_optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
            wide_optimizer,
            loss_scale='dynamic'
        )

    @tf.function
    def train_step(x, y, first_batch):
        with tf.GradientTape(persistent=True) as tape:
            y_pred = model(x, training=True)
            y_pred = tf.cast(y_pred, tf.float32)
            loss = compiled_loss(y, y_pred)
            linear_loss = wide_optimizer.get_scaled_loss(loss) if args.dtype == "fp16" else loss
            deep_loss = deep_optimizer.get_scaled_loss(loss) if args.dtype == "fp16" else loss

        if args.use_horovod:
            tape = hvd.DistributedGradientTape(tape)

        for metric in metrics:
            metric.update_state(y, y_pred)

        linear_vars = model.linear_model.trainable_variables
        dnn_vars = model.dnn_model.trainable_variables
        linear_grads = tape.gradient(linear_loss, linear_vars)
        dnn_grads = tape.gradient(deep_loss, dnn_vars)
        if args.dtype == "fp16":
            linear_grads = wide_optimizer.get_unscaled_gradients(linear_grads)
            dnn_grads = deep_optimizer.get_unscaled_gradients(dnn_grads)

        wide_optimizer.apply_gradients(zip(linear_grads, linear_vars))
        deep_optimizer.apply_gradients(zip(dnn_grads, dnn_vars))
        if first_batch and args.use_horovod:
            hvd.broadcast_variables(model.linear_model.variables, root_rank=0)
            hvd.broadcast_variables(model.dnn_model.variables, root_rank=0)
            hvd.broadcast_variables(wide_optimizer.variables(), root_rank=0)
            hvd.broadcast_variables(deep_optimizer.variables(), root_rank=0)
        return loss

    @tf.function
    def evaluation_step(x, y):
        predictions = model(x, training=False)
        predictions = tf.cast(predictions, tf.float32)
        loss = compiled_loss(y, predictions)

        for metric in metrics:
            metric.update_state(y, predictions)

        predictions = tf.reshape(predictions, [-1])
        predictions = tf.cast(predictions, tf.float64)
        display_ids = x[DISPLAY_ID_COLUMN]
        display_ids = tf.reshape(display_ids, [-1])
        labels = tf.reshape(y, [-1])
        sorted_ids = tf.argsort(display_ids)
        display_ids = tf.gather(display_ids, indices=sorted_ids)
        predictions = tf.gather(predictions, indices=sorted_ids)
        labels = tf.gather(labels, indices=sorted_ids)
        _, display_ids_idx, display_ids_ads_count = tf.unique_with_counts(display_ids, out_idx=tf.int64)
        pad_length = 30 - tf.reduce_max(display_ids_ads_count)
        preds = tf.RaggedTensor.from_value_rowids(predictions, display_ids_idx).to_tensor()
        labels = tf.RaggedTensor.from_value_rowids(labels, display_ids_idx).to_tensor()

        labels_mask = tf.math.reduce_max(labels, 1)
        preds_masked = tf.boolean_mask(preds, labels_mask)
        labels_masked = tf.boolean_mask(labels, labels_mask)
        labels_masked = tf.argmax(labels_masked, axis=1, output_type=tf.int32)
        labels_masked = tf.reshape(labels_masked, [-1, 1])

        preds_masked = tf.pad(preds_masked, [(0, 0), (0, pad_length)])
        _, predictions_idx = tf.math.top_k(preds_masked, 12)
        indices = tf.math.equal(predictions_idx, labels_masked)
        indices_mask = tf.math.reduce_any(indices, 1)
        masked_indices = tf.boolean_mask(indices, indices_mask)

        res = tf.argmax(masked_indices, axis=1)
        ap_matrix = tf.divide(1, tf.add(res, 1))
        ap_sum = tf.reduce_sum(ap_matrix)
        shape = tf.cast(tf.shape(indices)[0], tf.float64)
        display_id_counter.assign_add(shape)
        streaming_map.assign_add(ap_sum)
        return loss

    t0 = None
    t_batch = time.time()

    with writer.as_default():
        for epoch in range(1, args.num_epochs + 1):
            for step, (x, y) in enumerate(train_dataset):
                current_step = current_step_var.numpy().item()
                schedule(optimizer=deep_optimizer, current_step=current_step)

                for metric in metrics:
                    metric.reset_states()
                loss = train_step(x, y, epoch == 1 and step == 0)

                enable_logging_for_current_step = (current_step % args.log_every == 0)
                if not args.use_horovod or hvd.rank() == 0:
                    if enable_logging_for_current_step:
                        for metric in metrics:
                            tf.summary.scalar(f'{metric.name}', metric.result(), step=current_step)
                        tf.summary.scalar('loss', loss, step=current_step)
                        tf.summary.scalar('schedule', K.get_value(deep_optimizer.lr), step=current_step)
                        writer.flush()

                if args.benchmark:
                    if current_step > 0 and enable_logging_for_current_step:
                        batch_time = (time.time() - t_batch) / args.log_every
                        samplesps = args.global_batch_size / batch_time
                        dllogger.log(data={'batch_samplesps': samplesps}, step=(1, current_step))
                        tf.summary.scalar("examples/sec", samplesps, step=current_step)
                        tf.summary.scalar("global_step/sec", 1. / batch_time, step=current_step)
                        writer.flush()
                        t_batch = time.time()

                    boundary = max(args.benchmark_warmup_steps, 1)
                    if current_step == boundary:
                        t0 = time.time()
                    if current_step > boundary and args.benchmark_steps <= current_step:
                        train_time = time.time() - t0
                        epochs = args.benchmark_steps - max(args.benchmark_warmup_steps, 1)
                        train_throughput = (args.global_batch_size * epochs) / train_time
                        dllogger.log(
                            data={'train_throughput': train_throughput},
                            step=tuple()
                        )
                        return

                else:
                    if enable_logging_for_current_step:
                        train_data = {metric.name: f'{metric.result().numpy():.4f}' for metric in metrics}
                        train_data['loss'] = f'{loss.numpy():.4f}'
                        dllogger.log(data=train_data, step=(current_step, args.num_epochs * steps))

                    if step == steps:
                        break

                current_step_var.assign_add(1)
            if args.benchmark:
                continue

            for metric in metrics:
                metric.reset_states()
            eval_loss.reset_states()

            for step, (x, y) in enumerate(eval_dataset):
                loss = evaluation_step(x, y)
                eval_loss.update_state(loss)

            map_metric = tf.divide(streaming_map, display_id_counter) if not args.use_horovod else \
                hvd.allreduce(tf.divide(streaming_map, display_id_counter))

            map_metric = map_metric.numpy()
            eval_loss_reduced = eval_loss.result() if not args.use_horovod else \
                hvd.allreduce(eval_loss.result())

            metrics_reduced = {
                f'{metric.name}_val': metric.result() if not args.use_horovod else
                hvd.allreduce(metric.result()) for metric in metrics
            }

            for name, result in metrics_reduced.items():
                tf.summary.scalar(f'{name}', result, step=steps * epoch)
            tf.summary.scalar('loss_val', eval_loss_reduced, step=steps * epoch)
            tf.summary.scalar('map_val', map_metric, step=steps * epoch)
            writer.flush()

            eval_data = {name: f'{result.numpy():.4f}' for name, result in metrics_reduced.items()}
            eval_data.update({
                'loss_val': f'{eval_loss_reduced.numpy():.4f}',
                'streaming_map_val': f'{map_metric:.4f}'
            })
            dllogger.log(data=eval_data, step=(steps * epoch, args.num_epochs * steps))

            if not args.use_horovod or hvd.rank() == 0:
                manager.save()

            display_id_counter.assign(0)
            streaming_map.assign(0)
        if not args.use_horovod or hvd.rank() == 0:
            dllogger.log(data=eval_data, step=tuple())


def evaluate(args, model, config):
    logger = logging.getLogger('tensorflow')

    deep_optimizer = backward_compatible_optimizers.RMSprop(
        learning_rate=args.deep_learning_rate,
        rho=0.5
    )

    wide_optimizer = backward_compatible_optimizers.Ftrl(
        learning_rate=args.linear_learning_rate
    )

    compiled_loss = tf.keras.losses.BinaryCrossentropy()
    eval_loss = tf.keras.metrics.Mean()

    metrics = [
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.AUC()
    ]

    if args.dtype == "fp16":
        deep_optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
            deep_optimizer,
            loss_scale='dynamic'
        )
        wide_optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
            wide_optimizer,
            loss_scale='dynamic'
        )

    current_step = 0
    current_step_var = tf.Variable(0, trainable=False, dtype=tf.int64)
    display_id_counter = tf.Variable(0., trainable=False, dtype=tf.float64)
    streaming_map = tf.Variable(0., name='STREAMING_MAP', trainable=False, dtype=tf.float64)

    checkpoint = tf.train.Checkpoint(
        deep_optimizer=deep_optimizer,
        wide_optimizer=wide_optimizer,
        model=model,
        current_step=current_step_var
    )
    manager = tf.train.CheckpointManager(
        checkpoint=checkpoint,
        directory=os.path.join(args.model_dir, 'checkpoint'),
        max_to_keep=1
    )

    if args.use_checkpoint:
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
        if manager.latest_checkpoint:
            logger.warning(f'Model restored from checkpoint {args.model_dir}')
        else:
            logger.warning(f'Failed to restore model from checkpoint {args.model_dir}')

    @tf.function
    def evaluation_step(x, y):
        predictions = model(x, training=False)
        predictions = tf.cast(predictions, tf.float32)
        loss = compiled_loss(y, predictions)

        for metric in metrics:
            metric.update_state(y, predictions)

        predictions = tf.reshape(predictions, [-1])
        predictions = tf.cast(predictions, tf.float64)
        display_ids = x[DISPLAY_ID_COLUMN]
        display_ids = tf.reshape(display_ids, [-1])
        labels = tf.reshape(y, [-1])
        sorted_ids = tf.argsort(display_ids)
        display_ids = tf.gather(display_ids, indices=sorted_ids)
        predictions = tf.gather(predictions, indices=sorted_ids)
        labels = tf.gather(labels, indices=sorted_ids)
        _, display_ids_idx, display_ids_ads_count = tf.unique_with_counts(display_ids, out_idx=tf.int64)
        pad_length = 30 - tf.reduce_max(display_ids_ads_count)
        preds = tf.RaggedTensor.from_value_rowids(predictions, display_ids_idx).to_tensor()
        labels = tf.RaggedTensor.from_value_rowids(labels, display_ids_idx).to_tensor()

        labels_mask = tf.math.reduce_max(labels, 1)
        preds_masked = tf.boolean_mask(preds, labels_mask)
        labels_masked = tf.boolean_mask(labels, labels_mask)
        labels_masked = tf.argmax(labels_masked, axis=1, output_type=tf.int32)
        labels_masked = tf.reshape(labels_masked, [-1, 1])

        preds_masked = tf.pad(preds_masked, [(0, 0), (0, pad_length)])
        _, predictions_idx = tf.math.top_k(preds_masked, 12)
        indices = tf.math.equal(predictions_idx, labels_masked)
        indices_mask = tf.math.reduce_any(indices, 1)
        masked_indices = tf.boolean_mask(indices, indices_mask)

        res = tf.argmax(masked_indices, axis=1)
        ap_matrix = tf.divide(1, tf.add(res, 1))
        ap_sum = tf.reduce_sum(ap_matrix)
        shape = tf.cast(tf.shape(indices)[0], tf.float64)
        display_id_counter.assign_add(shape)
        streaming_map.assign_add(ap_sum)
        return loss

    eval_dataset = config['eval_dataset']

    t0 = None
    t_batch = None

    for step, (x, y) in enumerate(eval_dataset):
        loss = evaluation_step(x, y)
        eval_loss.update_state(loss)
        if args.benchmark:
            boundary = max(args.benchmark_warmup_steps, 1)
            if current_step == boundary:
                t0 = time.time()
            if current_step > boundary:
                batch_time = time.time() - t_batch
                samplesps = args.eval_batch_size / batch_time
                if not args.use_horovod or hvd.rank() == 0:
                    dllogger.log(data={'batch_samplesps': samplesps}, step=(1, current_step))

                if args.benchmark_steps <= current_step:
                    valid_time = time.time() - t0
                    epochs = args.benchmark_steps - max(args.benchmark_warmup_steps, 1)
                    valid_throughput = (args.eval_batch_size * epochs) / valid_time
                    if not args.use_horovod or hvd.rank() == 0:
                        dllogger.log(
                            data={'validation_throughput': valid_throughput},
                            step=tuple()
                        )
                    return

        else:
            if step % args.log_every == 0:
                valid_data = {metric.name: f'{metric.result().numpy():.4f}' for metric in metrics}
                valid_data['loss'] = f'{loss.numpy():.4f}'
                if not args.use_horovod or hvd.rank() == 0:
                    dllogger.log(data=valid_data, step=(step,))
        current_step += 1
        t_batch = time.time()

    map_metric = tf.divide(streaming_map, display_id_counter) if not args.use_horovod else \
        hvd.allreduce(tf.divide(streaming_map, display_id_counter))
    eval_loss_reduced = eval_loss.result() if not args.use_horovod else \
        hvd.allreduce(eval_loss.result())

    metrics_reduced = {
        f'{metric.name}_val': metric.result() if not args.use_horovod else
        hvd.allreduce(metric.result()) for metric in metrics
    }

    eval_data = {name: f'{result.numpy():.4f}' for name, result in metrics_reduced.items()}
    eval_data.update({
        'loss_val': f'{eval_loss_reduced.numpy():.4f}',
        'streaming_map_val': f'{map_metric.numpy():.4f}'
    })

    dllogger.log(data=eval_data, step=(step,))
