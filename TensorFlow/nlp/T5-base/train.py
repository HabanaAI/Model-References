#!/usr/bin/env python3
#
# Copyright (c) 2020 Snapthat
# Source: https://github.com/snapthat/TF-T5-text-to-text/blob/master/snapthatT5/notebooks/TF-T5-%20Training.ipynb
#
###############################################################################
# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import argparse
import datasets
import datetime
import os
import tensorflow as tf
import transformers

from TensorFlow.common.debug import dump_callback
from TensorFlow.common.tb_utils import (
    TensorBoardWithHParamsV2, ExamplesPerSecondKerasHookV2)

import dataset
from model import T5


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='T5-base Finetuning (Q&A)')
parser.add_argument('-ds', '--data_dir', type=str, default='/data/huggingface',
                    help='path to directory that contains SQUAD dataset and pretrained T5-base model')
parser.add_argument('-d', '--dtype', metavar='DT',
                    help='data type: fp32 or bf16', type=str, choices=['fp32', 'bf16'], default='bf16')
parser.add_argument('-b', '--batch_size', type=int,
                    metavar='N', help='batch size', default=16)
parser.add_argument('--lr', type=float, metavar='LR',
                    help='learning rate', default=0.001)
parser.add_argument('--val_batch_size', type=int,
                    metavar='N', help='validation batch size', default=10)
parser.add_argument('--model_dir', type=str, default='/tmp/t5_base',
                    help='directory for storing model and logs')
parser.add_argument('--save_summary_steps', type=int, default=1,
                    help='steps between saving summaries to TensorBoard (disabled when 0)')
parser.add_argument('-e', '--epochs', type=int, default=1,
                    help='number of epochs')
parser.add_argument('-s', '--steps', type=int, default=None,
                    help='number of steps in epoch (None means whole dataset)')
parser.add_argument('-nc', '--no_checkpoints', default=False, action='store_true',
                    help='disable saving checkpoints')
parser.add_argument('--no_hpu', action='store_true',
                    help='do not load Habana modules = train on CPU/GPU')
parser.add_argument('--deterministic_dataset', default=False, action='store_true',
                    help='disables dataset shuffling')
parser.add_argument('--no_dropout', default=False, action='store_true',
                    help='disables dropout')
parser.add_argument('--no_eval', default=False, action='store_true',
                    help='disables evaluation')
parser.add_argument('--dump_config', type=str, default=None,
                    help='Side-by-side config file. Internal, do not use.')
params = parser.parse_args()

print(f"Using TF {tf.__version__}, datasets {datasets.__version__}, transformers {transformers.__version__}")

# Load Habana module in order to train on HPU (Gaudi)
if not params.no_hpu:
    from habana_frameworks.tensorflow import load_habana_module
    load_habana_module()

# Load dataset
assert os.path.exists(params.data_dir), (
    f'"{params.data_dir}" does not exist! Use "prepare_data.py" to create required data.')

train_ds = datasets.load_from_disk(
    os.path.join(params.data_dir, 'squad', 'train'))
valid_ds = datasets.load_from_disk(
    os.path.join(params.data_dir, 'squad', 'valid'))

print("Example data from the mapped dataset: \n", next(iter(train_ds)))

tf_train_ds = dataset.to_tf_dataset(train_ds)
tf_valid_ds = dataset.to_tf_dataset(valid_ds)

tf_train_ds = dataset.create_dataset(tf_train_ds, batch_size=params.batch_size,
                                     shuffling=not params.deterministic_dataset,
                                     cache_path=None)
tf_valid_ds = dataset.create_dataset(tf_valid_ds, batch_size=params.val_batch_size,
                                     shuffling=False, cache_path=None,
                                     drop_remainder=(valid_ds.num_rows % params.val_batch_size == 0))

# Configure callbacks
callbacks = []
if params.save_summary_steps > 0:
    log_dir = os.path.join(params.model_dir, 'logs')
    callbacks.append(TensorBoardWithHParamsV2(
        hparams={**vars(params), 'precision': params.dtype},
        log_dir=log_dir, histogram_freq=0, profile_batch=0,
        update_freq=params.save_summary_steps))
    callbacks.append(ExamplesPerSecondKerasHookV2(
        every_n_steps=params.save_summary_steps, output_dir=log_dir,
        batch_size=params.batch_size))

# Prepare model
if params.dtype == 'bf16':
    tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
optimizer = tf.keras.optimizers.Adam(params.lr)
model_kwargs = {'dropout_rate': 0.0} if params.no_dropout else {}
model = T5.from_pretrained(os.path.join(
    params.data_dir, 't5_base'), **model_kwargs)
model.compile(optimizer=optimizer)

# Run training
steps = params.steps or (len(train_ds) // params.batch_size)
valid_steps = 0 if params.no_eval else (len(valid_ds) // params.val_batch_size)

with dump_callback(params.dump_config):
    model.fit(tf_train_ds.repeat(), epochs=params.epochs, steps_per_epoch=steps,
              callbacks=callbacks, validation_data=tf_valid_ds.repeat(),
              validation_steps=valid_steps)

if not params.no_checkpoints:
    model.save_pretrained(os.path.join(
        params.model_dir, 'checkpoints'))
