# Copyright (c) 2020 Snapthat
# Source: https://github.com/snapthat/TF-T5-text-to-text/blob/master/snapthatT5/notebooks/TF-T5-%20Training.ipynb
#
###############################################################################
# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
###############################################################################
import tensorflow as tf
import model
import numpy as np


def create_dataset(dataset, batch_size):
    num_examples = dataset.num_rows
    values = {}
    for i, x in enumerate(dataset):
        for key in x:
            if key not in ["input_ids", "attention_mask", "labels", "decoder_attention_mask"]:
                continue
            if i == 0:
                values[key] = np.zeros((num_examples, len(x[key])), dtype=int)
            values[key][i, :] = x[key]
    tensors = {}
    for key in values:
        tensors[key] = tf.constant(values[key], tf.int32, name=key)
    dataset = tf.data.Dataset.from_tensor_slices((tensors))
    dataset = dataset.batch(batch_size, drop_remainder=True).cache().repeat()

    if len(tf.config.list_logical_devices('HPU')) > 0:
        device = "/device:HPU:0"
        with tf.device(device):
            dataset = dataset.apply(tf.data.experimental.prefetch_to_device(device))
    else:
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
