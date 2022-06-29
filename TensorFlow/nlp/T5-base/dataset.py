# Copyright (c) 2020 Snapthat
# Source: https://github.com/snapthat/TF-T5-text-to-text/blob/master/snapthatT5/notebooks/TF-T5-%20Training.ipynb
#
###############################################################################
# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import tensorflow as tf
import model


def to_tf_dataset(dataset):
    columns = ['input_ids', 'attention_mask',
               'labels', 'decoder_attention_mask']
    dataset.set_format(type='tensorflow', columns=columns)
    return_types = {
        'input_ids': tf.int32,
        'attention_mask': tf.int32,
        'labels': tf.int32,
        'decoder_attention_mask': tf.int32
    }
    return_shapes = {
        'input_ids': tf.TensorShape([model.encoder_max_len]),
        'attention_mask': tf.TensorShape([model.encoder_max_len]),
        'labels': tf.TensorShape([model.decoder_max_len]),
        'decoder_attention_mask': tf.TensorShape([model.decoder_max_len])
    }
    ds = tf.data.Dataset.from_generator(
        lambda: dataset, return_types, return_shapes)
    return ds


def create_dataset(dataset, cache_path=None, batch_size=4,
                   buffer_size=1000, shuffling=True, drop_remainder=True):
    if cache_path is not None:
        dataset = dataset.cache(cache_path)
    if shuffling:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
