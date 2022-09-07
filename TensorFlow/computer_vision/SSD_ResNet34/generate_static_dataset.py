###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
import argparse
import tensorflow as tf
from TensorFlow.computer_vision.SSD_ResNet34.dataloader import SSDInputReader
from TensorFlow.computer_vision.SSD_ResNet34.constants import *


def serialize(sample):
    image = tf.io.serialize_tensor(sample[0])
    num_matched_boxes = tf.io.serialize_tensor(sample[1][NUM_MATCHED_BOXES])
    boxes = tf.io.serialize_tensor(sample[1][BOXES])
    classes = tf.io.serialize_tensor(sample[1][CLASSES])
    return tf.io.serialize_tensor(tf.concat(tf.stack([image, num_matched_boxes, boxes, classes]), 0))


def convert(input_pattern, output_dir, batch_size, steps, workers, worker_idx):
    reader = SSDInputReader(input_pattern + "*",
                            is_training=True, use_fake_data=False)
    params = dict(batch_size=batch_size, num_shards=workers,
                  shard_index=worker_idx, dtype=tf.float32, visualize_dataloader=False)
    dataset = reader(params)
    dataset = dataset.repeat()

    idx = 1
    for sample in dataset:
        if idx > steps:
            break

        worker_dir = os.path.join(output_dir, f'worker_{worker_idx}')
        os.makedirs(worker_dir, exist_ok=True)
        filename = f'train-{batch_size}-{idx:06d}.tfrecord'
        path = os.path.join(worker_dir, filename)

        sample = serialize(sample)
        with tf.io.TFRecordWriter(path) as file_writer:
            print(f"[{idx}/{steps}] Writing to {path}")
            file_writer.write(sample.numpy())

        idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_pattern', '-i',
                        help='Input file pattern', required=True)
    parser.add_argument('--output_dir', '-o',
                        help='Output directory', required=True)
    parser.add_argument('--batch_size', '-b', type=int,
                        help=f'Batch size. Each tfrecord file will contain exactly one batch', required=True)
    parser.add_argument('--steps', '-s', type=int,
                        help='Number of training steps', required=True)
    parser.add_argument('--workers', '-w', type=int,
                        help='Number of workers', default=1)
    args = parser.parse_args()

    for worker_idx in range(args.workers):
        convert(args.input_pattern, args.output_dir, args.batch_size,
                args.steps, args.workers, worker_idx)
