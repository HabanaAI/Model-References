# Copyright (C) 2021-2022 Habana Labs, Ltd. an Intel Company

""" Distributed MNIST Training using Keras

    Demonstration of multi-worker training using HPUStrategy.
"""

import argparse
import json
import os

import tensorflow as tf
import tensorflow_datasets as tfds
from mpi4py import MPI

BASE_TF_SERVER_PORT = 7850
DEFAULT_PER_WORKER_BATCHS_SIZE = 64
DEFAULT_DTYPE = "bf16"
DEFAULT_NUM_EPOCHS = 6
SHUFFLE_BUFFER_SIZE = 10000


num_workers = MPI.COMM_WORLD.Get_size()
worker_index = MPI.COMM_WORLD.Get_rank()
if not num_workers > 1:
    print("warning: Please run this script using mpirun with at least 2 workers.")


def set_tf_config():
    """ Makes a TensorFlow cluster information and sets it to TF_CONFIG environment variable.
    """
    tf_config = {
        "cluster": {
            "worker": [f"localhost:{BASE_TF_SERVER_PORT + index}" for index in range(num_workers)]
        },
        "task": {"type": "worker", "index": worker_index}
    }
    tf_config_text = json.dumps(tf_config)
    os.environ["TF_CONFIG"] = tf_config_text
    print(f"TF_CONFIG = {tf_config_text}")
    return tf_config_text


def parse_args():
    """ Parses the command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Multi-Worker Distributed MNIST Training Demo.")
    parser.add_argument("-d", "--device", type=str, default="hpu",
                        help="Device to use: either 'hpu' or 'cpu'")
    parser.add_argument("-b", "-bs", "--batch_size", type=int, default=DEFAULT_PER_WORKER_BATCHS_SIZE,
                        help="Batch size per replica")
    parser.add_argument("-t", "--dtype", type=str, default=None,
                        help="Datatype to use: either 'bf' or 'fp'")
    parser.add_argument("-e", "--epochs", type=int, default=DEFAULT_NUM_EPOCHS,
                        help="Number of training epochs")
    args = parser.parse_args()

    args.device = args.device.lower()
    assert args.device in [
        "hpu", "cpu"], "'device' must be either 'hpu' or 'cpu'."
    args.use_hpu = (args.device == "hpu")

    if args.dtype is None:
        args.dtype = "bf" if args.use_hpu else "fp"
    args.dtype = args.dtype.lower()
    assert args.dtype in ["bf", "bf16", "bfloat", "bfloat16", "fp",
                          "fp32", "float", "float32"], "'dtype' must be either 'bf' or 'fp'."
    args.use_bfloat = args.dtype in ["bf", "bf16", "bfloat", "bfloat16"]

    if (not args.use_hpu) and args.use_bfloat:
        assert not "Unable to use bfloat16 on CPU."

    return args


def train_mnist(strategy: tf.distribute.Strategy, batch_size: int, num_epochs: int):
    """ Train the distributed model on MNIST Dataset.
    """
    # Determine the total training batch size.
    batch_size_per_replica = batch_size
    total_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
    print(
        f"total_batch_size = {batch_size_per_replica} * {strategy.num_replicas_in_sync} workers = {total_batch_size}")

    # Load and preprocess the MNIST Dataset.
    # As tfds.load() may download the dataset if not cached, let the first worker do it first.
    for dataload_turn in range(2):
        if (dataload_turn == 0) == (worker_index == 0):
            print("Loading MNIST dataset...")
            datasets, info = tfds.load(
                name="mnist", with_info=True, as_supervised=True)
        MPI.COMM_WORLD.barrier()

    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.cast(label, tf.int32)
        return image, label

    train_dataset = datasets["train"]
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_dataset = train_dataset.with_options(options)
    train_dataset = train_dataset.map(
        preprocess).cache().shuffle(SHUFFLE_BUFFER_SIZE).batch(total_batch_size)

    test_dataset = datasets["test"]
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    test_dataset = test_dataset.with_options(options)
    test_dataset = test_dataset.map(
        preprocess).batch(total_batch_size)

    # Create and compile the distributed CNN model.
    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                32, 3, activation="relu", input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10)
        ])

        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=["accuracy"])

    # Train the model.
    print("Calling model.fit()...")
    model.fit(train_dataset, epochs=num_epochs, verbose=2)
    print("Calling model.evaluate()...")
    eval_results = model.evaluate(test_dataset, verbose=2)
    print(f"Evaluation results: {eval_results}")


if __name__ == "__main__":
    args = parse_args()

    # Set TF_CONFIG.
    set_tf_config()

    # Instantiate the distributed strategy class.
    if args.use_hpu:
        # Optionally enable automatic bfloat16 operations conversion.
        if args.use_bfloat:
            os.environ["TF_BF16_CONVERSION"] = "full"
            print(
                f"TF_BF16_CONVERSION = {os.environ['TF_BF16_CONVERSION']}")

        # Load Habana device support.
        from habana_frameworks.tensorflow import load_habana_module
        load_habana_module()

        # Use HPUStrategy (instead of MultiWorkerMirroredStrategy).
        from habana_frameworks.tensorflow.distribute import HPUStrategy
        strategy = HPUStrategy()
    else:
        strategy = tf.distribute.MultiWorkerMirroredStrategy()

    train_mnist(strategy, args.batch_size, args.epochs)
