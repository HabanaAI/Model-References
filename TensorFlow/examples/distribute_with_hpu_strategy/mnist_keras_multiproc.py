# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company

""" Distributed MNIST Training using Keras

    Demonstration of multi-worker training using HPUStrategy.
    Python multiprocessing library is used to provide multiprocessing istead of Open MPI.
"""

import argparse
import json
import multiprocessing as mp
import os

import tensorflow as tf
import tensorflow_datasets as tfds

BASE_TF_SERVER_PORT = 7850
DEFAULT_PER_WORKER_BATCHS_SIZE = 64
DEFAULT_DTYPE = "bf16"
DEFAULT_NUM_EPOCHS = 6
SHUFFLE_BUFFER_SIZE = 10000
DEFAULT_NUM_WORKERS = 4


def set_multiprocessing_spawn_method():
    """ Configure multiprocessing package to "spawn" a new process, rather than fork
        this one, so the state of TensorFlow library shall not be inherited in child processes.
        This is to mitigate issues, as the "parent" process imports TensorFlow as well (but it remains unused).
    """
    try:
        mp.set_start_method('spawn')
    except RuntimeError as ex:
        print(f"warning: Setting Python multiprocessing spawn method to 'spawn' failed. However, this should not affect the execution of this particular example.")
        print(str(ex))


def set_tf_config(worker_index: int, num_workers: int):
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
    print(f"[{worker_index}] TF_CONFIG = {tf_config_text}")
    return tf_config_text


def parse_args():
    """ Parses the command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Multi-Worker Distributed MNIST Training Demo.")
    parser.add_argument("-d", "--device", type=str, default="hpu",
                        help="device to use: either 'hpu' or 'cpu'")
    parser.add_argument("-b", "-bs", "--batch_size", type=int, default=DEFAULT_PER_WORKER_BATCHS_SIZE,
                        help="batch size per replica")
    parser.add_argument("-t", "--dtype", type=str, default=None,
                        help="datatype to use: either 'bf' or 'fp'")
    parser.add_argument("-e", "--epochs", type=int, default=DEFAULT_NUM_EPOCHS,
                        help="number of training epochs")
    parser.add_argument("-n", "--num_workers", type=int, default=DEFAULT_NUM_WORKERS,
                        help="number of workers")
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


def train_mnist(worker_index: int, num_workers: int, barrier: mp.Barrier, use_hpu: bool, batch_size: int, use_bfloat: bool, num_epochs: int):
    """ Train the distributed model on MNIST Dataset.
    """
    # Set TF_CONFIG.
    set_tf_config(worker_index, num_workers)

    # Instantiate the distributed strategy class.
    if use_hpu:
        # Optionally enable automatic bfloat16 operations conversion.
        if use_bfloat:
            os.environ["TF_BF16_CONVERSION"] = "full"
            print(
                f"[{worker_index}] TF_BF16_CONVERSION = {os.environ['TF_BF16_CONVERSION']}")

        # Load Habana device support.
        from habana_frameworks.tensorflow import load_habana_module
        load_habana_module()

        # Use HPUStrategy (instead of MultiWorkerMirroredStrategy).
        from habana_frameworks.tensorflow.distribute import HPUStrategy
        strategy = HPUStrategy()
    else:
        strategy = tf.distribute.MultiWorkerMirroredStrategy()

    # Determine the total training batch size.
    batch_size_per_replica = batch_size
    total_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
    print(
        f"[{worker_index}] total_batch_size = {batch_size_per_replica} * {strategy.num_replicas_in_sync} workers = {total_batch_size}")

    # Load and preprocess the MNIST Dataset.
    # As tfds.load() may download the dataset if not cached, let the first worker do it first.
    for dataload_turn in range(2):
        if (dataload_turn == 0) == (worker_index == 0):
            print("[{worker_index}] Loading MNIST dataset...")
            datasets, info = tfds.load(
                name="mnist", with_info=True, as_supervised=True)
        barrier.wait()

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
    print("[{worker_index}] Calling model.fit()...")
    model.fit(train_dataset, epochs=num_epochs, verbose=2)
    print("[{worker_index}] Calling model.evaluate()...")
    eval_results = model.evaluate(test_dataset, verbose=2)
    print(f"[{worker_index}] Evaluation results: {eval_results}")


if __name__ == "__main__":
    set_multiprocessing_spawn_method()

    args = parse_args()
    barrier = mp.Barrier(parties=args.num_workers)
    processes = [mp.Process(target=train_mnist, args=(
        i, args.num_workers, barrier, args.use_hpu, args.batch_size, args.use_bfloat, args.epochs)) for i in range(args.num_workers)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
