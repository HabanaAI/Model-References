
# Multi-Worker Training Using HPUStrategy Example

This directory provides example training scripts which run `tensorflow.distribute` multi-worker training using `habana_frameworks.tensorflow.distribute.HPUStrategy` class.
The scripts perform CNN model training based on TensorFlow Keras API to recognize hand-written MNIST characters.

# Table of Contents
* [Model-References](../../../README.md)
* [Setup](#setup)
* [Run Using Open MPI](#run-using-open-mpi)
* [Run Using Python Multiprocessing Library](#run-using-python-multiprocessing-library)

## Setup
Multi-worker training requires at least two processes working in parallel.
Each worker must be set to run as separate processes, not threads, as only a single process can acquire a single Gaudi device.

This example shows two methods of spawning multiple concurrent processes:
 * Using Open MPI - `mpirun` tool is used for invoking multiple instances of the same Python training script (interpreter).
 * Using `python.multiprocessing` - a Python multiprocessing library is used for spawning Python interpreter processes directly from within the training script.

## Run Using Open MPI
To run the script using Open MPI on eight local Gaudi devices, run:

```bash
$ pip install tensorflow_datasets mpi4py
$ NUM_WORKERS=8 ./run_mnist_keras.sh
```

This script accepts several input parameters that can be used:

```bash
$ NUM_WORKERS=4 ./run_mnist_keras.sh --device cpu --batch_size 256 --dtype fp --epochs 1
```
## Run Using Python Multiprocessing Library
To run the script using Python multiprocessing library, run:

```bash
$ PYTHON=${PYTHON:-python3}
$ $PYTHON -m mnist_keras_multiproc
```

By default, this script uses four workers, however, other input parameters can be used:

```bash
$ $PYTHON -m mnist_keras_multiproc --device cpu --batch_size 256 --dtype fp --num_workers 2
```
