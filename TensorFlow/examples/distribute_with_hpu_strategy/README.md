
# Demonstration of Multi-Worker Training Using HPUStrategy

This directory provides an example training scripts, which exercise `tensorflow.distribute` multi-worker training using `habana_frameworks.tensorflow.distribute.HPUStrategy` class.
The scripts perform a training of a CNN model based on TensorFlow Keras API to recognize hand-written MNIST characters.

Multi-worker training requires at least two processes working in parallel.
It is a must that workers are separate processes, not threads, as a single process is capable of acquiring a single Gaudi device.

This example shows 2 methods of spawning multiple concurrent processes:
 * Using *Open MPI* - where `mpirun` tool is used for invoking multiple instances of the same Python training script (interpreter).
 * Using *python.multiprocessing* - where a Python multiprocessing library is used for spawning Python interpreter processes directly from within the training script.

To run the script that uses Open MPI on 8 local Gaudi devices, call:

```bash
$ pip install tensorflow_datasets mpi4py
$ NUM_WORKERS=8 ./run_mnist_keras.sh
```

This script accepts several input parameters you can try:

```bash
$ NUM_WORKERS=4 ./run_mnist_keras.sh --device cpu --batch_size 256 --dtype fp --epochs 1
```

Another option is to run the script that uses Python multiprocessing library (instead of *Open MPI*).
To run it, call:

```bash
$ PYTHON=${PYTHON:-python3}
$ $PYTHON -m mnist_keras_multiproc
```

By default, this script uses 4 workers, however you can try other input parameters:

```bash
$ $PYTHON -m mnist_keras_multiproc --device cpu --batch_size 256 --dtype fp --num_workers 2
```
