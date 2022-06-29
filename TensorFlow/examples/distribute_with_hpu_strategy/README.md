
# Demonstration of Multi-Worker Training using HPUStrategy

This directory contains an example training scripts, which exercise `tensorflow.distribute` multi-worker training using `habana_frameworks.tensorflow.distribute.HPUStrategy` class.
The scripts perform a training of a CNN model based on TensorFlow Keras API to recognize hand-written MNIST characters.

To run the script that uses Open MPI on 8 local Gaudi devices, call:

```bash
$ pip install tensorflow_datasets mpi4py
$ NUM_WORKERS=8 ./run_mnist_keras.sh
```

This script accepts several input parameters you can try:

```bash
$ NUM_WORKERS=4 ./run_mnist_keras.sh --device cpu --batch_size 256 --dtype fp --epochs 1
```

Another option is to run the script that uses Python multiprocessing library instead of Open MPI.
To run it, call:

```bash
$ PYTHON=${PYTHON:-python3}
$ $PYTHON -m mnist_keras_multiproc
```

By default this script uses 4 workers, but you can try other input parameters:

```bash
$ $PYTHON -m mnist_keras_multiproc --device cpu --batch_size 256 --dtype fp --num_workers 2
```