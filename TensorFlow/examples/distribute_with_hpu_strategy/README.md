
# Demonstration of Multi-Worker Training using HPUStrategy

This directory contains an example training script, which exercises `tensorflow.distribute` multi-worker training using `habana_frameworks.tensorflow.distribute.HPUStrategy` class.
The script performs a training of a CNN model based on TensorFlow Keras API to recognize hand-written MNIST characters.

Note: HPUStrategy is only supported with TensorFlow 2.4.1 for this release.

To run the script on using 8 local Gaudi devices, call:

```bash
$ NUM_WORKERS=8 ./run_mnist_keras.sh
```

The script accepts several input parameters you can try:

```bash
$ NUM_WORKERS=4 ./run_mnist_keras.sh --device cpu --batch_size 256 --dtype fp --epochs 1
```
