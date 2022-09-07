# ResNet-50 Transfer Learning Demo

This section implements a ResNet-50 transfer learning demo that performs training to recognize five types of flowers from a publicly available dataset.

## Table of Contents
 
  - [Model-References](../../../../../README.md)
  - [Model Overview](#model-overview)
  - [Setup](#setup)
  - [Getting the Checkpoint](#getting-the-checkpoint)
  - [Training](#training)

## Model Overview
The script utilizes a ResNet-50 model architecture defined in `Model-References/TensorFlow/computer_vision/Resnets/resnet_keras/resnet_model.py`.
It removes the last two layers to transform the model from a classifier to a feature extractor and marks all the remaining layers as non-trainable.
Then, it adds additional trainable layers on top of that.  
The pre-trained ResNet Keras model is a modified version of the original version located in the [TensorFlow model garden](https://github.com/tensorflow/models/tree/master/official/legacy/image_classification/resnet). It uses a custom training loop, supports 50 layers and can work both with SGD and LARS optimizers.

## Setup
Please follow the instructions provided in the [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the
environment including the `$PYTHON` environment variable.
The guide will walk you through the process of setting up your system to run the model on Gaudi.


The dataset can be downloaded for inspection using the following link (the script downloads and preprocesses it automatically):
https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz  

## Getting the Checkpoint
To run the transfer learning demo, first you need to have a pre-trained checkpoint.
Habana provides ResNet-50 checkpoints pre-trained on Habana Gaudi.
To download, go to [ResNet50 Catalog](https://developer.habana.ai/catalog/resnet-for-tensorflow/), select the checkpoint to obtain its URL, and run the following commands:

```bash
cd Model-References/TensorFlow/computer_vision/Resnets/resnet_keras/transfer_learning_demo
mkdir pretrained_checkpoint
wget </url/of/pretrained_checkpoint.tar.gz>
tar -xvf <pretrained_checkpoint.tar.gz> -C pretrained_checkpoint && rm <pretrained_checkpoint.tar.gz>
```

Alternatively, you can create the checkpoint from scratch by training ResNet-50 on Habana Gaudi with the following commands:

```bash
cd Model-References/TensorFlow/computer_vision/Resnets/resnet_keras/transfer_learning_demo
mkdir pretrained_checkpoint && cd ..
$PYTHON resnet_ctl_imagenet_main.py -bs 256 -dt bf16 -dlit bf16 --experimental_preloading -dd /data/tensorflow/imagenet/tf_records -te 40 -ebe 40 --optimizer LARS -md transfer_learning_demo/pretrained_checkpoint/ --enable_checkpoint_and_export
```

The command above runs 40 epochs of ResNet-50 with LARS optimizer on 1 HPU and saves the checkpoint under `Model-References/TensorFlow/computer_vision/Resnets/resnet_keras/transfer_learning_demo/pretrained_checkpoint/`.
In order to run the training faster, depending on how many HPUs you have at your disposal, you can use `mpirun`. Please refer to this [ResNet-50 Keras model README](../README.md) for more information.

## Training
In order to run the transfer learning demo, run the following command:

```bash
$PYTHON transfer_learning_demo.py --checkpoint_path Model-References/TensorFlow/computer_vision/Resnets/resnet_keras/transfer_learning_demo/pretrained_checkpoint/ckpt-25000
```

It will run 20 epochs on the downloaded and preprocessed dataset. The expected validation accuracy is above 90%.