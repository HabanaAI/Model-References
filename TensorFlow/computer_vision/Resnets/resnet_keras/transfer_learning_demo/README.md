# ResNet-50 Transfer Learning Demo

This section implements a ResNet-50 transfer learning demo that performs training to recognize five types of flowers from a publicly available dataset.

## Table of Contents
  - [Model-References](../../../../../README.md)
  - [Model Overview](#model-overview)
  - [Setup](#setup)
  - [Getting a Pretrained Checkpoint](#getting-a-pretrained-checkpoint)
  - [Getting a Pretrained Model](#getting-a-pretrained-model)
  - [Training](#training)

## Model Overview
The script utilizes a ResNet-50 model architecture defined in `Model-References/TensorFlow/computer_vision/Resnets/resnet_keras/resnet_model.py`.
It removes the last two layers to transform the model from a classifier to a feature extractor and marks all the remaining layers as non-trainable.
Then, it adds additional trainable layers on top of that.  
The pre-trained ResNet Keras model is a modified version of the original version located in the [TensorFlow Model Garden](https://github.com/tensorflow/models/tree/master/official/legacy/image_classification/resnet). It uses a custom training loop, supports 50 layers and can work both with SGD and LARS optimizers.

## Setup
Please follow the instructions provided in the [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the
environment including the `$PYTHON` environment variable.
The guide will walk you through the process of setting up your system to run the model on Gaudi.


The dataset can be downloaded for inspection using the following link (the script downloads and preprocesses it automatically):
https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz  

### Install Model Requirements

1. In the docker container, go to the `transfer_learning_demo` directory:
```bash
cd /root/Model-References/TensorFlow/computer_vision/Resnets/resnet_keras/transfer_learning_demo
```
2. Install the required packages using pip:
```bash
$PYTHON -m pip install -r requirements.txt
```

## Getting a Pretrained Checkpoint
To run the transfer learning demo, first you need to have a pre-trained checkpoint (or a model - see [Getting a pretrained model](#getting-a-pretrained-model) section).
Habana provides ResNet-50 checkpoints pre-trained on Intel® Gaudi® AI Accelerator.
To download, go to [ResNet50 Catalog](https://developer.habana.ai/catalog/resnet-for-tensorflow/), select the checkpoint to obtain its URL, and run the following commands:

```bash
cd Model-References/TensorFlow/computer_vision/Resnets/resnet_keras/transfer_learning_demo
mkdir pretrained_checkpoint
wget </url/of/pretrained_checkpoint.tar.gz>
tar -xvf <pretrained_checkpoint.tar.gz> -C pretrained_checkpoint && rm <pretrained_checkpoint.tar.gz>
```

Alternatively, you can create the checkpoint from scratch by training ResNet-50 on Intel Gaudi AI Accelerator with the following commands:

```bash
cd Model-References/TensorFlow/computer_vision/Resnets/resnet_keras/transfer_learning_demo
mkdir pretrained_checkpoint && cd ..
$PYTHON resnet_ctl_imagenet_main.py -bs 256 -dt bf16 -dlit bf16 --experimental_preloading -dd /data/tensorflow/imagenet/tf_records -te 40 -ebe 40 --optimizer LARS --base_learning_rate 9.5 --warmup_epochs 3 --lr_schedule polynomial --label_smoothing 0.1 --weight_decay 0.0001 --single_l2_loss_op -md transfer_learning_demo/pretrained_checkpoint/ --enable_checkpoint_and_export
```

The command above runs 40 epochs of ResNet-50 with LARS optimizer on 1 HPU and saves the checkpoint under `Model-References/TensorFlow/computer_vision/Resnets/resnet_keras/transfer_learning_demo/pretrained_checkpoint/`.
In order to run the training faster, depending on how many HPUs you have at your disposal, you can use `mpirun`. Please refer to this [ResNet-50 Keras model README](../README.md) for more information.

## Getting a Pretrained Model
Instead of using pretrained checkpoints (see [Getting a pretrained checkpoint](#getting-a-pretrained-checkpoint) section), you can also use pretrained models.
To download, go to [ResNet50 Catalog](https://developer.habana.ai/catalog/resnet-for-tensorflow/), select the saved model to obtain its URL, and run the following commands:

```bash
cd Model-References/TensorFlow/computer_vision/Resnets/resnet_keras/transfer_learning_demo
mkdir pretrained_model
wget </url/of/pretrained_model.tar.gz>
tar -xvf <pretrained_model.tar.gz> -C pretrained_model && rm <pretrained_model.tar.gz>
```

Alternatively, you can create the model from scratch by training ResNet-50 on Intel Gaudi AI Accelerator with the following commands:

```bash
cd Model-References/TensorFlow/computer_vision/Resnets/resnet_keras/transfer_learning_demo
mkdir pretrained_model && cd ..
$PYTHON resnet_ctl_imagenet_main.py -bs 256 -dt bf16 -dlit bf16 --experimental_preloading -dd /data/tensorflow/imagenet/tf_records -te 40 -ebe 40 --optimizer LARS --base_learning_rate 9.5 --warmup_epochs 3 --lr_schedule polynomial --label_smoothing 0.1 --weight_decay 0.0001 --single_l2_loss_op -md transfer_learning_demo/pretrained_model/ --save_full_model
```

The command above runs 40 epochs of ResNet-50 with LARS optimizer on 1 HPU and saves the model under `Model-References/TensorFlow/computer_vision/Resnets/resnet_keras/transfer_learning_demo/pretrained_model/`.
In order to run the training faster, depending on how many HPUs you have at your disposal, you can use `mpirun`. Please refer to [ResNet-50 Keras model README](../README.md) for more information.

## Training
In order to run the transfer learning demo, run one of the following commands:

To run with pretrained checkpoint:
```bash
$PYTHON transfer_learning_demo.py --checkpoint_path Model-References/TensorFlow/computer_vision/Resnets/resnet_keras/transfer_learning_demo/pretrained_checkpoint/ckpt-25000
```

To run with pretrained model:
```bash
$PYTHON transfer_learning_demo.py --saved_model_path Model-References/TensorFlow/computer_vision/Resnets/resnet_keras/transfer_learning_demo/pretrained_model/resnet_model.h5
```

It will run 20 epochs on the downloaded and preprocessed dataset. The expected validation accuracy is above 90%.