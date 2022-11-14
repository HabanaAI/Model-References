# VGG-Segnet Model for TensorFlow

This directory provides a script and recipe to train a VGG-Segnet model for semantic segmentation in Keras to achieve state of the art accuracy, and is tested and maintained by Habana.
For further information on performance, refer to [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

For further information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents

* [Model-References](../../../README.md)
* [Model overview](#model-overview)
* [Setup](#setup)
* [Training Examples](#training-examples)
* [Results](#results)
* [Supported Configuration](#supported-configuration)
* [Changelog](#Changelog)
* [Known Issues](#known-issues)


## Model Overview

This repository implements semantic segmentation in Keras using VGG-Segnet. It is adapted from
[image-segmentation-keras](https://github.com/divamgupta/image-segmentation-keras), and based on the following paper: [SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](https://arxiv.org/pdf/1511.00561.pdf).

The VGG-Segnet model consists of a VGG backbone, followed by furthur layers of convolution and upsampling. The final layer is of the same size as the input with the same number of channels as the number of classes. Thus the network predicts a class for each pixel in the input image.

<p align="center">
  <img src="https://www.researchgate.net/profile/Arsal-Syed-2/publication/335495371/figure/fig2/AS:892800157638656@1589871553065/SegNets-encoder-decoder-architecture-based-off-VGG16-with-fully-connected-layers-removed.ppm" width="50%" >
</p>

### Model Changes

The following are the major changes that were implemented to the original model from [image-segmentation-keras](https://github.com/divamgupta/image-segmentation-keras):

* This repository focuses on VGG backbone, while the original one has other backbone options.
* Experiments were done in both BF16 and float32 mode instead of float32 only in the original repository.

## Setup

Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) to set up the
environment including the `$PYTHON` environment variable.
The guide will walk you through the process of setting up your system to run the model on Gaudi.


### Clone Habana Model-References

In the docker container, clone this repository and switch to the branch that matches your SynapseAI version. You can run the [`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the SynapseAI version.

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References /root/Model-References
```

**Note:** If Model-References repository path is not in the PYTHONPATH, make sure you update it:
```bash
export PYTHONPATH=$PYTHONPATH:/root/Model-References
```

### Install Model Requirements

1. In the docker container, go to the VGG-Segnet directory:

```bash
cd /root/Model-References/TensorFlow/computer_vision/Segnet
```

2. Install the required packages using pip:

```bash
$PYTHON -m pip install -r requirements.txt
```

### Install the Module

1. Download VGG weights:

```shell
cd /root/Model-References/TensorFlow/computer_vision/Segnet
wget https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
```

2. Install the module:

```shell
cd /root/Model-References/TensorFlow/computer_vision/Segnet
$PYTHON setup.py install
```

### Prepare the Data for Training

To prepare the data for training, you need to make two folders:

* Images Folder - For all the training images.
* Annotations Folder - For the corresponding ground truth segmentation images.

And the folders should apply to the following:

* The files names of the annotation images should be the same as the files names of the RGB images.
* The size of the annotation image for the corresponding RGB image should be the same.
* For each pixel in the RGB image, the class label of that pixel in the annotation image would be the value of the blue pixel.

The following is an example code for generating annotation images:

```python
import cv2
import numpy as np

ann_img = np.zeros((30,30,3)).astype('uint8')
ann_img[ 3 , 4 ] = 1 # this would set the label of pixel 3,4 as 1

cv2.imwrite( "ann_1.png" ,ann_img )
```

**NOTE:** Use only bmp or png format for the annotation images.

#### Download the Sample Prepared Dataset

To start off, download and extract the following dataset:

https://drive.google.com/file/d/0B0d9ZiqAgFkiOHR1NTJhWVJMNEU/view?usp=sharing

You will get a folder named `dataset1/`. You can use `./data_prepare.sh` to download the data.


## Training Examples

After implementing the data preparation steps described above (or `./data_prepare.sh` has been run),
you can start the training with `$PYTHON -m keras_segmentation train`.

### Single Card and Multi-Card Training Examples

**Run training on 1 HPU:**

- 1 HPU in BF16 mode with batch size 16, height 320, width 640, epochs 125 and learning rate 0.00001:

  ```bash
  $PYTHON -m keras_segmentation train --train_images="dataset1/images_prepped_train/" --train_annotations="dataset1/annotations_prepped_train/" --val_images="dataset1/images_prepped_test/" --val_annotations="dataset1/annotations_prepped_test/" --n_classes=12 --input_height=320 --input_width=640 --model_name="vgg_segnet" --data_type="bf16" --epochs=125 --batch_size=16 --learning_rate=0.00001
  ```

- 1 HPU in FP32 mode with batch size 16, height 320, width 640, learning rate 0.00001:

  ```bash
  $PYTHON -m keras_segmentation train --train_images="dataset1/images_prepped_train/" --train_annotations="dataset1/annotations_prepped_train/" --val_images="dataset1/images_prepped_test/" --val_annotations="dataset1/annotations_prepped_test/" --n_classes=12 --input_height=320 --input_width=640 --model_name="vgg_segnet" --epochs=125 --batch_size=16 --learning_rate=0.00001
  ```

**Run training on 8 HPUs:**

**NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration).

- 8 HPUs in BF16 mode with batch size 16, height 320, width 640, epochs 125 and learning rate 0.00002:

  ```bash
  mpirun --allow-run-as-root --tag-output --merge-stderr-to-stdout --output-filename /tmp/demo_segnet --bind-to core --map-by socket:PE=6 -np 8 $PYTHON -m keras_segmentation train --train_images=dataset1/images_prepped_train/ --train_annotations=dataset1/annotations_prepped_train/ --n_classes=12 --model_name=vgg_segnet --val_images=dataset1/images_prepped_test/ --val_annotations=dataset1/annotations_prepped_test/ --input_height=320 --input_width=640 --epochs=125 --data_type=bf16 --distributed --num_workers_per_hls=8 --batch_size=16 --learning_rate=0.00002
  ```

- 8 HPUs in FP32 mode with batch size 16, height 320, width 640, epochs 125 and learning rate 0.00002:

  ```bash
  mpirun --allow-run-as-root --tag-output --merge-stderr-to-stdout --output-filename /tmp/demo_segnet --bind-to core --map-by socket:PE=6 -np 8 $PYTHON -m keras_segmentation train --train_images=dataset1/images_prepped_train/ --train_annotations=dataset1/annotations_prepped_train/ --n_classes=12 --model_name=vgg_segnet --val_images=dataset1/images_prepped_test/ --val_annotations=dataset1/annotations_prepped_test/ --input_height=320 --input_width=640 --epochs=125 --distributed --num_workers_per_hls=8 --batch_size=16 --learning_rate=0.00002
  ```

## Results

Example results:

![Sample Segmentation](sample_images/seg_mask.gif)

## Supported Configuration

| Device | SynapseAI Version | TensorFlow Version(s)  |
|:------:|:-----------------:|:-----:|
| Gaudi  | 1.7.0             | 2.10.0 |
| Gaudi  | 1.7.0             | 2.8.3 |

## Changelog

### 1.4.0

* Replaced references to custom demo script by community entry points in README.
* Removed setup_jemalloc from demo script.
* Fixed 8x run on GPU.
* Removed extra model fit.
* Fixed accuracy and performance reporting.
* Fixed the sequence of commands for Segnet setup.

### 1.3.0

* Enabled support for 8 HPUs.
* Changed `python` or `python3` to `$PYTHON` to execute correct version based on environment setup.

### 1.2.0

* Removed `tf.compat.v1.disable_eager_execution()`.
* Removed unused tensorflow_addons import.

## Known Issues

In this repository, we use cross entropy as the loss. That only works if the relative distribution of the classes are uniform. In case the dataset has a skewed distribution of classes, then a different loss function should be used, such as dice.
