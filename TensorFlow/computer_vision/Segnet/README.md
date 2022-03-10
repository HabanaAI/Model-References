# Implementation of VGG-Segnet for Semantic Segmentation in Keras.
For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

Please visit [this page](https://developer.habana.ai/resources/habana-training-models/#performance) for performance information.

## Table of Contents
For more information about training deep learning models on Gaudi, visit developer.habana.ai.
* [Model-References](../../../README.md)
* [Model overview](#model-overview)
* [Setup](#setup)
* [Training](#training-the-model)
* [Changelog](#Changelog)
* [Known Issues](#known-issues)


## Model Overview
This repo implements semantic segmentation in Keras using VGG-Segnet. It is adapted from [here](https://github.com/divamgupta/image-segmentation-keras). The paper this repo follows is [here](https://arxiv.org/pdf/1511.00561.pdf)

VGG-Segnet consists of a VGG backbone, followed by furthur layers of convolution and upsampling. The final layer is of the same size as the input with same number of channels as number of classes. Thus the network predicts a class for each pixel in the input image.

<p align="center">
  <img src="https://www.researchgate.net/profile/Arsal-Syed-2/publication/335495371/figure/fig2/AS:892800157638656@1589871553065/SegNets-encoder-decoder-architecture-based-off-VGG16-with-fully-connected-layers-removed.ppm" width="50%" >
</p>


Changes made to the reference [github](https://github.com/divamgupta/image-segmentation-keras) are:
* This repo focusses on VGG backbone, while the original one has other backbone options
* Experiments were done in both bf16 and float32 mode instead of float32 only in the original repo


## Setup
Please follow the instructions given in the following link for setting up the
environment including the `$PYTHON` environment variable: [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html).
This guide will walk you through the process of setting up your system to run
the model on Gaudi.


Download VGG weights:
```shell
wget https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
```

### Installing

Install the module

```shell
git clone https://github.com/HabanaAI/Model-References /root/Model-References
cd /root/Model-References
export PYTHONPATH=/root/Model-References:$PYTHONPATH
cd /root/Model-References/TensorFlow/computer_vision/Segnet
python3 setup.py install
```

### Install Model Requirements

In the docker container, go to the VGG-Segnet directory
```bash
cd /root/Model-References/TensorFlow/computer_vision/Segnet
```
Install required packages using pip
```bash
$PYTHON -m pip install -r requirements.txt
```

### Preparing the data for training

You need to make two folders

* Images Folder - For all the training images
* Annotations Folder - For the corresponding ground truth segmentation images

The filenames of the annotation images should be same as the filenames of the RGB images.

The size of the annotation image for the corresponding RGB image should be same.

For each pixel in the RGB image, the class label of that pixel in the annotation image would be the value of the blue pixel.

Example code to generate annotation images :

```python
import cv2
import numpy as np

ann_img = np.zeros((30,30,3)).astype('uint8')
ann_img[ 3 , 4 ] = 1 # this would set the label of pixel 3,4 as 1

cv2.imwrite( "ann_1.png" ,ann_img )
```

Only use bmp or png format for the annotation images.

#### Download the sample prepared dataset

To start off, download and extract the following dataset:

https://drive.google.com/file/d/0B0d9ZiqAgFkiOHR1NTJhWVJMNEU/view?usp=sharing

You will get a folder named `dataset1/`. You can use `./data_prepare.sh` to download the data


## Training the Model

`demo_segnet.py` is the starting point of running. It can run distributed or single runs. Internally it calls `python3 -m keras_segmentation train`. It is assumed steps described in data preparation are done (or `./data_prepare.sh` has been run)

Some sample command lines are present here:

i. Example: Running on 1 HPU in BF16 mode.
```
python3 demo_segnet.py train --train_images="dataset1/images_prepped_train/" --train_annotations="dataset1/annotations_prepped_train/" --val_images="dataset1/images_prepped_test/" --val_annotations="dataset1/annotations_prepped_test/" --n_classes=12 --input_height=320 --input_width=640 --model_name="vgg_segnet" --data_type="bf16" --epochs=125 --batch_size=16
```
ii. Example: Running on 1 HPU in FP32 mode.
```
python3 demo_segnet.py train --train_images="dataset1/images_prepped_train/" --train_annotations="dataset1/annotations_prepped_train/" --val_images="dataset1/images_prepped_test/" --val_annotations="dataset1/annotations_prepped_test/" --n_classes=12 --input_height=320 --input_width=640 --model_name="vgg_segnet" --epochs=125 --batch_size=16
```

iii. Example: Running on 8 HPU in BF16 mode.
```
python3 demo_segnet.py train --train_images=dataset1/images_prepped_train/ --train_annotations=dataset1/annotations_prepped_train/ --n_classes=12 --model_name=vgg_segnet --val_images=dataset1/images_prepped_test/ --val_annotations=dataset1/annotations_prepped_test/ --input_height=320 --input_width=640 --epochs=125 --data_type="bf16" --distributed --num_workers_per_hls=8 --batch_size=4
```

iv. Example: Running on 8 HPU in FP32 mode.
```
python3 demo_segnet.py train --train_images=dataset1/images_prepped_train/ --train_annotations=dataset1/annotations_prepped_train/ --n_classes=12 --model_name=vgg_segnet --val_images=dataset1/images_prepped_test/ --val_annotations=dataset1/annotations_prepped_test/ --input_height=320 --input_width=640 --epochs=125 --distributed --num_workers_per_hls=8 --batch_size=4
```

## Results

Example results:

![Sample Segmentation](sample_images/seg_mask.gif)

# Changelog
### 1.2.0
* remove `tf.compat.v1.disable_eager_execution()`
* Removed unused tensorflow_addons import
### 1.3.0
- enable support for 8 HPUs
- Change `python` or `python3` to `$PYTHON` to execute correct version based on environment setup.

# Known Issues

In this repo, we use cross entropy as the loss. That only works if the relative distribution of the classes are uniform. In case the dataset has a skewed distribution of classes, then a different loss function should be used, such as dice.
