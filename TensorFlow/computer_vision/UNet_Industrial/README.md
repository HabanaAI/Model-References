# UNet Industrial for TensorFlow

This directory provides a script and recipe to train a UNet Industrial Defect Segmentation model to achieve state of the art accuracy, and is tested and maintained by Habana.
For further information on performance, refer to [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

For further information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).


## Table of Contents

* [Model-References](../../../README.md)
* [Model Overview](#model-overview)
* [Setup](#setup)
* [Training and Examples](#training-and-examples)
* [Advanced](#advanced)
* [Supported Configuration](#supported-configuration)
* [Changelog](#changelog)

## Model Overview

The UNet model is a modified version of the original model located in [UNet model](https://arxiv.org/abs/1505.04597), called TinyUNet, which performs efficiently and with a high accuracy on the industrial dataset [DAGM2007].
It is constructed with the same structure as the original UNet, including repeatedly 3 downsampling blocks and 3 upsampling blocks.
The modified version mainly differ from the original UNet in reducing the dimension of the model layers and the model capacity. This change reduces the model to over-fit on a small dataset such as DAGM2007.

The script is based on [NVIDIA UNet Industrial Defect Segmentation for TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/abe062867f8904d6ef37966d79c754b7d1be9dca/TensorFlow/Segmentation/UNet_Industrial).
We converted the training scripts to TensorFlow 2, added Habana device support and modified Horovod usage to Horovod function wrappers and global `hvd` object.

### Model Changes

The following are the major changes that were implemented to the original model from [NVIDIA UNet Industrial Defect Segmentation for TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Segmentation/UNet_Industrial):

* Converted all the scripts to Tensorflow 2.x version by using [tf_upgrade_v2](https://www.tensorflow.org/guide/upgrade?hl=en) tool.
* Changed some scripts to run the model on Gaudi. This includes loading Habana TensorFlow modules and using multi Gaudi card helpers.
* Added support for using bfloat16 precision instead of float16.
* Added further TensorBoard and performance logging options.
* Removed access to the ``tf.contrib`` library for Tensorflow 2:
    * Added ``utils/hparam.py`` file.
    * Removed ``tf.contrib.mixed_precision.ExponentialUpdateLossScaleManager`` from script.
* Replaced the ``/gpu:0`` device with ``/device:HPU:0``.

### Default Configuration

This model trains 2500 iterations, with the following configuration:

* Global Batch Size: 16

* Optimizer RMSProp:
    * decay: 0.9
    * momentum: 0.8
    * centered: True

* Learning Rate Schedule: Exponential Step Decay
    * decay: 0.8
    * steps: 500
    * initial learning rate: 1e-4

* Weight Initialization: He Uniform Distribution (introduced by [Kaiming He et al. in 2015](https://arxiv.org/abs/1502.01852) to address issues related ReLU activations in deep neural networks)

* Loss Function: Adaptive
    * When DICE Loss < 0.3, Loss = Binary Cross Entropy
    * Else, Loss = DICE Loss

* Data Augmentation
    * Random Horizontal Flip (50% chance)
    * Random Rotation 90°

* Activation Functions:
    * ReLU is used for all layers
    * Sigmoid is used at the output to ensure that the outputs are between [0, 1]

* Weight decay: 1e-5


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
export PYTHONPATH=$PYTHONPATH:/root/to/Model-References
```

### Install Model Requirements

1. In the docker container, go to the UNet_Industrial directory:

```bash
cd /root/Model-References/TensorFlow/computer_vision/UNet_Industrial
```

2. Install the required packages using pip:

```bash
$PYTHON -m pip install -r requirements.txt
```

### Download and Pre-process Dataset

Download and pre-process the dataset DAGM2007 by executing the following:

```bash
./download_and_preprocess_dagm2007.sh /data
```

**IMPORTANT:** Some files of the dataset require an account to be downloaded. The script will invite you to download the files manually and put them in the correct directory.


## Training and Examples

### Single Card and Multi-Card Training Examples

**Run training on 1 HPU:**

```bash
$PYTHON main.py --data_dir <path/to/dataset> --dtype <precision> --results_dir <path/to/result_dir> --dataset_classID <dataset_classID> --exec_mode train_and_evaluate --warmup_step 10
```

Run training on 1 HPU with batch size 16, bfloat16 precision:
```bash
$PYTHON main.py --data_dir /data/DAGM2007_dataset --dtype bf16 --results_dir /tmp/unet_industrial --dataset_classID 1 --exec_mode train_and_evaluate --warmup_step 10
```

**Run training on 8 HPUs:**

**NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration).

```bash
mpirun --allow-run-as-root --tag-output --merge-stderr-to-stdout --output-filename /root/tmp/unet_industrial_log --bind-to core --map-by socket:PE=6 -np 8 $PYTHON main.py --data_dir <path/to/dataset> --dtype <precision> --results_dir <path/to/result_dir> --dataset_classID <dataset_classID> --num_workers_per_hls 8 --batch_size 2 --exec_mode train_and_evaluate --warmup_step 10
```

Run training on 8 HPUs with batch size 2, bfloat16 precision:

```bash
mpirun --allow-run-as-root --tag-output --merge-stderr-to-stdout --output-filename /root/tmp/unet_industrial_log --bind-to core --map-by socket:PE=6 -np 8 $PYTHON main.py --data_dir /data/DAGM2007_dataset --dtype bf16 --results_dir /tmp/unet_industrial --dataset_classID 1 --num_workers_per_hls 8 --batch_size 2 --exec_mode train_and_evaluate --warmup_step 10
```

## Advanced

The following sections provide further details on the available options.

* `--exec_mode=train_and_evaluate`: Which execution mode to run the model into, choice: "train" ,"evaluate", or "train_and_evaluate".
* `--iter_unit=batch`: Will the model be run for X batches or X epochs ?
* `--num_iter=2500`: Number of iterations to run.
* `--batch_size=16`: Size of each minibatch per HPU.
* `--results_dir=/path/to/results`: Directory in which to write training logs, summaries, and checkpoints.
* `--data_dir=/path/to/dataset`: Directory which contains the DAGM2007 dataset.
* `--dataset_name=DAGM2007`: Name of the dataset used in this run (only DAGM2007 is supported atm).
* `--dataset_classID=1`: ClassID to consider to train or evaluate the network (used for DAGM).

## Supported Configuration

| Device | SynapseAI Version | TensorFlow Version(s)  |
|:------:|:-----------------:|:-----:|
| Gaudi  | 1.6.1             | 2.9.1 |
| Gaudi  | 1.6.1             | 2.8.2 |

## Changelog

### 1.4.0

* Added support to import horovod-fork package directly instead of using Model-References' TensorFlow.common.horovod_helpers. Wrapped horovod import with a try-catch block so that installing the library is not required when the model is running on a single card.

* Replaced references to custom demo script by community entry points in the README and `train_and_evaluate.sh`.
