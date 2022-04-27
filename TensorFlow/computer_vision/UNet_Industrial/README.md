# UNet Industrial Defect Segmentation

This repository provides a script to train the segmentation model for Tensorflow on Habana Gaudi<sup>TM</sup> device.
For more information, visit [developer.habana.ai](https://developer.habana.ai/resources/).

Please visit [this page](https://developer.habana.ai/resources/habana-training-models/#performance) for performance information.

## Table of Contents

- [Model-References](../../../README.md)
- [Model Overview](#model-overview)
- [Setup](#setup)
- [Training the Model](#training-the-model)
- [Advanced](#advanced)
- [Supported Configuration](#supported-configuration)
- [Changelog](#changelog)

## Model Overview
This UNet model is a variant of the original version [UNet model](https://arxiv.org/abs/1505.04597), called TinyUNet, which performs efficiently and with high accuracy on the industrial dataset [DAGM2007].
It is constructed with the same structure as the original UNet, including repeatedly 3 downsampling blocks and 3 upsampling blocks.
The main difference of it to original UNet is it reduces the dimension of model layers and the model capacity, this change reduces the model to over-fit on a small dataset like DAGM2007.

The script is based on [NVIDIA UNet Industrial Defect Segmentation for TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/abe062867f8904d6ef37966d79c754b7d1be9dca/TensorFlow/Segmentation/UNet_Industrial)
We converted the training scripts to TensorFlow 2, added Habana device support, and modified Horovod usage to Horovod function wrappers and global `hvd` object.

### Model Changes
Major changes were done to the original model from  [NVIDIA UNet Industrial Defect Segmentation for TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Segmentation/UNet_Industrial):

* All scripts have been converted to Tensorflow 2.x version by using [tf_upgrade_v2](https://www.tensorflow.org/guide/upgrade?hl=en) tool;
* Some scripts were changed in order to run the model on Gaudi. It includes loading Habana TensorFlow modules and using multi Gaudi card helpers;
* Model is using bfloat16 precision instead of float16;
* Additional tensorboard and performance logging options were added;
* Library ``tf.contrib`` is not accessible for Tensorflow 2. File ``utils/hparam.py`` was added and ``tf.contrib.mixed_precision.ExponentialUpdateLossScaleManager`` was removed from script;
* Replace device ``/gpu:0`` with ``/device:HPU:0``.

### Default configuration
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
Please follow the instructions given in the following link for setting up the
environment including the `$PYTHON` environment variable: [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html).
This guide will walk you through the process of setting up your system to run
the model on Gaudi.

### Clone Habana Model-References repository and go to UNet_Industrial folder:
In the docker container, clone this repository and switch to the branch that matches your SynapseAI version.
(Run the
[`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version).

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
cd Model-References/TensorFlow/computer_vision/UNet_Industrial
```
### Install Model Requirements

In the docker container, go to the UNet_Industrial directory
```bash
cd /root/Model-References/TensorFlow/computer_vision/UNet_Industrial
```
Install required packages using pip
```bash
$PYTHON -m pip install -r requirements.txt
```

### Download and preprocess the dataset: DAGM2007

In order to download the dataset. You can execute the following:

```bash
./download_and_preprocess_dagm2007.sh /data
```

**Important Information:** Some files of the dataset require an account to be downloaded, the script will invite you to download them manually and put them in the correct directory.



## Training the Model
The model was tested both in single Gaudi and 8x Gaudi cards configurations.

### Add Tensorflow packages from model_garden to python path:
```bash
export PYTHONPATH=$PYTHONPATH:/root/Model-References/
```

### Run training on single Gaudi card

```bash
$PYTHON main.py --data_dir <path/to/dataset> --dtype <precision> --results_dir <path/to/result_dir> --dataset_classID <dataset_classID> --exec_mode train_and_evaluate
```

For example:
- single Gaudi card training with batch size 16, bfloat16 precision:
    ```bash
    $PYTHON main.py --data_dir /data/DAGM2007_dataset --dtype bf16 --results_dir /tmp/unet_industrial --dataset_classID 1 --exec_mode train_and_evaluate
    ```

### Run training on single server (8 Gaudi card)
*mpirun map-by PE attribute value may vary on your setup and should be calculated as:<br>
socket:PE = floor((number of physical cores) / (number of gaudi devices per each node))*

```bash
mpirun --allow-run-as-root --tag-output --merge-stderr-to-stdout --output-filename /root/tmp/unet_industrial_log --bind-to core --map-by socket:PE=4 -np 8 $PYTHON main.py --data_dir <path/to/dataset> --dtype <precision> --results_dir <path/to/result_dir> --dataset_classID <dataset_classID> --num_workers_per_hls 8 --batch_size 2 --exec_mode train_and_evaluate
```

For example:
- 8 Gaudi cards training with batch size 2, bfloat16 precision:
    ```bash
    mpirun --allow-run-as-root --tag-output --merge-stderr-to-stdout --output-filename /root/tmp/unet_industrial_log --bind-to core --map-by socket:PE=4 -np 8 $PYTHON main.py --data_dir /data/DAGM2007_dataset --dtype bf16 --results_dir /tmp/unet_industrial --dataset_classID 1 --num_workers_per_hls 8 --batch_size 2 --exec_mode train_and_evaluate
    ```



## Advanced
The following sections provide greater details of available options.
```
--exec_mode=train_and_evaluate Which execution mode to run the model into, choice: "train" ,"evaluate", or "train_and_evaluate"

--iter_unit=batch Will the model be run for X batches or X epochs ?

--num_iter=2500 Number of iterations to run.

--batch_size=16 Size of each minibatch per HPU.

--results_dir=/path/to/results Directory in which to write training logs, summaries, and checkpoints.

--data_dir=/path/to/dataset Directory which contains the DAGM2007 dataset.

--dataset_name=DAGM2007 Name of the dataset used in this run (only DAGM2007 is supported atm).

--dataset_classID=1 ClassID to consider to train or evaluate the network (used for DAGM).
```

## Supported Configuration

| Device | SynapseAI Version | TensorFlow Version(s)  |
|:------:|:-----------------:|:-----:|
| Gaudi  | 1.4.0             | 2.8.0 |
| Gaudi  | 1.4.0             | 2.7.1 |

## Changelog
### 1.4.0
* Import horovod-fork package directly instead of using Model-References' TensorFlow.common.horovod_helpers; wrapped horovod import with a try-catch block so that the user is not required to install this library when the model is being run on a single card
* References to custom demo script were replaced by community entry points in README.