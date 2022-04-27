# CycleGAN for Image Translation

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents

* [Model-References](../../../README.md)
* [Model overview](#model-overview)
* [Setup](#setup)
* [Training](#training)
    - [Run single-card training](#run-single-card-training)
    - [Run 8-cards training](#run-8-cards-training)
* [Advanced](#advanced)
* [Changelog](#changelog)
* [Known Issues](#known-issues)

## Model Overview

CycleGAN is a model that aims to solve the image-to-image translation problem. The goal of the image-to-image translation problem is to learn the mapping between an input image and an output image using a training set of aligned image pairs. However, obtaining paired examples isn't always feasible. CycleGAN tries to learn this mapping without requiring paired input-output images, using cycle-consistent adversarial networks. More details in [paper](https://arxiv.org/pdf/1703.10593.pdf)

### Default configuration

- Batch size = 2
- Epochs = 200
- Data type = bf16
- Buffer = 256
- Use hooks = True
- Horovod workers = 1
- Generator Learning rate = 4e-4
- Discrimantor  Learning rate = 2e-4
- Monitor frequency transformed test images value = 5
- Save model frequency value = 1
- Cosine Decay Delay = 100

## Setup
Please follow the instructions given in the following link for setting up the
environment including the `$PYTHON` environment variable: [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html).
This guide will walk you through the process of setting up your system to run
the model on Gaudi.

### Clone Habana Model-References

In the docker container, clone this repository and switch to the branch that
matches your SynapseAI version. (Run the
[`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.)

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References /root/Model-References
```
Go to the CycleGAN directory:
```bash
cd /root/Model-References/TensorFlow/computer_vision/CycleGAN
```
Add Model-References to PYTHONPATH
```bash
export PYTHONPATH=/root/Model-References:$PYTHONPATH
```

### Install Model Requirements

In the docker container, go to the CycleGAN directory
```bash
cd /root/Model-References/TensorFlow/computer_vision/CycleGAN
```
Install required packages using pip
```python
$PYTHON -m pip install -r requirements.txt
```

## Training

### Download dataset

If there is no dataset at given path (--dataset_dir flag) it will be downloaded before training execution.

### Run single-card training

Using `cycle_gan.py` with all default parameters:
```python
$PYTHON cycle_gan.py
```

For example the following command will train the topology on single Gaudi, batch size 2, 200 epochs, precision bf16 and remaining default hyperparameters.
```python
$PYTHON cycle_gan.py -e 200 -b 2 -d bf16
```

### Run 8-cards training
*mpirun map-by PE attribute value may vary on your setup and should be calculated as:<br>
socket:PE = floor((number of physical cores) / (number of gaudi devices per each node))*

With mpirun using `cycle_gan.py` the `hvd_workers` parameter can be set for multi card training.

For example to run 8 Gaudi cards training via mpirun with batch size 2, 200 epochs, precision bf16 and other default hyperparameters.
*<br>mpirun map-by PE attribute value may vary on your setup and should be calculated as:<br>
socket:PE = floor((number of physical cores) / (number of gaudi devices per each node))*

```bash
mpirun --allow-run-as-root --tag-output --merge-stderr-to-stdout --output-filename /tmp/cycle_gan -np 8 $PYTHON cycle_gan.py --use_horovod --hvd_workers 8 -e 200 -b 2 -d bf16
```

### Examples
*mpirun map-by PE attribute value may vary on your setup and should be calculated as:<br>
socket:PE = floor((number of physical cores) / (number of gaudi devices per each node))*


| Command | Notes |
| ------- | ----- |
|`$PYTHON cycle_gan.py -d bf16 -b 2 --use_hooks --logdir <path/to/logdir>`| Single-card training in bf16 |
|`$PYTHON cycle_gan.py -d fp32 -b 2 --use_hooks --logdir <path/to/logdir>`| Single-card training in fp32 |
|`mpirun --allow-run-as-root --tag-output --merge-stderr-to-stdout --output-filename /tmp/cyclegan -np 8 $PYTHON cycle_gan.py -d bf16 -b 2 --use_hooks --logdir <path/to/logdir> --hvd_workers 8`| 8-cards training in bf16 |
|`mpirun --allow-run-as-root --tag-output --merge-stderr-to-stdout --output-filename /tmp/cyclegan -np 8 $PYTHON cycle_gan.py -d fp32 -b 2 --use_hooks --logdir <path/to/logdir> --hvd_workers 8`| 8-cards training in fp32 |

NOTE: It is expected for 8-card training to take a bit longer than 1-card training, because dataset is not sharded between workers for this topology.

## Advanced

The following section provides details of running training.

### Scripts and sample code

Important files in the `root/Model-References/TensorFlow/computer_vision/CycleGAN` directory are:

* `cycle_gan.py`: The main training script of the CycleGAN model.
* `arguments.py`: The file containing list of all arguments.

### Parameters

Modify the training behavior through the various flags present in the `arguments.py` file, which is imported in `cycle_gan.py` file. Some of the important parameters in the
`arguments.py` script are as follows:

-  `-d` or `--data_type`                             Data type, possible values: fp32, bf16 (default: bf16)
-  `-b` or `--batch_size`                            Batch size (default: 2)
-  `-e` or `--epochs`                                Number of epochs for training (default: 200)
-  `--steps_per_epoch`                               Steps per epoch
-  `--logdir`                                        Path where all logs will be stored (default: `./model_checkpoints/exp_alt_pool_no_tv`)
-  `--use_horovod`                                   Use Horovod for distributed training
-  `--dataset_dir`                                   Path to dataset. If dataset doesn't exist, it will be downloaded (default: `./dataset/`)
-  `--use_hooks`                                     Whether to use hooks during training. If used, stores value as True
-  `--generator_lr`                                  Generator learning rate (default: 4e-4)
-  `--save_freq`                                     How often save model (default: 1)
-  `--cosine_decay_delay`                            After how many epoch start decaying learning rates (default: 100)

## Supported Configuration

| Device | SynapseAI Version | TensorFlow Version(s)  |
|:------:|:-----------------:|:-----:|
| Gaudi  | 1.4.0             | 2.8.0 |
| Gaudi  | 1.4.0             | 2.7.1 |

## Changelog
### 1.2.0
- Learning Rate scaling changed from linear to sqrt to improve training convergence
- Initialization changed to improve convergence
- Added optional warmup and gradient clipping
- InstanceNorm kernel is used to improve performance
- Updated requirements.txt
### 1.3.0
- Updated requirements.txt
### 1.4.0
- Import horovod-fork package directly instead of using Model-References' TensorFlow.common.horovod_helpers; wrapped horovod import with a try-catch block so that the user is not required to install this library when the model is being run on a single card
- References to custom demo script were replaced by community entry points in README.
## Known Issues
- Sporadic training divergence on multi-cards run (1/20 runs)
