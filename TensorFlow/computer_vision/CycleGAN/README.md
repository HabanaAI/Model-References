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
environment including the `$PYTHON` environment variable: [Gaudi Setup and
Installation Guide](https://github.com/HabanaAI/Setup_and_Install). Please
answer the questions in the guide according to your preferences. This guide will
walk you through the process of setting up your system to run the model on
Gaudi.

### Clone Habana Model-References

In the docker container, clone this repository and switch to the branch that
matches your SynapseAI version. (Run the
[`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
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
```bash
$PYTHON -m pip install -r requirements.txt
```

## Training

### Download dataset

If there is no dataset at given path (--dataset_dir flag) it will be downloaded before training execution.

### Run single-card training

Both `demo_cycle_gan.py` and `cycle_gan.py` script can be used to run training. `demo_cycle_gan.py` is a thin wrapper for `cycle_gan.py`, that reduces boilerplate when running multi-node training.

Using `demo_cycle_gan.py` with all default parameters:
```bash
$PYTHON demo_cycle_gan.py
```

For example the following command will train the topology on single Gaudi, batch size 2, 200 epochs, precision bf16 and remaining default hyperparameters.
```bash
$PYTHON demo_cycle_gan.py -e 200 -b 2 -d bf16
```

### Run 8-cards training

When `demo_cycle_gan.py` is used, then `hvd_workers` parameter can be set. HCL configuration and hyperparameters adjustment will be performed automatically.

For example to train the topology on 8 gaudi cards, batch size 2, 200 epochs, precision bf16 and other default hyperparameters.
```bash
$PYTHON demo_cycle_gan.py --hvd_workers 8 -e 200 -b 2 -d bf16
```

Equivalent command for mpirun using `cycle_gan.py`:

For example to run 8 Gaudi cards training via mpirun with batch size 2, 200 epochs, precision bf16 and other default hyperparameters.
```bash
mpirun --allow-run-as-root --tag-output --merge-stderr-to-stdout --output-filename /tmp/cycle_gan -np 8 $PYTHON cycle_gan.py --use_horovod --hvd_workers 8 --epochs 200 --batch_size 2 --data_type bf16
```

### Examples

| Command | Notes |
| ------- | ----- |
|`$PYTHON demo_cycle_gan.py -d bf16 -b 2 --use_hooks --logdir <path/to/logdir>`| Single-card training in bf16 |
|`$PYTHON demo_cycle_gan.py -d fp32 -b 2 --use_hooks --logdir <path/to/logdir>`| Single-card training in fp32 |
|`$PYTHON demo_cycle_gan.py -d bf16 -b 2 --use_hooks --logdir <path/to/logdir> --hvd_workers 8`| 8-cards training in bf16 |
|`$PYTHON demo_cycle_gan.py -d fp32 -b 2 --use_hooks --logdir <path/to/logdir> --hvd_workers 8`| 8-cards training in fp32 |

NOTE: It is expected for 8-card training to take a bit longer than 1-card training, because dataset is not sharded between workers for this topology.

## Advanced

The following section provides details of running training.

### Scripts and sample code

Important files in the `root/Model-References/TensorFlow/computer_vision/CycleGAN` directory are:

* `demo_cycle_gan.py`: Serves as a wrapper script for the training file `cycle_gan.py`. It allows to run single card or distributed training as it contains the `--hvd_workers` argument, which let as decide on how many cards execute training.
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

## Changelog
### 1.2.0
- Learning Rate scaling changed from linear to sqrt to improve training convergence
- Initialization changed to improve convergence
- Added optional warmup and gradient clipping
- InstanceNorm kernel is used to improve performance

## Known Issues
- Sporadic training divergence on multi-cards run (1/20 runs)