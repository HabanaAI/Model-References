# CycleGAN for TensorFlow

This directory provides a script and recipe to train a CycleGAN Model for image translation to achieve state of the art accuracy, and is tested and maintained by Habana.
For further information on performance, refer to [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

For further information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents

* [Model-References](../../../README.md)
* [Model overview](#model-overview)
* [Setup](#setup)
* [Training Examples](#training-examples)
* [Advanced](#advanced)
* [Supported Configuration](#supported-configuration)
* [Changelog](#changelog)
* [Known Issues](#known-issues)

## Model Overview

The CycleGAN model aims to solve the image-to-image translation problem. The goal of the image-to-image translation issue is to learn the mapping between an input image and an output image using a training set of aligned image pairs. However, obtaining paired examples isn't always feasible. CycleGAN tries to learn this mapping without requiring paired input-output images, using cycle-consistent adversarial networks. For further details, refer to [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf).

### Default Configuration

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

1. In the docker container, go to the CycleGAN directory:

```bash
cd /root/Model-References/TensorFlow/computer_vision/CycleGAN
```
2. Install the required packages using pip:

```python
$PYTHON -m pip install -r requirements.txt
```

### Download the Dataset

If there is no dataset at the given path (`--dataset_dir flag`), it will be automatically downloaded from [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/cycle_gan#cycle_ganhorse2zebra) before the training execution.

## Training and Examples

### Single Card and Multi-Card Training Examples

**Run training on 1 HPU:**

Use `cycle_gan.py` with all the default parameters:

```python
$PYTHON cycle_gan.py
```

Run training on 1 HPU, batch size 2, 200 epochs, precision BF16 and remaining default hyperparameters:

```python
$PYTHON cycle_gan.py -e 200 -b 2 -d bf16
```

**Run training on 8 HPUs:**

**NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration).

Run training on 8 HPUs via mpirun with batch size 2, 200 epochs, precision BF16 and other default hyperparameters:

```bash
mpirun --allow-run-as-root --tag-output --merge-stderr-to-stdout --output-filename /tmp/cycle_gan -np 8 $PYTHON cycle_gan.py --use_horovod --hvd_workers 8 -e 200 -b 2 -d bf16
```
**NOTE:** With mpirun using `cycle_gan.py`, the `hvd_workers` parameter can be set for multi-card training.

## Advanced

The following sections provide further details on the scripts in the directory and available parameters.

### Scripts and Sample Code

The following lists the critical files in the root directory: `root/Model-References/TensorFlow/computer_vision/CycleGAN`:

* `cycle_gan.py`: The main training script of the CycleGAN model.
* `arguments.py`: The file containing list of all arguments.

### Parameters

Modify the training behavior through the various flags present in the `arguments.py` file, which is imported in `cycle_gan.py` file. The following lists the important parameters in the
`arguments.py`:

-  `-d` or `--data_type`                             Data type, possible values: fp32, bf16 (default: bf16)
-  `-b` or `--batch_size`                            Batch size (default: 2)
-  `-e` or `--epochs`                                Number of epochs for training (default: 200)
-  `--steps_per_epoch`                               Steps per epoch
-  `--logdir`                                        Path where all logs will be stored (default: `./model_checkpoints/exp_alt_pool_no_tv`)
-  `--use_horovod`                                   Use Horovod for distributed training
-  `--dataset_dir`                                   Path to dataset. If dataset doesn't exist, it will be downloaded (default: `./dataset/`)
-  `--use_hooks`                                     Whether to use hooks during training. If used, stores value as True
-  `--generator_lr`                                  Generator learning rate (default: 4e-4)
-  `--discriminator_lr`                              Discriminator learning rate (default: 2e-4)
-  `--save_freq`                                     How often save model (default: 200)
-  `--cosine_decay_delay`                            After how many epoch start decaying learning rates. Needs to be smaller than the number of epochs (default: 100)

## Supported Configuration

| Device | SynapseAI Version | TensorFlow Version(s)  |
|:------:|:-----------------:|:-----:|
| Gaudi  | 1.7.0             | 2.10.0 |
| Gaudi  | 1.7.0             | 2.8.3 |

## Changelog

### 1.7.0

* Added TimeToTrain callback for dumping evaluation timestamps.
* Changed default save_freq argument to 200 to match other models when saving after a full run.

### 1.4.0

* Added support to import horovod-fork package directly instead of using Model-References' TensorFlow.common.horovod_helpers. Wrapped horovod import with a try-catch block so that installing the library is not required when the model is running on a single card.
* Replaced references to custom demo script by community entry points in README.

### 1.3.0

* Updated requirements.txt.

### 1.2.0

* Changed Learning Rate scaling from linear to sqrt to improve training convergence.
* Changed Initialization changed to improve convergence.
* Added optional warmup and gradient clipping.
* Added support for InstanceNorm kernel to improve performance.
* Updated requirements.txt.

## Known Issues

* Sporadic training divergence on multi-cards run (1/20 runs).
