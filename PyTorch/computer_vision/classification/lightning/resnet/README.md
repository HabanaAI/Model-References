# ResNet50 for PyTorch Lightning  

This directory provides a script to train a "PyTorch Lightning ResNet50" to achieve state-of-the-art accuracy, and is tested and maintained by Habana. To obtain model performance data, refer to the
[Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

For more information about training deep learning models using Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents
  * [Model-References](../../../../../README.md)
  * [Setup](#setup)
  * [Training Examples ](#training-examples)
  * [Supported Configurations](#supported-configurations)
  * [Changelog](#changelog)

## Setup
Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) 
to set up the environment including the `$PYTHON` environment variable. To achieve the best performance, please follow the methods outlined in the [Optimizing Training Platform guide](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html).
The guides will walk you through the process of setting up your system to run the model on Gaudi.  

### Clone Habana Model-References
In the docker container, clone this repository and switch to the branch that
matches your SynapseAI version. You can run the
[`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.
```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```
### Install Model Requirements
In the docker container, go to the resnet directory:
```bash
cd /root/Model-References/PyTorch/computer_vision/clasification/lightning/resnet/
pip install -r requirements.txt
```

### Training Data

ImageNet 2012 dataset needs to be organized as per PyTorch requirements. PyTorch requirements are specified in the link below which contains scripts to organize ImageNet data.
https://github.com/soumith/imagenet-multiGPU.torch

## Training Examples

**Note:** It is assumed that the working directory is `Model-References/PyTorch/computer_vision/clasification/lightning/resnet/`.

**Run training with single HPU:**
  ```python
  $PYTHON resnet50_PTL.py --data_path /data/pytorch/datasets/imagenet/ILSVRC2012/ --epochs 4 --print_freq 1 --max_train_batches 200 --hpu 1 \
    --autocast --custom_lr_values 0.1 0.01 0.001 0.0001 --custom_lr_milestones 0 30 60 80 
  ```
**Run training with single HPU in benchmark mode:**
  ```python
  $PYTHON resnet50_PTL.py --data_path /data/pytorch/datasets/imagenet/ILSVRC2012/ --epochs 4 --benchmark --max_train_batches 200 --hpu 1 \
    --autocast --custom_lr_values 0.1 0.01 0.001 0.0001 --custom_lr_milestones 0 30 60 80 
  ```
**Run training with 8 HPUs:**
  ```python
  $PYTHON resnet50_PTL.py --data_path /data/pytorch/datasets/imagenet/ILSVRC2012/ --epochs 4 --print_freq 1 --max_train_batches 200 --hpu 8 \
    --autocast --custom_lr_values 0.275 0.45 0.625 0.8 0.08 0.008 0.0008 --custom_lr_milestones 1 2 3 4 30 60 80
  ```
**Run training with 8 HPUs in benchmark mode:**
  ```python
  $PYTHON resnet50_PTL.py --data_path /data/pytorch/datasets/imagenet/ILSVRC2012/ --epochs 4 --benchmark --max_train_batches 200 --hpu 8 \
    --autocast --custom_lr_values 0.275 0.45 0.625 0.8 0.08 0.008 0.0008 --custom_lr_milestones 1 2 3 4 30 60 80
  ```
## Supported Configurations

| Validated on | SynapseAI Version | PyTorch Lightning Version | Lightning Habana Version | Mode |
|-----|-----|-----|-----|-----|
| Gaudi | 1.11.0 | 2.0.7 | 1.0.1 | Training |

## Changelog
### 1.7.0
 - Added support for lazy mode.
 - Added support for BF16 mixed precision.
### 1.8.0
 - Added support for HABANA data loader.
### 1.9.0
 - Added work around to disable dynamic shape support for HPU  to mitigate performance issues.
### 1.11.0
 - Dynamic Shapes will be enabled by default in future releases. It is currently disabled in training script.
### 1.12.0
 - Increased bucket_cap_mb size as torch dynamo need the first_bucket cap to be smaller than bucket_bytes_cap.
 - Deprecated support for HMP
 - Upgraded to be supported with Lightning 2.0.7 and lightning-Habana plugin 1.0.1
 - Eager mode support is deprecated.