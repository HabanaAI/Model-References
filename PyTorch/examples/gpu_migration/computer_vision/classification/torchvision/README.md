# ResNet50 for PyTorch with GPU Migration
This folder contains scripts to train ResNet50 model on Intel® Gaudi® AI accelerator to achieve state-of-the-art accuracy.
For more information on training and inference of deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).
For model performance data, refer to the [Intel Gaudi Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

## Table of Contents
  - [Model-References](../../../../../../README.md)
  - [Model Overview](#model-overview)
  - [Setup](#setup)
  - [Media Loading Acceleration](#media-loading-acceleration)
  - [Training and Examples](#training-and-examples)
  - [Known Issues](#known-issues)
  - [Changelog](#changelog)
  - [Enabling the Model from Scratch](#enabling-the-model-from-scratch)
  - [GPU Migration Logs](#gpu-migration-logs)

## Model Overview
The model has been enabled using an experimental feature called GPU migration, in addition to another [ResNet50 model](../../../../../computer_vision/classification/torchvision/README.md) enabled with a more traditional approach.

## Setup
Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the environment including the `$PYTHON` environment variable.
The guide will walk you through the process of setting up your system to run the model on Gaudi.

### Clone Intel Gaudi Model-References
In the docker container, clone this repository and switch to the branch that matches your Intel Gaudi software version.
You can run the [`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the Intel Gaudi software version.

```bash
git clone -b [Intel Gaudi software version] https://github.com/HabanaAI/Model-References
cd Model-References/PyTorch/examples/gpu_migration/computer_vision/classification/torchvision
```

### Setting up the Dataset
ImageNet 2012 dataset needs to be organized according to PyTorch requirements, and as specified in the scripts of [imagenet-multiGPU.torch](https://github.com/soumith/imagenet-multiGPU.torch).

## Media Loading Acceleration
Gaudi 2 offers a dedicated hardware engine for Media Loading operations.
For further details, please refer to [Intel Gaudi Media Loader](https://docs.habana.ai/en/latest/PyTorch/Reference/Using_Media_Loader_with_PyTorch/Media_Loader_PT.html).

### Enabling GPU Migration Toolkit
GPU Migration Toolkit can be enabled with PT_HPU_GPU_MIGRATION=1 flag
(by default PT_HPU_GPU_MIGRATION=0).
```bash
export PT_HPU_GPU_MIGRATION=1
```

## Training and Examples

The following commands assume that ImageNet dataset is available at `/data/pytorch/imagenet/ILSVRC2012/` directory.

- To see the available training parameters, run:
```bash
$PYTHON -u train.py --help
```
### Conversion from Float16 to Bfloat16 data type

HPUs prefer usage of BFloat16 over Float16 data type for models training/inference. To enable automatic conversion from Float16 to Bfloat16 data type, use PT_HPU_CONVERT_FP16_TO_BF16_FOR_MIGRATION=1 flag (by default PT_HPU_CONVERT_FP16_TO_BF16_FOR_MIGRATION=0). For example:
```bash
PT_HPU_CONVERT_FP16_TO_BF16_FOR_MIGRATION=1 PT_HPU_LAZY_MODE=0 $PYTHON train.py \ 
--batch-size=256 --model=resnet50 --device=cuda --data-path=/data/pytorch/imagenet/ILSVRC2012 \ 
--workers=8 --epochs=90 --opt=sgd --amp --use-torch-compile 
```

### Single Card and Multi-Card Training Examples

**Run training on 1 HPU with torch.compile:**

Run training on 1 HPU, torch.compile mode, batch size 256, 90 epochs, SGD optimizer, mixed precision (BF16):
```bash
PT_HPU_CONVERT_FP16_TO_BF16_FOR_MIGRATION=1 PT_HPU_LAZY_MODE=0 $PYTHON train.py \ 
--batch-size=256 --model=resnet50 --device=cuda --data-path=/data/pytorch/imagenet/ILSVRC2012 \ 
--workers=8 --epochs=90 --opt=sgd --amp --use-torch-compile
```

**Run training on 8 HPUs with torch.compile:**

To run multi-card training, make sure the host machine has 512 GB of RAM installed.
Also, ensure you followed the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) to install and set up docker, so that the docker has access to all eight Gaudi cards required for multi-card training.

Run training on 8 HPUs, torch.compile mode, batch size 256, 90 epochs, SGD optimizer, mixed precision (BF16):
```bash
PT_HPU_CONVERT_FP16_TO_BF16_FOR_MIGRATION=1 PT_HPU_LAZY_MODE=0 torchrun --nproc_per_node 8 train.py \ 
--batch-size=256 --model=resnet50 --device=cuda --data-path=/data/pytorch/imagenet/ILSVRC2012 \ 
--workers=8 --epochs=90 --opt=sgd --amp --use-torch-compile
```

## Known Issues
* The accuracy on Gaudi 2 is slightly lower due to the use of a different data loader.
* Training on Ubuntu22.04 results in segmentation fault. To mitigate that, remove TcMalloc from LD_PRELOAD env variable before running the workload.

## Changelog
### 1.20.0
* Changed bash examples to use torch.compile
### 1.17.0
* Replaced `import habana_frameworks.torch.gpu_migration` with PT_HPU_GPU_MIGRATION environment variable.
### 1.13.0
* Added experimental torch.compile feature support.
### 1.10.0
* Removed PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES environment variable.
### 1.9.0
Major changes done to the original model from [the original GitHub repository](https://github.com/pytorch/vision/tree/900982fccb88d1220cac5b1dad9ae37dd7554f2e/references/classification) repository:
* Changed README.
* Added `import habana_frameworks.torch.gpu_migration`.
* Added `mark_steps()`.
* Added performance optimizations.
* Added custom learning rate scheduler.

## Enabling the Model from Scratch
Intel Gaudi provides scripts ready-to-use on Gaudi.
Listed below are the steps to enable the model from a reference source.

This section outlines the overall procedure for enabling any given model with GPU migration feature.
However, model-specific modifications will be required to enable the functionality and improve performance.

1. Clone the original GitHub repository and reset it to the commit this example is based on:
```bash
git clone https://github.com/pytorch/vision.git && cd vision/references/classification && git reset --hard 900982fccb88d1220cac5b1dad9ae37dd7554f2e
```

2. Apply a set of patches.
You can stop at any patch to see which steps have been performed to reach a particular level of functionality and performance.

The first patch adds the bare minimum to run the model on HPU. For purely functional changes (without performance optimization), run the following command:
```bash
git apply Model-References/PyTorch/examples/gpu_migration/computer_vision/classification/torchvision/patches/minimal_changes.diff
```

*Note:* If you encounter issues with shared memory in this example, you can resolve them by either reducing the batch size or by applying the next patch which modifies the data loader to handle this problem.

3. To improve performance, apply the following patch:
```bash
git apply Model-References/PyTorch/examples/gpu_migration/computer_vision/classification/torchvision/patches/performance_improvements.diff
```

4. To achieve state-of-the-art accuracy, change the learning rate scheduler by running the command below:
```bash
git apply Model-References/PyTorch/examples/gpu_migration/computer_vision/classification/torchvision/patches/lr_scheduler.diff
```

To start training with any of these patches, refer to the commands from the [Single Card and Multi-Card Training Examples](#single-card-and-multi-card-training-examples) section.

## GPU Migration Logs
You can review GPU Migration logs under [gpu_migration_logs/gpu_migration_958.log](gpu_migration_logs/gpu_migration_958.log).
For further information, refer to [GPU Migration Toolkit documentation](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Model_Porting/GPU_Migration_Toolkit/GPU_Migration_Toolkit.html#enabling-logging-feature).
