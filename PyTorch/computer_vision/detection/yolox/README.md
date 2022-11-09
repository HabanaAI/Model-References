# YOLOX for PyTorch
This repository provides scripts to train YOLOX model on Habana Gaudi device to achieve state-of-the-art
accuracy. To obtain model performance data, refer to the [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).
For more information about training deep learning models using Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

The YOLOX demo included in this release is YOLOX-S in lazy mode training for different batch sizes with
FP32 and BF16 mixed precision.

## Table of Contents
- [Model-References](../../../../README.md)
- [Model Overview](#model-overview)
- [Setup](#setup)
- [Training Examples](#training-examples)
- [Supported Configurations](#supported-configurations)
- [Changelog](#changelog)
- [Known Issues](#known-issues)


## Model Overview

YOLOX is an anchor-free object detector that adopts the architecture of YOLO with DarkNet53 backbone.
The anchor-free mechanism greatly reduces the number of model parameters and therefore simplifies the
detector. Additionally, YOLOX also provides improvements to the previous YOLO series such as decoupled head,
advanced label assigning strategy, and strong data augmentation. The decoupled head contains a 1x1 conv
layer, followed by two parallel branches with two 3x3 conv layers for classification and regression tasks
respectively, which helps the model converge faster with better accuracy. The advanced label assignment,
SimOTA, selects the top k predictions with the lowest cost as the positive samples for a ground truth object.
SimOTA not only reduces training time by approximating the assignment instead of using an optimization
algorithm, but also improves AP of the model. Additionally, Mosaic and MixUp image augmentation are applied
to the training process to further improve the accuracy. Equipped with these latest advanced techniques,
YOLOX remarkably achieves a better trade-off between training speed and accuracy than other counterparts
in all model sizes.

This repository is an implementation of PyTorch version YOLOX, based on the source code from [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/MegEngine/YOLOX).
More details can be found in the paper [YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/abs/2107.08430) by Zhen Ge, Songtao Liu,
Feng Wang, Zeming Li, and Jian Sun.


## Setup
Please follow the instructions provided in the [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the
environment including the `$PYTHON` environment variable.
The guide will walk you through the process of setting up your system to run the model on Gaudi.

### Clone Habana Model-References
In the docker container, clone this repository and switch to the branch that
matches your SynapseAI version. You can run the
[`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the SynapseAI version

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```

Go to PyTorch YOLOX directory:
```bash
cd Model-References/PyTorch/computer_vision/detection/yolox
```

### Install Model Requirements
Install the required packages:

```bash
pip install -r requirements.txt
pip install -v -e .
```

### Setting up the Dataset
Download COCO 2017 dataset from http://cocodataset.org with following commands:

```
cd Model-References/PyTorch/computer_vision/detection/yolox
source download_dataset.sh
```

You can either set the dataset location to the `YOLOX_DATADIR` environment variable:

```bash
export YOLOX_DATADIR=/data/COCO
```

Or create a sub-directory, `datasets`, and create a symbolic link from the COCO dataset path to the 'datasets' sub-directory.

```bash
mkdir datasets
ln -s /data/COCO ./datasets/COCO
```

Alternatively, you can pass the COCO dataset location to the `--data_dir` argument of the training commands.

## Training Examples
### Run Single Card and Multi-Card Training Examples
**Run training on 1 HPU:**
* Lazy mode, FP32 data type, train for 500 steps:
    ```bash
    PT_HPU_MAX_COMPOUND_OP_SIZE=100 $PYTHON tools/train.py \
        --name yolox-s --batch-size 16 --data_dir /data/COCO --hpu steps 500 output_dir ./yolox_output
    ```

* Lazy mode, BF16 data type. train for 500 steps:
    ```bash
    PT_HPU_MAX_COMPOUND_OP_SIZE=100 $PYTHON tools/train.py \
        --name yolox-s --batch-size 16 --data_dir /data/COCO --hpu --hmp \
        steps 500 output_dir ./yolox_output
    ```

**Run training on 8 HPUs:**
* Lazy mode, FP32 data type, train for 2 epochs:
    ```bash
    PT_HPU_MAX_COMPOUND_OP_SIZE=100 $PYTHON tools/train.py \
        --name yolox-s --devices 8 --batch-size 128 --data_dir /data/COCO --hpu max_epoch 2 output_dir ./yolox_output
    ```

* Lazy mode, BF16 data type. train for 2 epochs:
    ```bash
    PT_HPU_MAX_COMPOUND_OP_SIZE=100 $PYTHON tools/train.py \
        --name yolox-s --devices 8 --batch-size 128 --data_dir /data/COCO --hpu --hmp \
        max_epoch 2 output_dir ./yolox_output
    ```

* Lazy mode, BF16 data type, train for 300 epochs:
    ```bash
    PT_HPU_MAX_COMPOUND_OP_SIZE=100 $PYTHON tools/train.py \
        --name yolox-s --devices 8 --batch-size 128 --data_dir /data/COCO --hpu --hmp \
        print_interval 100 max_epoch 300 save_history_ckpt False eval_interval 300 output_dir ./yolox_output
    ```
# Supported Configurations
| Device | SynapseAI Version | PyTorch Version |
|--------|-------------------|-----------------|
| Gaudi  | 1.6.1             | 1.12.0          |

## Changelog
### Training Script Modifications
The following are the changes made to the training scripts:

1. Added source code to enable training on CPU.
2. Added source code to support Habana devices.

   1) Enabled Habana Mixed Precision (hmp) data type.

   2) Added support to run training in lazy mode.

   3) Re-implemented loss function with TorchScript and deployed the function to CPU

   4) Enabled distributed training with Habana HCCL backend on 8 HPUs.

   5) mark_step() is called to trigger execution.

## Known Issues
Eager mode is not supported.