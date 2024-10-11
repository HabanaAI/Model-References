# YOLOX for PyTorch
This repository provides scripts to train YOLOX model on Intel® Gaudi® AI accelerator to achieve state-of-the-art
accuracy. To obtain model performance data, refer to the [Intel Gaudi Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).
For more information about training deep learning models using Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/). Before you get started, make sure to review the [Supported Configurations](#supported-configurations).

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
Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) 
to set up the environment including the `$PYTHON` environment variable. To achieve the best performance, please follow the methods outlined in the [Optimizing Training Platform Guide](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html).
The guides will walk you through the process of setting up your system to run the model on Gaudi.  

### Clone Intel Gaudi Model-References
In the docker container, clone this repository and switch to the branch that
matches your Intel Gaudi software version. You can run the
[`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the Intel Gaudi software version

```bash
git clone -b [Intel Gaudi software version] https://github.com/HabanaAI/Model-References
```

Go to PyTorch YOLOX directory:
```bash
cd Model-References/PyTorch/computer_vision/detection/yolox
```

### Install Model Requirements
Install the required packages and add current directory to PYTHONPATH:

```bash
pip install -r requirements.txt
pip install -v -e .
export PYTHONPATH=$PWD:$PYTHONPATH
```

### Setting up the Dataset
Download COCO 2017 dataset from http://cocodataset.org using the following commands:

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
**NOTE:** YOLOX only supports Lazy mode.

**Run training on 1 HPU:**
* FP32 data type, train for 500 steps:
    ```bash
    $PYTHON tools/train.py \
        --name yolox-s --devices 1 --batch-size 16 --data_dir /data/COCO --hpu steps 500 output_dir ./yolox_output
    ```

* BF16 data type. train for 500 steps:
    ```bash
    PT_HPU_AUTOCAST_LOWER_PRECISION_OPS_LIST=ops_bf16_yolox.txt PT_HPU_AUTOCAST_FP32_OPS_LIST=ops_fp32_yolox.txt $PYTHON tools/train.py \
        --name yolox-s --devices 1 --batch-size 16 --data_dir /data/COCO --hpu --autocast \
        steps 500 output_dir ./yolox_output
    ```

**Run training on 8 HPUs:**

**NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Scaling_Guide/DDP_Based_Scaling.html#mpirun-configuration).

* FP32 data type, train for 2 epochs:
    ```bash
    export MASTER_ADDR=localhost
    export MASTER_PORT=12355
    mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
    $PYTHON tools/train.py \
        --name yolox-s --devices 8 --batch-size 128 --data_dir /data/COCO --hpu max_epoch 2 output_dir ./yolox_output
    ```

* BF16 data type. train for 2 epochs:
    ```bash
    export MASTER_ADDR=localhost
    export MASTER_PORT=12355
    PT_HPU_AUTOCAST_LOWER_PRECISION_OPS_LIST=ops_bf16_yolox.txt PT_HPU_AUTOCAST_FP32_OPS_LIST=ops_fp32_yolox.txt mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
    $PYTHON tools/train.py \
        --name yolox-s --devices 8 --batch-size 128 --data_dir /data/COCO --hpu --autocast\
        max_epoch 2 output_dir ./yolox_output
    ```

* BF16 data type, train for 300 epochs:
    ```bash
    export MASTER_ADDR=localhost
    export MASTER_PORT=12355
    PT_HPU_AUTOCAST_LOWER_PRECISION_OPS_LIST=ops_bf16_yolox.txt PT_HPU_AUTOCAST_FP32_OPS_LIST=ops_fp32_yolox.txt mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
    $PYTHON tools/train.py \
        --name yolox-s --devices 8 --batch-size 128 --data_dir /data/COCO --hpu --autocast \
        print_interval 100 max_epoch 300 save_history_ckpt False eval_interval 300 output_dir ./yolox_output
    ```

# Validation examples
### Run Single Card and Multi-Card Validation Examples

**Pretrained model:** you can get one on [this page](https://github.com/Megvii-BaseDetection/YOLOX?tab=readme-ov-file#standard-models). For example, you can use next command to download **pretrained yolox-s** model:
```bash
curl -L -O https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
```

**Run validation on 1 HPU:**
* FP32 data type:
    ```bash
    $PYTHON tools/eval.py -n yolox-s -c path/to/yolox_s.pth --data_dir path/to/data/COCO -b 256 -d 1 --conf 0.001 --data_num_workers 4 --hpu --fuse --cpu-post-processing --warmup_steps 4
    ```

* BF16 data type:
    ```bash
    PT_HPU_AUTOCAST_LOWER_PRECISION_OPS_LIST=ops_bf16_yolox.txt PT_HPU_AUTOCAST_FP32_OPS_LIST=ops_fp32_yolox.txt \
    $PYTHON tools/eval.py -n yolox-s -c path/to/yolox_s.pth --data_dir path/to/data/COCO -b 256 -d 1 --conf 0.001 --hpu --autocast --fuse --cpu-post-processing --warmup_steps 4
    ```

**Run validation on 2 HPUs:**

**NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Scaling_Guide/DDP_Based_Scaling.html#mpirun-configuration).

* FP32 data type:
    ```bash
    export MASTER_ADDR=localhost
    export MASTER_PORT=12355
    mpirun -n 2 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
    $PYTHON tools/eval.py -n yolox-s -c path/to/yolox_s.pth --data_dir path/to/data/COCO -b 1024 -d 2 --conf 0.001 --hpu --fuse --cpu-post-processing --warmup_steps 2
    ```

* BF16 data type:
    ```bash
    export MASTER_ADDR=localhost
    export MASTER_PORT=12355
    PT_HPU_AUTOCAST_LOWER_PRECISION_OPS_LIST=ops_bf16_yolox.txt PT_HPU_AUTOCAST_FP32_OPS_LIST=ops_fp32_yolox.txt \
    mpirun -n 2 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
    $PYTHON tools/eval.py -n yolox-s -c path/to/yolox_s.pth --data_dir path/to/data/COCO -b 1024 -d 2 --conf 0.001 --hpu --autocast --fuse --cpu-post-processing --warmup_steps 2
    ```

# Supported Configurations
| Device | Intel Gaudi Software Version | PyTorch Version |
|--------|------------------------------|-----------------|
| Gaudi  | 1.17.1                       | 2.3.1          |

## Changelog
### 1.12.0
* Removed PT_HPU_LAZY_MODE environment variable.
* Removed flag use_lazy_mode.
* Removed HMP data type.
* Updated run commands which allows for overriding the default lower precision and FP32 lists of ops.

### 1.10.0
* Enabled mixed precision training using PyTorch autocast on Gaudi.
### Training Script Modifications
The following are the changes made to the training scripts:

* Added source code to enable training on CPU.
* Added source code to support Gaudi devices.

   * Enabled HMP data type.

   * Added support to run training in Lazy mode.

   * Re-implemented loss function with TorchScript and deployed the function to CPU.

   * Enabled distributed training with HCCL backend on 8 HPUs.

   * mark_step() is called to trigger execution.

## Known Issues
Eager mode is not supported.
