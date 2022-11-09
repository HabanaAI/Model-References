# UNet2D and UNet3D for PyTorch and PyTorch Lightning

This directory provides a script and recipe to train the UNet2D and UNet3D models to achieve state of the art accuracy, and is tested and maintained by Habana. For further information on performance, refer to [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

For further information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources).

## Table of Contents
* [Model-References](../../../../README.md)
* [Model Overview](#model-overview)
* [Setup](#setup)
* [Media Loading Acceleration](#media-loading-acceleration)
* [Training Examples](#training-examples)
* [Advanced](#advanced)
* [Supported Configurations](#supported-configurations)
* [Changelog](#changelog)
* [Known Issues](#known-issues)

## Model Overview

The supported UNet2D and UNet3D are based on PyTorch and PyTorch Lightning. The PyTorch Lightning implementations are based on an earlier implementation from [NVIDIA's nnUNet](https://github.com/NVIDIA/DeepLearningExamples/tree/2b20ca80cf7f08585e90a11c5b025fa42e4866c8/PyTorch/Segmentation/nnUNet).
Habana accelerator support is enabled with PyTorch Lightning version 1.7.7, which is installed along with the release dockers. For further details on the changes applied to the original model, refer to [Training Script Modifications](#training-script-modifications).

The following are the demos included in this release:
- For UNet2D, Eager mode and Lazy mode training for BS64 with FP32 & BF16 mixed precision.
- For UNet3D, Eager mode and Lazy mode training for BS2 with FP32 & BF16 mixed precision.

## Setup

Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) to set up the environment including the `$PYTHON` environment variable. The guide will walk you through the process of setting up your system to run the model on Gaudi.

### Clone Habana Model-References

In the docker container, clone this repository and switch to the branch that matches your SynapseAI version. You can run the [`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the SynapseAI version.

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```
**NOTE:** If the repository is not in the PYTHONPATH, make sure you update it:
```bash
export PYTHONPATH=/path/to/Model-References:$PYTHONPATH
```

### Install Model Requirements

1. Go to PyTorch UNet directory:
```bash
cd Model-References/PyTorch/computer_vision/segmentation/Unet
```

2. Install the required packages:
```bash
pip install -r ./requirements.txt
```

### Download BraTS Dataset

1. Create a /data directory if not present:
```bash
mkdir /data
```
2. Download the dataset:
```python
$PYTHON download.py --task 01
```
**NOTE:** The script downloads the dataset in /data directory by default.

### Pre-process the Dataset

- To pre-process the dataset for UNet2D, run:

```python
$PYTHON preprocess.py --task 01 --dim 2
```

- To process the dataset for UNet3D, run:

```python
$PYTHON preprocess.py --task 01 --dim 3
```

**NOTE:** The script pre-processes the dataset downloaded in the above steps from `/data` directory and creates `01_2d` directory for UNet2D and `01_3d` directory for UNet3D model inside `/data` directory. Consequently, the dataset is available at `/data/pytorch/unet/01_2d` directory for UNet2D and `/data/pytorch/unet/01_3d` directory for UNet3D.

## Media Loading Acceleration
Gaudi2 offers a dedicated hardware engine for Media Loading operations. For more details, please refer to [Habana Media Loader page](https://docs.habana.ai/en/latest/PyTorch/Habana_Media_Loader_PT/Media_Loader_PT.html)

## Training Examples

**NOTE:** The training examples are applicable for first-gen Gaudi and **Gaudi2**

### Single Card and Multi-Card Training Examples

```bash
mkdir -p /tmp/Unet/results/fold_0
```

**Run training on 1 HPU:**

**NOTE:** The following commands use PyTorch Lightning by default. To use PyTorch, add '--framework pytorch' to the run command.

- UNet2D in lazy mode, BF16 mixed precision, batch size 64, fold 0:

```bash
$PYTHON -u  main.py --results /tmp/Unet/results/fold_0 --task 01 \
        --logname res_log --fold 0 --hpus 1 --gpus 0 --data /data/pytorch/unet/01_2d \
        --seed 1 --num_workers 8 --affinity disabled --norm instance --dim 2 \
        --optimizer fusedadamw --exec_mode train --learning_rate 0.001 --hmp \
        --hmp-bf16 ./config/ops_bf16_unet.txt --hmp-fp32 ./config/ops_fp32_unet.txt \
        --deep_supervision --batch_size 64 --val_batch_size 64
```
- UNet2D in lazy mode, BF16 mixed precision, batch size 64, fold 0 (with hardware decode support on **Gaudi2**)::

```bash
$PYTHON -u  main.py --results /tmp/Unet/results/fold_0 --task 01 \
        --logname res_log --fold 0 --hpus 1 --gpus 0 --data /data/pytorch/unet/01_2d \
        --seed 1 --num_workers 8 --affinity disabled --norm instance --dim 2 \
        --optimizer fusedadamw --exec_mode train --learning_rate 0.001 --hmp \
        --hmp-bf16 ./config/ops_bf16_unet.txt --hmp-fp32 ./config/ops_fp32_unet.txt \
        --deep_supervision --batch_size 64 --val_batch_size 64 \
        --habana_loader
```
- UNet2D in eager mode, BF16 mixed precision, batch size 64, fold 0:

```bash
$PYTHON -u  main.py --results /tmp/Unet/results/fold_0 --task 01 \
        --logname res_log --fold 0 --hpus 1 --gpus 0 \
        --data /data/pytorch/unet/01_2d --seed 1 --num_workers 8 --affinity disabled \
        --norm instance --dim 2 --optimizer fusedadamw --exec_mode train \
        --learning_rate 0.001 --hmp --hmp-bf16 ./config/ops_bf16_unet.txt \
        --hmp-fp32 ./config/ops_fp32_unet.txt --deep_supervision --batch_size 64 \
        --val_batch_size 64 --run-lazy-mode False
```
- UNet2D in eager mode, FP32 precision, batch size 64, fold 2:

```bash
$PYTHON -u  main.py --results /tmp/Unet/results/fold_0 --task 01 \
        --logname res_log --fold 2 --hpus 1 --gpus 0 \
        --data /data/pytorch/unet/01_2d --seed 1 --num_workers 8 --affinity disabled \
        --norm instance --dim 2 --optimizer fusedadamw --exec_mode train \
        --learning_rate 0.001 --deep_supervision --batch_size 64 --val_batch_size 64 --run-lazy-mode False
```
- UNet2D in lazy mode, BF16 mixed precision, batch size 64, benchmarking:

```bash
$PYTHON -u main.py --results /tmp/Unet/results/fold_0 --task 1 --logname res_log \
        --fold 0 --hpus 1 --gpus 0 --data /data/pytorch/unet/01_2d --seed 123 \
        --num_workers 1 --affinity disabled --norm instance --dim 2 --optimizer fusedadamw \
        --exec_mode train --learning_rate 0.001 --hmp --hmp-bf16 ./config/ops_bf16_unet.txt \
        --hmp-fp32 ./config/ops_fp32_unet.txt --batch_size 64 \
        --val_batch_size 64 --benchmark --min_epochs 1 --max_epochs 2  --train_batches 150 --test_batches 150
```

- UNet3D in lazy mode, BF16 mixed precision, batch size 2, fold 0:

```bash
$PYTHON -u main.py --results /tmp/Unet/results/fold_0 --task 01 --logname res_log \
        --fold 0 --hpus 1 --gpus 0 --data /data/pytorch/unet/01_3d --seed 1 --num_workers 8 \
        --affinity disabled --norm instance --dim 3 --optimizer fusedadamw \
        --exec_mode train --learning_rate 0.001  --hmp --hmp-bf16 ./config/ops_bf16_unet.txt \
        --hmp-fp32 ./config/ops_fp32_unet.txt --deep_supervision --batch_size 2 --val_batch_size 2
```

- UNet3D in lazy mode, BF16 mixed precision, batch size 2, fold 0 (with hardware decode support on **Gaudi2**):

```bash
$PYTHON -u main.py --results /tmp/Unet/results/fold_0 --task 01 --logname res_log \
        --fold 0 --hpus 1 --gpus 0 --data /data/pytorch/unet/01_3d --seed 1 --num_workers 8 \
        --affinity disabled --norm instance --dim 3 --optimizer fusedadamw \
        --exec_mode train --learning_rate 0.001  --hmp --hmp-bf16 ./config/ops_bf16_unet.txt \
        --hmp-fp32 ./config/ops_fp32_unet.txt --deep_supervision --batch_size 2 --val_batch_size 2 \
        --habana_loader
```
- UNet3D in lazy mode, BF16 mixed precision, batch size 2, benchmarking:

```bash
$PYTHON -u main.py --results /tmp/Unet/results/fold_0 --task 1 --logname res_log \
--fold 0 --hpus 1 --gpus 0 --data /data/pytorch/unet/01_3d --seed 1 --num_workers 1 \
--affinity disabled --norm instance --dim 3 --optimizer fusedadamw \
--exec_mode train --learning_rate 0.001 --hmp --hmp-bf16 ./config/ops_bf16_unet.txt \
--hmp-fp32 ./config/ops_fp32_unet.txt --batch_size 2 \
--val_batch_size 2 --benchmark --min_epochs 1 --max_epochs 2  --train_batches 150 --test_batches 150
```

**Run traning on 8 HPUs:**

**NOTE:** UNet2D and UNet3D for multicard is supported only through PyTorch Lightning. Support for PyTorch based scripts will be available in subsequent releases.

To run multi-card demo, make sure to set the following prior to the training:
- The host machine has 512 GB of RAM installed.
- The docker is installed and set up as per the [Gaudi Setup and Installation Guide](https://github.com/HabanaAI/Setup_and_Install), so that the docker has access to all 8 cards required for multi-card demo. Multi-card configuration for UNet2D and UNet3D training up to 1 server, with 8 Gaudi/**Gaudi2** cards, has been verified.
- All server network interfaces are up. You can change the state of each network interface managed by the habanalabs driver by running the following command:
   ```
   sudo ip link set <interface_name> up
   ```
**NOTE:** To identify if a specific network interface is managed by the habanalabs driver type, run:
   ```
   sudo ethtool -i <interface_name>
   ```
- UNet2D in lazy mode, BF16 mixed precision, batch size 64, world-size 8, fold 0:
```bash
$PYTHON -u main.py --results /tmp/Unet/results/fold_0 --task 1 --logname res_log \
        --fold 0 --hpus 8 --gpus 0 --data /data/pytorch/unet/01_2d --seed 123 --num_workers 8 \
        --affinity disabled --norm instance --dim 2 --optimizer fusedadamw --exec_mode train \
        --learning_rate 0.001 --hmp --hmp-bf16 ./config/ops_bf16_unet.txt \
        --hmp-fp32 ./config/ops_fp32_unet.txt --deep_supervision --batch_size 64 \
        --val_batch_size 64 --min_epochs 30 --max_epochs 10000 --train_batches 0 --test_batches 0
```
- UNet2D in lazy mode, BF16 mixed precision, batch size 64, world-size 8, fold 0 (with hardware decode support on **Gaudi2**):
```bash
$PYTHON -u main.py --results /tmp/Unet/results/fold_0 --task 1 --logname res_log \
        --fold 0 --hpus 8 --gpus 0 --data /data/pytorch/unet/01_2d --seed 123 --num_workers 8 \
        --affinity disabled --norm instance --dim 2 --optimizer fusedadamw --exec_mode train \
        --learning_rate 0.001 --hmp --hmp-bf16 ./config/ops_bf16_unet.txt \
        --hmp-fp32 ./config/ops_fp32_unet.txt --deep_supervision --batch_size 64 \
        --val_batch_size 64 --min_epochs 30 --max_epochs 10000 --train_batches 0 --test_batches 0 \
        --habana_loader
```
- UNet2D in eager mode, BF16 mixed precision, batch size 64, world-size 8, fold 0:
```bash
$PYTHON -u  main.py --results /tmp/Unet/results/fold_0 --task 01 --logname res_log \
        --fold 0 --hpus 8 --gpus 0 --data /data/pytorch/unet/01_2d --seed 1 --num_workers 8 \
        --affinity disabled --norm instance --dim 2 --optimizer fusedadamw --exec_mode train \
        --learning_rate 0.001 --hmp --hmp-bf16 ./config/ops_bf16_unet.txt \
        --hmp-fp32 ./config/ops_fp32_unet.txt --deep_supervision --batch_size 64 \
        --val_batch_size 64 --run-lazy-mode False
```
- UNet2D in lazy mode, BF16 mixed precision, batch size 64, world-size 8, benchmarking:
```bash
$PYTHON -u main.py --results /tmp/Unet/results/fold_0 --task 1 --logname res_log \
        --fold 0 --hpus 8 --gpus 0 --data /data/pytorch/unet/01_2d --seed 123 --num_workers 1 \
        --affinity disabled --norm instance --dim 2 --optimizer fusedadamw --exec_mode train \
        --learning_rate 0.001 --hmp --hmp-bf16 ./config/ops_bf16_unet.txt \
        --hmp-fp32 ./config/ops_fp32_unet.txt --batch_size 64 \
        --val_batch_size 64  --benchmark --min_epochs 1 --max_epochs 2 --train_batches 150 --test_batches 150
```
- UNet3D in Lazy mode, bf16 mixed precision, Batch Size 2, world-size 8
```bash
$PYTHON -u main.py --results /tmp/Unet/results/fold_0 --task 01 --logname res_log \
        --fold 0 --hpus 8 --gpus 0 --data /data/pytorch/unet/01_3d --seed 1 --num_workers 8 \
        --affinity disabled --norm instance --dim 3 --optimizer fusedadamw --exec_mode train \
        --learning_rate 0.001 --hmp --hmp-bf16 ./config/ops_bf16_unet.txt \
        --hmp-fp32 ./config/ops_fp32_unet.txt --deep_supervision --batch_size 2 --val_batch_size 2
```
- UNet3D in Lazy mode, bf16 mixed precision, Batch Size 2, world-size 8  (with hardware decode support on **Gaudi2**)
```bash
$PYTHON -u main.py --results /tmp/Unet/results/fold_0 --task 01 --logname res_log \
        --fold 0 --hpus 8 --gpus 0 --data /data/pytorch/unet/01_3d --seed 1 --num_workers 8 \
        --affinity disabled --norm instance --dim 3 --optimizer fusedadamw --exec_mode train \
        --learning_rate 0.001 --hmp --hmp-bf16 ./config/ops_bf16_unet.txt \
        --hmp-fp32 ./config/ops_fp32_unet.txt --deep_supervision --batch_size 2 --val_batch_size 2 \
        --habana_loader
```
- UNet3D in lazy mode, BF16 mixed precision, batch size 2, world-size 8, benchmarking:
```bash
$PYTHON -u main.py --results /tmp/Unet/results/fold_0 --task 1 --logname res_log --fold 0 \
        --hpus 8 --gpus 0 --data /data/pytorch/unet/01_2d --seed 123 --num_workers 1 \
        --affinity disabled --norm instance --dim 3 --optimizer fusedadamw --exec_mode train \
        --learning_rate 0.001 --hmp --hmp-bf16 ./config/ops_bf16_unet.txt \
        --hmp-fp32 ./config/ops_fp32_unet.txt --batch_size 2 \
        --val_batch_size 2  --benchmark --min_epochs 1 --max_epochs 2 --train_batches 150 --test_batches 150
```

## Advanced

### Parameters

To see the available training parameters, run the following command:

```bash
$PYTHON -u main.py --help
```

## Supported Configurations

| Device | SynapseAI Version | PyTorch Lightning Version | PyTorch Version |
|-----|-----|-----|-----|
| Gaudi | 1.7.0 | 1.7.7 | 1.12.0 |
| **Gaudi2** | 1.7.0 | 1.7.7 | 1.12.0 |

## Changelog

### 1.7.0
 - Updated script to make use of TQDM progress bar to override progressbar refresh rate.
 - Upgraded Unet to work with pytorch-lightning 1.7.7.
 - Removed mark_step handling in script as it is taken care in pytorch lightning plugins.
 - Added support for Habana Media Loader with hardware decode support for Gaudi2.
### 1.6.0
 - Added `optimizer_zero_grad` hook and changed `progress_bar_refresh_rate` to improve performance.
 - Added support for 1 and 8 card training on **Gaudi2**.
 - Added PyTorch support (without PyTorch Lightning) for single Gaudi device with a new flag ("--framework pytorch") in the run command.
### 1.5.0
 - Changes done to use vanilla PyTorch Lightning 1.6.4 which includes HPU device support.
 - Removed support for channels last format.
 - Weights and other dependent parameters need not be permuted anymore.
### 1.4.0
 - Default execution mode modified to lazy mode.
### 1.3.0
 - All ops in validation are executed on HPU.
 - Changes to improve time-to-train for UNet3D.
 - Removed support for specifying frequency of validation.
### 1.2.0
 - Bucket size has been increased to 125MB.
 - Enabled HCCL flow for distributed training.

### Training Script Modifications
The following are the changes made to the training scripts:

* Added support for Habana devices:
   - Loading Habana specific library.
   - Certain environment variables are defined for Habana device.
   - Added support to run training in lazy mode in addition to the eager mode.
   - `mark_step()` is performed to trigger execution.
   - Changes to enable scripts on PyTorch Lightning 1.4.0 as base scripts used older version of PyTorch Lightning.
   - Added support to use HPU accelerator plugin, DDP plugin(for multi-card training) and mixed precision plugin provided with the installed PyTorch Lightning package.

* Improved performance:
  - Optimized FusedAdamW operator is used in place of torch.optim.AdamW.
  - Added dice.py with code from monai package and replaced slice with split operator in the forward method.
  - Added monai_sliding_window_inference.py with code from monai package and modified to avoid recomputation of importance map every iteration.
  - Changes to configure the gradient reduction bucket size, set gradients as bucket for all-reduce use static graphs for multi-HPU training.
  - Changed progress_bar_refresh_rate while instantiating Trainer as a workaround for https://github.com/Lightning-AI/lightning/issues/13179.

* Changes to run DALI dataloader on CPU & make data-loading deterministic.
* Metric was copied to `pl_metric.py` from older version of PyTorch Lightning(1.0.4). Implementation in PyTorch Lightning 1.4.0(torch.metric) is different and incompatible.
* PyTorch Lightning metrics is deprecated since PyTorch Lightning 1.3 and suggested to change to torchmetrics. Since `stat_scores` implementation is different and incompatible, older version was copied here from PyTorch Lightning 1.0.
* As a workaround for  https://github.com/NVIDIA/DALI/issues/3865, validation loss is not computed in odd epochs. Other validation metrics are computed every epoch. All metrics are logged only for even epochs.

## Known Issues
- For multi-card traning, the supported UNet2D and UNet3D are based on PyTorch Lightning only. PyTorch based UNet2D and UNet3D (without Lightning) is currently not supported on multiple-card.
- For single-card training, PyTorch based UNet2D and UNet3D (without Lightning) have only been verified on first-gen Gaudi.
- Placing mark_step() arbitrarily may lead to undefined behavior. Recommend to keep mark_step() as shown in provided scripts.
- Only scripts & configurations mentioned in this README are supported and verified.
