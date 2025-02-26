# UNet2D and UNet3D for PyTorch Lightning

This directory provides a script and recipe to train the UNet2D and UNet3D models to achieve state of the art accuracy. It also contains scripts to run inference on the UNet2D and UNet3D models on Intel® Gaudi® AI accelerator. These scripts are tested and maintained by Intel Gaudi. For further information on performance, refer to [Intel Gaudi Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance). Before you get started, make sure to review the [Supported Configurations](#supported-configurations).

For further information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources).

## Table of Contents
* [Model-References](../../../../README.md)
* [Model Overview](#model-overview)
* [Setup](#setup)
* [Media Loading Acceleration](#medialoadingacceleration)
* [Training Examples](#training-examples)
* [Pre-trained Checkpoint](#pre-trained-checkpoint)
* [Inference Examples](#inference-examples)
* [Accuracy Evaluation](#accuracy-evaluation)
* [Advanced](#advanced)
* [Supported Configurations](#supported-configurations)
* [Changelog](#changelog)
* [Known Issues](#known-issues)

## Model Overview

The supported UNet2D and UNet3D are based on PyTorch Lightning. The PyTorch Lightning implementations are based on an earlier implementation from [NVIDIA's nnUNet](https://github.com/NVIDIA/DeepLearningExamples/tree/2b20ca80cf7f08585e90a11c5b025fa42e4866c8/PyTorch/Segmentation/nnUNet).
Gaudi support is enabled with PyTorch Lightning, which is installed along with the release dockers. For further details on the changes applied to the original model, refer to [Training Script Modifications](#training-script-modifications).

The following are the demos included in this release:
- For UNet2D, torch.compile mode training for BS64 with FP32 & BF16 mixed precision.
- For UNet3D, torch.compile mode training for BS2 with FP32 & BF16 mixed precision.
- For UNet2D, inference for BS64 with FP32 & BF16 mixed precision.
- For UNet3D, inference for BS2 with FP32 & BF16 mixed precision.

## Setup

Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) 
to set up the environment including the `$PYTHON` environment variable. To achieve the best performance, please follow the methods outlined in the [Optimizing Training Platform Guide](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html).
The guides will walk you through the process of setting up your system to run the model on Gaudi.  

### Clone Intel Gaudi Model-References

In the docker container, clone this repository and switch to the branch that matches your Intel Gaudi software version. You can run the [`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the Intel Gaudi software version.

```bash
git clone -b [Intel Gaudi software version] https://github.com/HabanaAI/Model-References
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
**On Ubuntu22.04**
```bash
pip install -r ./requirements_u22.txt
```
**On Ubuntu24.04**
```bash
pip install -r ./requirements_u24.txt
```
**On others**
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
$PYTHON preprocess.py --task 01 --dim 2 --results /data/pytorch/unet/
$PYTHON preprocess.py --task 01 --dim 2 --exec_mode val --results /data/pytorch/unet/
$PYTHON preprocess.py --task 01 --dim 2 --exec_mode test --results /data/pytorch/unet/

```

- To process the dataset for UNet3D, run:

```python
$PYTHON preprocess.py --task 01 --dim 3 --results /data/pytorch/unet/
$PYTHON preprocess.py --task 01 --dim 3 --exec_mode val --results /data/pytorch/unet/
$PYTHON preprocess.py --task 01 --dim 3 --exec_mode test --results /data/pytorch/unet/
```

**NOTE:** The script pre-processes the dataset downloaded in the above steps from `/data` directory and based on top of results directory it creates `01_2d` directory for UNet2D and `01_3d` directory for UNet3D model inside `/data` directory. Consequently, the dataset is available at `/data/pytorch/unet/01_2d` directory for UNet2D and `/data/pytorch/unet/01_3d` directory for UNet3D.


## Media Loading Acceleration

Gaudi 2 offers a dedicated hardware engine for Media Loading operations. For more details, please refer to [Intel Gaudi Media Loader](https://docs.habana.ai/en/latest/PyTorch/Reference/Using_Media_Loader_with_PyTorch/Media_Loader_PT.html).

## Training Examples

**NOTE:** The training examples are applicable for first-gen Gaudi and **Gaudi 2** with **torch.compile** mode. When using Eager mode, replace the `--use_torch_compile` with `--run-lazy-mode=False` in the examples below.

### Single Card and Multi-Card Training Examples

```bash
export PT_HPU_LAZY_MODE=0
mkdir -p /tmp/Unet/results/fold_0
```

**Run training on 1 HPU:**

**NOTE:** The following commands use PyTorch Lightning by default. To use media loader on Gaudi 2, add `--habana_loader` to the run commands.

- UNet2D in torch.compile mode, BF16 mixed precision, batch size 64, fold 0:

```bash
$PYTHON -u  main.py --results /tmp/Unet/results/fold_0 --task 01 \
        --logname res_log --fold 0 --hpus 1 --gpus 0 --data /data/pytorch/unet/01_2d \
        --seed 1 --num_workers 8 --affinity disabled --norm instance --dim 2 \
        --optimizer fusedadamw --exec_mode train --learning_rate 0.001 --autocast \
        --deep_supervision --batch_size 64 --val_batch_size 64 --use_torch_compile
```

- UNet2D in torch.compile mode, BF16 mixed precision, batch size 64, benchmarking:

```bash
$PYTHON -u main.py --results /tmp/Unet/results/fold_0 --task 1 --logname res_log \
        --fold 0 --hpus 1 --gpus 0 --data /data/pytorch/unet/01_2d --seed 123 \
        --num_workers 1 --affinity disabled --norm instance --dim 2 --optimizer fusedadamw \
        --exec_mode train --learning_rate 0.001 --autocast --batch_size 64 \
        --val_batch_size 64 --benchmark --min_epochs 1 --max_epochs 2  --train_batches 150 --test_batches 150 --use_torch_compile
```

- UNet3D in torch.compile mode, BF16 mixed precision, batch size 2, fold 0:

```bash
$PYTHON -u main.py --results /tmp/Unet/results/fold_0 --task 01 --logname res_log \
        --fold 0 --hpus 1 --gpus 0 --data /data/pytorch/unet/01_3d --seed 1 --num_workers 8 \
        --affinity disabled --norm instance --dim 3 --optimizer fusedadamw \
        --exec_mode train --learning_rate 0.001  --autocast --deep_supervision --batch_size 2 --val_batch_size 2 --use_torch_compile
```
- UNet3D in torch.compile mode, BF16 mixed precision, batch size 2, benchmarking:

```bash
$PYTHON -u main.py --results /tmp/Unet/results/fold_0 --task 1 --logname res_log \
        --fold 0 --hpus 1 --gpus 0 --data /data/pytorch/unet/01_3d --seed 1 --num_workers 1 \
        --affinity disabled --norm instance --dim 3 --optimizer fusedadamw \
        --exec_mode train --learning_rate 0.001 --autocast --batch_size 2 \
        --val_batch_size 2 --benchmark --min_epochs 1 --max_epochs 2  --train_batches 150 --test_batches 150 --use_torch_compile
```

**Run training on 8 HPUs:**

**NOTE:** The following commands use PyTorch Lightning by default. To use media loader on Gaudi 2, add `--habana_loader` to the run commands.

To run multi-card demo, make sure to set the following prior to the training:
- The host machine has 512 GB of RAM installed.
- The docker is installed and set up as per the [Gaudi Setup and Installation Guide](https://github.com/HabanaAI/Setup_and_Install), so that the docker has access to all 8 cards required for multi-card demo. Multi-card configuration for UNet2D and UNet3D training up to 1 server, with 8 Gaudi/**Gaudi 2** cards, has been verified.
- All server network interfaces are up. You can change the state of each network interface managed by the `habanalabs` driver by running the following command:
   ```
   sudo ip link set <interface_name> up
   ```
**NOTE:** To identify if a specific network interface is managed by the `habanalabs` driver type, run:
   ```
   sudo ethtool -i <interface_name>
   ```
- UNet2D in torch.compile mode, BF16 mixed precision, batch size 64, world-size 8, fold 0:
```bash
$PYTHON -u main.py --results /tmp/Unet/results/fold_0 --task 1 --logname res_log \
        --fold 0 --hpus 8 --gpus 0 --data /data/pytorch/unet/01_2d --seed 123 --num_workers 8 \
        --affinity disabled --norm instance --dim 2 --optimizer fusedadamw --exec_mode train \
        --learning_rate 0.001 --autocast --deep_supervision --batch_size 64 \
        --val_batch_size 64 --min_epochs 30 --max_epochs 10000 --train_batches 0 --test_batches 0 --use_torch_compile
```

- UNet2D in torch.compile mode, BF16 mixed precision, batch size 64, world-size 8, benchmarking:
```bash
$PYTHON -u main.py --results /tmp/Unet/results/fold_0 --task 1 --logname res_log \
        --fold 0 --hpus 8 --gpus 0 --data /data/pytorch/unet/01_2d --seed 123 --num_workers 1 \
        --affinity disabled --norm instance --dim 2 --optimizer fusedadamw --exec_mode train \
        --learning_rate 0.001 --autocast --batch_size 64 \
        --val_batch_size 64  --benchmark --min_epochs 1 --max_epochs 2 --train_batches 150 --test_batches 150 --use_torch_compile
```
- UNet3D in torch.compile mode, BF16 mixed precision, Batch Size 2, world-size 8
```bash
$PYTHON -u main.py --results /tmp/Unet/results/fold_0 --task 01 --logname res_log \
        --fold 0 --hpus 8 --gpus 0 --data /data/pytorch/unet/01_3d --seed 1 --num_workers 8 \
        --affinity disabled --norm instance --dim 3 --optimizer fusedadamw --exec_mode train \
        --learning_rate 0.001 --autocast --deep_supervision --batch_size 2 --val_batch_size 2 --use_torch_compile
```
- UNet3D in torch.compile mode, BF16 mixed precision, batch size 2, world-size 8, benchmarking:
```bash
$PYTHON -u main.py --results /tmp/Unet/results/fold_0 --task 1 --logname res_log --fold 0 \
        --hpus 8 --gpus 0 --data /data/pytorch/unet/01_3d --seed 123 --num_workers 1 \
        --affinity disabled --norm instance --dim 3 --optimizer fusedadamw --exec_mode train \
        --learning_rate 0.001 --autocast --batch_size 2 \
        --val_batch_size 2  --benchmark --min_epochs 1 --max_epochs 2 --train_batches 150 --test_batches 150 --use_torch_compile
```

## Pre-trained Checkpoint

To run the inference example, a pre-trained checkpoint is required. Intel Gaudi provides UNet2D and UNet3D checkpoints pre-trained on Gaudi.
For example, the relevant checkpoint for UNet2D can be downloaded from [UNet2D Catalog](https://developer.habana.ai/catalog/unet2d-for-pytorch/).
The relevant checkpoint for UNet3D can be downloaded from [UNet3D Catalog](https://developer.habana.ai/catalog/unet-3d-for-pytorch/).
```bash
cd Model-References/PyTorch/computer_vision/segmentation/Unet
mkdir pretrained_checkpoint
wget </url/of/pretrained_checkpoint.tar.gz>
tar -xvf <pretrained_checkpoint.tar.gz> -C pretrained_checkpoint && rm <pretrained_checkpoint.tar.gz>
```


## Inference Examples

The following commands assume that:
- Pre-processed dataset is available at `/data/pytorch/unet/` directory.
  Alternative location for the dataset can be specified using the `--data` argument.

  **NOTE** When using the provided pre-trained checkpoint, ensure the data path in the checkpoint (.ckpt) file matches the location of your pre-processed dataset.
- Pre-trained checkpoint is available at `pretrained_checkpoint/pretrained_checkpoint.pt`.
  Alternative file name for the pretrained checkpoint can be specified using the `--ckpt_path` argument.

**NOTE:** The following commands use PyTorch Lightning by default. To use media loader on Gaudi 2, add `--habana_loader` to the run commands. Default `--measurement_type` is `throughput` to get perf but to get actual latency add `--measurement_type latency` to below run commands. When using Eager mode, replace the `--use_torch_compile` with `--run-lazy-mode=False` in the examples below.

### Single Card Inference Examples

```bash
export PT_HPU_LAZY_MODE=0
mkdir -p /tmp/Unet/results/fold_3
```

**Run inference on 1 HPU:**

**Benchmark Inference**

- UNet2D in torch.compile mode, BF16 mixed precision, batch Size 64:
  ```bash
  $PYTHON main.py --exec_mode predict --task 01 --hpus 1 --fold 3 --val_batch_size 64 --dim 2 --data=/data/pytorch/unet/01_2d --results=/tmp/Unet/results/fold_3 --autocast --benchmark --test_batches 150 --use_torch_compile
  ```
- UNet2D in torch.compile mode, FP32 precision, batch size 64:
  ```bash
  $PYTHON main.py --exec_mode predict --task 01 --hpus 1 --fold 3 --val_batch_size 64 --dim 2 --data=/data/pytorch/unet/01_2d --results=/tmp/Unet/results/fold_3 --benchmark --test_batches 150 --use_torch_compile
  ```
- UNet3D in torch.compile mode, BF16 mixed precision, batch size 2:
  ```bash
  $PYTHON main.py --exec_mode predict --task 01 --hpus 1 --fold 3 --val_batch_size 2 --dim 3 --data=/data/pytorch/unet/01_3d --results=/tmp/Unet/results/fold_3 --autocast --benchmark --test_batches 150 --use_torch_compile
  ```
- UNet3D in torch.compile mode, FP32 precision, batch size 2:
  ```bash
  $PYTHON main.py --exec_mode predict --task 01 --hpus 1 --fold 3 --val_batch_size 2 --dim 3 --data=/data/pytorch/unet/01_3d --results=/tmp/Unet/results/fold_3 --benchmark --test_batches 150 --use_torch_compile
  ```

**Inference**

- UNet2D in torch.compile mode, BF16 mixed precision, batch size 64:
  ```bash
  $PYTHON main.py --exec_mode predict --task 01 --hpus 1 --fold 3 --seed 123 --val_batch_size 64 --dim 2 --data=/data/pytorch/unet/01_2d --results=/tmp/Unet/results/fold_3 --autocast --ckpt_path pretrained_checkpoint/pretrained_checkpoint.pt --use_torch_compile
  ```
- UNet2D in torch.compile mode, FP32 precision, batch size 64:
  ```bash
  $PYTHON main.py --exec_mode predict --task 01 --hpus 1 --fold 3 --seed 123 --val_batch_size 64 --dim 2 --data=/data/pytorch/unet/01_2d --results=/tmp/Unet/results/fold_3 --ckpt_path pretrained_checkpoint/pretrained_checkpoint.pt --use_torch_compile
  ```
- UNet3D in torch.compile mode, BF16 mixed precision, batch size 2:
  ```bash
  $PYTHON main.py --exec_mode predict --task 01 --hpus 1 --fold 3 --seed 123 --val_batch_size 2 --dim 3 --data=/data/pytorch/unet/01_3d --results=/tmp/Unet/results/fold_3 --autocast --ckpt_path pretrained_checkpoint/pretrained_checkpoint.pt --use_torch_compile
  ```
- UNet3D in torch.compile mode, FP32 precision, batch size 2:
  ```bash
  $PYTHON main.py --exec_mode predict --task 01 --hpus 1 --fold 3 --seed 123 --val_batch_size 2 --dim 3 --data=/data/pytorch/unet/01_3d --results=/tmp/Unet/results/fold_3 --ckpt_path pretrained_checkpoint/pretrained_checkpoint.pt --use_torch_compile
  ```


## Accuracy Evaluation

```bash
export PT_HPU_LAZY_MODE=0
mkdir -p /tmp/Unet/results/fold_3
```

### Checkpoint Accuracy Evaluation Using Validation Data
**NOTE:** The following commands use PyTorch Lightning by default. To use media loader on Gaudi 2, add `--habana_loader` to the run commands. When using Eager mode, replace the `--use_torch_compile` with `--run-lazy-mode=False` in the examples below.

**NOTE** When using the provided pre-trained checkpoint, ensure the data path in the checkpoint (.ckpt) file matches the location of your pre-processed dataset.

  - UNet2D in torch.compile mode, FP32 mixed precision, batch size 64:
    ```bash
    $PYTHON main.py --exec_mode=evaluate --data=/data/pytorch/unet/01_2d --hpus=1 --fold=3 --seed 123 --batch_size=64 --val_batch_size=64 --task=01 --dim=2 --results=/tmp/Unet/results/fold_3 --ckpt_path pretrained_checkpoint/pretrained_checkpoint.pt --use_torch_compile
    ```
  - UNet2D in torch.compile mode, BF16 mixed precision, batch size 64:
    ```bash
    $PYTHON main.py --exec_mode=evaluate --data=/data/pytorch/unet/01_2d --hpus=1 --fold=3 --seed 123 --batch_size=64 --val_batch_size=64 --autocast --task=01 --dim=2 --results=/tmp/Unet/results/fold_3 --ckpt_path pretrained_checkpoint/pretrained_checkpoint.pt --use_torch_compile
    ```
  - UNet3D in torch.compile mode, FP32 precision, batch size 2:
    ```bash
    $PYTHON main.py --exec_mode=evaluate --data=/data/pytorch/unet/01_3d/ --hpus=1 --fold=3 --seed 123 --batch_size=2 --val_batch_size=2 --task=01 --dim=3 --results=/tmp/Unet/results/fold_3 --ckpt_path pretrained_checkpoint/pretrained_checkpoint.pt --use_torch_compile
    ```
  - UNet3D in torch.compile mode, BF16 precision, batch size 2:
    ```bash
    $PYTHON main.py --exec_mode=evaluate --data=/data/pytorch/unet/01_3d/ --hpus=1 --fold=3 --seed 123 --batch_size=2 --val_batch_size=2 --autocast --task=01 --dim=3 --results=/tmp/Unet/results/fold_3 --ckpt_path pretrained_checkpoint/pretrained_checkpoint.pt --use_torch_compile
    ```

### Checkpoint Accuracy Evaluation Using Test Data with Target Labels
  - The above Inference commands can be used with `--save_preds` and predictions will be saved in a folder.
  - Using above saved predictions and target labels folder as shown in the below command to get accuracy.
    ```bash
    $PYTHON evaluate.py --preds <prediction_results_path> --lbls <labels_path>
    ```

## Advanced

### Parameters

To see the available training parameters, run the following command:

```bash
$PYTHON -u main.py --help
```

## Supported Configurations

| Models | Cards | Intel Gaudi Software Version | PyTorch Lightning Version | Validated on G1     | Validated on G2     | Validated on G3 |
|--------|-------|------------------------------|---------------------------|---------------------|---------------------|-----------------|
| Unet2D | 1x    | 1.20.0                       | 2.5.0.post0               | Training, Inference | Training, Inference | -               |
| Unet2D | 8x    | 1.20.0                       | 2.5.0.post0               | Training            | Training            | -               | 
| Unet3D | 1x    | 1.20.0                       | 2.5.0.post0               | Training, Inference | Training, Inference | Training*       |
| Unet3D | 8x    | 1.20.0                       | 2.5.0.post0               | Training            | Training            | Training*       |

*Disclaimer: only functional checks done

## Changelog
### 1.20.0
  - Unet is supported for python > '3.8', due to security issue
  - Setting of default value for PT_HPU_EAGER_ENABLE_GRADIENT_VIEW_LAYOUT_OPT=1 flag. Vision models should run with flag set to 1 to allow for higher performance.
### 1.18.0
  - Default execution mode modified to torch.compile mode.
  - Lazy mode support is deprecated.
### 1.17.0
  - Added support for torch.compile and Eager mode inference.
### 1.15.0
  - Added support for torch.compile and Eager mode training.
### 1.14.0
 - Upgraded dali dataloader package "nvidia-dali-cuda110" to 1.32.0.
 - Added support for Gaudi on Ubuntu22.04.
### 1.13.0
 - Enabled using HPU Graphs by default.
 - Added option to enable HPU Graphs in training via `--hpu_graphs` flag.
### 1.12.0
 - Removed HMP, switched to autocast.
 - Eager mode support is deprecated.
### 1.11.0
 - Dynamic shapes will be enabled by default in future releases. It is currently enabled in training script as a temporary solution.
 - UNet2D/3D training using native PyTorch scripts (without PyTorch Lightning) is deprecated.
### 1.10.0
 - Enabled dynamic shapes.
 - Enabled HPUProfiler using habana-lightning-plugins.
### 1.9.0
 - Disabled dynamic shapes.
 - Upgraded pytorch-lightning to 1.9.4 version.
 - Enabled usage of PyTorch autocast.
 - Initial release for inference support on UNet3D.
 - Removed support for Gaudi on Ubuntu22.04.
 - Refactored code to support on Ubuntu22.04 without DALI dataloader on Gaudi 2.
 - Installation instructions are different for Ubuntu20.04 and Ubuntu22.04.
 - HPU Graphs is the default inference mode.
 - Removed newly added scripts to support inference.
 - Inference is supported through existing scripts only.
### 1.8.0
 - Initial release for inference support on UNet2D
### 1.7.0
 - Updated script to make use of TQDM progress bar to override progressbar refresh rate.
 - Upgraded Unet to work with pytorch-lightning 1.7.7.
 - Removed mark_step handling in script as it is taken care in pytorch lightning plugins.
### 1.6.0
 - Added `optimizer_zero_grad` hook and changed `progress_bar_refresh_rate` to improve performance.
 - Added support for 1 and 8 card training on **Gaudi 2**.
 - Added PyTorch support (without PyTorch Lightning) for single Gaudi device with a new flag (`--framework pytorch`) in the run command.
### 1.5.0
 - Changes done to use vanilla PyTorch Lightning 1.6.4 which includes HPU device support.
 - Removed support for channels last format.
 - Weights and other dependent parameters need not be permuted anymore.
### 1.4.0
 - Default execution mode modified to Lazy mode.
### 1.3.0
 - All ops in validation are executed on HPU.
 - Changes to improve time-to-train for UNet3D.
 - Removed support for specifying frequency of validation.
### 1.2.0
 - Bucket size has been increased to 125MB.
 - Enabled HCCL flow for distributed training.

### Training Script Modifications
The following are the changes made to the training scripts:

* Added support for Gaudi devices:
   - Loading Intel Gaudi specific library.
   - Certain environment variables are defined for Gaudi.
   - Added support to run training in Lazy mode in addition to the Eager mode.
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
* Added HPU Graphs support to reduce latency for inference.

## Known Issues
- Placing mark_step() arbitrarily may lead to undefined behavior. Recommend to keep mark_step() as shown in provided scripts.
- Only scripts & configurations mentioned in this README are supported and verified.
