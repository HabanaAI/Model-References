# Table of Contents
- [Unet2D & Unet3D for PyTorch](#unet2d--unet3d-for-pytorch)
  - [Model Overview](#model-overview)
  - [Setup](#setup)
    - [Install the requirements](#install-the-requirements)
    - [Set up BraTS dataset](#set-up-brats-dataset)
  - [Training the Model](#training-the-model)
  - [Multinode Training](#multinode-training)
- [Changelog](#changelog)
  - [1.3.0](#130)
  - [1.2.0](#120)
- [Known Issues](#known-issues)
- [Training Script Modifications](#training-script-modifications)

# Unet2D & Unet3D for PyTorch
This folder contains scripts to train Unet2D & Unet3D model on Habana Gaudi<sup>TM</sup> device to achieve state-of-the-art accuracy. Please visit [this page](https://developer.habana.ai/resources/habana-training-models/#performance) for performance information.

The Unet2D demos included in this release are Eager mode and Lazy mode training for BS64 with FP32 & BF16 mixed precision.
The Unet3D demos included in this release are Eager mode and Lazy mode training for BS2 with FP32 & BF16 mixed precision.

## Model Overview
The base model used is from [GitHub: nnUNet](https://github.com/NVIDIA/DeepLearningExamples/tree/2b20ca80cf7f08585e90a11c5b025fa42e4866c8/PyTorch/Segmentation/nnUNet). As in the base scripts, Pytorch Lightning is used. Please refer to a below section for a summary of changes in training scripts.

## Setup
Please follow the instructions given in the following link for setting up the
environment including the `$PYTHON` environment variable: [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html).
This guide will walk you through the process of setting up your system to run
the model on Gaudi.

The base training and modelling scripts for training are based on a clone of
https://github.com/NVIDIA/DeepLearningExamples/tree/2b20ca80cf7f08585e90a11c5b025fa42e4866c8/PyTorch/Segmentation/nnUNet with certain changes in training scripts.
Please refer to later sections on training script and model modifications for a summary of
modifications to the original files.

In the docker container, clone this repository and switch to the branch that
matches your SynapseAI version. (Run the
[`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.)

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```
Go to PyTorch Unet directory:
```bash
cd Model-References/PyTorch/computer_vision/segmentation/Unet
```
Note: If the repository is not in the PYTHONPATH, make sure you update it.
```bash
export PYTHONPATH=/path/to/Model-References:$PYTHONPATH
```
### Install the requirements
It is necessary to install the required packages in `requirements.txt` to run the model and download dataset.
```bash
pip install -r ./requirements.txt
```

### Set up BraTS dataset
Create a /data directory if not present:
```bash
mkdir /data
```
To download the dataset run:
```python
$PYTHON download.py --task 01
```
NOTE: The script downloads the dataset in /data directory by default.

To preprocess it for Unet2D run:
```python
$PYTHON preprocess.py --task 01 --dim 2
```

To process it for Unet3D run:
```python
$PYTHON preprocess.py --task 01 --dim 3
```

NOTE: The script preprocess the dataset downloaded in above step from /data directory and creates directory `01_2d` for Unet2D and directory `01_3d` for Unet3D model inside `/data` directory.

## Training the Model

The following commands assume that dataset is available at /root/software/data/pytorch/unet/01_2d directory for Unet2D model and /root/software/data/pytorch/unet/01_3d directory for Unet3D model.

Please refer to the following command for available training parameters:
```
cd Model-References/PyTorch/computer_vision/segmentation/Unet
pip install -r requirements.txt
mkdir /tmp/Unet/results/fold_0
```
i. Example: Unet2D in Lazy mode, bf16 mixed precision, Batch Size 64, Fold 0
```python
$PYTHON -u  main.py --results /tmp/Unet/results/fold_0 --task 01 --logname res_log --fold 0 --hpus 1 --gpus 0 --data /root/software/data/pytorch/unet/01_2d --seed 1 --num_workers 8 --affinity disabled --norm instance --dim 2 --optimizer fusedadamw --exec_mode train --learning_rate 0.001 --run_lazy_mode --hmp --hmp-bf16 ./config/ops_bf16_unet.txt --hmp-fp32 ./config/ops_fp32_unet.txt --deep_supervision --batch_size 64 --val_batch_size 64
```
ii. Example: Unet2D in Eager mode, bf16 mixed precision, Batch Size 64, Fold 0
```python
$PYTHON -u  main.py --results /tmp/Unet/results/fold_0 --task 01 --logname res_log --fold 0 --hpus 1 --gpus 0 --data /root/software/data/pytorch/unet/01_2d --seed 1 --num_workers 8 --affinity disabled --norm instance --dim 2 --optimizer fusedadamw --exec_mode train --learning_rate 0.001 --hmp --hmp-bf16 ./config/ops_bf16_unet.txt --hmp-fp32 ./config/ops_fp32_unet.txt --deep_supervision --batch_size 64 --val_batch_size 64
```
iii. Example: Unet2D in Eager mode, FP32 precision, Batch Size 64, Fold 2
```python
$PYTHON -u  main.py --results /tmp/Unet/results/fold_0 --task 01 --logname res_log --fold 2 --hpus 1 --gpus 0 --data /root/software/data/pytorch/unet/01_2d --seed 1 --num_workers 8 --affinity disabled --norm instance --dim 2 --optimizer fusedadamw --exec_mode train --learning_rate 0.001 --deep_supervision --batch_size 64 --val_batch_size 64
```
iv. Example: Unet2D in Lazy mode, bf16 mixed precision, Batch Size 64, benchmarking
```python
$PYTHON -u  main.py --results /tmp/Unet/results/fold_0 --task 01 --logname res_log --fold 0 --hpus 1 --gpus 0 --data /root/software/data/pytorch/unet/01_2d --seed 123 --num_workers 1 --affinity disabled --norm instance --dim 2 --optimizer fusedadamw --exec_mode train --learning_rate 0.001 --run_lazy_mode --hmp --hmp-bf16 ./config/ops_bf16_unet.txt --hmp-fp32 ./config/ops_fp32_unet.txt --benchmark --max_epochs 2 --min_epochs 1 --warmup 50 --batch_size 64 --val_batch_size 64 --train_batches 150 --test_batches 150
```
```python
$PYTHON -u demo_unet.py  --task 01 --fold 0 --hpus 1 --gpus 0 --data /root/software/data/pytorch/unet/01_2d --seed 123 --num_workers 1 --norm instance --dim 2 --optimizer fusedadamw --exec_mode train --learning_rate 0.001 --mode lazy --data_type bf16 --benchmark --max_epochs 2 --min_epochs 1 --batch_size 64 --val_batch_size 64 --train_batches 150 --test_batches 150
```
v. Example: Unet3D in Lazy mode, bf16 mixed precision, Batch Size 2 Fold 0
```python
$PYTHON -u main.py --results /tmp/Unet/results/fold_0 --task 01 --logname res_log --fold 0 --hpus 1 --gpus 0 --data /root/software/data/pytorch/unet/01_3d --seed 1 --num_workers 8 --affinity disabled --norm instance --dim 3 --optimizer fusedadamw --exec_mode train --learning_rate 0.001 --run_lazy_mode --hmp --hmp-bf16 ./config/ops_bf16_unet.txt --hmp-fp32 ./config/ops_fp32_unet.txt --deep_supervision --batch_size 2 --val_batch_size 2
```
vi. Example: Unet3D in Lazy mode, bf16 mixed precision, Batch Size 2, Benchmarking
```python
$PYTHON -u demo_unet.py  --task 01 --fold 0 --hpus 1 --gpus 0 --data /root/software/data/pytorch/unet/01_3d --seed 1 --num_workers 1 --norm instance --dim 3 --optimizer fusedadamw --exec_mode train --learning_rate 0.001 --mode lazy --data_type bf16 --benchmark --max_epochs 2 --min_epochs 1 --batch_size 2 --val_batch_size 2 --train_batches 150 --test_batches 150
```

## Multinode Training
To run multi-node demo, make sure the host machine has 512 GB of RAM installed.
Also ensure you followed the [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html)
to install and set up docker, so that the docker has access to all the 8 cards
required for multi-node demo. Multinode configuration for Unet2D & Unet3D
training upto 1 server, with 8 Gaudi cards, has been verified.

Before execution of the multi-node demo scripts, make sure all network interfaces are up. You can change the state of each network interface managed by the habanalabs driver using the following command:
```
sudo ip link set <interface_name> up
```
To identify if a specific network interface is managed by the habanalabs driver type, run:
```
sudo ethtool -i <interface_name>
```

i. Example: Unet2D in Lazy mode, bf16 mixed precision, Batch Size 64, world-size 8, Fold 0
```python
$PYTHON -u  main.py --results /tmp/Unet/results/fold_0 --task 01 --logname res_log --fold 0 --hpus 8 --gpus 0 --data /root/software/data/pytorch/unet/01_2d --seed 1 --num_workers 8 --affinity disabled --norm instance --dim 2 --optimizer fusedadamw --exec_mode train --learning_rate 0.001 --run_lazy_mode --hmp --hmp-bf16 ./config/ops_bf16_unet.txt --hmp-fp32 ./config/ops_fp32_unet.txt --deep_supervision --batch_size 64 --val_batch_size 64
```
```python
$PYTHON -u demo_unet.py  --task 01 --fold 0 --hpus 8 --gpus 0 --data /root/software/data/pytorch/unet/01_2d --seed 123 --num_workers 8 --norm instance --dim 2 --optimizer fusedadamw --exec_mode train --learning_rate 0.001 --mode lazy --data_type bf16 --deep_supervision --batch_size 64 --val_batch_size 64
```
ii. Example: Unet2D in Eager mode, bf16 mixed precision, Batch Size 64, world-size 8, Fold 0
```python
$PYTHON -u  main.py --results /tmp/Unet/results/fold_0 --task 01 --logname res_log --fold 0 --hpus 8 --gpus 0 --data /root/software/data/pytorch/unet/01_2d --seed 1 --num_workers 8 --affinity disabled --norm instance --dim 2 --optimizer fusedadamw --exec_mode train --learning_rate 0.001 --hmp --hmp-bf16 ./config/ops_bf16_unet.txt --hmp-fp32 ./config/ops_fp32_unet.txt --deep_supervision --batch_size 64 --val_batch_size 64
```
iii. Example: Unet2D in Eager mode, FP32 precision, Batch Size 64, world-size 8, Fold 2
```python
$PYTHON -u  main.py --results /tmp/Unet/results/fold_0 --task 01 --logname res_log --fold 2 --hpus 8 --gpus 0 --data /root/software/data/pytorch/unet/01_2d --seed 1 --num_workers 8 --affinity disabled --norm instance --dim 2 --optimizer fusedadamw --exec_mode train --learning_rate 0.001 --deep_supervision --batch_size 64 --val_batch_size 64
```
iv. Example: Unet2D in Lazy mode, bf16 mixed precision, Batch Size 64, world-size 8, benchmarking
```python
$PYTHON -u  main.py --results /tmp/Unet/results/fold_0 --task 01 --logname res_log --fold 0 --hpus 8 --gpus 0 --data /root/software/data/pytorch/unet/01_2d --seed 123 --num_workers 1 --affinity disabled --norm instance --dim 2 --optimizer fusedadamw --exec_mode train --learning_rate 0.001 --run_lazy_mode --hmp --hmp-bf16 ./config/ops_bf16_unet.txt --hmp-fp32 ./config/ops_fp32_unet.txt --benchmark --max_epochs 2 --min_epochs 1 --warmup 50 --batch_size 64 --val_batch_size 64 --train_batches 150 --test_batches 150
```
```python
$PYTHON -u demo_unet.py --task 01 --fold 0 --hpus 8 --gpus 0 --data /root/software/data/pytorch/unet/01_2d --seed 123 --num_workers 1 --norm instance --dim 2 --optimizer fusedadamw --exec_mode train --learning_rate 0.001 --mode lazy --data_type bf16 --benchmark --max_epochs 2 --min_epochs 1 --batch_size 64 --val_batch_size 64 --train_batches 150 --test_batches 150
```
v. Example: Unet3D in Lazy mode, bf16 mixed precision, Batch Size 2, world-size 8
```python
$PYTHON -u main.py --results /tmp/Unet/results/fold_0 --task 01 --logname res_log --fold 0 --hpus 8 --gpus 0 --data /root/software/data/pytorch/unet/01_3d --seed 1 --num_workers 8 --affinity disabled --norm instance --dim 3 --optimizer fusedadamw --exec_mode train --learning_rate 0.001 --run_lazy_mode --hmp --hmp-bf16 ./config/ops_bf16_unet.txt --hmp-fp32 ./config/ops_fp32_unet.txt --deep_supervision --batch_size 2 --val_batch_size 2
```

vi. Example: Unet3D in Lazy mode, bf16 mixed precision, Batch Size 2, world-size 8, benchmarking
```python
$PYTHON -u demo_unet.py  --task 01 --fold 0 --hpus 8 --gpus 0 --data /root/software/data/pytorch/unet/01_3d --seed 1 --num_workers 1 --norm instance --dim 3 --optimizer fusedadamw --exec_mode train --learning_rate 0.001 --mode lazy --data_type bf16 --benchmark --max_epochs 2 --min_epochs 1 --batch_size 2 --val_batch_size 2 --train_batches 150 --test_batches 150
```

# Changelog
## 1.3.0
 - All ops in validation are executed on HPU
 - Changes to improve time-to-train for Unet3D
 - Removed support for specifying frequency of validation
## 1.2.0
 - Bucket size has been increased to 125MB
 - Enabled HCCL flow for distributed training

# Known Issues
- channels-last = false is not supported. Train Unet2D & Unet3D with channels-last=true. This will be fixed in a subsequent release.
- Placing mark_step() arbitrarily may lead to undefined behavior. Recommend to keep mark_step() as shown in provided scripts.
- Only scripts & configurations mentioned in this README are supported and verified.

# Training Script Modifications
The following are the changes made to the training scripts:

1. Added support for Habana devices

   a. loading Habana specific library.

   b. Certain environment variables are defined for habana device.

   c. Added support to run training in lazy mode in addition to the eager mode.

   d. mark_step() is performed to trigger execution.

   e. Changes to enable scripts on Pytorch Lightning 1.4.0 as base scripts used older version of Pytorch Lightning.

   f. Added support to use HPU accelerator plugin, DDP plugin(for multi-card training) & mixed precision plugin
   provided with installed Pytorch Lightning package.

2. Metric was copied to pl_metric.py from older version of  Pytorch Lightning(1.0.4). Implementation in Pytorch Lightning 1.4.0(torch.metric) is different and incompatible.
3. Pytorch Lightning metrics is deprecated since Pytorch Lightning 1.3 and suggested to change to torchmetrics. Since stat_scores implementation is different and incompatible, older version was copied here from Pytorch Lightning 1.0.
4. To improve performance

   a. Enhance with channels last data format (NHWC) support.

   b. Permute convolution weight tensors for better performance.

   c. Checkpoint saving involves getting trainable params and other state variables to CPU and permuting the weight tensors.

   d. Optimized FusedAdamW operator is used in place of torch.optim.AdamW.

   e. Added dice.py with code from monai package and replaced slice with split operator in the forward method.

   f. Added monai_sliding_window_inference.py with code from monai package and modified to avoid recomputation of importance map every iteration.

   g. Changes to configure the gradient reduction bucket size, set gradients as bucket for all-reduce use static graphs for multinode training.

5.  Changes to run DALI dataloader on CPU & make dataloading deterministic.
