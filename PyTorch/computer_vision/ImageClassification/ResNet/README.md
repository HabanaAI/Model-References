# ResNet for PyTorch
This folder contains scripts to train ResNet50 & ResNext101 model on Habana Gaudi<sup>TM</sup> device to achieve state-of-the-art accuracy. Please visit [this page](https://developer.habana.ai/resources/habana-training-models/#performance) for performance information.

The ResNet50 demos included in this release are Eager mode and Lazy mode training for BS128 with FP32 and BS256 with BF16 mixed precision.
The ResNext101 demos included in this release are Eager mode and Lazy mode training for BS64 with FP32 and BS128 with BF16 mixed precision.

## Table of Contents
- [ResNet for PyTorch](#resnet-for-pytorch)
  - [Model Overview](#model-overview)
  - [Setup](#setup)
    - [Set up dataset](#set-up-dataset)
  - [Training the Model](#training-the-model)
  - [Multinode Training](#multinode-training)
      - [Docker ssh port setup for multi-server training](#docker-ssh-port-setup-for-multi-hls-training)
      - [Setup password-less ssh between all connected servers used in the scale-out training](#setup-password-less-ssh-between-all-connected-hls-systems-used-in-the-scale-out-training)
- [Known Issues](#known-issues)
- [Training Script Modifications](#training-script-modifications)

## Model Overview
The base model used is from [GitHub-PyTorch-Vision](https://github.com/pytorch/vision.git). A copy of the model file [resnet.py](model/resnet.py) is used to make
certain model changes functionally equivalent to the original model. Please refer to a below section for a summary of model changes, changes to training script and the original files.

## Setup
Please follow the instructions given in the following link for setting up the
environment including the `$PYTHON` environment variable: [Gaudi Setup and
Installation Guide](https://github.com/HabanaAI/Setup_and_Install). Please
answer the questions in the guide according to your preferences. This guide will
walk you through the process of setting up your system to run the model on
Gaudi.

The base training and modelling scripts for training are based on a clone of
https://github.com/pytorch/vision.git with certain changes for modeling and training script.
Please refer to later sections on training script and model modifications for a summary of
modifications to the original files.

In the docker container, clone this repository and switch to the branch that
matches your SynapseAI version. (Run the
[`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.)

```
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```
Go to PyTorch ResNet directory:
```bash
cd Model-References/PyTorch/computer_vision/ImageClassification/ResNet
```
Note: If the repository is not in the PYTHONPATH, make sure you update it.
```bash
export PYTHONPATH=/path/to/Model-References:$PYTHONPATH
```
### Set up dataset

Imagenet 2012 dataset needs to be organized as per PyTorch requirement. PyTorch requirements are specified in the link below which contains scripts to organize Imagenet data. https://github.com/soumith/imagenet-multiGPU.


## Training the Model

The following commands assume that Imagenet dataset is available at /root/software/data/pytorch/imagenet/ILSVRC2012/ directory.

Please refer to the following command for available training parameters:
```
cd Model-References/PyTorch/computer_vision/ImageClassification/ResNet

```
i. Example: ResNet50, lazy mode, bf16 mixed precision, Batch Size 256, Custom learning rate
```python
$PYTHON -u demo_resnet.py  --world-size 1 --dl-worker-type MP --batch-size 256 --model resnet50 --device hpu --workers 12 --print-freq 20 --channels-last True --dl-time-exclude False --deterministic --data-path /root/software/data/pytorch/imagenet/ILSVRC2012 --mode lazy --epochs 90 --data-type bf16  --custom-lr-values 0.1,0.01,0.001,0.0001 --custom-lr-milestones 0,30,60,80
```
ii. Example: ResNext101 lazy mode, bf16 mixed precision, Batch size 128, Custom learning rate
```python
$PYTHON -u demo_resnet.py  --world-size 1 --dl-worker-type MP --batch-size 128 --model resnext101_32x4d --device hpu --workers 12 --print-freq 20 --channels-last True --dl-time-exclude False --deterministic --data-path /root/software/data/pytorch/imagenet/ILSVRC2012 --mode lazy --epochs 100 --data-type bf16  --custom-lr-values 0.1,0.01,0.001,0.0001 --custom-lr-milestones 0,30,60,80
```


## Multinode Training
To run multi-node demo, make sure the host machine has 512 GB of RAM installed.
Also ensure you followed the [Gaudi Setup and
Installation Guide](https://github.com/HabanaAI/Setup_and_Install) to install and set up docker,
so that the docker has access to all the 8 cards required for multi-node demo. Multinode configuration for ResNet 50 training up to 2 servers, each with 8 Gaudi cards, have been verified.

Before execution of the multi-node demo scripts, make sure all server network interfaces are up. You can change the state of each network interface managed by the habanalabs driver using the following command:
```
sudo ip link set <interface_name> up
```
To identify if a specific network interface is managed by the habanalabs driver type, run:
```
sudo ethtool -i <interface_name>
```
#### Docker ssh port setup for multi-server training

Multi-server training works by setting these environment variables:

- **`MULTI_HLS_IPS`**: set this to a comma-separated list of host IP addresses. Network id is derived from first entry of this variable

This example shows how to setup for a 2-server training configuration. The IP addresses used are only examples:

```bash
export MULTI_HLS_IPS="10.10.100.101,10.10.100.102"

# When using demo script, export the following variable to the required port number (default 3022)
export DOCKER_SSHD_PORT=3022
```
By default, the Habana docker uses `port 22` for ssh. The default port configured in the demo script is `port 3022`. Run the following commands to configure the selected port number , `port 3022` in example below.

```bash
sed -i 's/#Port 22/Port 3022/g' /etc/ssh/sshd_config
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
service ssh restart
```
#### Setup password-less ssh between all connected servers used in the scale-out training

1. Configure password-less ssh between all nodes:

   Do the following in all the nodes' docker sessions:
   ```bash
   mkdir ~/.ssh
   cd ~/.ssh
   ssh-keygen -t rsa -b 4096
   ```
   Copy id_rsa.pub contents from every node's docker to every other node's docker's ~/.ssh/authorized_keys (all public keys need to be in all hosts' authorized_keys):
   ```bash
   cat id_rsa.pub > authorized_keys
   vi authorized_keys
   ```
   Copy the contents from inside to other systems.
   Paste all hosts' public keys in all hosts' “authorized_keys” file.

2. On each system:
   Add all hosts (including itself) to known_hosts. The IP addresses used below are just for illustration:
   ```bash
   ssh-keyscan -p 3022 -H 10.10.100.101 >> ~/.ssh/known_hosts
   ssh-keyscan -p 3022 -H 10.10.100.102 >> ~/.ssh/known_hosts
   ```

i. Example: ResNet50 ,lazy mode: bf16 mixed precision, Batch size 256, world-size 8, custom learning rate, 8x on single server, include dataloading time  in throughput computation
```python
$PYTHON -u demo_resnet.py  --world-size 8 --batch-size 256 --model resnet50 --device hpu --print-freq 1 --channels-last True --deterministic --data-path /root/software/data/pytorch/imagenet/ILSVRC2012 --mode lazy --epochs 90 --data-type bf16  --custom-lr-values 0.275,0.45,0.625,0.8,0.08,0.008,0.0008 --custom-lr-milestones 1,2,3,4,30,60,80 --dl-time-exclude=False

ii. Example: ResNet50 ,lazy mode: bf16 mixed precision, Batch size 256, world-size 8, custom learning rate, 8x on single server, exclude dataloading time in throughput computation
```python
$PYTHON -u demo_resnet.py  --world-size 8 --batch-size 256 --model resnet50 --device hpu --print-freq 1 --channels-last True --deterministic --data-path /root/software/data/pytorch/imagenet/ILSVRC2012 --mode lazy --epochs 90 --data-type bf16  --custom-lr-values 0.275,0.45,0.625,0.8,0.08,0.008,0.0008 --custom-lr-milestones 1,2,3,4,30,60,80 --dl-time-exclude=True

```
iii. Example: ResNet50 ,lazy mode: bf16 mixed precision, Batch size 256, world-size 16, custom learning rate, 16x on multiple servers, include dataloading time in throughput computation
```python
MULTI_HLS_IPS="<node1 ipaddr>,<node2 ipaddr>" $PYTHON -u demo_resnet.py  --world-size 16 --batch-size 256 --model resnet50 --device hpu --workers 4 --print-freq 1 --channels-last True --deterministic --data-path /root/software/data/pytorch/imagenet/ILSVRC2012 --mode lazy --epochs 90 --data-type bf16  --custom-lr-values 0.475,0.85,1.225,1.6,0.16,0.016,0.0016 --custom-lr-milestones 1,2,3,4,30,60,80 --process-per-node 8 --dl-time-exclude=False
```
iv. Example: ResNext101 ,lazy mode: bf16 mixed precision, Batch size 128, world-size 8, single server, include dataloading time in throughput computation
```python
$PYTHON -u demo_resnet.py --world-size 8 --batch-size 128 --model resnext101_32x4d --device hpu --print-freq 1 --channels-last True --deterministic --data-path /root/software/data/pytorch/imagenet/ILSVRC2012 --mode lazy --epochs 100 --data-type bf16  --dl-time-exclude=False
```
# Known Issues
- channels-last = false is not supported. Train ResNet with channels-last=true. This will be fixed in a subsequent release.
- Crash observed while using multi-processing workers for dataloader with multinode training.
- Placing mark_step() arbitrarily may lead to undefined behaviour. Recommend to keep mark_step() as shown in provided scripts.

# Training Script Modifications
The following are the changes added to the training script (train.py) and utilities(utils.py):
1. Added support for Habana devices – load Habana specific library.
2. Enhance with channels last data format (NHWC) support.
3. Added multi-node (distributed) and mixed precision support.
4. Modified image copying to device as non-blocking to gain better performance.
5. Permute convolution weight tensors & and any other dependent tensors like 'momentum' for better performance.
6. Checkpoint saving involves getting trainable params and other state variables to CPU and permuting the weight tensors. Hence, checkpoint saving is by default disabled. It can be enabled using –save-checkpoint option.
7. Accuracy calculation is performed on the CPU.
8. Essential train operations are wrapped into a function to limit the variable scope and thus reduce memory consumption.
9. Used copy of ResNet model file (torchvision/models/resnet.py) instead of importing model from torch vision package. This is to add modifications (functionally equivalent to the original) to the model.
10. Loss tensor is brought to CPU for printing stats.
11. Skip printing eval phase stats if eval steps is zero.
12. Limited validation phase batch size to 32. This limitation will be removed in future releases.
13. Dropped last batch if it is of partial batch size. This is a workaround to avoid creating extra graphs due to changed batch size, hence avoiding extra memory usage. This will be removed in future releases.
14. Modified training script to use mpirun for distributed training. Introduced mpi barrier to sync the processes and calculate training time by excluding the data load time.
15. Added –deterministic flag to make data loading deterministic.
16. Modified script to avoid broadcast at the beginning of iteration.
17. Enabled Pinned memory for data loader using flag pin_memory=True.
18. Added additional parameter to DistributedDataParallel to increase the size of the first bucket to combine all the all-reduce calls to a single call.
19. Added support to run Resnet training in lazy mode in addition to the eager mode.
    mark_step() is performed to trigger execution of the graph.
20. Optimized FusedSGD operator is used in place of torch.optim.SGD for lazy mode.
21. Control of a few Habana lazy mode optimization passes in bridge.
22. Added support in script to override default LR scheduler with custom schedule passed as parameters to script.
23. Added support for multithreaded dataloader.
24. Added support for lowering print frequency of loss.
25. Added support for including dataloader time in performance computation. Disabled by default.
26. Certain environment variables are defined for habana device.
27. Added support for training on multiple servers.
28. Changes for dynamic loading of HCL library.
29. Changed start_method for torch multiprocessing.

The model changes are listed below:
1. Replaced nn.AdaptiveAvgPool with functionally equivalent nn.AvgPool.
2. Replaced in-place nn.Relu and torch.add_ with non-in-place equivalents.
3. Added support for resnext101_32x4d.
