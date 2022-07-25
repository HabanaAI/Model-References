# Table of Contents
- [Classification for PyTorch](#classification-for-pytorch)
  - [Model-References](../../../../README.md)
  - [Model Overview](#model-overview)
  - [Setup](#setup)
    - [Set up dataset](#set-up-dataset)
  - [Media Loading Acceleration](#media-loading-acceleration)
  - [Training the Model](#training-the-model)
  - [Multinode Training](#multinode-training)
      - [Docker ssh port setup for multi-server training](#docker-ssh-port-setup-for-multi-server-training)
      - [Run multi-server over host NICs](#run-multi-server-over-host-nics)
      - [Setup password-less ssh between all connected servers used in the scale-out training](#setup-password-less-ssh-between-all-connected-servers-used-in-the-scale-out-training)
- [Supported Configurations](#supported-configurations)
- [Changelog](#changelog)
  - [1.5.0](#150)
  - [1.4.1](#141)
  - [1.4.0](#140)
- [Known Issues](#known-issues)
- [Training Script Modifications](#training-script-modifications)

# Classification for PyTorch
This folder contains scripts to train ResNet50, ResNet152, ResNext101, MobileNet_v2 & GoogLeNet model on Habana Gaudi<sup>TM</sup> device to achieve state-of-the-art accuracy. Please visit [this page](https://developer.habana.ai/resources/habana-training-models/#performance) for performance information.

The ResNet50 demos included in this release are Eager mode and Lazy mode training for BS128 with FP32 and BS256 with BF16 mixed precision.
The ResNet152 demos included in this release are Eager mode and Lazy mode training for BS128 with BF16 mixed precision.
The ResNext101 demos included in this release are Eager mode and Lazy mode training for BS64 with FP32 and BS128 with BF16 mixed precision.
The MobileNet_v2 demos included in this release are Eager mode and Lazy mode training for BS256 with FP32 and BF16 mixed precision.
The GoogLeNet demos included in this release are Eager mode and Lazy mode training for BS128 with FP32 and BF256 mixed precision.

## Model Overview
The base model used is from [GitHub: PyTorch-Vision](https://github.com/pytorch/vision/tree/release/0.10/torchvision/models). A copy of the resnet model file [resnet.py](model/resnet.py) is used to make
certain model changes functionally equivalent to the original model. Please refer to a below section for a summary of model changes, changes to training script and the original files.

## Setup
Please follow the instructions given in the following link for setting up the
environment including the `$PYTHON` environment variable: [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html).
This guide will walk you through the process of setting up your system to run
the model on Gaudi.

The base training and modelling scripts for training are based on a clone of
https://github.com/pytorch/vision.git with certain changes for modeling and training script.
Please refer to later sections on training script and model modifications for a summary of
modifications to the original files.

In the docker container, clone this repository and switch to the branch that
matches your SynapseAI version. (Run the
[`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.)

> ### ⚠ Exception for MobileNet_v2 and ResNet152
> The latest SynapseAI version supported by MobileNet_v2 and Resnet152 is **1.4.1**

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```
Go to PyTorch torchvision directory:
```bash
cd Model-References/PyTorch/computer_vision/classification/torchvision
```
Note: If the repository is not in the PYTHONPATH, make sure you update it.
```bash
export PYTHONPATH=/path/to/Model-References:$PYTHONPATH
```
Install the requirements:
```bash
pip install -r requirements.txt
```
### Set up dataset

Imagenet 2012 dataset needs to be organized as per PyTorch requirement. PyTorch requirements are specified in the link below which contains scripts to organize Imagenet data.
https://github.com/soumith/imagenet-multiGPU.torch

## Media Loading Acceleration
Gaudi2 offers dedicated hardware engine for Media Loading operations. For more details, please refer to [Habana Media Loader page](https://docs.habana.ai/en/latest/PyTorch/Habana_Media_Loader_PT/Media_Loader_PT.html)

## Training the Model

The following commands assume that Imagenet dataset is available at /data/pytorch/imagenet/ILSVRC2012/ directory.

Please refer to the following command for available training parameters for ResNet50, ResNet152, ResNext101 and MobileNet_v2:
```bash
$PYTHON -u train.py --help
```
Please refer to the following command for available training parameters for GoogleNet:
```bash
$PYTHON -u main.py --help
```

i. Example: ResNet50, lazy mode, bf16 mixed precision, Batch Size 256, Custom learning rate, Habana dataloader (with hardware decode support on **Gaudi2**), 1x on a single HLS1 (**HLS2 on Gaudi2**) machine
```bash
$PYTHON -u train.py --dl-worker-type HABANA --batch-size 256 --model resnet50 --device hpu --workers 8 --print-freq 20 --dl-time-exclude False --deterministic --data-path /data/pytorch/imagenet/ILSVRC2012 --epochs 90 --hmp --hmp-bf16 ./ops_bf16_Resnet.txt --hmp-fp32 ./ops_fp32_Resnet.txt  --lr 0.1 --custom-lr-values 0.1 0.01 0.001 0.0001 --custom-lr-milestones 0 30 60 80
```
ii. Example: ResNext101 lazy mode, bf16 mixed precision, Batch size 256, Custom learning rate, Habana dataloader (with hardware decode support on **Gaudi2**), 1x on single **HLS2** machine
```bash
$PYTHON -u train.py --dl-worker-type HABANA --batch-size 256 --model resnext101_32x4d --device hpu --workers 8 --print-freq 20 --dl-time-exclude False --deterministic --data-path /data/pytorch/imagenet/ILSVRC2012 --epochs 100 --hmp --hmp-bf16 ./ops_bf16_Resnet.txt --hmp-fp32 ./ops_fp32_Resnet.txt --lr 0.1 --custom-lr-values 0.1 0.01 0.001 0.0001 --custom-lr-milestones 0 30 60 80
```
iii. Example: ResNext101 lazy mode, bf16 mixed precision, Batch size 128, Custom learning rate, 1x on a single HLS1 machine
```bash
$PYTHON -u train.py --dl-worker-type HABANA --batch-size 128 --model resnext101_32x4d --device hpu --workers 8 --print-freq 20 --dl-time-exclude False --deterministic --data-path /data/pytorch/imagenet/ILSVRC2012 --epochs 100 --hmp --hmp-bf16 ./ops_bf16_Resnet.txt --hmp-fp32 ./ops_fp32_Resnet.txt --lr 0.1 --custom-lr-values 0.1 0.01 0.001 0.0001 --custom-lr-milestones 0 30 60 80
```
iv. Example: ResNet152, lazy mode, bf16 mixed precision, Batch Size 128, Custom learning rate, 1x on a single HLS1 machine
```bash
$PYTHON -u train.py --dl-worker-type HABANA --batch-size 128 --model resnet152 --device hpu --workers 8 --print-freq 20 --dl-time-exclude False --deterministic --data-path /data/pytorch/imagenet/ILSVRC2012 --epochs 90 --hmp --hmp-bf16 ./ops_bf16_Resnet.txt --hmp-fp32 ./ops_fp32_Resnet.txt --lr 0.1 --custom-lr-values 0.1 0.01 0.001 0.0001 --custom-lr-milestones 0 30 60 80
```
v. Example: MobileNet_v2, lazy mode, bf16 mixed precision, Batch Size 256, 1x on HLS1 with default pytorch dataloader
```bash
$PYTHON -u train.py --batch-size 256 --model mobilenet_v2 --device hpu --print-freq 10 --deterministic --data-path /data/pytorch/imagenet/ILSVRC2012 --epochs 90 --hmp --hmp-bf16 ./ops_bf16_Mobilenet.txt --hmp-fp32 ./ops_fp32_Mobilenet.txt --dl-time-exclude=False --lr 0.045 --wd 0.00004 --lr-step-size 1 --lr-gamma 0.98 --momentum 0.9
```
vi. Example: GoogLeNet, Batch Size 256, bf16 precision, lazy mode, 1x on a single HLS1 machine.
```bash
$PYTHON -u main.py --batch-size 256 --data-path /data/pytorch/imagenet/ILSVRC2012 --hmp --hmp-bf16 ./ops_bf16_googlenet.txt --hmp-fp32 ./ops_fp32_googlenet.txt --device hpu --dl-worker-type HABANA --epochs 90 --lr 0.1 --enable-lazy --model googlenet --seed 123 --no-aux-logits --print-interval 20 --workers 8
```
vii. Example: GoogLeNet, Batch Size 128, fp32 precision, lazy mode, 1x on a single HLS1 machine.
```bash
$PYTHON -u main.py --batch-size 128 --data-path /data/pytorch/imagenet/ILSVRC2012 --device hpu --dl-worker-type HABANA --epochs 90 --lr 0.07 --enable-lazy --model googlenet --seed 123 --no-aux-logits --print-interval 20 --workers 8
```

## Multinode Training
To run multi-node demo, make sure the host machine has 512 GB of RAM installed.
Also ensure you followed the [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html)
to install and set up docker, so that the docker has access to all the 8 cards
required for multi-node demo. Multinode configuration for ResNet 50 training up
to 2 servers, each with 8 Gaudi cards, have been verified.

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

By default, the Habana docker uses `port 22` for ssh. If the default port is occupied then the port number can also be changed. Run the following commands to configure the selected port number , `port 3022` in example below.

```bash
sed -i 's/#Port 22/Port 3022/g' /etc/ssh/sshd_config
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
service ssh restart
```

#### Run multi-server over host NICs

- Environment variable `HCCL_OVER_TCP=1` must be set to enable multi-server in HCCL using raw TCP sockets.
- Environment variable `HCCL_OVER_OFI=1` must be set to enable multi-server in HCCL using Libfabric.

```bash
# Set this to the network interface name or subnet that will be used by HCCL to communicate e.g.:
export HCCL_SOCKET_IFNAME=interface_name
e.g.
export HCCL_SOCKET_IFNAME=eth0
#or
export HCCL_SOCKET_IFNAME=eth0,eth1

export HCCL_OVER_TCP=1
#or
export HCCL_OVER_OFI=1
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
   ssh-keyscan -p 3022 -H 10.3.124.124 >> ~/.ssh/known_hosts
   ssh-keyscan -p 3022 -H 10.3.124.175 >> ~/.ssh/known_hosts
   ```

**NOTE:** mpirun map-by PE attribute value may vary on your setup. Please refer to the instructions on [mpirun Configuration](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Scaling_Guide/DDP_Based_Scaling.html#mpirun-configuration) for calculation.

i. Example: ResNet50, lazy mode: bf16 mixed precision, Batch size 256, custom learning rate, 8x on single server, print-frequency 1 and include dataloading time in throughput computation
```bash
export MASTER_ADDR=localhost
export MASTER_PORT=12355
mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=resnet50 --device=hpu --batch-size=256 --epochs=90 --print-freq=1 --output-dir=. --seed=123 --hmp --hmp-bf16 ./ops_bf16_Resnet.txt --hmp-fp32 ./ops_fp32_Resnet.txt --custom-lr-values 0.275 0.45 0.625 0.8 0.08 0.008 0.0008 --custom-lr-milestones 1 2 3 4 30 60 80 --deterministic --dl-time-exclude=False
```

ii. Example: ResNet50, lazy mode: bf16 mixed precision, Batch size 256, custom learning rate, 8x on single server, exclude dataloading time in throughput computation
```bash
export MASTER_ADDR=localhost
export MASTER_PORT=12355
mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=resnet50 --device=hpu --batch-size=256 --epochs=90 --print-freq=1 --output-dir=. --seed=123 --hmp --hmp-bf16 ./ops_bf16_Resnet.txt --hmp-fp32 ./ops_fp32_Resnet.txt --custom-lr-values 0.275 0.45 0.625 0.8 0.08 0.008 0.0008 --custom-lr-milestones 1 2 3 4 30 60 80 --deterministic --dl-time-exclude=True
```

iii. Example: ResNet50, lazy mode: bf16 mixed precision, Batch size 256, custom learning rate, 8x on single server, include dataloading time in throughput computation, use habana_dataloader (with hardware decode support on **Gaudi2**), worker(decoder) instances 8
```bash
export MASTER_ADDR=localhost
export MASTER_PORT=12355
mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=resnet50 --device=hpu --batch-size=256 --epochs=90 --workers=8 --print-freq=1 --output-dir=. --seed=123 --hmp --hmp-bf16 ./ops_bf16_Resnet.txt --hmp-fp32 ./ops_fp32_Resnet.txt --custom-lr-values 0.275 0.45 0.625 0.8 0.08 0.008 0.0008 --custom-lr-milestones 1 2 3 4 30 60 80 --deterministic --dl-time-exclude=False --dl-worker-type="HABANA"
```

iv. Example: ResNet50, lazy mode: bf16 mixed precision, Batch size 256, custom learning rate, 8x on single server, include dataloading time in throughput computation, print-frequency 10 and native pyTorch dataloader
```bash
export MASTER_ADDR=localhost
export MASTER_PORT=12355
mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=resnet50 --device=hpu --batch-size=256 --epochs=90 --workers=10 --dl-worker-type=MP --print-freq=10 --output-dir=. --seed=123 --hmp --hmp-bf16 ./ops_bf16_Resnet.txt --hmp-fp32 ./ops_fp32_Resnet.txt --custom-lr-values 0.275 0.45 0.625 0.8 0.08 0.008 0.0008 --custom-lr-milestones 1 2 3 4 30 60 80 --deterministic --dl-time-exclude=False
```

v. Example: ResNet50, lazy mode: bf16 mixed precision, Batch size 256, custom learning rate, 16x on multiple servers using Gaudi NIC, include dataloading time in throughput computation
- NOTE: MODIFY IP ADDRESS BELOW TO MATCH YOUR SYSTEM.
    - **`-H`**: set this to a comma-separated list of host IP addresses
    - **`--mca btl_tcp_if_include`**: Provide network interface associated with IP address. More details: [Open MPI documentation](https://www.open-mpi.org/faq/?category=tcp#tcp-selection). If you get mpirun `btl_tcp_if_include` errors, try un-setting this environment variable and let the training script automatically detect the network interface associated with the host IP address.
    - **`HCCL_SOCKET_IFNAME`**: HCCL_SOCKET_IFNAME defines the prefix of the network interface name that is used for HCCL sideband TCP communication. If not set, the first network interface with a name that does not start with lo or docker will be used.
```bash
export MASTER_ADDR=10.3.124.124
export MASTER_PORT=12355
mpirun --allow-run-as-root --mca plm_rsh_args -p3022 --bind-to core --map-by ppr:4:socket:PE=7 -np 16 --mca btl_tcp_if_include 10.3.124.124/16 --merge-stderr-to-stdout --prefix $MPI_ROOT -H 10.3.124.124:8,10.3.124.175:8 -x GC_KERNEL_PATH -x PYTHONPATH -x MASTER_ADDR -x MASTER_PORT $PYTHON -u train.py --batch-size=256 --model=resnet50 --device=hpu --workers=8 --print-freq=1 --deterministic --data-path=/data/pytorch/imagenet/ILSVRC2012 --epochs=90 --hmp --hmp-bf16 ./ops_bf16_Resnet.txt --hmp-fp32 ./ops_fp32_Resnet.txt --custom-lr-values 0.475 0.85 1.225 1.6 0.16 0.016 0.0016 --custom-lr-milestones 1 2 3 4 30 60 80 --dl-time-exclude=False --dl-worker-type=HABANA
```

vi. Example: ResNet50, lazy mode: bf16 mixed precision, Batch size 256, custom learning rate, 16x on multiple servers using raw TCP based Host NIC, include dataloading time in throughput computation
- NOTE: MODIFY IP ADDRESS BELOW TO MATCH YOUR SYSTEM.
    - **`-H`**: set this to a comma-separated list of host IP addresses
    - **`--mca btl_tcp_if_include`**: Provide network interface associated with IP address. More details: [Open MPI documentation](https://www.open-mpi.org/faq/?category=tcp#tcp-selection). If you get mpirun `btl_tcp_if_include` errors, try un-setting this environment variable and let the training script automatically detect the network interface associated with the host IP address.
    - **`HCCL_SOCKET_IFNAME`**: HCCL_SOCKET_IFNAME defines the prefix of the network interface name that is used for HCCL sideband TCP communication. If not set, the first network interface with a name that does not start with lo or docker will be used.
```bash
export MASTER_ADDR=10.3.124.124
export MASTER_PORT=12355
export HCCL_OVER_TCP=1
mpirun --allow-run-as-root --mca plm_rsh_args -p3022 --bind-to core --map-by ppr:4:socket:PE=7 -np 16 --mca btl_tcp_if_include 10.3.124.124/16 --merge-stderr-to-stdout --prefix $MPI_ROOT -H 10.3.124.124:8,10.3.124.175:8 -x GC_KERNEL_PATH -x HCCL_OVER_TCP -x PYTHONPATH -x MASTER_ADDR -x MASTER_PORT $PYTHON -u train.py --batch-size=256 --model=resnet50 --device=hpu --workers=8 --print-freq=1 --deterministic --data-path=/data/pytorch/imagenet/ILSVRC2012 --epochs=90 --hmp --hmp-bf16 ./ops_bf16_Resnet.txt --hmp-fp32 ./ops_fp32_Resnet.txt --custom-lr-values 0.475 0.85 1.225 1.6 0.16 0.016 0.0016 --custom-lr-milestones 1 2 3 4 30 60 80 --dl-time-exclude=False --dl-worker-type=HABANA
```

vii. Example: ResNet50, lazy mode: bf16 mixed precision, Batch size 256, custom learning rate, 16x on multiple servers using libFabric based Host NIC, include dataloading time in throughput computation
- NOTE: MODIFY IP ADDRESS BELOW TO MATCH YOUR SYSTEM.
    - **`-H`**: set this to a comma-separated list of host IP addresses
    - **`--mca btl_tcp_if_include`**: Provide network interface associated with IP address. More details: [Open MPI documentation](https://www.open-mpi.org/faq/?category=tcp#tcp-selection). If you get mpirun `btl_tcp_if_include` errors, try un-setting this environment variable and let the training script automatically detect the network interface associated with the host IP address.
    - **`HCCL_SOCKET_IFNAME`**: HCCL_SOCKET_IFNAME defines the prefix of the network interface name that is used for HCCL sideband TCP communication. If not set, the first network interface with a name that does not start with lo or docker will be used.
```bash
export MASTER_ADDR=10.3.124.124
export MASTER_PORT=12355
export HCCL_OVER_OFI=1
mpirun --allow-run-as-root --mca plm_rsh_args -p3022 --bind-to core --map-by ppr:4:socket:PE=7 -np 16 --mca btl_tcp_if_include 10.3.124.124/16 --merge-stderr-to-stdout --prefix $MPI_ROOT -H 10.3.124.124:8,10.3.124.175:8 -x GC_KERNEL_PATH -x HCCL_OVER_OFI -x PYTHONPATH -x MASTER_ADDR -x MASTER_PORT $PYTHON -u train.py --batch-size=256 --model=resnet50 --device=hpu --workers=8 --print-freq=1 --deterministic --data-path=/data/pytorch/imagenet/ILSVRC2012 --epochs=90 --hmp --hmp-bf16 ./ops_bf16_Resnet.txt --hmp-fp32 ./ops_fp32_Resnet.txt --custom-lr-values 0.475 0.85 1.225 1.6 0.16 0.016 0.0016 --custom-lr-milestones 1 2 3 4 30 60 80 --dl-time-exclude=False --dl-worker-type=HABANA
```

viii. Example: ResNext101, lazy mode: bf16 mixed precision, Batch size 128, 8x on single server, include dataloading time in throughput computation
```bash
export MASTER_ADDR=localhost
export MASTER_PORT=12355
mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=resnext101_32x4d --device=hpu --batch-size=128 --epochs=100 --print-freq=1 --output-dir=. --seed=123 --hmp --hmp-bf16 ./ops_bf16_Resnet.txt --hmp-fp32 ./ops_fp32_Resnet.txt --deterministic --dl-time-exclude=False
```

ix. Example: ResNext101, lazy mode: bf16 mixed precision, Batch size 256, 8x on single server, use habana_dataloader (with hardware decode support on **Gaudi2**), include dataloading time in throughput computation
```bash
export MASTER_ADDR=localhost
export MASTER_PORT=12355
mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=resnext101_32x4d --device=hpu --batch-size=256 --epochs=100 --print-freq=1 --output-dir=. --seed=123 --hmp --hmp-bf16 ./ops_bf16_Resnet.txt --hmp-fp32 ./ops_fp32_Resnet.txt --deterministic --dl-time-exclude=False --dl-worker-type=HABANA
```

x. Example: Mobilenet_v2, lazy mode: bf16 mixed precision, Batch size 256, 8x on single server, include dataloading time in throughput computation
```bash
export MASTER_ADDR=localhost
export MASTER_PORT=12355
mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=mobilenet_v2 --device=hpu --batch-size=256 --epochs=90 --print-freq=10 --output-dir=. --seed=123 --hmp --hmp-bf16 ./ops_bf16_Mobilenet.txt --hmp-fp32 ./ops_fp32_Mobilenet.txt --lr=0.36 --wd=0.00004 --lr-step-size=1 --lr-gamma=0.98 --momentum=0.9 --deterministic --dl-time-exclude=False
```

xi. Example: Mobilenet_v2, lazy mode: bf16 mixed precision, Batch size 256, 8x on single server, exclude dataloading time in throughput computation
```bash
export MASTER_ADDR=localhost
export MASTER_PORT=12355
mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=mobilenet_v2 --device=hpu --batch-size=256 --epochs=90 --print-freq=10 --output-dir=. --seed=123 --hmp --hmp-bf16 ./ops_bf16_Mobilenet.txt --hmp-fp32 ./ops_fp32_Mobilenet.txt --lr=0.36 --wd=0.00004 --lr-step-size=1 --lr-gamma=0.98 --momentum=0.9 --deterministic --dl-time-exclude=True
```

xii. Example: Mobilenet_v2, lazy mode: bf16 mixed precision, Batch size 256, 8x on single server, include dataloading time in throughput computation, use habana_dataloader, worker(decoder) instances 8
```bash
export MASTER_ADDR=localhost
export MASTER_PORT=12355
mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=mobilenet_v2 --device=hpu --batch-size=256 --epochs=90 --workers=8 --print-freq=10 --output-dir=. --seed=123 --hmp --hmp-bf16 ./ops_bf16_Mobilenet.txt --hmp-fp32 ./ops_fp32_Mobilenet.txt --lr=0.36 --wd=0.00004 --lr-step-size=1 --lr-gamma=0.98 --momentum=0.9 --deterministic --dl-time-exclude=False --dl-worker-type="HABANA"
```

xiii. Example: Mobilenet_v2, lazy mode: bf16 mixed precision, Batch size 256, custom learning rate, 8x on single server, include dataloading time in throughput computation, print-frequency 10
```bash
export MASTER_ADDR=localhost
export MASTER_PORT=12355
mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=mobilenet_v2 --device=hpu --batch-size=256 --epochs=90 --workers=8 --print-freq=10 --output-dir=. --seed=123 --hmp --hmp-bf16 ./ops_bf16_Mobilenet.txt --hmp-fp32 ./ops_fp32_Mobilenet.txt --lr=0.36 --wd=0.00004 --lr-step-size=1 --lr-gamma=0.98 --momentum=0.9 --deterministic --dl-time-exclude=False --dl-worker-type="HABANA"
```

xiv: Example: GoogLeNet, Batch Size 256, bf16 precision, lazy mode, 8x on single server.
```bash
export MASTER_ADDR=localhost
export MASTER_PORT=12355
mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON main.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=googlenet --device=hpu --batch-size=256 --epochs=90 --lr=0.2828 --enable-lazy --print-interval=20 --dl-worker-type=HABANA --no-aux-logits --hmp --hmp-bf16 ./ops_bf16_googlenet.txt --hmp-fp32 ./ops_fp32_googlenet.txt --workers=8
```

xv: Example: GoogLeNet, Batch Size 128, fp32 mixed precision, lazy mode, 8x on single server.
```bash
export MASTER_ADDR=localhost
export MASTER_PORT=12355
mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON main.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=googlenet --device=hpu --batch-size=128 --epochs=90 --lr=0.2 --enable-lazy --print-interval=20 --dl-worker-type=HABANA --seed=123 --no-aux-logits --workers=8
```

xvi: Example: GoogLeNet, Batch Size 256, bf16 precision, lazy mode, 8x on a single HLS1 machine, use habana_dataloader, worker(decoder) instances 8 with print interval as 20
```bash
export MASTER_ADDR=localhost
export MASTER_PORT=12355
mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON main.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=googlenet --device=hpu --batch-size=256 --epochs=90 --lr=0.2828 --enable-lazy --print-interval=20 --dl-worker-type=HABANA --seed=123 --no-aux-logits --hmp --hmp-bf16 ./ops_bf16_googlenet.txt --hmp-fp32 ./ops_fp32_googlenet.txt --workers=8
```

# Supported Configurations

**Resnet50, ResNext101**

| Device | SynapseAI Version | PyTorch Version |
|-----|-----|-----|
| Gaudi | 1.5.0 | 1.11.0 |
| **Gaudi2** | 1.5.0 | 1.11.0 |

**MobileNet_v2 and ResNet152**

| Device | SynapseAI Version | PyTorch Version |
|-----|-----|-----|
| Gaudi | 1.4.1 | 1.10.2 |

**GoogLeNet**

| Device | SynapseAI Version | PyTorch Version |
|-----|-----|-----|
| Gaudi | 1.5.0 | 1.11.0 |

# Changelog
## 1.5.0
 - Extended support for habana_dataloader with hardware decode support for **Gaudi2** to support 8 instances on Resnet50/Resnext101
 - Removed channels-last=true support
 - Weights and other dependent parameters need not be permuted anymore

## 1.4.1
 - Added support for habana_dataloader with hardware decode support for **Gaudi2**
## 1.4.0
 - Default execution mode modified to lazy mode

# Known Issues
- channels-last = true is not supported. Train ResNet & MobileNet_v2 with channels-last=false.
- Placing mark_step() arbitrarily may lead to undefined behaviour. Recommend to keep mark_step() as shown in provided scripts.
- Habana_dataloader only supports Imagenet (JPEG) and 8 worker instances on Gaudi.
- Habana_dataloader with hardware decode is supported for training with ImageNet (JPEG) for 1 **Gaudi2** card for Resnet50/Resnext101.
- Only scripts & configurations mentioned in this README are supported and verified.

# Training Script Modifications
The following are the changes added to the training script (train.py) and utilities(utils.py):

1. Added support for Habana devices

   a. Load Habana specific library.

   b. Certain environment variables are defined for habana device.

   c. Added support to run Resnet, MobileNet_v2 & GoogLeNet training in lazy mode in addition to the eager mode.
    mark_step() is performed to trigger execution of the graph.

   d. Added multi-HPU (including multiple node) and mixed precision support.

   c. Modified training script to use mpirun for distributed training.

   f. Changes for dynamic loading of HCCL library.

   g. Changed start_method for torch multiprocessing.

   h. GoogLeNet training script is cloned from https://github.com/pytorch/examples/tree/master/imagenet with modifications.

   i. Checkpoint saving is by default disabled. It can be enabled using –save-checkpoint option.

2. Dataloader related changes

    a. Added –deterministic flag to make data loading deterministic with Pytorch dataloader.

    b. Added support for including dataloader time in performance computation. To match behavior of base script, it is disabled by default.

    c. Added CPU based habana_dataloader with faster image preprocessing (only JPEG) support. It is enabled by default.

    d. Added habana_dataloader with hardware decode support for Imagenet dataset on **Gaudi2**.

3. To improve performance

   a. Optimized FusedSGD operator is used in place of torch.optim.SGD for lazy mode.

   b. Added support for lowering print frequency of loss.

   c. Modified script to avoid broadcast at the beginning of iteration.

   d. Added additional parameter to DistributedDataParallel to increase the size of the bucket to combine all the all-reduce calls to a single call.

   e. Use CPU based habana_dataloader.

   f. Enable pinned memory for data loader.

   g. Accuracy calculation is performed on the CPU as lower performance observed with "top-k" on device

   h. Added extra mark_step() before loss.backward() for MobileNet_v2.

   i. For GoogLeNet, --print-interval value of 20 is recommended for torch multiprocess dataloader optimum performance. It can be set lower for more granularity at the cost of performance.

   j. Copy of input & targets is made non-blocking.

   k. Size of first gradient bucket is adjusted to minimize the number of allreduce.

4. Skip printing eval phase stats if eval steps is zero.

5. Added support in script to override default LR scheduler with custom schedule passed as parameters to script.

6. For GoogLeNet, first two batches' training and evaluation time is not included in the average batch time performance numbers due to long wait times of initial data loading to device.

The model changes are listed below:

1. Added support for resnext101_32x4d.
