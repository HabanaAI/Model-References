
# Classification for PyTorch
This folder contains scripts to train ResNet50, ResNet152, ResNeXt101, MobileNetV2 & GoogLeNet models on Habana Gaudi device to achieve state-of-the-art accuracy. To obtain model performance data, refer to the [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

For more information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).


## Table of Contents
  - [Model-References](../../../../README.md)
  - [Model Overview](#model-overview)
  - [Setup](#setup)
  - [Media Loading Acceleration](#media-loading-acceleration)
  - [Training and Examples](#training-and-examples)
  - [Pre-trained Model](#pre-trained-model)
  - [Supported Configurations](#supported-configurations)
  - [Changelog](#changelog)
  - [Known Issues](#known-issues)


## Model Overview
The base model used is from [GitHub: PyTorch-Vision](https://github.com/pytorch/vision/tree/release/0.10/torchvision/models). A copy of the ResNet model file [resnet.py](model/resnet.py) is used to make
certain model changes functionally equivalent to the original model. The base training and modelling scripts for training are based on a clone of https://github.com/pytorch/vision.git with certain changes for modeling and training script. Please refer to later sections on training script and model modifications for a summary of
modifications to the original files.

- ResNet50 - Eager mode and Lazy mode training for BS128 with FP32 and BS256 with BF16 mixed precision.
- ResNet152 - Eager mode and Lazy mode training for BS128 with BF16 mixed precision.
- ResNeXt101 - Eager mode and Lazy mode training for BS64 with FP32 and BS128 with BF16 mixed precision.
- MobileNetV2 - Eager mode and Lazy mode training for BS256 with FP32 and BF16 mixed precision.
- GoogLeNet demos - Eager mode and Lazy mode training for BS128 with FP32 and BF256 mixed precision.

## Setup
Please follow the instructions provided in the [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the
environment including the `$PYTHON` environment variable.
The guide will walk you through the process of setting up your system to run the model on Gaudi.


### Clone Habana Model-References
In the docker container, clone this repository and switch to the branch that
matches your SynapseAI version. You can run the
[`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.

> ### ⚠ Exception for MobileNetV2
> The latest SynapseAI version supported by MobileNetV2 is **1.4.1**

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```
Go to PyTorch torchvision directory:
```bash
cd Model-References/PyTorch/computer_vision/classification/torchvision
```
**Note:** If the repository is not in the PYTHONPATH, make sure to update by running the below:
```bash
export PYTHONPATH=/path/to/Model-References:$PYTHONPATH
```

### Install Model Requirements
Install the required packages using pip:
```bash
pip install -r requirements.txt
```
### Set Up Dataset
ImageNet 2012 dataset needs to be organized as per PyTorch requirements. PyTorch requirements are specified in the link below which contains scripts to organize ImageNet data.
https://github.com/soumith/imagenet-multiGPU.torch

## Media Loading Acceleration
Gaudi2 offers a dedicated hardware engine for Media Loading operations. For more details, please refer to [Habana Media Loader page](https://docs.habana.ai/en/latest/PyTorch/Habana_Media_Loader_PT/Media_Loader_PT.html)

## Training and Examples

The following commands assume that ImageNet dataset is available at `/data/pytorch/imagenet/ILSVRC2012/` directory.

- To see the available training parameters for ResNet50, ResNet152, ResNeXt101 and MobileNetV2, run:
```bash
$PYTHON -u train.py --help
```
- To see the available training parameters for GoogLeNet, run:
```bash
$PYTHON -u main.py --help
```

### Single Card and Multi-Card Training Examples
**Run training on 1 HPU:**
- ResNet50, lazy mode, BF16 mixed precision, batch Size 256, custom learning rate, Habana dataloader (with hardware decode support on **Gaudi2**), 1 HPU on a single server:
  ```bash
  $PYTHON -u train.py --dl-worker-type HABANA --batch-size 256 --model resnet50 --device hpu --workers 8 --print-freq 20 --dl-time-exclude False --deterministic --data-path /data/pytorch/imagenet/ILSVRC2012 --epochs 90 --hmp --hmp-bf16 ./ops_bf16_Resnet.txt --hmp-fp32 ./ops_fp32_Resnet.txt  --lr 0.1 --custom-lr-values 0.1 0.01 0.001 0.0001 --custom-lr-milestones 0 30 60 80
  ```
- ResNeXt101 lazy mode, BF16 mixed precision, batch size 256, custom learning rate, Habana dataloader (with hardware decode support on **Gaudi2**), 1 HPU on a single server:
  ```bash
  $PYTHON -u train.py --dl-worker-type HABANA --batch-size 256 --model resnext101_32x4d --device hpu --workers 8 --print-freq 20 --dl-time-exclude False --deterministic --data-path /data/pytorch/imagenet/ILSVRC2012 --epochs 100 --hmp --hmp-bf16 ./ops_bf16_Resnet.txt --hmp-fp32 ./ops_fp32_Resnet.txt --lr 0.1 --custom-lr-values 0.1 0.01 0.001 0.0001 --custom-lr-milestones 0 30 60 80
  ```
- ResNeXt101 lazy mode, BF16 mixed precision, batch size 128, custom learning rate, 1 HPU on a single server:
  ```bash
  $PYTHON -u train.py --dl-worker-type HABANA --batch-size 128 --model resnext101_32x4d --device hpu --workers 8 --print-freq 20 --dl-time-exclude False --deterministic --data-path /data/pytorch/imagenet/ILSVRC2012 --epochs 100 --hmp --hmp-bf16 ./ops_bf16_Resnet.txt --hmp-fp32 ./ops_fp32_Resnet.txt --lr 0.1 --custom-lr-values 0.1 0.01 0.001 0.0001 --custom-lr-milestones 0 30 60 80
  ```
- ResNet152, lazy mode, BF16 mixed precision, batch size 128, custom learning rate, 1 HPU on a single server:
  ```bash
  $PYTHON -u train.py --dl-worker-type HABANA --batch-size 128 --model resnet152 --device hpu --workers 8 --print-freq 20 --dl-time-exclude False --deterministic --data-path /data/pytorch/imagenet/ILSVRC2012 --epochs 90 --hmp --hmp-bf16 ./ops_bf16_Resnet.txt --hmp-fp32 ./ops_fp32_Resnet.txt --lr 0.1 --custom-lr-values 0.1 0.01 0.001 0.0001 --custom-lr-milestones 0 30 60 80
  ```
- MobileNetV2, lazy mode, BF16 mixed precision, batch size 256, 1 HPU on a single server with default PyTorch dataloader:
  ```bash
  $PYTHON -u train.py --batch-size 256 --model mobilenet_v2 --device hpu --print-freq 10 --deterministic --data-path /data/pytorch/imagenet/ILSVRC2012 --epochs 90 --hmp --hmp-bf16 ./ops_bf16_Mobilenet.txt --hmp-fp32 ./ops_fp32_Mobilenet.txt --dl-time-exclude=False --lr 0.045 --wd 0.00004 --lr-step-size 1 --lr-gamma 0.98 --momentum 0.9
  ```
- GoogLeNet, batch size 256, BF16 precision, lazy mode, 1 HPU on a single server:
  ```bash
  $PYTHON -u main.py --batch-size 256 --data-path /data/pytorch/imagenet/ILSVRC2012 --hmp --hmp-bf16 ./ops_bf16_googlenet.txt --hmp-fp32 ./ops_fp32_googlenet.txt --device hpu --dl-worker-type HABANA --epochs 90 --lr 0.1 --enable-lazy --model googlenet --seed 123 --no-aux-logits --print-interval 20 --workers 8
  ```
- GoogLeNet, batch size 128, FP32 precision, lazy mode, 1 HPU on a single server:
  ```bash
  $PYTHON -u main.py --batch-size 128 --data-path /data/pytorch/imagenet/ILSVRC2012 --device hpu --dl-worker-type HABANA --epochs 90 --lr 0.07 --enable-lazy --model googlenet --seed 123 --no-aux-logits --print-interval 20 --workers 8
  ```
**Run training on 8 HPUs:**

To run multi-card training, make sure the host machine has 512 GB of RAM installed.
Also ensure you followed the [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html)
to install and set up docker, so that the docker has access to the eight Gaudi cards
required for multi-card training.

**NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Scaling_Guide/DDP_Based_Scaling.html#mpirun-configuration).

- ResNet50, lazy mode, BF16 mixed precision, batch size 256, custom learning rate, 8 HPUs on a single server, print-frequency 1 and include dataloading time in throughput computation:
  ```bash
  export MASTER_ADDR=localhost
  export MASTER_PORT=12355
  mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=resnet50 --device=hpu --batch-size=256 --epochs=90 --print-freq=1 --output-dir=. --seed=123 --hmp --hmp-bf16 ./ops_bf16_Resnet.txt --hmp-fp32 ./ops_fp32_Resnet.txt --custom-lr-values 0.275 0.45 0.625 0.8 0.08 0.008 0.0008 --custom-lr-milestones 1 2 3 4 30 60 80 --deterministic --dl-time-exclude=False
  ```
- ResNet50, lazy mode, BF16 mixed precision, batch size 256, custom learning rate, 8 HPUs on a single server, exclude dataloading time in throughput computation:
  ```bash
  export MASTER_ADDR=localhost
  export MASTER_PORT=12355
  mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=resnet50 --device=hpu --batch-size=256 --epochs=90 --print-freq=1 --output-dir=. --seed=123 --hmp --hmp-bf16 ./ops_bf16_Resnet.txt --hmp-fp32 ./ops_fp32_Resnet.txt --custom-lr-values 0.275 0.45 0.625 0.8 0.08 0.008 0.0008 --custom-lr-milestones 1 2 3 4 30 60 80 --deterministic --dl-time-exclude=True
  ```
- ResNet50, lazy mode, BF16 mixed precision, batch size 256, custom learning rate, 8 HPUs on a single server, include dataloading time in throughput computation, use `habana_dataloader` (with hardware decode support on **Gaudi2**), 8 worker (decoder) instances:
  ```bash
  export MASTER_ADDR=localhost
  export MASTER_PORT=12355
  mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=resnet50 --device=hpu --batch-size=256 --epochs=90 --workers=8 --print-freq=1 --output-dir=. --seed=123 --hmp --hmp-bf16 ./ops_bf16_Resnet.txt --hmp-fp32 ./ops_fp32_Resnet.txt --custom-lr-values 0.275 0.45 0.625 0.8 0.08 0.008 0.0008 --custom-lr-milestones 1 2 3 4 30 60 80 --deterministic --dl-time-exclude=False --dl-worker-type="HABANA"
  ```
- ResNet50, lazy mode, BF16 mixed precision, batch size 256, custom learning rate, 8 HPUs on a single server, include dataloading time in throughput computation, print-frequency 10 and native PyTorch dataloader:
  ```bash
  export MASTER_ADDR=localhost
  export MASTER_PORT=12355
  mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=resnet50 --device=hpu --batch-size=256 --epochs=90 --workers=10 --dl-worker-type=MP --print-freq=10 --output-dir=. --seed=123 --hmp --hmp-bf16 ./ops_bf16_Resnet.txt --hmp-fp32 ./ops_fp32_Resnet.txt --custom-lr-values 0.275 0.45 0.625 0.8 0.08 0.008 0.0008 --custom-lr-milestones 1 2 3 4 30 60 80 --deterministic --dl-time-exclude=False
  ```
- ResNeXt101, lazy mode, BF16 mixed precision, batch size 128, 8 HPUs on s single server, include dataloading time in throughput computation:
  ```bash
  export MASTER_ADDR=localhost
  export MASTER_PORT=12355
  mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=resnext101_32x4d --device=hpu --batch-size=128 --epochs=100 --print-freq=1 --output-dir=. --seed=123 --hmp --hmp-bf16 ./ops_bf16_Resnet.txt --hmp-fp32 ./ops_fp32_Resnet.txt --deterministic --dl-time-exclude=False
  ```
- ResNeXt101, lazy mode, BF16 mixed precision, batch size 256, 8 HPUs on a single server, use `habana_dataloader` (with hardware decode support on **Gaudi2**), include dataloading time in throughput computation:
  ```bash
  export MASTER_ADDR=localhost
  export MASTER_PORT=12355
  mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=resnext101_32x4d --device=hpu --batch-size=256 --epochs=100 --print-freq=1 --output-dir=. --seed=123 --hmp --hmp-bf16 ./ops_bf16_Resnet.txt --hmp-fp32 ./ops_fp32_Resnet.txt --deterministic --dl-time-exclude=False --dl-worker-type=HABANA
  ```

- MobileNetV2, lazy mode, BF16 mixed precision, batch size 256, 8 HPUs on a single server, include dataloading time in throughput computation:
  ```bash
  export MASTER_ADDR=localhost
  export MASTER_PORT=12355
  mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=mobilenet_v2 --device=hpu --batch-size=256 --epochs=90 --print-freq=10 --output-dir=. --seed=123 --hmp --hmp-bf16 ./ops_bf16_Mobilenet.txt --hmp-fp32 ./ops_fp32_Mobilenet.txt --lr=0.36 --wd=0.00004 --lr-step-size=1 --lr-gamma=0.98 --momentum=0.9 --deterministic --dl-time-exclude=False
  ```
- MobileNetV2, lazy mode, BF16 mixed precision, batch size 256, 8 HPUs on a single server, exclude dataloading time in throughput computation:
  ```bash
  export MASTER_ADDR=localhost
  export MASTER_PORT=12355
  mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=mobilenet_v2 --device=hpu --batch-size=256 --epochs=90 --print-freq=10 --output-dir=. --seed=123 --hmp --hmp-bf16 ./ops_bf16_Mobilenet.txt --hmp-fp32 ./ops_fp32_Mobilenet.txt --lr=0.36 --wd=0.00004 --lr-step-size=1 --lr-gamma=0.98 --momentum=0.9 --deterministic --dl-time-exclude=True
  ```

- MobileNetV2, lazy mode, BF16 mixed precision, batch size 256, 8 HPUs on a single server, include dataloading time in throughput computation, use `habana_dataloader`, 8 worker (decoder) instances:
  ```bash
  export MASTER_ADDR=localhost
  export MASTER_PORT=12355
  mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=mobilenet_v2 --device=hpu --batch-size=256 --epochs=90 --workers=8 --print-freq=10 --output-dir=. --seed=123 --hmp --hmp-bf16 ./ops_bf16_Mobilenet.txt --hmp-fp32 ./ops_fp32_Mobilenet.txt --lr=0.36 --wd=0.00004 --lr-step-size=1 --lr-gamma=0.98 --momentum=0.9 --deterministic --dl-time-exclude=False --dl-worker-type="HABANA"
  ```
- MobileNetV2, lazy mode, BF16 mixed precision, batch size 256, custom learning rate, 8 HPUs on a single server, include dataloading time in throughput computation, print-frequency 10:
  ```bash
  export MASTER_ADDR=localhost
  export MASTER_PORT=12355
  mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=mobilenet_v2 --device=hpu --batch-size=256 --epochs=90 --workers=8 --print-freq=10 --output-dir=. --seed=123 --hmp --hmp-bf16 ./ops_bf16_Mobilenet.txt --hmp-fp32 ./ops_fp32_Mobilenet.txt --lr=0.36 --wd=0.00004 --lr-step-size=1 --lr-gamma=0.98 --momentum=0.9 --deterministic --dl-time-exclude=False --dl-worker-type="HABANA"
  ```
- GoogLeNet, batch Size 256, BF16 precision, lazy mode, 8 HPUs on a single server:
  ```bash
  export MASTER_ADDR=localhost
  export MASTER_PORT=12355
  mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON main.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=googlenet --device=hpu --batch-size=256 --epochs=90 --lr=0.2828 --enable-lazy --print-interval=20 --dl-worker-type=HABANA --no-aux-logits --hmp --hmp-bf16 ./ops_bf16_googlenet.txt --hmp-fp32 ./ops_fp32_googlenet.txt --workers=8
  ```
- GoogLeNet, batch size 128, FP32 mixed precision, lazy mode, 8 HPUs on a single server:
  ```bash
  export MASTER_ADDR=localhost
  export MASTER_PORT=12355
  mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON main.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=googlenet --device=hpu --batch-size=128 --epochs=90 --lr=0.2 --enable-lazy --print-interval=20 --dl-worker-type=HABANA --seed=123 --no-aux-logits --workers=8
  ```
- GoogLeNet, batch size 256, BF16 precision, lazy mode, 8 HPUs on a single server, use `habana_dataloader`, 8 worker (decoder) instances with print interval as 20:
  ```bash
  export MASTER_ADDR=localhost
  export MASTER_PORT=12355
  mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON main.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=googlenet --device=hpu --batch-size=256 --epochs=90 --lr=0.2828 --enable-lazy --print-interval=20 --dl-worker-type=HABANA --seed=123 --no-aux-logits --hmp --hmp-bf16 ./ops_bf16_googlenet.txt --hmp-fp32 ./ops_fp32_googlenet.txt --workers=8
  ```
### Multi-Server Training Setup
To run multi-server training, make sure the host machine has 512 GB of RAM installed.
Also ensure you followed the [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html)
to install and set up docker, so that the docker has access to all Gaudi cards
required for multi-server training. Multi-server configuration for ResNet50 training up
to two servers, each with eight Gaudi cards, have been verified.

Before execution of the multi-server demo scripts, make sure all server network interfaces are up. You can change the state of each network interface managed by the habanalabs driver using the following command:
```
sudo ip link set <interface_name> up
```
To identify if a specific network interface is managed by the habanalabs driver type, run:
```
sudo ethtool -i <interface_name>
```
#### Docker ssh Port Setup for Multi-Server Training

By default, the Habana docker uses `port 22` for ssh. If the default port is occupied then the port number can also be changed. Run the following commands to configure the selected port number , `port 3022` in the example below:

```bash
sed -i 's/#Port 22/Port 3022/g' /etc/ssh/sshd_config
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
service ssh restart
```

#### Run Multi-Server over Host NICs

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

#### Set up Password-less ssh

To set up password-less ssh between all connected servers used in scale-out training, follow the below steps:
1. Do the following in all the servers' docker sessions:
   ```bash
   mkdir ~/.ssh
   cd ~/.ssh
   ssh-keygen -t rsa -b 4096
   ```
   a. Copy id_rsa.pub contents from every server's docker to every other server's docker's ~/.ssh/authorized_keys (all public keys need to be in all hosts' authorized_keys):
   ```bash
   cat id_rsa.pub > authorized_keys
   vi authorized_keys
   ```
   b. Copy the contents from inside to other systems.

   c. Paste all hosts' public keys in all hosts' “authorized_keys” file.

2. On each system, add all hosts (including itself) to known_hosts. The IP addresses used below are just for illustration:
   ```bash
   ssh-keyscan -p 3022 -H 10.3.124.124 >> ~/.ssh/known_hosts
   ssh-keyscan -p 3022 -H 10.3.124.175 >> ~/.ssh/known_hosts
   ```

### Multi-Server Training Examples
**Run training on 16 HPUs:**

**NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Scaling_Guide/DDP_Based_Scaling.html#mpirun-configuration).

- ResNet50, lazy mode, BF16 mixed precision, batch size 256, custom learning rate, 16 HPUs on multiple servers using Gaudi NIC, include dataloading time in throughput computation:
    - **`-H`**: Set this to a comma-separated list of host IP addresses. Make sure to modify IP address below to match your system.
    - **`--mca btl_tcp_if_include`**: Provide network interface associated with IP address. More details: [Open MPI documentation](https://www.open-mpi.org/faq/?category=tcp#tcp-selection). If you get mpirun `btl_tcp_if_include` errors, try un-setting this environment variable and let the training script automatically detect the network interface associated with the host IP address.
    - **`HCCL_SOCKET_IFNAME`**: HCCL_SOCKET_IFNAME defines the prefix of the network interface name that is used for HCCL sideband TCP communication. If not set, the first network interface with a name that does not start with lo or docker will be used.
    - `$MPI_ROOT` environment variable is set automatically during Setup. See [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) for details.
  ```bash
  export MASTER_ADDR=10.3.124.124
  export MASTER_PORT=12355
  mpirun --allow-run-as-root --mca plm_rsh_args -p3022 --bind-to core --map-by ppr:4:socket:PE=7 -np 16 --mca btl_tcp_if_include 10.3.124.124/16 --merge-stderr-to-stdout --prefix $MPI_ROOT -H 10.3.124.124:8,10.3.124.175:8 -x GC_KERNEL_PATH -x PYTHONPATH -x MASTER_ADDR -x MASTER_PORT $PYTHON -u train.py --batch-size=256 --model=resnet50 --device=hpu --workers=8 --print-freq=1 --deterministic --data-path=/data/pytorch/imagenet/ILSVRC2012 --epochs=90 --hmp --hmp-bf16 ./ops_bf16_Resnet.txt --hmp-fp32 ./ops_fp32_Resnet.txt --custom-lr-values 0.475 0.85 1.225 1.6 0.16 0.016 0.0016 --custom-lr-milestones 1 2 3 4 30 60 80 --dl-time-exclude=False --dl-worker-type=HABANA
  ```

- ResNet50, lazy mode, BF16 mixed precision, batch size 256, custom learning rate, 16 HPUs on multiple servers using raw TCP based Host NIC, include dataloading time in throughput computation:
    - **`-H`**: Set this to a comma-separated list of host IP addresses. Make sure to modify IP address below to match your system.
    - **`--mca btl_tcp_if_include`**: Provide network interface associated with IP address. More details: [Open MPI documentation](https://www.open-mpi.org/faq/?category=tcp#tcp-selection). If you get mpirun `btl_tcp_if_include` errors, try un-setting this environment variable and let the training script automatically detect the network interface associated with the host IP address.
    - **`HCCL_SOCKET_IFNAME`**: HCCL_SOCKET_IFNAME defines the prefix of the network interface name that is used for HCCL sideband TCP communication. If not set, the first network interface with a name that does not start with lo or docker will be used.
    - `$MPI_ROOT` environment variable is set automatically during Setup. See [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) for details.
  ```bash
  export MASTER_ADDR=10.3.124.124
  export MASTER_PORT=12355
  export HCCL_OVER_TCP=1
  mpirun --allow-run-as-root --mca plm_rsh_args -p3022 --bind-to core --map-by ppr:4:socket:PE=7 -np 16 --mca btl_tcp_if_include 10.3.124.124/16 --merge-stderr-to-stdout --prefix $MPI_ROOT -H 10.3.124.124:8,10.3.124.175:8 -x GC_KERNEL_PATH -x HCCL_OVER_TCP -x PYTHONPATH -x MASTER_ADDR -x MASTER_PORT $PYTHON -u train.py --batch-size=256 --model=resnet50 --device=hpu --workers=8 --print-freq=1 --deterministic --data-path=/data/pytorch/imagenet/ILSVRC2012 --epochs=90 --hmp --hmp-bf16 ./ops_bf16_Resnet.txt --hmp-fp32 ./ops_fp32_Resnet.txt --custom-lr-values 0.475 0.85 1.225 1.6 0.16 0.016 0.0016 --custom-lr-milestones 1 2 3 4 30 60 80 --dl-time-exclude=False --dl-worker-type=HABANA
  ```

- ResNet50, lazy mode, BF16 mixed precision, batch size 256, custom learning rate, 16 HPUs on multiple servers using libFabric based Host NIC, include dataloading time in throughput computation:
    - **`-H`**: Set this to a comma-separated list of host IP addresses. Make sure to modify IP address below to match your system.
    - **`--mca btl_tcp_if_include`**: Provide network interface associated with IP address. More details: [Open MPI documentation](https://www.open-mpi.org/faq/?category=tcp#tcp-selection). If you get mpirun `btl_tcp_if_include` errors, try un-setting this environment variable and let the training script automatically detect the network interface associated with the host IP address.
    - **`HCCL_SOCKET_IFNAME`**: HCCL_SOCKET_IFNAME defines the prefix of the network interface name that is used for HCCL sideband TCP communication. If not set, the first network interface with a name that does not start with lo or docker will be used.
  ```bash
  export MASTER_ADDR=10.3.124.124
  export MASTER_PORT=12355
  export HCCL_OVER_OFI=1
  mpirun --allow-run-as-root --mca plm_rsh_args -p3022 --bind-to core --map-by ppr:4:socket:PE=7 -np 16 --mca btl_tcp_if_include 10.3.124.124/16 --merge-stderr-to-stdout --prefix $MPI_ROOT -H 10.3.124.124:8,10.3.124.175:8 -x GC_KERNEL_PATH -x HCCL_OVER_OFI -x PYTHONPATH -x MASTER_ADDR -x MASTER_PORT $PYTHON -u train.py --batch-size=256 --model=resnet50 --device=hpu --workers=8 --print-freq=1 --deterministic --data-path=/data/pytorch/imagenet/ILSVRC2012 --epochs=90 --hmp --hmp-bf16 ./ops_bf16_Resnet.txt --hmp-fp32 ./ops_fp32_Resnet.txt --custom-lr-values 0.475 0.85 1.225 1.6 0.16 0.016 0.0016 --custom-lr-milestones 1 2 3 4 30 60 80 --dl-time-exclude=False --dl-worker-type=HABANA
  ```

## Pre-trained Model
PyTorch ResNet50 is trained on Habana Gaudi cards and the saved model file is created. You can use it for fine-tuning or transfer learning tasks with your own datasets. To download the saved model file, please refer to [Habana Catalog](https://developer.habana.ai/catalog/description-resnet-for-pytorch/) to obtain the URL.


## Supported Configurations

**ResNet50, ResNeXt101**

| Device | SynapseAI Version | PyTorch Version |
|-----|-----|-----|
| Gaudi  | 1.6.1 | 1.12.0 |
| Gaudi2 | 1.6.1 | 1.12.0 |

**MobileNetV2**

| Device | SynapseAI Version | PyTorch Version |
|-----|-----|-----|
| Gaudi | 1.4.1 | 1.10.2 |

**GoogLeNet and ResNet152**

| Device | SynapseAI Version | PyTorch Version |
|-----|-----|-----|
| Gaudi | 1.6.1 | 1.12.0 |

## Changelog
### 1.5.0
 - Extended support for habana_dataloader with hardware decode support for Gaudi2 to support 8 instances on ResNet50/ResNeXt101.
 - Removed channels-last=true support.
### 1.4.1
 - Added support for habana_dataloader with hardware decode support for Gaudi2.
### 1.4.0
 - Default execution mode modified to lazy mode.

### Training Script Modifications
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

2. Dataloader related changes

    a. Added –deterministic flag to make data loading deterministic with Pytorch dataloader.

    b. Added support for including dataloader time in performance computation. To match behavior of base script, it is disabled by default.

    c. Added CPU based habana_dataloader with faster image preprocessing (only JPEG) support. It is enabled by default.

    d. Added habana_dataloader with hardware decode support for Imagenet dataset on **Gaudi2**.

3. To improve performance

   a. Permute convolution weight tensors & and any other dependent tensors like 'momentum' for better performance.

   b. Checkpoint saving involves getting trainable params and other state variables to CPU and permuting the weight tensors. Hence, checkpoint saving is by default disabled. It can be enabled using –save-checkpoint option.

   c. Optimized FusedSGD operator is used in place of torch.optim.SGD for lazy mode.

   d. Added support for lowering print frequency of loss.

   e. Modified script to avoid broadcast at the beginning of iteration.

   f. Added additional parameter to DistributedDataParallel to increase the size of the bucket to combine all the all-reduce calls to a single call.

   g. Use CPU based habana_dataloader.

   h. Enable pinned memory for data loader.

   i. Accuracy calculation is performed on the CPU as lower performance observed with "top-k" on device

   j. Added extra mark_step() before loss.backward() for MobileNet_v2.

   k. For GoogLeNet, --print-interval value of 20 is recommended for torch multiprocess dataloader optimum performance. It can be set lower for more granularity at the cost of performance.

   l. Copy of input & targets is made non-blocking.

   m. Size of first gradient bucket is adjusted to minimize the number of allreduce.

4. Skip printing eval phase stats if eval steps is zero.

5. Added support in script to override default LR scheduler with custom schedule passed as parameters to script.

6. For GoogLeNet, first two batches' training and evaluation time is not included in the average batch time performance numbers due to long wait times of initial data loading to device.

The model changes are listed below:

1. Added support for resnext101_32x4d.

## Known Issues
- channels-last = true is not supported. Train ResNet & MobileNet_v2 with channels-last=false.
- Placing mark_step() arbitrarily may lead to undefined behaviour. Recommend to keep mark_step() as shown in provided scripts.
- Habana_dataloader only supports Imagenet (JPEG) and 8 worker instances on Gaudi.
- Habana_dataloader with hardware decode support only supports training with Imagenet (JPEG) up to 8 cards on **Gaudi2** for Resnet50/Resnext101.
- Only scripts & configurations mentioned in this README are supported and verified.