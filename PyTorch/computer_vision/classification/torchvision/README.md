# Classification for PyTorch
This folder contains scripts to train ResNet50, ResNet152, ResNeXt101, MobileNetV2 & GoogLeNet models on Intel® Gaudi® AI Accelerator to achieve state-of-the-art accuracy. It also contains the scripts to run inference on ResNet50 and ResNeXt101 models on Intel® Gaudi® AI Accelerator. To obtain model performance data, refer to the [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

For more information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).


## Table of Contents
  - [Model-References](../../../../README.md)
  - [Model Overview](#model-overview)
  - [Setup](#setup)
  - [Media Loading Acceleration](#media-loading-acceleration)
  - [Training and Examples](#training-and-examples)
  - [Pre-trained Model and Checkpoint](#pre-trained-model-and-checkpoint)
  - [Inference and Examples](#inference-and-examples)
  - [Supported Configurations](#supported-configurations)
  - [Changelog](#changelog)
  - [Training Script Modifications](#training-script-modifications)
  - [Known Issues](#known-issues)


## Model Overview
The base model used is from [GitHub: PyTorch-Vision](https://github.com/pytorch/vision/tree/release/0.10/torchvision/models). A copy of the ResNet model file [resnet.py](model/resnet.py) is used to make
certain model changes functionally equivalent to the original model. The base training and modelling scripts for training are based on a clone of https://github.com/pytorch/vision.git with certain changes for modeling and training script. Please refer to later sections on training script and model modifications for a summary of
modifications to the original files.

- ResNet50 - Lazy mode training for BS128 with FP32 and BS256 with BF16 mixed precision.
- ResNet50 - Inference for BS256 with FP32 and BF16 mixed precision.
- ResNet152 - Lazy mode training for BS128 with BF16 mixed precision.
- ResNeXt101 - Lazy mode training for BS64 with FP32 and BS128 with BF16 mixed precision.
- ResNeXt101 - Inference for BS256 with FP32 and BF16 mixed precision.
- MobileNetV2 - Lazy mode training for BS256 with FP32 and BF16 mixed precision.
- GoogLeNet demos - Lazy mode training for BS128 with FP32 and BF256 mixed precision.

**Note**: Inference on ResNet50 and ResNeXt101 32x4d models are currently enabled only on **Gaudi2**.

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
  $PYTHON -u train.py --dl-worker-type HABANA --batch-size 256 --model resnet50 --device hpu --workers 8 --print-freq 20 --dl-time-exclude False --deterministic --data-path /data/pytorch/imagenet/ILSVRC2012 --epochs 90 --autocast  --lr 0.1 --custom-lr-values 0.1 0.01 0.001 0.0001 --custom-lr-milestones 0 30 60 80
  ```

- ResNet50, lazy mode, BF16 mixed precision, batch size 256, eval every 4th epoch with offset 3, base learning rate 2.5, label smoothing 0.1, FusedLARS with polynomial decay LR scheduler, 1 HPU on a single server, include dataloading time in throughput computation, Habana dataloader (with hardware decode support on **Gaudi2**), 8 worker (decoder) instances:
  ```bash
  $PYTHON -u train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=resnet50 --device=hpu --batch-size=256 --epochs=35 --workers=8 --print-freq=1200 --output-dir=. --autocast  --dl-time-exclude=False --dl-worker-type="HABANA" --optimizer=lars -eoe 3 -ebe 4 --lars_base_learning_rate 2.5 --label-smoothing=0.1
  ```
- ResNet50, eager mode, BF16 mixed precision, batch size 256, eval every 4th epoch with offset 3, base learning rate 2.5, label smoothing 0.1, FusedLARS with polynomial decay LR scheduler, 1 HPU on a single server, include dataloading time in throughput computation, Habana dataloader (with hardware decode support on **Gaudi2**), 8 worker (decoder) instances:
  ```bash
  export PT_HPU_LAZY_MODE=0
  $PYTHON -u train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=resnet50 --device=hpu --batch-size=256 --epochs=35 --workers=8 --print-freq=1200 --output-dir=. --autocast  --dl-time-exclude=False --dl-worker-type="HABANA" --optimizer=lars -eoe 3 -ebe 4 --lars_base_learning_rate 2.5 --label-smoothing=0.1 --run-lazy-mode=False
  ```
- ResNet50, eager mode with torch.compile enabled, BF16 mixed precision, batch size 256, eval every 4th epoch with offset 3, base learning rate 2.5, label smoothing 0.1, FusedLARS with polynomial decay LR scheduler, 1 HPU on a single server, include dataloading time in throughput computation, Habana dataloader (with hardware decode support on **Gaudi2**), 8 worker (decoder) instances:
  ```bash
  export PT_HPU_LAZY_MODE=0
  $PYTHON -u train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=resnet50 --device=hpu --batch-size=256 --epochs=35 --workers=8 --print-freq=1200 --output-dir=. --autocast  --dl-time-exclude=False --dl-worker-type="HABANA" --optimizer=lars -eoe 3 -ebe 4 --lars_base_learning_rate 2.5 --label-smoothing=0.1 --run-lazy-mode=False --use_torch_compile
  ```
- ResNeXt101 lazy mode, BF16 mixed precision, batch size 256, custom learning rate, Habana dataloader (with hardware decode support on **Gaudi2**), 1 HPU on a single server:
  ```bash
  $PYTHON -u train.py --dl-worker-type HABANA --batch-size 256 --model resnext101_32x4d --device hpu --workers 8 --print-freq 20 --dl-time-exclude False --deterministic --data-path /data/pytorch/imagenet/ILSVRC2012 --epochs 100 --autocast --lr 0.1 --custom-lr-values 0.1 0.01 0.001 0.0001 --custom-lr-milestones 0 30 60 80
  ```
- ResNeXt101 lazy mode, BF16 mixed precision, batch size 128, custom learning rate, 1 HPU on a single server:
  ```bash
  $PYTHON -u train.py --dl-worker-type HABANA --batch-size 128 --model resnext101_32x4d --device hpu --workers 8 --print-freq 20 --dl-time-exclude False --deterministic --data-path /data/pytorch/imagenet/ILSVRC2012 --epochs 100 --autocast --lr 0.1 --custom-lr-values 0.1 0.01 0.001 0.0001 --custom-lr-milestones 0 30 60 80
  ```
- ResNet152, lazy mode, BF16 mixed precision, batch size 128, custom learning rate, 1 HPU on a single server:
  ```bash
  $PYTHON -u train.py --dl-worker-type HABANA --batch-size 128 --model resnet152 --device hpu --workers 8 --print-freq 20 --dl-time-exclude False --deterministic --data-path /data/pytorch/imagenet/ILSVRC2012 --epochs 90 --autocast --lr 0.1 --custom-lr-values 0.1 0.01 0.001 0.0001 --custom-lr-milestones 0 30 60 80
  ```
- MobileNetV2, lazy mode, BF16 mixed precision, batch size 256, 1 HPU on a single server with default PyTorch dataloader:
  ```bash
  $PYTHON -u train.py --batch-size 256 --model mobilenet_v2 --device hpu --print-freq 10 --deterministic --data-path /data/pytorch/imagenet/ILSVRC2012 --epochs 90 --autocast --dl-time-exclude=False --lr 0.045 --wd 0.00004 --lr-step-size 1 --lr-gamma 0.98 --momentum 0.9
  ```
- GoogLeNet, batch size 256, BF16 precision, lazy mode, 1 HPU on a single server:
  ```bash
  $PYTHON -u main.py --batch-size 256 --data-path /data/pytorch/imagenet/ILSVRC2012 --autocast  --device hpu --dl-worker-type HABANA --epochs 90 --lr 0.1 --enable-lazy --model googlenet --seed 123 --no-aux-logits --print-interval 20 --workers 8
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
  mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
  $PYTHON train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=resnet50 --device=hpu --batch-size=256 --epochs=90 --print-freq=1 --output-dir=. --seed=123 --autocast --custom-lr-values 0.275 0.45 0.625 0.8 0.08 0.008 0.0008 --custom-lr-milestones 1 2 3 4 30 60 80 --deterministic --dl-time-exclude=False
  ```
- ResNet50, lazy mode, BF16 mixed precision, batch size 256, custom learning rate, 8 HPUs on a single server, exclude dataloading time in throughput computation:
  ```bash
  export MASTER_ADDR=localhost
  export MASTER_PORT=12355
  mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
  $PYTHON train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=resnet50 --device=hpu --batch-size=256 --epochs=90 --print-freq=1 --output-dir=. --seed=123 --autocast --custom-lr-values 0.275 0.45 0.625 0.8 0.08 0.008 0.0008 --custom-lr-milestones 1 2 3 4 30 60 80 --deterministic --dl-time-exclude=True
  ```
- ResNet50, lazy mode, BF16 mixed precision, batch size 256, custom learning rate, 8 HPUs on a single server, include dataloading time in throughput computation, use `habana_dataloader` (with hardware decode support on **Gaudi2**), 8 worker (decoder) instances:
  ```bash
  export MASTER_ADDR=localhost
  export MASTER_PORT=12355
  mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
  $PYTHON train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=resnet50 --device=hpu --batch-size=256 --epochs=90 --workers=8 --print-freq=1 --output-dir=. --seed=123 --autocast --custom-lr-values 0.275 0.45 0.625 0.8 0.08 0.008 0.0008 --custom-lr-milestones 1 2 3 4 30 60 80 --deterministic --dl-time-exclude=False --dl-worker-type="HABANA"
  ```
- ResNet50, lazy mode, BF16 mixed precision, batch size 256, eval every 4th epoch with offset 3, label smoothing 0.1, FusedLARS with polynomial decay LR scheduler, 8 HPUs on a single server, include dataloading time in throughput computation, use `habana_dataloader` (with hardware decode support on **Gaudi2**), 8 worker (decoder) instances:
  ```bash
  export MASTER_ADDR=localhost
  export MASTER_PORT=12355
  mpirun -n 8 --bind-to core --map-by slot:PE=6 --rank-by core --report-bindings --allow-run-as-root \
  $PYTHON train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=resnet50 --device=hpu --batch-size=256 --epochs=35 --workers=8 --print-freq=150 --output-dir=. --autocast --dl-time-exclude=False --dl-worker-type="HABANA" --optimizer=lars -eoe 3 -ebe 4 --label-smoothing=0.1
  ```

  - ResNet50, eager mode, BF16 mixed precision, batch size 256, eval every 4th epoch with offset 3, label smoothing 0.1, FusedLARS with polynomial decay LR scheduler, 8 HPUs on a single server, include dataloading time in throughput computation, use `habana_dataloader` (with hardware decode support on **Gaudi2**), 8 worker (decoder) instances:
  ```bash
  export PT_HPU_LAZY_MODE=0
  export MASTER_ADDR=localhost
  export MASTER_PORT=12355
  mpirun -n 8 --bind-to core --map-by slot:PE=6 --rank-by core --report-bindings --allow-run-as-root \
  $PYTHON train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=resnet50 --device=hpu --batch-size=256 --epochs=35 --workers=8 --print-freq=150 --output-dir=. --autocast --dl-time-exclude=False --dl-worker-type="HABANA" --optimizer=lars -eoe 3 -ebe 4 --label-smoothing=0.1 --run-lazy-mode=False
  ```

  - ResNet50, eager mode with torch.compile enabled, BF16 mixed precision, batch size 256, eval every 4th epoch with offset 3, label smoothing 0.1, FusedLARS with polynomial decay LR scheduler, 8 HPUs on a single server, include dataloading time in throughput computation, use `habana_dataloader` (with hardware decode support on **Gaudi2**), 8 worker (decoder) instances:
  ```bash
  export PT_HPU_LAZY_MODE=0
  export MASTER_ADDR=localhost
  export MASTER_PORT=12355
  mpirun -n 8 --bind-to core --map-by slot:PE=6 --rank-by core --report-bindings --allow-run-as-root \
  $PYTHON train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=resnet50 --device=hpu --batch-size=256 --epochs=35 --workers=8 --print-freq=150 --output-dir=. --autocast --dl-time-exclude=False --dl-worker-type="HABANA" --optimizer=lars -eoe 3 -ebe 4 --label-smoothing=0.1 --run-lazy-mode=False --use_torch_compile
  ```
- ResNet50, lazy mode, BF16 mixed precision, batch size 256, custom learning rate, 8 HPUs on a single server, include dataloading time in throughput computation, print-frequency 10 and native PyTorch dataloader:
  ```bash
  export MASTER_ADDR=localhost
  export MASTER_PORT=12355
  mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
  $PYTHON train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=resnet50 --device=hpu --batch-size=256 --epochs=90 --workers=10 --dl-worker-type=MP --print-freq=10 --output-dir=. --seed=123 --autocast --custom-lr-values 0.275 0.45 0.625 0.8 0.08 0.008 0.0008 --custom-lr-milestones 1 2 3 4 30 60 80 --deterministic --dl-time-exclude=False
  ```
- ResNeXt101, lazy mode, BF16 mixed precision, batch size 128, 8 HPUs on s single server, include dataloading time in throughput computation:
  ```bash
  export MASTER_ADDR=localhost
  export MASTER_PORT=12355
  mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
  $PYTHON train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=resnext101_32x4d --device=hpu --batch-size=128 --epochs=100 --print-freq=1 --output-dir=. --seed=123 --autocast --deterministic --dl-time-exclude=False
  ```
- ResNeXt101, lazy mode, BF16 mixed precision, batch size 256, 8 HPUs on a single server, use `habana_dataloader` (with hardware decode support on **Gaudi2**), include dataloading time in throughput computation:
  ```bash
  export MASTER_ADDR=localhost
  export MASTER_PORT=12355
  mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
  $PYTHON train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=resnext101_32x4d --device=hpu --batch-size=256 --epochs=100 --print-freq=1 --output-dir=. --seed=123 --autocast --deterministic --dl-time-exclude=False --dl-worker-type=HABANA
  ```

- MobileNetV2, lazy mode, BF16 mixed precision, batch size 256, 8 HPUs on a single server, include dataloading time in throughput computation:
  ```bash
  export MASTER_ADDR=localhost
  export MASTER_PORT=12355
  mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
  $PYTHON train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=mobilenet_v2 --device=hpu --batch-size=256 --epochs=90 --print-freq=10 --output-dir=. --seed=123 --autocast  --lr=0.36 --wd=0.00004 --lr-step-size=1 --lr-gamma=0.98 --momentum=0.9 --deterministic --dl-time-exclude=False
  ```
- MobileNetV2, lazy mode, BF16 mixed precision, batch size 256, 8 HPUs on a single server, exclude dataloading time in throughput computation:
  ```bash
  export MASTER_ADDR=localhost
  export MASTER_PORT=12355
  mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
  $PYTHON train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=mobilenet_v2 --device=hpu --batch-size=256 --epochs=90 --print-freq=10 --output-dir=. --seed=123 --autocast  --lr=0.36 --wd=0.00004 --lr-step-size=1 --lr-gamma=0.98 --momentum=0.9 --deterministic --dl-time-exclude=True
  ```

- MobileNetV2, lazy mode, BF16 mixed precision, batch size 256, 8 HPUs on a single server, include dataloading time in throughput computation, use `habana_dataloader`, 8 worker (decoder) instances:
  ```bash
  export MASTER_ADDR=localhost
  export MASTER_PORT=12355
  mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
  $PYTHON train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=mobilenet_v2 --device=hpu --batch-size=256 --epochs=90 --workers=8 --print-freq=10 --output-dir=. --seed=123 --autocast  --lr=0.36 --wd=0.00004 --lr-step-size=1 --lr-gamma=0.98 --momentum=0.9 --deterministic --dl-time-exclude=False --dl-worker-type="HABANA"
  ```
- MobileNetV2, lazy mode, BF16 mixed precision, batch size 256, custom learning rate, 8 HPUs on a single server, include dataloading time in throughput computation, print-frequency 10:
  ```bash
  export MASTER_ADDR=localhost
  export MASTER_PORT=12355
  mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
  $PYTHON train.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=mobilenet_v2 --device=hpu --batch-size=256 --epochs=90 --workers=8 --print-freq=10 --output-dir=. --seed=123 --autocast  --lr=0.36 --wd=0.00004 --lr-step-size=1 --lr-gamma=0.98 --momentum=0.9 --deterministic --dl-time-exclude=False --dl-worker-type="HABANA"
  ```
- GoogLeNet, batch Size 256, BF16 precision, lazy mode, 8 HPUs on a single server:
  ```bash
  export MASTER_ADDR=localhost
  export MASTER_PORT=12355
  mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root $PYTHON main.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=googlenet --device=hpu --batch-size=256 --epochs=90 --lr=0.2828 --enable-lazy --print-interval=20 --dl-worker-type=HABANA --no-aux-logits --autocast  --workers=8
  ```
- GoogLeNet, batch size 128, FP32 mixed precision, lazy mode, 8 HPUs on a single server:
  ```bash
  export MASTER_ADDR=localhost
  export MASTER_PORT=12355
  mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root $PYTHON main.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=googlenet --device=hpu --batch-size=128 --epochs=90 --lr=0.2 --enable-lazy --print-interval=20 --dl-worker-type=HABANA --seed=123 --no-aux-logits --workers=8
  ```
- GoogLeNet, batch size 256, BF16 precision, lazy mode, 8 HPUs on a single server, use `habana_dataloader`, 8 worker (decoder) instances with print interval as 20:
  ```bash
  export MASTER_ADDR=localhost
  export MASTER_PORT=12355
  mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root $PYTHON main.py --data-path=/data/pytorch/imagenet/ILSVRC2012 --model=googlenet --device=hpu --batch-size=256 --epochs=90 --lr=0.2828 --enable-lazy --print-interval=20 --dl-worker-type=HABANA --seed=123 --no-aux-logits --autocast  --workers=8
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

- Set this to the network interface name or subnet that will be used by HCCL to communicate e.g.:  

```bash
export HCCL_SOCKET_IFNAME=interface_name
e.g.
export HCCL_SOCKET_IFNAME=eth0
#or
export HCCL_SOCKET_IFNAME=eth0,eth1
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

- ResNet50, lazy mode, BF16 mixed precision, batch size 256, FusedLARS with polynomial decay LR scheduler, 16 HPUs on multiple servers using Gaudi NIC, include dataloading time in throughput computation:
    - **`-H`**: Set this to a comma-separated list of host IP addresses. Make sure to modify IP address below to match your system.
    - **`--mca btl_tcp_if_include`**: Provide network interface associated with IP address. More details: [Open MPI documentation](https://www.open-mpi.org/faq/?category=tcp#tcp-selection). If you get mpirun `btl_tcp_if_include` errors, try un-setting this environment variable and let the training script automatically detect the network interface associated with the host IP address.
    - **`HCCL_SOCKET_IFNAME`**: HCCL_SOCKET_IFNAME defines the prefix of the network interface name that is used for HCCL sideband TCP communication. If not set, the first network interface with a name that does not start with lo or docker will be used.
    - `$MPI_ROOT` environment variable is set automatically during Setup. See [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) for details.
  ```bash
  export MASTER_ADDR=10.3.124.124
  export MASTER_PORT=12355
  mpirun --allow-run-as-root --mca plm_rsh_args -p3022 --bind-to core --map-by ppr:4:socket:PE=6 -np 16 --mca btl_tcp_if_include 10.3.124.124/16 --merge-stderr-to-stdout --prefix $MPI_ROOT -H 10.3.124.124:8,10.3.124.175:8 -x PYTHONPATH -x MASTER_ADDR -x MASTER_PORT \
  $PYTHON -u train.py --batch-size=256 --model=resnet50 --device=hpu --workers=8 --print-freq=100 --epochs=40 -ebe 4 --data-path=/data/pytorch/imagenet/ILSVRC2012 --dl-time-exclude=False --dl-worker-type=HABANA  --autocast --output-dir=. --seed=123 \
  --optimizer=lars --label-smoothing=0.1 --lars-weight-decay=0.0001 --lars_base_learning_rate=13 --lars_warmup_epochs=7 --lars_decay_epochs=41
  ```

- ResNet50, lazy mode, BF16 mixed precision, batch size 256, FusedLARS with polynomial decay LR scheduler, 16 HPUs on multiple servers using libFabric based Host NIC, include dataloading time in throughput computation:
    - **`-H`**: Set this to a comma-separated list of host IP addresses. Make sure to modify IP address below to match your system.
    - **`--mca btl_tcp_if_include`**: Provide network interface associated with IP address. More details: [Open MPI documentation](https://www.open-mpi.org/faq/?category=tcp#tcp-selection). If you get mpirun `btl_tcp_if_include` errors, try un-setting this environment variable and let the training script automatically detect the network interface associated with the host IP address.
    - **`HCCL_SOCKET_IFNAME`**: HCCL_SOCKET_IFNAME defines the prefix of the network interface name that is used for HCCL sideband TCP communication. If not set, the first network interface with a name that does not start with lo or docker will be used.
  ```bash
  export MASTER_ADDR=10.3.124.124
  export MASTER_PORT=12355
  mpirun --allow-run-as-root --mca plm_rsh_args -p3022 --bind-to core --map-by ppr:4:socket:PE=6 -np 16 --mca btl_tcp_if_include 10.3.124.124/16 --merge-stderr-to-stdout --prefix $MPI_ROOT -H 10.3.124.124:8,10.3.124.175:8 -x PYTHONPATH -x MASTER_ADDR -x RDMAV_FORK_SAFE=1 \
  -x FI_EFA_USE_DEVICE_RDMA=1 -x MASTER_PORT \
  $PYTHON -u train.py --batch-size=256 --model=resnet50 --device=hpu --workers=8 --print-freq=100 --epochs=40 -ebe 4 --data-path=/data/pytorch/imagenet/ILSVRC2012 --dl-time-exclude=False --dl-worker-type=HABANA --autocast --output-dir=. --seed=123 \
  --optimizer=lars --label-smoothing=0.1 --lars-weight-decay=0.0001 --lars_base_learning_rate=13 --lars_warmup_epochs=7 --lars_decay_epochs=41
  ```

## Pre-trained Model and Checkpoint
 Habana provides ResNet50 pre-trained on Gaudi models and checkpoints.
You can use it for fine-tuning or transfer learning tasks with your own datasets or for inference examples.
For e.g. the relevant checkpoint for ResNet50 can be downloaded from [ResNet50 Catalog](https://developer.habana.ai/catalog/description-resnet-for-pytorch/).
```bash
cd Model-References/PyTorch/computer_vision/classification/torchvision
mkdir pretrained_checkpoint
wget </url/of/pretrained_checkpoint.tar.gz>
tar -xvf <pretrained_checkpoint.tar.gz> -C pretrained_checkpoint && rm <pretrained_checkpoint.tar.gz>
```

## Inference and Examples

- To see the available training parameters for ResNet50 and ResNeXt101, run:
```bash
$PYTHON -u inference.py --help
```

The following commands assume that:
1. ImageNet dataset is available at `/data/pytorch/imagenet/ILSVRC2012/` directory.
   Alternative location for the dataset can be specified using the -data argument.
2. Pre-trained checkpoint is available at `pretrained_checkpoint/pretrained_checkpoint.pt`.
   Alternative file name for the pretrained checkpoint can be specific using the -ckpt argument.

### Single Card Inference Examples

All the configurations will print the following metrics for performance and accuracy:
- Performance - average latency (ms), performance (images/second)
- Accuracy - top_1, top_5 accuracy (%)

**Run inference on 1 HPU:**
- ResNet50, eager mode with torch.compile enabled, BF16 mixed precision, batch Size 256, Habana dataloader (with hardware decode support on **Gaudi2**), 1 HPU on a single server:
  ```bash
  export PT_HPU_LAZY_MODE=0
  $PYTHON -u inference.py -t HPUModel -m resnet50 -b 256 --benchmark -dt bfloat16 --accuracy --compile
  ```
- ResNet50, lazy mode, BF16 mixed precision, batch Size 256, Habana dataloader (with hardware decode support on **Gaudi2**), 1 HPU on a single server:
  ```bash
  $PYTHON -u inference.py -t HPUModel -m resnet50 -b 256 --benchmark -dt bfloat16 --accuracy
  ```
- ResNeXt101, lazy mode, BF16 mixed precision, batch size 256, Habana dataloader (with hardware decode support on **Gaudi2**), 1 HPU on a single server:
  ```bash
  $PYTHON -u inference.py -t HPUModel -m resnext101_32x4d -b 256 --benchmark -dt bfloat16 --accuracy
  ```
- ResNet50, with HPU graphs, BF16 mixed precision, batch size 256, Habana dataloader (with hardware decode support on **Gaudi2**), 1 HPU on a single server:
  ```bash
  $PYTHON -u inference.py -t HPUGraphModel -m resnet50 -b 256 --benchmark -dt bfloat16 --accuracy
  ```
- ResNeXt101, with HPU graphs, BF16 mixed precision, batch size 256, Habana dataloader (with hardware decode support on **Gaudi2**), 1 HPU on a single server:
  ```bash
  $PYTHON -u inference.py -t HPUGraphModel -m resnext101_32x4d -b 256 --benchmark -dt bfloat16 --accuracy
  ```
- ResNet50, with torch.jit.trace model, BF16 mixed precision, batch size 256, Habana dataloader (with hardware decode support on **Gaudi2**), 1 HPU on a single server:
  ```bash
  $PYTHON -u inference.py -t HPUJITModel -m resnet50 -b 256 --benchmark -dt bfloat16 --accuracy
  ```
- ResNeXt101, with torch.jit.trace model, BF16 mixed precision, batch size 256, Habana dataloader (with hardware decode support on **Gaudi2**), 1 HPU on a single server:
  ```bash
  $PYTHON -u inference.py -t HPUJITModel -m resnext101_32x4d -b 256 --benchmark -dt bfloat16 --accuracy
  ```
- ResNet50, eager mode with torch.compile enabled, FP32 precision, batch Size 256, Habana dataloader (with hardware decode support on **Gaudi2**), 1 HPU on a single server:
  ```bash
  export PT_HPU_LAZY_MODE=0
  $PYTHON -u inference.py -t HPUModel -m resnet50 -b 256 --benchmark -dt float32 --accuracy --compile
  ```
- ResNet50, lazy mode, FP32 precision, batch Size 256, Habana dataloader (with hardware decode support on **Gaudi2**), 1 HPU on a single server:
  ```bash
  $PYTHON -u inference.py -t HPUModel -m resnet50 -b 256 --benchmark -dt float32 --accuracy
  ```
- ResNeXt101, lazy mode, FP32 precision, batch size 256, Habana dataloader (with hardware decode support on **Gaudi2**), 1 HPU on a single server:
  ```bash
  $PYTHON -u inference.py -t HPUModel -m resnext101_32x4d -b 256 --benchmark -dt float32 --accuracy
  ```
- ResNet50, with HPU graphs, FP32 precision, batch size 256, Habana dataloader (with hardware decode support on **Gaudi2**), 1 HPU on a single server:
  ```bash
  $PYTHON -u inference.py -t HPUGraphModel -m resnet50 -b 256 --benchmark -dt float32 --accuracy
  ```
- ResNeXt101, with HPU graphs, FP32 precision, batch size 256, Habana dataloader (with hardware decode support on **Gaudi2**), 1 HPU on a single server:
  ```bash
  $PYTHON -u inference.py -t HPUGraphModel -m resnext101_32x4d -b 256 --benchmark -dt float32 --accuracy
  ```
- ResNet50, with torch.jit.trace model, FP32 precision, batch size 256, Habana dataloader (with hardware decode support on **Gaudi2**), 1 HPU on a single server:
  ```bash
  $PYTHON -u inference.py -t HPUJITModel -m resnet50 -b 256 --benchmark -dt float32 --accuracy
  ```
- ResNeXt101, with torch.jit.trace model, FP32 precision, batch size 256, Habana dataloader (with hardware decode support on **Gaudi2**), 1 HPU on a single server:
  ```bash
  $PYTHON -u inference.py -t HPUJITModel -m resnext101_32x4d -b 256 --benchmark -dt float32 --accuracy
  ```

This model recommends using the ["HPU graph"](https://docs.habana.ai/en/latest/PyTorch/Inference_on_Gaudi/Inference_using_HPU_Graphs/Inference_using_HPU_Graphs.html) model type to minimize the host time spent in the `forward()` call.
If HPU graphs are disabled, there could be noticeable host time spent in interpreting the lines in the `forward()` call, which cause latency to increase.

## Supported Configurations

**Training**

**ResNet50**

| Validated on | SynapseAI Version | PyTorch Version | Mode |
|-----|-----|-----|---------|
| Gaudi  | 1.15.0 | 2.2.0 | Training |
| Gaudi2 | 1.15.0 | 2.2.0 | Training |
| Gaudi2 | 1.15.0 | 2.2.0 | Inference |

**ResNeXt101**

| Validated on | SynapseAI Version | PyTorch Version | Mode |
|-----|-----|-----|---------|
| Gaudi  | 1.10.0 | 2.0.1 | Training |
| Gaudi2 | 1.15.0 | 2.2.0 | Training |
| Gaudi2 | 1.15.0 | 2.2.0 | Inference |

**MobileNetV2**

| Validated on | SynapseAI Version | PyTorch Version | Mode |
|-----|-----|-----|--------|
| Gaudi | 1.14.0 | 2.1.1 | Training |

**ResNet152**

| Validated on | SynapseAI Version | PyTorch Version | Mode |
|-----|-----|-----|--------|
| Gaudi | 1.15.0 | 2.2.0 | Training |

**GoogLeNet**
| Validated on | SynapseAI Version | PyTorch Version | Mode |
|-----|-----|-----|--------|
| Gaudi | 1.12.1 | 2.0.1 | Training |

## Changelog
### 1.14.0
 - Added model saving functionality.
 - Added torch.compile support for inference - performance improvement feature for PyTorch eager mode. Supported only for Resnet50.
 - Lazy and HPU graphs support for Resnet50 inference is deprecated.
### 1.13.0
 - Added torch.compile support - performance improvement feature for PyTorch eager mode. Supported only for Resnet50 LARS.
### 1.12.0
 - Removed support for HMP.
 - Eager mode support is deprecated.
### 1.11.0
 - Dynamic Shapes will be enabled by default in future releases. It is currently enabled for all models
   except for ResNet50 LARS in the model training script, which serves as a temporary solution.
 - Enabled using HPU Graphs by default on Gaudi2.
### 1.9.0
 - Disabled auto dynamic shape support for Habana devices in ResNet50 LARS.
 - Enabled usage of PyTorch autocast.
### 1.8.0
 - Added Media API implementation for image processing on Gaudi2.
 - Added support for FusedLARS with polynomial decay LR scheduler.
 - Added configurable frequency of eval.
 - Changed CrossEntropyLoss to use reduce='sum' and division instead of mean.
 - Added upper limit of print frequency.
 - Fixed images per second print.
 - Added configurable label smoothing parameter to loss function.
 - Added tensorboard logging.
 - Initial release of inference script for ResNet50 and ResNeXt101 32x4d.
### 1.5.0
 - Extended support for habana_dataloader with hardware decode support for Gaudi2 to support 8 instances on ResNet50/ResNeXt101.
 - Removed channels-last=true support.
### 1.4.1
 - Added support for habana_dataloader with hardware decode support for Gaudi2.
### 1.4.0
 - Default execution mode modified to lazy mode.

## Training Script Modifications
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

   i. Disable dynamic shapes with PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES=0.

2. Dataloader related changes

    a. Added –deterministic flag to make data loading deterministic with PyTorch dataloader.

    b. Added support for including dataloader time in performance computation. To match behavior of base script, it is disabled by default.

    c. Added CPU based habana_dataloader with faster image preprocessing (only JPEG) support. It is enabled by default.

    d. Added habana_dataloader with hardware decode support for Imagenet dataset on **Gaudi2**.

    e. Added Media API implementation for habana_dataloader on **Gaudi2**.

3. To improve performance

   a. Permute convolution weight tensors & and any other dependent tensors like 'momentum' for better performance.

   b. Checkpoint saving involves getting trainable params and other state variables to CPU and permuting the weight tensors. Hence, checkpoint saving is by default disabled. It can be enabled using --save-checkpoint option.

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

   n. Added support for FusedLARS with label smoothing and polynominal decay LR scheduler, enabled by --optimizer=lars

4. Skip printing eval phase stats if eval steps is zero.

5. Added support in script to override default LR scheduler with custom schedule passed as parameters to script.

6. For GoogLeNet, first two batches' training and evaluation time is not included in the average batch time performance numbers due to long wait times of initial data loading to device.

7. Added new parameters
  a. '-ebe' or '--epochs_between_evals' - integer with default value=1 - number of epochs to be completed before evaluation

  b. '-eoe' or '--eval_offset_epochs' - integer with default value=0 - offsets the epoch on which the evaluation starts

  c. '--lars_base_learning_rate' - float with default value=9.0 - base learning rate used with FusedLARS in polynomial decay

  d. '--lars_end_learning_rate' - float with default value=0.0001 - end learning rate used with FusedLARS in polynomial decay

  e. '--lars_warmup_epochs' - integer with default value=3 - number of warmup epochs for FusedLARS

  f. '--lars_decay_epochs' - integer with default value=36 - number of decay epochs for FusedLARS

  g. '--lwd' or '--lars-weight-decay' - float with default value=5e-5 - weight decay for FusedLARS

  h. '--label-smoothing' - float with default value=0.0 - CrossEntropyLoss's label smoothing

  i. '--optimizer' - string with default value='sgd' - optimizer selection parameter, available values are 'sgd' and 'lars'

  j. '--use_torch_compile' - use torch.compile feature to run the model - supported only for Resnet50 LARS

The model changes are listed below:

1. Added support for resnext101_32x4d.

## Known Issues
- channels-last = true is not supported. Train ResNet & MobileNet_v2 with channels-last=false.
- Placing mark_step() arbitrarily may lead to undefined behaviour. Recommend to keep mark_step() as shown in provided scripts.
- Habana_dataloader only supports Imagenet (JPEG) and 8 worker instances on Gaudi.
- Habana_dataloader with hardware decode support only supports training with Imagenet (JPEG) up to 8 cards on **Gaudi2** for Resnet50/Resnext101.
- Only scripts & configurations mentioned in this README are supported and verified.
