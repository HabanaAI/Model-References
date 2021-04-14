# Resnet50 for PyTorch
This folder contains scripts to train Resnet50 model on Habana Gaudi<sup>TM</sup> device to achieve state-of-the-art accuracy.

The Resnet50 demos included in this release are Eager mode and Lazy mode training for BS128 with FP32 and BS256 with BF16 mixed precision.
## Table of Contents

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

* [Model-References](../../../../README.md)
* [Model Overview](#model-overview)
* [Setup](#setup)
* [Training](#training)
* [Known Issues](#known-issues)

## Model Overview
The base Renet50 model used is from [GitHub-PyTorch-Vision](https://github.com/pytorch/vision.git). A copy of the model file [resnet.py](models/resnet.py) is used to make
certain model changes functionally equivalent to the original model. Please refer to a below section for a summary of ResNet50 model changes, changes to training script and the original files.

### Training Script Modifications
The following are the changes added to the training script (train.py) and utilities(utils.py):
1. Added support for Habana devices – load Habana specific library.
2. Enhance with channels last data format (NHWC) support.
3. Added multi-node (distributed) and mixed precision support
4. Modified image copying to device as non-blocking to gain better performance.
5. Permute convolution weight tensors for better performance.
6. Checkpoint saving involves getting trainable params and other state variables to CPU and permuting the weight tensors. Hence, checkpoint saving is by default disabled. It can be enabled using –save-checkpoint option.
7. Accuracy calculation is performed on the CPU.
8. Essential train operations are wrapped into a function to limit the variable scope and thus reduce memory consumption.
9. Used copy of ResNet model file (torchvision/models/resnet.py) instead of importing model from torch vision package. This is to add modifications (functionally equivalent to the original) to the model.
10. Loss tensor is bought to CPU for printing stats.
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
20. Optimized FusedSGD operator is used in place of torch.optim.SGD for lazy mode
21. Control of a few Habana lazy mode optimization passes in bridge
22. Added support in script to override default LR scheduler with custom schedule passed as parameters to script

The ResNet50 model changes are listed below:
1. Replaced nn.AdaptiveAvgPool with functionally equivalent nn.AvgPool.
2. Replaced in-place nn.Relu and torch.add_ with non-in-place equivalents.

## Setup

### Requirements
* Docker version 19.03.12 or newer
* Sudo access to install required drivers/firmware

### Install Drivers
Follow steps in the [Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) to install the driver.

<br />

### Install container runtime
<details>
<summary>Ubuntu distributions</summary>

### Setup package fetching
1. Download and install the public key:  
```
curl -X GET https://vault.habana.ai/artifactory/api/gpg/key/public | sudo apt-key add -
```
2. Create an apt source file /etc/apt/sources.list.d/artifactory.list.
3. Add the following content to the file:
```
deb https://vault.habana.ai/artifactory/debian focal main
```
4. Update Debian cache:  
```
sudo dpkg --configure -a
sudo apt-get update
```  
### Install habana-container-runtime:
Install the `habana-container-runtime` package:
```
sudo apt install -y habanalabs-container-runtime=0.14.0-420
```
### Docker Engine setup

To register the `habana` runtime, use the method below that is best suited to your environment.
You might need to merge the new argument with your existing configuration.

#### Daemon configuration file
```bash
sudo tee /etc/docker/daemon.json <<EOF
{
    "runtimes": {
        "habana": {
            "path": "/usr/bin/habana-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF
sudo systemctl restart docker
```

You can optionally reconfigure the default runtime by adding the following to `/etc/docker/daemon.json`:
```
"default-runtime": "habana"
```
</details>

<details>
<summary>CentOS distributions</summary>

### Setup package fetching
1. Create /etc/yum.repos.d/Habana-Vault.repo.
2. Add the following content to the file:
```
[vault]

name=Habana Vault

baseurl=https://vault.habana.ai/artifactory/centos7

enabled=1

gpgcheck=0

gpgkey=https://vault.habana.ai/artifactory/centos7/repodata/repomod.xml.key

repo_gpgcheck=0
```
3. Update YUM cache by running the following command:
```
sudo yum makecache
```
4. Verify correct binding by running the following command:
```
yum search habana
```
This will search for and list all packages with the word Habana.

### Install habana-container-runtime:
Install the `habana-container-runtime` package:
```
sudo yum install habanalabs-container-runtime-0.14.0-420* -y
```
### Docker Engine setup

To register the `habana` runtime, use the method below that is best suited to your environment.
You might need to merge the new argument with your existing configuration.

#### Daemon configuration file
```bash
sudo tee /etc/docker/daemon.json <<EOF
{
    "runtimes": {
        "habana": {
            "path": "/usr/bin/habana-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF
sudo systemctl restart docker
```

You can optionally reconfigure the default runtime by adding the following to `/etc/docker/daemon.json`:
```
"default-runtime": "habana"
```
</details>

<details>
<summary>Amazon linux distributions</summary>

### Setup package fetching
1. Create /etc/yum.repos.d/Habana-Vault.repo.
2. Add the following content to the file:
```
[vault]

name=Habana Vault

baseurl=https://vault.habana.ai/artifactory/AmazonLinux2

enabled=1

gpgcheck=0

gpgkey=https://vault.habana.ai/artifactory/AmazonLinux2/repodata/repomod.xml.key

repo_gpgcheck=0
```
3. Update YUM cache by running the following command:
```
sudo yum makecache
```
4. Verify correct binding by running the following command:
```
yum search habana
```
This will search for and list all packages with the word Habana.

### Install habana-container-runtime:
Install the `habana-container-runtime` package:
```
sudo yum install habanalabs-container-runtime-0.14.0-420* -y
```
### Docker Engine setup

To register the `habana` runtime, use the method below that is best suited to your environment.
You might need to merge the new argument with your existing configuration.

#### Daemon configuration file
```bash
sudo tee /etc/docker/daemon.json <<EOF
{
    "runtimes": {
        "habana": {
            "path": "/usr/bin/habana-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF
sudo systemctl restart docker
```

You can optionally reconfigure the default runtime by adding the following to `/etc/docker/daemon.json`:
```
"default-runtime": "habana"
```
</details>
<br />

### Get the Habana PyTorch docker image

| Ubuntu version      |                              Command Line                                                                  |
| ------------------- | ---------------------------------------------------------------------------------------------------------- |
| Ubuntu 18.04        | `docker pull vault.habana.ai/gaudi-docker/0.14.0/ubuntu18.04/habanalabs/pytorch-installer:0.14.0-420`      |
| Ubuntu 20.04        | `docker pull vault.habana.ai/gaudi-docker/0.14.0/ubuntu20.04/habanalabs/pytorch-installer:0.14.0-420`      |

Note: For the sake of simplicity the Readme instructions follows the assumption that docker image is Ubuntu version 20.04

### Follow the docker and Model-References git clone setup steps.

1. Stop running dockers
```bash
docker stop $(docker ps -a -q)
```

2. Pull docker image
```bash
docker pull vault.habana.ai/gaudi-docker/0.14.0/ubuntu20.04/habanalabs/pytorch-installer:0.14.0-420
```

3. Run docker
```bash
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice -v /sys/kernel/debug:/sys/kernel/debug -v $HOME/datasets/:/root/dataset --net=host vault.habana.ai/gaudi-docker/0.14.0/ubuntu20.04/habanalabs/pytorch-installer:0.14.0-420
```

4. In the docker container, clone the repository and go to PyTorch ResNet directory:
```bash
git clone https://github.com/HabanaAI/Model-References
cd Model-References/PyTorch/computer_vision/ImageClassification/ResNet
```

5. Run `./demo_resnet -h` for command-line options

### Set up dataset
Imagenet 2012 dataset needs to be organized as per PyTorch requirement. PyTorch requirements are specified in the [link](https://github.com/soumith/imagenet-multiGPU.torch) which contains scripts to organize Imagenet data. https://github.com/soumith/imagenet-multiGPU.torch

## Training
Clone the Model-References git.
Set up the data set as mentioned in the section [Set up dataset](#set-up-dataset).

Note: Parameter –-data-path (or -p) points to the path of the dateset location. The below training commands assumes that Imagenet dataset is available at `/root/datasets/imagenet/ILSVRC2012/` folder.

### Training on Single node:
Please refer to the following command for available training parameters:
```
cd Model-References/PyTorch/computer_vision/ImageClassification/ResNet
Run `./demo_resnet -h` for command-line options
```
i. eager mode: bf16 mixed precision, BS256
```
./demo_resnet --data-type bf16 -b 256 --mode eager -p /root/datasets/imagenet/ILSVRC2012/
```
ii. eager mode: FP32, BS128
```
./demo_resnet --data-type fp32 -b 128 --mode eager -p /root/datasets/imagenet/ILSVRC2012/
```
iii. lazy mode: bf16 mixed precision, BS256
```
./demo_resnet --data-type bf16 --batch-size 256 -p /root/datasets/imagenet/ILSVRC2012/ --mode lazy
```
iv. lazy mode: FP32, BS128
```
./demo_resnet --data-type fp32 --batch-size 128 -p /root/datasets/imagenet/ILSVRC2012/ --mode lazy
```

### Multinode Training
To run multi-node demo, make sure the host machine has 512 GB of RAM installed. Modify the docker run command to pass 8 Gaudi cards to the docker container. This ensures the docker has access to all the 8 cards required for multi-node demo.
```
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice -v /sys/kernel/debug:/sys/kernel/debug -v $HOME/datasets/:/root/dataset --net=host vault.habana.ai/gaudi-docker/0.14.0/ubuntu20.04/habanalabs/pytorch-installer:0.14.0-420
```
Before execution of the multi-node demo scripts, make sure all HLS-1 network interfaces are up. You can change the state of each network interface managed by the habanalabs driver using the following command:
```
sudo ip link set <interface_name> up
```
To identify if a specific network interface is managed by the habanalabs driver type, run:
```
sudo ethtool -i <interface_name>
```
Number of nodes can be configured using -w option in the demo script. The commands below are for 8 cards (1HLS) which is the supported configuration.

i. eager mode: bf16 mixed precision, BS256
```
./demo_resnet_dist --data-type bf16 --batch-size 256 -p /root/datasets/imagenet/ILSVRC2012/ --mode eager
```
ii. eager mode: FP32, BS128
```
./demo_resnet_dist --data-type fp32 --batch-size 128 -p /root/datasets/imagenet/ILSVRC2012/ --mode eager
```
iii. lazy mode: BF16 mixed precision, BS256, custom lr
```
./demo_resnet_dist --data-type bf16 --batch-size 256 -p /root/datasets/imagenet/ILSVRC2012/ --mode lazy --custom-lr-milestones 1 2 3 4 30 60 80 --custom-lr-values 0.275 0.45 0.625 0.8 0.08 0.008 0.0008
```
iv. lazy mode: FP32, BS128, custom lr
```
./demo_resnet_dist --data-type fp32 --batch-size 128 -p /root/datasets/imagenet/ILSVRC2012/ --mode lazy --custom-lr-milestones 1 2 3 4 30 60 80 --custom-lr-values 0.175 0.25 0.325 0.4 0.04 0.004 0.0004
```

## Known Issues
- channels-last = false is not supported. Train RN50 with channels-last=true. This will be fixed in a subsequent release.
- Full training with BS256, BF16 reaches 75.7% accuracy instead of 76%
