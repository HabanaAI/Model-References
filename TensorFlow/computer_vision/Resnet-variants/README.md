# ResNet Variants for TensorFlow

This repository provides a script and recipe to train the ResNet v1.5 model to achieve state-of-the-art accuracy, and is tested and maintained by Habana.

## Table Of Contents
* [Model Overview](#model-overview)
* [Setup](#setup)
* [Running ResNet](#running-resnet)
* [Training Results](#training-results)


## Model Overview
The ResNet v1.5 model is a modified version of the original ResNet v1 model. It supports layers 50, 101, and 152.

### Hyperparameters
This model uses the the following hyperparameters:

* Momentum (0.9).
* Learning rate (LR) = 0.1 for 256 batch size, for other batch sizes we linearly scale the learning
  rate. 
* Learning rate schedule - TBD
* For bigger batch sizes (512 and up) we use linear warmup of the learning rate.
during the first 5 epochs according to [Training ImageNet in 1 hour](https://arxiv.org/abs/1706.02677).
* Weight decay: 3.0517578125e-05 (1/32768).
* We do not apply Weight decay on batch norm trainable parameters (gamma/bias).
* Label Smoothing: 0.1.

### Data Augmentation
This model uses the following data augmentation:

* For training:
    * Normalization.
    * Random resized crop to 224x224.
        * Scale from 8% to 100%.
        * Aspect ratio from 3/4 to 4/3.
    * Random horizontal flip.
* For inference:
    * Normalization.
    * Scale to 256x256.
    * Center crop to 224x224.

## Setup

### Requirements TODO: Clarify version requirements
* Docker version 19.03.12 or newer
* Sudo access to install required drivers/firmware  
  OR  
  If they are already installed, skip to [Quick Setup Guide](#quick-setup-guide)

### Install drivers

1. Stop running dockers
```
docker stop $(docker ps -a -q)
```
2. Remove old packages habanalabs-dkms
```
sudo dpkg -P habanalabs-dkms
```
3. Download and install habanalabs-dkms --  **TODO: This is an internal link, does not work, would need external Artifacotry** 
```
sudo wget https://artifactory.habana-labs.com/repo-ubuntu18.04/pool/qa/h/habanalabs/habanalabs-dkms_0.10.0-294_all.deb
sudo dpkg -i habanalabs-dkms_0.10.0-294_all.deb
```
4. Unload the driver
```
sudo rmmod habanalabs
```
5. Update firmware. Be careful when updating firmware - it cannot be interrupted. Otherwise, the system may become non-functional.
```
sudo hl-fw-loader --yes
```
6. Load the driver
```
sudo modprobe habanalabs
```

## Quick Setup Guide

The ResNet v1.5 script operates on ImageNet 1k, a popular image classification dataset from the ILSVRC challenge.

If you want to use an alternative dataset, ensure that the data directory is organized like ImageNet:

```
  data_dir/train-00000-of-01024
  data_dir/train-00001-of-01024
   ...
  data_dir/train-00127-of-01024

and

  data_dir/validation-00000-of-00128
  data_dir/validation-00001-of-00128
  ...
  data_dir/validation-00127-of-00128
```

1. Sign up for an account with [ImageNet](http://image-net.org/) to gain access to the data. Look for the sign up
page, create an account, and request an access key. 

2. Use [this
script](https://github.com/tensorflow/models/blob/cee4aff18b08daf15114351cef826eb1ee7c8519/inception/inception/data/download_and_preprocess_imagenet.sh) to download and convert the data to TFRecord format. We recommend placing the
dataset on a local drive for best performance. Enter your ImageNet username and password when prompted.

2. Stop running dockers
```
docker stop $(docker ps -a -q)
```

3. Download docker ---   **Note THIS IS an INTERNAL repo, this will be converted to Docker Hub for final usage**
```
docker pull artifactory.habana-labs.com/docker-local/0.10.0/ubuntu18.04/habanalabs/tensorflow-installer:0.10.0-294
```

4. Run docker ---  Note: this will Need the external Artifacotry for this to work.  
**NOTE:** This assumes Imagenet dataset is under /opt/datasets/imagenet on the host. Modify accordingly.  
```
docker run -td --device=/dev/hl_controlD0:/dev/hl_controlD0 --device=/dev/hl_controlD1:/dev/hl_controlD1 --device=/dev/hl_controlD2:/dev/hl_controlD2 --device=/dev/hl_controlD3:/dev/hl_controlD3 --device=/dev/hl_controlD4:/dev/hl_controlD4 --device=/dev/hl_controlD5:/dev/hl_controlD5 --device=/dev/hl_controlD6:/dev/hl_controlD6 --device=/dev/hl_controlD7:/dev/hl_controlD7 --device=/dev/hl0:/dev/hl0 --device=/dev/hl1:/dev/hl1 --device=/dev/hl2:/dev/hl2 --device=/dev/hl3:/dev/hl3 --device=/dev/hl4:/dev/hl4 --device=/dev/hl5:/dev/hl5 --device=/dev/hl6:/dev/hl6 --device=/dev/hl7:/dev/hl7 -e DISPLAY=$DISPLAY -e LOG_LEVEL_ALL=6 -e TF_MODULES_RELEASE_BUILD=/usr/lib/habanalabs/ -v /sys/kernel/debug:/sys/kernel/debug -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v /tmp:/tmp -v /opt/datasets/imagenet:/home/user1/tensorflow_datasets/imagenet --net=host --user user1 --workdir=/home/user1 artifactory.habana-labs.com/docker-local/0.10.0/ubuntu18.04/habanalabs/tensorflow-installer:0.10.0-294
```
OPTIONAL with mounted shared folder to transfer files out of docker:
```
docker run -td --device=/dev/hl_controlD0:/dev/hl_controlD0 --device=/dev/hl_controlD1:/dev/hl_controlD1 --device=/dev/hl_controlD2:/dev/hl_controlD2 --device=/dev/hl_controlD3:/dev/hl_controlD3 --device=/dev/hl_controlD4:/dev/hl_controlD4 --device=/dev/hl_controlD5:/dev/hl_controlD5 --device=/dev/hl_controlD6:/dev/hl_controlD6 --device=/dev/hl_controlD7:/dev/hl_controlD7 --device=/dev/hl0:/dev/hl0 --device=/dev/hl1:/dev/hl1 --device=/dev/hl2:/dev/hl2 --device=/dev/hl3:/dev/hl3 --device=/dev/hl4:/dev/hl4 --device=/dev/hl5:/dev/hl5 --device=/dev/hl6:/dev/hl6 --device=/dev/hl7:/dev/hl7 -e DISPLAY=$DISPLAY -e LOG_LEVEL_ALL=6 -e TF_MODULES_RELEASE_BUILD=/usr/lib/habanalabs/ -v /sys/kernel/debug:/sys/kernel/debug -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v /tmp:/tmp -v ~/shared:/home/user1/shared -v /opt/dataset/imagenet:/home/user1/tensorflow_datasets/imagenet --net=host --user user1 --workdir=/home/user1 artifactory.habana-labs.com/docker-local/0.10.0/ubuntu18.04/habanalabs/tensorflow-installer:0.10.0-294
```

5. Check name of your docker
```
docker ps
```

6. Run bash in your docker
```
docker exec -ti <NAME> bash 
```

7. Install git.
  **Check your proxy settings. If on Intel network, use the following:**
  ```
  export HTTP_PROXY=http://proxy-chain.intel.com:911
  export HTTPS_PROXY=http://proxy-chain.intel.com:912
  export ftp_proxy=http://proxy-chain.intel.com:911
  export socks_proxy=http://proxy-us.intel.com:1080
  export no_proxy=intel.com,.intel.com,localhost,127.0.0.1
  export http_proxy=$HTTP_PROXY
  export https_proxy=$HTTPS_PROXY
  ```
    
  ```
  sudo -E apt install git -y
  ```  
  
8. [OPTIONAL] Install vim  
  ```
  sudo -E apt install vim -y
  ```

## Running ResNet

Clone the repository and go to resnet directory:
```
git clone https://github.com/habana-labs-demo/ResnetModelExample.git
cd TensorFlow/computer_vision/resnet
```

Note: If the repository is not in the PYTHONPATH, make sure you update it.
```
export PYTHONPATH=/path/to/ResnetModelExample:$PYTHONPATH
```
 
### Single-Card Training

#### The resnet_single_card.sh script 

The script for training and evaluating the ResNet model has a variety of paramaters.
```
usage: resnet_single_card.sh [arguments]

mandatory arguments:
  -d <data_type>,    --dtype <data_type>          Data type, possible values: fp32, bf16

optional arguments:
  -rs <resnet_size>, --resnet-size <resnet_size>  ResNet size, default 50, possible values: 50, 101, 152
  -b <batch_size>,   --batch-size <batch_size>    Batch size, default 128 for fp32, 256 for bf16
  -e <epochs>,       --epochs <epochs>            Number of epochs, default to 1
  -a <data_dir>,     --data-dir <data_dir>        Data dir, defaults to /software/data/tf/data/imagenet/tf_records/
  -m <model_dir>,    --model-dir <model_dir>      Model dir, defaults to /home/user1/tmp/resnet50/
  -o,                --use-horovod                Use horovod for training
  -s <steps>,        --steps <steps>              Max train steps
  -l <steps>,        --eval-steps <steps>         Max evaluation steps
  -n,                --no-eval                    Don't do evaluation
  -v <steps>,        --display-steps <steps>      How often display step status
  -c <steps>,        --checkpoint-steps <steps>   How often save checkpoint
  -r                 --recover                    If crashed restart training from last checkpoint. Requires -s to be set
  -k                 --no-experimental-preloading Disables support for 'data.experimental.prefetch_to_device' TensorFlow operator. If not set:
                                                  - loads extension dynpatch_prf_remote_call.so (via LD_PRELOAD)
                                                  - sets environment variable HBN_TF_REGISTER_DATASETOPS to 1
                                                  - this feature is experimental and works only with single node
```
### Multi-Card Training

#### The resnet_multi_card.sh script

The script requires additional parameters.

```
export IMAGENET_DIR=/path/to/tensorflow_datasets/imagenet/
export HCL_CONFIG_PATH=/path/to/hcl_config
export RESNET_SIZE=<resnet-size>
./resnet_multi_card.sh
```

## Training Results
        
The following sections provide details on how we achieved our training performance and accuracy.

### Validation accuracy results
    
#### Validation accuracy: single card

Our results were obtained by running the `resnet_single_card` training script in the 0.9.2-49 container.
```
./resnet_single_card.sh -d bf16 -rs  50 -e 90 -a /path/to/tensorflow_datasets/imagenet/ -m /path/to/tensorflow-training/ckpt_rn50_bf16
./resnet_single_card.sh -d bf16 -rs 101 -e 90 -a /path/to/tensorflow_datasets/imagenet/ -m /path/to/tensorflow-training/ckpt_rn101_bf16
./resnet_single_card.sh -d bf16 -rs 152 -e 90 -a /path/to/tensorflow_datasets/imagenet/ -m /path/to/tensorflow-training/ckpt_rn152_bf16
```

| Model | Epochs | Batch Size / Card | Accuracy BF16 - (top1) | Accuracy - BF16 (top5) |
|-----------|--------|------------------|-----------------|----------------------------|
| resnet50  | 90     | 256              | 0.74813930      | 0.92277520         |
| resnet101 | 90     | 256              | 0.76800570      | 0.93463904         |
| resnet152 | 90     | 256              | 0.76414450      | 0.93047770         |

#### Validation accuracy: Habana HLS-1 (8x Habana HL-205)

Our results were obtained by running the `resnet_multi_card` training script in the 0.10.0-294 container.
```
export IMAGENET_DIR=/path/to/tensorflow_datasets/imagenet/
export HCL_CONFIG_PATH=/path/to/hcl_config
RESNET_SIZE=50 ./resnet_multi_card.sh
RESNET_SIZE=101 HABANA_INITIAL_WORKSPACE_SIZE_MB=2000 ./resnet_multi_card.sh 
RESNET_SIZE=152 HABANA_INITIAL_WORKSPACE_SIZE_MB=2000 ./resnet_multi_card.sh 
```

| Model | Epochs | Batch Size / Card | Accuracy BF16 - (top1) | Accuracy - BF16 (top5) |
|-----------|--------|------------------|-----------------|----------------------------|
| resnet50  | 90     | 256              | 0.74554690      | 0.92195310         |
| resnet101 | 90     | 256              | 0.76480466      | 0.93261720         |
| resnet152 | 90     | 256              | 0.76843750      | 0.93214840         |

#### Training loss
<p float="left">
  <img src="./imgs/resnet50_bf16_1_card.png" width="430" />
  <img src="./imgs/resnet50_bf16_8_cards.png" width="430" /> 
</p>

<p float="left">
  <img src="./imgs/resnet101_bf16_1_card.png" width="430" />
  <img src="./imgs/resnet101_bf16_8_cards.png" width="430" /> 
</p>

<p float="left">
  <img src="./imgs/resnet152_bf16_1_card.png" width="430" />
  <img src="./imgs/resnet152_bf16_8_cards.png" width="430" /> 
</p>

### Training performance results

Our results were obtained by running the `resnet_multi_card` training scripts in the 0.10.0-294 container. The performance numbers (in images per second) were averaged over 1500 iterations, excluding the first iteration.
```
export IMAGENET_DIR=/path/to/tensorflow_datasets/imagenet/
export HCL_CONFIG_PATH=/path/to/hcl_config

export TF_ENABLE_BF16_CONVERSION=true
export TF_ALLOW_CONTROL_EDGES_IN_HABANA_OPS=1
export HABANA_USE_PREALLOC_BUFFER_FOR_ALLREDUCE=true
export HABANA_USE_STREAMS_FOR_HCL=true
export TF_ALLOW_CONTROL_EDGES_IN_HABANA_OP=1
export TF_PRELIMINARY_CLUSTER_SIZE=200

RESNET_SIZE=50 ./resnet_multi_card.sh
RESNET_SIZE=101 HABANA_INITIAL_WORKSPACE_SIZE_MB=2000 ./resnet_multi_card.sh 
RESNET_SIZE=152 HABANA_INITIAL_WORKSPACE_SIZE_MB=2000 ./resnet_multi_card.sh 
```

| Model     | Cards | Batch Size / Card | Throughput (ave) - BF16 | 
|-----------|-------|-------------------|-------------------------|
| resnet50  |   8   |        256        |      10823.332 img/s    | 
| resnet101 |   8   |        256        |       7566.344 img/s    |
| resnet152 |   8   |        256        |       5410.350 img/s    |

#### Training time for 90 epochs

Our results were obtained by running the commands below.
```
export IMAGENET_DIR=/path/to/tensorflow_datasets/imagenet/
export HCL_CONFIG_PATH=/path/to/hcl_config
RESNET_SIZE=50  ./resnet_multi_card.sh 
RESNET_SIZE=101 ./resnet_multi_card.sh  
RESNET_SIZE=152 ./resnet_multi_card.sh  
```

| Model     | Cards | Batch Size / Card | Total time to train | 
|-----------|-------|-------------------|---------------------|
| resnet50  |   8   |        256        |        TBDh       | 
| resnet101 |   8   |        256        |        TBDh       |
| resnet152 |   8   |        256        |        TBDh       | 

