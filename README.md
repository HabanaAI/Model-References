# ResNet-50 v1.5 for TensorFlow

This repository provides a script and recipe to train the ResNet-50 v1.5 model to achieve state-of-the-art accuracy, and is tested and maintained by Habana

## Table Of Contents
* [Model overview](#model-overview)
* [Setup](#setup)
    * [Requirements](#requirements)
* [Advanced](#advanced)
* [Results](#results)


## Model Overview

### Hyperparameters
This model uses the the following hyperparameters:

* Momentum (0.875).
* Learning rate (LR) = 0.256 for 256 batch size, for other batch sizes we linearly scale the learning
  rate. 
* Learning rate schedule - we use cosine LR schedule.
* For bigger batch sizes (512 and up) we use linear warmup of the learning rate.
during the first 5 epochs according to [Training ImageNet in 1 hour](https://arxiv.org/abs/1706.02677).
* Weight decay: 3.0517578125e-05 (1/32768).
* We do not apply Weight decay on batch norm trainable parameters (gamma/bias).
* Label Smoothing: 0.1.
* We train for:
    * 90 Epochs -> 90 epochs is a standard for ResNet family networks.
    * 250 Epochs -> best possible accuracy. 
* For 250 epoch training we also use [MixUp regularization](https://arxiv.org/pdf/1710.09412.pdf).


## Loading Docker

1. Stop running dockers
```
sudo docker stop $(sudo docker ps -a -q)
```
2. Remove old packages habanalabs-dkms
```
sudo dpkg -P habanalabs-dkms
```
3. Download and install habanalabs-dkms
```
sudo wget https://artifactory.habana-labs.com/repo-ubuntu18.04/pool/testing/h/habanalabs/habanalabs-dkms_0.8.0-490_all.deb​​​​​​​
sudo dpkg -i habanalabs-dkms_0.8.0-490_all.deb
```
4. Unload LKD
```
sudo rmmod habanalabs
```
5. Update firmware. Be careful when updating firmware - it can not be interrupted. otherwise, the card may be bricked.
```
sudo hl-fw-loader --yes
```
6. Load LKD
```
sudo modprobe habanalabs
```
7. Download docker
```
sudo docker pull artifactory.habana-labs.com/docker-local/0.8.0/ubuntu18.04/habanalabs/tensorflow-installer:0.8.0-490
```
8. Run docker
```
sudo docker run -td --privileged -e DISPLAY= -e LOG_LEVEL_ALL=6 -e TF_MODULES_RELEASE_BUILD=/usr/lib/habanalabs/ --device=/dev/hlv200  -v /sys/kernel/debug:/sys/kernel/debug -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v /tmp:/tmp --net=host --user user1 --workdir=/home/user1 artifactory.habana-labs.com/docker-local/0.8.0/ubuntu18.04/habanalabs/tensorflow-installer:0.8.0-490
```
9. Check name of your docker
```
sudo docker ps
```
10. Run bash in your docker
```
sudo docker exec -ti <NAME> bash 
```
11. OPTIONAL You may want to install VIM
```
sudo apt install vim -y
```
12. Now you can run demo (directory demo). This directory originates from tensorflow-training/demo.

## Running the Demo

### RS-50 Demo
This is the script for running RS50 model
```
usage: demo_resnet50 [arguments]

mandatory arguments:
  -d <data_type>,    --dtype <data_type>          Data type, possible values: fp32, bf16

optional arguments:
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

These are the options
```
example:
  demo_resnet50 -d bf16
  demo_resnet50 -d fp32
  demo_resnet50 -d bf16 -s 1000
  demo_resnet50 -d bf16 -s 1000 -l 50
  demo_resnet50 -d bf16 -e 9
  demo_resnet50 -d fp32 -e 9 -b 128 -a home/user1/tensorflow_datasets/imagenet/tf_records/ -m /home/user1/tensorflow-training/demo/ck_81_080_450_bs128
```

**here's the location of the data set**
**here's the location of the Docker Container**

## Training Output

## Training Results

#### Training performance benchmark

To benchmark the training performance on a specific batch size, run:

* For Single Gaudi 
    * FP32 

        `demo_resnet50 -d fp32 -b 128 -a home/user1/tensorflow_datasets/imagenet/tf_records/ -m /home/user1/tensorflow-training/demo/ck_81_080_450_bs128 `
        
    * BF16

        `demo_resnet50 -d bf16 -b 256 -a home/user1/tensorflow_datasets/imagenet/tf_records/ -m /home/user1/tensorflow-training/demo/ck_81_080_450_bs256 `
        
* For multiple Gaudi cards
    * FP32 

        `demo_resnet50_hvd.sh - - - `
        
    * BF16
    
        `demo_resnet50_hvd.sh - - - `

        

### Results

The following sections provide details on how we achieved for training performance and accuracy

#### Training accuracy results

##### Training accuracy: Single Card

Our results were obtained by running the `/demo/reset50` training script in the "container"

| Epochs | Batch Size / GPU | Accuracy FP32 - (top1) | Accuracy - BF16 (top1) | 
|--------|------------------|-----------------|----------------------------|
| 90     | 128              | 77.01           | n/a                        |
| 90     | 256              | n/a             | 76.93                      |


##### Training accuracy: Habana HLS-1 (8x Habana HL-205)
Our results were obtained by running the `/demo/reset50` training script in the "container"

| Epochs | Batch Size / GPU | Accuracy FP32 - (top1) | Accuracy - BF16 (top1) | 
|--------|------------------|-----------------|----------------------------|
| 90     | 128              | 77.01           | n/a                        |
| 90     | 256              | n/a             | 76.93                      |

**Example training loss plot**

![TrainingLoss](./imgs/train_loss.png)

#### Training performance results

##### Training performance: Habana HLS-1 (8x Habana HL-205)
Our results were obtained by running the `/demo/reset50` training script in the "container". Performance numbers (in images per second) were averaged over an entire training epoch.


| GPUs | Batch Size / GPU | Throughput - FP32 | Throughput - BF16 | 
|----|---------------|---------------|------------------------|
| 1  | 256 | TBD img/s  | TBD img/s    | 
| 8  | 256 | TBD img/s  | TBD img/s   | 

#### Training Time for 90 Epochs

##### Training time: Habana HLS-1 (8x Habana HL-205)

Our results were estimated based on the [training performance results](#training-performance-nvidia-dgx-a100-8x-a100-40g) 
on NVIDIA DGX A100 with (8x A100 40G) GPUs.

| GPUs | Time to train - FP32 | Time to train - BF16 |
|---|--------|---------|
| 1 | ~TBDh   | ~TBDh   |
| 8 | ~TBDh    | ~TBDh   | 

### Change Log





