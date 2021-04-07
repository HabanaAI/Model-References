# Resnet50 for PyTorch

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

This folder contains scripts to train Resnet50 model on Habana Gaudi<sup>TM</sup> device to achieve state-of-the-art accuracy.

The Resnet50 demos included in this release are Eager mode training for BS128 with FP32 and BS256 with BF16 mixed precision.

## Model Overview

## Docker Setup
These models are tested with Habana PyTorch docker container 0.13.0-380 for Ubuntu 20.04 and some dependencies contained within it.

### Requirements
* Docker version 19.03.12 or newer
* Sudo access to install required drivers/firmware

### Install Drivers
Follow steps in the [Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) to install the drivers.

## Model Setup
The base training and modelling scripts for training are based on a clone of
https://github.com/NVIDIA/DeepLearningExamples.
Please follow the docker and Model-References git clone setup steps.

1. Stop running dockers
```
docker stop $(docker ps -a -q)
```

2. Pull docker image
```
docker pull vault.habana.ai/gaudi-docker/0.13.0/ubuntu20.04/habanalabs/pytorch-installer:0.13.0-380
```

3. Run docker
```
docker run -td --privileged -v $HOME/datasets/:/root/dataset --net=host vault.habana.ai/gaudi-docker/0.13.0/ubuntu20.04/habanalabs/pytorch-installer:0.13.0-380
```

4. Check name of your docker
 ```
docker ps
 ```

5. Run bash in your docker
 ```
docker exec -ti <NAME> bash
 ```
6. In the docker container, clone the repository and go to PyTorch ResNet directory:
 ```
git clone https://github.com/HabanaAI/Model-References
cd Model-References/PyTorch/computer-vision/ImageClassification/ResNet
 ```
7. Run `./demo_resnet -h` for command-line options

### Set up dataset
Imagenet 2012 dataset needs to be organized as per PyTorch requirement. PyTorch requirements are specified in the link below which contains scripts to organize Imagenet data. https://github.com/soumith/imagenet-multiGPU.


## Training the Model
Clone the Model-References git.
Set up the data set as mentioned in the section "Set up dataset".
ResNet demo supports training in Eager mode. By default, demo_resnet trains Imagenet dataset using the below parameters:
```
python resnet/train.py --data-path= \
/root/datasets/imagenet/ILSVRC2012/ --model=resnet50 \
--device=habana --batch-size=128 --epochs=1 --workers=0 --print-freq=1 \
--run-trace-mode
```
Note: Parameter –-data-path (or -p) points to the path of the dateset location. The above command assumes that Imagenet dataset is available at /root/datasets/imagenet/ folder.

Please refer to the following command for available training parameters:
```
python Model-References/PyTorch/computer-vision/ImageClassification/ResNet/train.py --help
```
```
cd Model-References/PyTorch/computer-vision/ImageClassification/ResNet
Run `./demo_resnet -h` for command-line options
```
i. eager mode: bf16 mixed precision, BS256
```
./demo_resnet --data-type bf16 -b 256 --mode eager -p $DOCKER_HOME/software/data/pytorch/imagenet/ILSVRC2012/ --num-train-steps 100 --num-eval-steps 30
```
ii. eager mode: FP32, BS128
```
./demo_resnet --data-type fp32 -b 128 --mode eager -p $DOCKER_HOME/software/data/pytorch/imagenet/ILSVRC2012/ --num-train-steps 100 --num-eval-steps 30
```

## Multinode Training
To run multi-node demo, make sure the host machine has 512 GB of RAM installed. Modify the docker run command to pass 8 Gaudi cards to the docker container. This ensures the docker has access to all the 8 cards required for multi-node demo.
```
docker run -ti --device=/dev:/dev -v /dev:/dev/ -v $HOME/datasets:/root/datasets --net=host \ --user root --workdir=/root vault.habana.ai/gaudi-docker/0.13.0/ubuntu20.04/habanalabs/pytorch-installer:0.13.0-380
```
Number of nodes can be configured using -w option in the demo script.
Use the below command to run multi-node ResNet demo using 8 cards (1HLS) in BF16 mixed precision:
```
./demo_resnet_dist --data-type bf16 --batch-size 256 -p /root/datasets/imagenet/ILSVRC2012/
```
Use the below command to run multi-node ResNet demo using 8 cards (1HLS) in FP32:
```
./demo_resnet_dist --data-type fp32 --batch-size 128 -p /root/datasets/imagenet/ILSVRC2012/
```

# Known Issues
- Pytorch RN50: channel-last = false is not supported. Train RN50 with channel-last=true. This will be fixed in a subsequent release.

# Training Script Modifications

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

The ResNet50 model changes are listed below:
1. Replaced nn.AdaptiveAvgPool with functionally equivalent nn.AvgPool.
2. Replaced in-place nn.Relu and torch.add_ with non-in-place equivalents.