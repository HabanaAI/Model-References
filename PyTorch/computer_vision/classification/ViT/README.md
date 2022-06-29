# Table of Contents
- [Vision Transformer](#vision-transformer)
  - [Model Overview](#model-overview)
  - [Changes](#changes)
  - [Setup](#setup)
    - [Install the requirements](#install-the-requirements)
    - [Download Pre-trained model](#download-pre-trained-model)
    - [Setting up the dataset](#setting-up-the-dataset)
  - [Training the Model](#training-the-model)
  - [Known Issues](#known-issues)

# Vision Transformer
This folder contains scripts to train Vision Transformer model on Habana Gaudi<sup>TM</sup> device to achieve state-of-the-art accuracy. Please visit [this page](https://developer.habana.ai/resources/habana-training-models/#performance) for performance information.

The Vision Transformer demos included in this release are Eager mode and Lazy mode training for different batch sizes with FP32 & BF16 mixed precision.

## Model Overview
Pytorch reimplementation done in [Pytorch Image Models(timm)](https://github.com/rwightman/pytorch-image-models) of [Google's repository for the ViT model](https://github.com/google-research/vision_transformer) that was released with the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) by Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby.

This paper show that Transformers applied directly to image patches and pre-trained on large datasets work really well on image recognition task.

![fig1](./img/figure1.png)

Vision Transformer achieve State-of-the-Art in image recognition task with standard Transformer encoder and fixed-size patches. In order to perform classification, author use the standard approach of adding an extra learnable "classification token" to the sequence.

![fig2](./img/figure2.png)

## Changes
The following are the changes made to the training scripts:

1. Added support for Habana devices

   a. loading Habana specific library.

   b. Certain environment variables are defined for habana device.

   c. Added support to run training in lazy mode in addition to the eager mode.

   d. mark_step() is performed to trigger execution.

   e. Added support to use HPU accelerator plugin, DDP plugin(for multi-card training) & mixed precision plugin
   provided with installed Pytorch Lightning package.

2. To improve performance

   a. Enhance with channels last data format (NHWC) support.

   b. Permute convolution weight tensors for better performance.

   c. Use fusedSGD instead of default SGD

   d. Move the div before the matmul in attention module

## Setup
Please follow the instructions given in the following link for setting up the
environment including the `$PYTHON` environment variable: [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html).
This guide will walk you through the process of setting up your system to run
the model on Gaudi.

In the docker container, clone this repository and switch to the branch that
matches your SynapseAI version. (Run the
[`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.)

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```

Go to PyTorch Vision Transformer directory:
```bash
cd /path/to/ViT_directory
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

### Download Pre-trained model
Below are the Google's official checkpoints:
* [Available models](https://console.cloud.google.com/storage/vit_models/): ViT-B_16(**85.8M**), R50+ViT-B_16(**97.96M**), ViT-B_32(**87.5M**), ViT-L_16(**303.4M**), ViT-L_32(**305.5M**), ViT-H_14(**630.8M**)
  * imagenet21k pre-train models
    * ViT-B_16, ViT-B_32, ViT-L_16, ViT-L_32, ViT-H_14
  * imagenet21k pre-train + imagenet2012 fine-tuned models
    * ViT-B_16-224, ViT-B_16, ViT-B_32, ViT-L_16-224, ViT-L_16, ViT-L_32
  * Hybrid Model([Resnet50](https://github.com/google-research/big_transfer) + Transformer)
    * R50-ViT-B_16

```
# imagenet21k pre-train
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz

# imagenet21k pre-train + imagenet2012 fine-tuning
wget https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/{MODEL_NAME}.npz
```

### Setting up the dataset
Imagenet 2012 dataset needs to be organized as per PyTorch requirement. PyTorch requirements are specified in the link below which contains scripts to organize Imagenet data.

https://github.com/soumith/imagenet-multiGPU.torch#data-processing

The training images for imagenet are already in appropriate subfolders (like n07579787, n07880968).
You need to get the validation groundtruth and move the validation images into appropriate subfolders.
To do this, download ILSVRC2012_img_train.tar ILSVRC2012_img_val.tar and use the following commands:

```bash
# extract train data
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
# extract validation data
cd ../ && mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```

NOTE: Going forward we assume the above ImageNet train and validation dataset is downloaded and available at path `/root/software/data/pytorch/imagenet/ILSVRC2012/`

## Training the Model

### Run Single Card Training

Train on 1 Gaudi card, batch size=32, Gradient accumulation=1 , FP32 :
```bash
$PYTHON train.py --name imagenet1k_TF --dataset imagenet1K --data_path /root/software/data/pytorch/imagenet/ILSVRC2012 --model_type ViT-B_16 --pretrained_dir ./ViT-B_16.npz --num_steps 20000 --eval_every 1000 --train_batch_size 32 --gradient_accumulation_steps 1 --img_size 384 --learning_rate 0.06
```

Train on 1 Gaudi card, batch size=32, Gradient accumulation=1 , mixed precision (BF16) :
```bash
$PYTHON train.py --name imagenet1k_TF --dataset imagenet1K --data_path /root/software/data/pytorch/imagenet/ILSVRC2012 --model_type ViT-B_16 --pretrained_dir ./ViT-B_16.npz --num_steps 20000 --eval_every 1000 --train_batch_size 32 --gradient_accumulation_steps 1 --img_size 384 --learning_rate 0.06 --hmp --hmp-opt-level O1
```

Train on 1 Gaudi card, batch size=512, Gradient accumulation=16 , FP32 :
```bash
$PYTHON train.py --name imagenet1k_TF --dataset imagenet1K --data_path /root/software/data/pytorch/imagenet/ILSVRC2012 --model_type ViT-B_16 --pretrained_dir ./ViT-B_16.npz --num_steps 20000 --eval_every 1000 --train_batch_size 512 --gradient_accumulation_steps 16 --img_size 384 --learning_rate 0.06
```

Train on 1 Gaudi card, batch size=512, Gradient accumulation=16 , mixed precision (BF16) :
```bash
$PYTHON train.py --name imagenet1k_TF --dataset imagenet1K --data_path /root/software/data/pytorch/imagenet/ILSVRC2012 --model_type ViT-B_16 --pretrained_dir ./ViT-B_16.npz --num_steps 20000 --eval_every 1000 --train_batch_size 512 --gradient_accumulation_steps 16 --img_size 384 --learning_rate 0.06 --hmp --hmp-opt-level O1
```

## Multi-HPU Training
To run multi-HPU demo, make sure the host machine has 512 GB of RAM installed.
Also ensure you followed the [Gaudi Setup and
Installation Guide](https://github.com/HabanaAI/Setup_and_Install) to install and set up docker,
so that the docker has access to all the 8 cards required for multi-HPU demo.

Before execution of the multi-HPU demo scripts, make sure all server network interfaces are up. You can change the state of each network interface managed by the habanalabs driver using the following command:
```
sudo ip link set <interface_name> up
```
To identify if a specific network interface is managed by the habanalabs driver type, run:
```
sudo ethtool -i <interface_name>
```

Finally to train the Swin Transformer on multiple HPUs with below script.

Train on 8 Gaudi card, batch size=512, Gradient accumulation=2 , mixed precision (BF16) :

*mpirun map-by PE attribute value may vary on your setup and should be calculated as:<br>
socket:PE = floor((number of physical cores) / (number of gaudi devices per each node))*
```bash
mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON -u train.py --name imagenet1k_TF --dataset imagenet1K --data_path /root/software/data/pytorch/imagenet/ILSVRC2012 --model_type ViT-B_16 --pretrained_dir ./ViT-B_16.npz --num_steps 20000 --eval_every 1000 --train_batch_size 64 --gradient_accumulation_steps 2 --img_size 384 --learning_rate 0.06 --hmp --hmp-opt-level O1
```

# Supported Configurations

| Device | SynapseAI Version | PyTorch Version |
|-----|-----|-----|
| Gaudi | 1.4.1 | 1.10.2 |

## Known Issues
- Placing mark_step() arbitrarily may lead to undefined behavior. Recommend to keep mark_step() as shown in provided scripts.
- Only scripts & configurations mentioned in this README are supported and verified.
