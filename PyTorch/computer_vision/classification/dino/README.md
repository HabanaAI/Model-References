# Self-Supervised Vision Transformers with DINO
This repository provides scripts to train and evaluate self-supervised Vision Transformer with DINO, and is maintained by Habana Labs, Ltd. an Intel Company.
For performance information please visit [this page](https://developer.habana.ai/resources/habana-training-models/#performance).
For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

# Table of Contents
- [Model-References](../../../../README.md)
- [Model Overview](#model-overview)
- [Setup](#setup)
  - [Install model requirements](#install-model-requirements)
  - [Training dataset](#dataset)
  - [Evaluation datasets dataset](#dataset)
- [Training](#training)
  - [Single-card](#single-card)
  - [Multi-card](#multi-card)
- [Evaluation](#evaluation)
  - [KNN](#knn)
    - [Single-card](#single-card-1)
    - [Multi-card](#multi-card-1)
  - [Linear](#linear)
    - [Single-card](#single-card-2)
    - [Multi-card](#multi-card-2)
- [Advanced](#advanced)
- [Supported Configurations](#supported-configurations)
- [Changelog](#changelog)
  - [1.5.0](#150)
- [Known issues](#known-issues)

# Model Overview
PyTorch implementation for DINO. The model is based on code from [facebookresearch/dino](https://github.com/facebookresearch/dino/tree/cb711401860da580817918b9167ed73e3eef3dcf) repository.
More details can be found here:

- [`blogpost`](https://ai.facebook.com/blog/dino-paws-computer-vision-with-self-supervised-transformers-and-10x-more-efficient-training)
- [`arXiv`](https://arxiv.org/abs/2104.14294)
- [`Yannic Kilcher's video`](https://www.youtube.com/watch?v=h3ij3F3cPIk)

# Setup
Please follow the instructions provided in the [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the
environment including the `$PYTHON` environment variable.
The guide will walk you through the process of setting up your system to run the model on Gaudi.

In the docker container, clone this repository and switch to the branch that matches your SynapseAI version. (Run the [`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the SynapseAI version).
```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
cd Model-References/PyTorch/computer_vision/classification/dino
```

## Install model Requirements
In the docker container, go to the model directory
```bash
cd Model-References/PyTorch/computer_vision/classification/dino
```
Install required packages using pip
```bash
$PYTHON -m pip install -r requirements.txt
```

## Training dataset
Please download and extract [ImageNet2012](https://imagenet.stanford.edu/) dataset.

> NOTE: Going forward we assume the above ImageNet train and validation dataset is downloaded and available at path `/data/pytorch/imagenet/ILSVRC2012/`

## Evaluation datasets
Different evaluation modes require different datasets:

| mode               | dataset                  | how to get                                                                                                                                                                                                                         | example location                                                            |
|--------------------|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| video segmentation | DAVIS 2017               | `git clone https://github.com/davisvideochallenge/davis-2017`<br>`cd davis-2017`<br>`./data/get_davis.sh`                                                                                                                          | `/data/pytorch/davis-2017/`                                                 |
| image retrieval    | Oxford & Paris revisited | `git clone https://github.com/filipradenovic/revisitop`                                                                                                                                                                            | `/data/pytorch/revisitop/roxford5k/`<br>`/data/pytorch/revisitop/rparis6k/` |
| copy detection     | copydays                 | `wget https://dl.fbaipublicfiles.com/vissl/datasets/copydays_original.tar.gz && tar xvf copydays_original.tar.gz`<br>`wget https://dl.fbaipublicfiles.com/vissl/datasets/copydays_strong.tar.gz && tar xvf copydays_strong.tar.gz` | `/data/pytorch/copydays/`                                                   |


# Training
> NOTE: In following commands `data_path` should point to `train` subdirectory of imagenet.

> NOTE: Running the model with bfloat16 precision improves training time and memory requirements, but gives slightly worse accuracy results.

## Single-card
To run self-supervised DINO training with *vit_small* backbone, *float32* precision and batch size 32 on a single card:
```bash
$PYTHON main_dino.py --arch vit_small --data_path /data/pytorch/imagenet/ILSVRC2012/train --output_dir ./dino_vit_small/
```
To run self-supervised DINO training with *vit_small* backbone, *bfloat16* precision and batch size 64 on a single card:
```bash
$PYTHON main_dino.py --arch vit_small --data_path /data/pytorch/imagenet/ILSVRC2012/train --output_dir ./dino_vit_small/ --hmp --batch_size_per_device 64
```

## Multi-card
To run self-supervised DINO training with *vit_small* backbone, *float32* precision and batch size 32 on 8 cards:
```bash
$PYTHON -m torch.distributed.launch --nproc_per_node=8 main_dino.py --arch vit_small --data_path /data/pytorch/imagenet/ILSVRC2012/train --output_dir ./dino_vit_small/
```
To run self-supervised DINO training with *vit_small* backbone, *bfloat16* precision and batch size 64 on 8 cards:
```bash
$PYTHON -m torch.distributed.launch --nproc_per_node=8 main_dino.py --arch vit_small --data_path /data/pytorch/imagenet/ILSVRC2012/train --output_dir ./dino_vit_small/ --hmp --batch_size_per_device 64
```

# Evaluation
Once self-supervised training is complete, you can run one of several available evaluation methods.

> NOTE: Going forward we assume weights from self-supervised training are in `./dino_vit_small/checkpoint.pth`

> NOTE: In following commands `data_path` should point to main directory of imagenet

## KNN
### Single-card
To run KNN-evaluation on a single card:
```bash
$PYTHON eval_knn.py --pretrained_weights ./dino_vit_small/checkpoint.pth --data_path /data/pytorch/imagenet/ILSVRC2012/
```
> NOTE: `data_path` should point to main directory of imagenet

### Multi-card
To run KNN-evaluation on 8 cards:
```bash
$PYTHON -m torch.distributed.launch --nproc_per_node=8 eval_knn.py --pretrained_weights ./dino_vit_small/checkpoint.pth --data_path /data/pytorch/imagenet/ILSVRC2012
```

## Linear
### Single-card
To run linear evaluation on a single card:
```bash
$PYTHON eval_linear.py --pretrained_weights ./dino_vit_small/checkpoint.pth --data_path /data/pytorch/imagenet/ILSVRC2012 --output_dir ./dino_vit_small_eval_linear/
```

### Multi-card
To run linear evaluation on 8 cards:
```bash
$PYTHON -m torch.distributed.launch --nproc_per_node=8 eval_linear.py --pretrained_weights ./dino_vit_small/checkpoint.pth --data_path /data/pytorch/imagenet/ILSVRC2012 --output_dir ./dino_vit_small_eval_linear/
```

## Copy detection
### Single-card
To run copy detection on a single card:
```bash
$PYTHON eval_copy_detection.py --pretrained_weights ./dino_vit_small/checkpoint.pth --data_path /data/pytorch/copydays
```

### Multi-card
To run copy detection on 8 cards:
```bash
$PYTHON -m torch.distributed.launch --nproc_per_node=8 eval_copy_detection.py --pretrained_weights ./dino_vit_small/checkpoint.pth --data_path /data/pytorch/copydays
```

## Image retrieval
### Single-card
To run image retrieval on a single card:
```bash
$PYTHON eval_image_retrieval.py --pretrained_weights ./dino_vit_small/checkpoint.pth --data_path /data/pytorch/revisitop --dataset roxford5k
$PYTHON eval_image_retrieval.py --pretrained_weights ./dino_vit_small/checkpoint.pth --data_path /data/pytorch/revisitop --dataset rparis6k
```

### Multi-card
To run image retrieval on 8 cards:
```bash
$PYTHON -m torch.distributed.launch --nproc_per_node=8 eval_image_retrieval.py --pretrained_weights ./dino_vit_small/checkpoint.pth --data_path /data/pytorch/revisitop --dataset roxford5k
$PYTHON -m torch.distributed.launch --nproc_per_node=8 eval_image_retrieval.py --pretrained_weights ./dino_vit_small/checkpoint.pth --data_path /data/pytorch/revisitop --dataset rparis6k
```

## Video segmentation
To run video segmentation:
```bash
$PYTHON eval_video_segmentation.py --pretrained_weights ./dino_vit_small/checkpoint.pth --data_path /data/pytorch/davis-2017
```

# Other tasks
## Visualizing attention
To visualize attention:
```bash
$PYTHON visualize_attention.py --pretrained_weights ./dino_vit_small/checkpoint.pth --image_path PATH_TO_SOURCE_IMAGE
```

## Video generation
To generate video with visualized attention:
```bash
$PYTHON video_generation.py --pretrained_weights ./dino_vit_small/checkpoint.pth --input_path PATH_TO_SOURCE_VIDEO
```

# Advanced
Each training/evaluation command can be run with `--help` flag to list all available parameters and their descriptions. For example:
```bash
$PYTHON main_dino.py --help
$PYTHON eval_knn.py --help
$PYTHON eval_linear.py --help
$PYTHON eval_image_retrieval.py --help
$PYTHON eval_video_segmentation.py --help
$PYTHON eval_copy_detection.py --help
$PYTHON visualize_attention.py --help
$PYTHON video_generation.py --help
```

# Supported Configurations
| Device | SynapseAI Version | PyTorch Version |
|--------|-------------------|-----------------|
| Gaudi  | 1.6.0             | 1.12.0          |

# Changelog
Major changes done to original model from [facebookresearch/dino](https://github.com/facebookresearch/dino/tree/cb711401860da580817918b9167ed73e3eef3dcf) repository:
* Some scripts were changed in order to add a possibilty to run the model on Gaudi. It includes loading habana pytorch module and changing tensors device assignment from `cuda` to `hpu`.
* Some temporary workarounds have been applied in scripts in order to enable the model on HPU device. It includes:
  * changing default batch_size_per_device to `32` for self-supervised part;
  * avoiding execution of torch.cat operator with empty tensors;
  * moving `dino_loss` to `cpu` device at the time of checkpoint saving due to a bug in PyTorch framework: https://github.com/pytorch/pytorch/issues/77533;
  * increasing number of chunks in knn_classifier from `100` to `200`.
  * moving `argsort` to `cpu`
* Support for Habana Mixed Precision (HMP) for HPU device has been implemented. It includes:
  * executing `hmp.convert()` function;
  * wrapping optimizer with `hmp.disable_casts()` context manager;
  * adding `ops_bf16.txt`, `ops_fp32.txt` files with bf16/fp32 conversion lists.
* Performance of the model has been increased by limiting synchronization between CPU and the device within gradient clipping implementation.
* Additional functionalities like tensorboard, throughput logging and limiting dataset size have been added.
## 1.5.0
* Initial release
## 1.6.0
* enabled additional tasks (eval_copy_detection, eval_video_segmentation, eval_image_retrieval, visualize_attention, video_generation)
* removed workaround for index_copy_
* removed workaround for bicubic interpolation mode
* fixed OOM for batch_size=64 on float32

# Known issues
