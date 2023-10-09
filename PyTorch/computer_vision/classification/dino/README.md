# Self-Supervised Vision Transformers with DINO
This repository provides scripts to train and evaluate self-supervised Vision Transformer with DINOand is tested and maintained by Habana.
For more information on training and inference of deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).
To obtain model performance data, refer to the [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

## Table of Contents
- [Model-References](../../../../README.md)
- [Model Overview](#model-overview)
- [Setup](#setup)
- [Training Examples](#training-examples)
- [Evaluation](#evaluation)
- [Supported Configurations](#supported-configurations)
- [Changelog](#changelog)

## Model Overview
This is a PyTorch implementation for DINO. The model is based on code from [facebookresearch/dino](https://github.com/facebookresearch/dino/tree/cb711401860da580817918b9167ed73e3eef3dcf) repository.

For further details, refer to:

- [`blogpost`](https://ai.facebook.com/blog/dino-paws-computer-vision-with-self-supervised-transformers-and-10x-more-efficient-training)
- [`arXiv`](https://arxiv.org/abs/2104.14294)
- [`Yannic Kilcher's video`](https://www.youtube.com/watch?v=h3ij3F3cPIk)

# Setup
Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) 
to set up the environment including the `$PYTHON` environment variable. To achieve the best performance, please follow the methods outlined in the [Optimizing Training Platform guide](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html).
The guides will walk you through the process of setting up your system to run the model on Gaudi.  

### Clone Habana Model-References
In the docker container, clone this repository and switch to the branch that matches your SynapseAI version.
You can run the [`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the SynapseAI version.
```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```

### Install Model Requirements
1. In the docker container, go to the model directory:
```bash
cd Model-References/PyTorch/computer_vision/classification/dino
```
2. Install the required packages using pip.
```bash
$PYTHON -m pip install -r requirements.txt
```

### Dataset Preparation
Download and extract [ImageNet2012](https://imagenet.stanford.edu/) dataset.

*NOTE:* It is assumed that the above ImageNet dataset is downloaded and available at path `/data/pytorch/imagenet/ILSVRC2012/`. 

#### Evaluation Datasets
Different evaluation modes require different datasets as described in the following table:

| Mode               | Dataset                  | How to get                                                                                                                                                                                                                         | Example Location                                                            |
|--------------------|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| Video Segmentation | DAVIS 2017               | `git clone https://github.com/davisvideochallenge/davis-2017`<br>`cd davis-2017`<br>`./data/get_davis.sh`                                                                                                                          | `/data/pytorch/davis-2017/`                                                 |
| Image Retrieval    | Oxford & Paris revisited | `git clone https://github.com/filipradenovic/revisitop`                                                                                                                                                                            | `/data/pytorch/revisitop/roxford5k/`<br>`/data/pytorch/revisitop/rparis6k/` |
| Copy Detection     | copydays                 | `wget https://dl.fbaipublicfiles.com/vissl/datasets/copydays_original.tar.gz && tar xvf copydays_original.tar.gz`<br>`wget https://dl.fbaipublicfiles.com/vissl/datasets/copydays_strong.tar.gz && tar xvf copydays_strong.tar.gz` | `/data/pytorch/copydays/`                                                   |
## Training Examples
*NOTE:*
  - In following commands, `data_path` should point to `train` subdirectory of imagenet.
  - Running the model with BF16 precision improves the training time and memory requirements, but may affect accuracy results.

### Single-card Training Examples
- Run self-supervised DINO training with *vit_small* backbone, FP32 precision and batch size 32 on a single card:
```bash
$PYTHON main_dino.py --arch vit_small --data_path /data/pytorch/imagenet/ILSVRC2012/train --output_dir ./dino_vit_small/
```
- Run self-supervised DINO training with *vit_small* backbone, BF16 precision and batch size 64 on a single card:
```bash
$PYTHON main_dino.py --arch vit_small --data_path /data/pytorch/imagenet/ILSVRC2012/train --output_dir ./dino_vit_small/ --autocast --batch_size_per_device 64
```

### Multi-card Training Examples
- Run self-supervised DINO training with *vit_small* backbone, FP32 precision and batch size 32 on 8 cards:
```bash
$PYTHON -m torch.distributed.launch --nproc_per_node=8 main_dino.py --arch vit_small --data_path /data/pytorch/imagenet/ILSVRC2012/train --output_dir ./dino_vit_small/
```
- Run self-supervised DINO training with *vit_small* backbone, BF16 precision and batch size 64 on 8 cards:
```bash
$PYTHON -m torch.distributed.launch --nproc_per_node=8 main_dino.py --arch vit_small --data_path /data/pytorch/imagenet/ILSVRC2012/train --output_dir ./dino_vit_small/ --autocast --batch_size_per_device 64
```

## Evaluation
Once self-supervised training is completed, you can run one of the available evaluation methods.

*NOTE:*
  - It is assumed that the weights from self-supervised training are located in `./dino_vit_small/checkpoint.pth`. 
  - In following commands, `data_path` should point to `train` subdirectory of imagenet.

### KNN 
**Single-card KNN Examples**

To run KNN-evaluation on a single card, execute the following command:
```bash
$PYTHON eval_knn.py --pretrained_weights ./dino_vit_small/checkpoint.pth --data_path /data/pytorch/imagenet/ILSVRC2012/
```
*NOTE:* In following commands, `data_path` should point to `train` subdirectory of imagenet.

**Multi-card KNN Examples**

To run KNN-evaluation on 8 cards, execute the following command:
```bash
$PYTHON -m torch.distributed.launch --nproc_per_node=8 eval_knn.py --pretrained_weights ./dino_vit_small/checkpoint.pth --data_path /data/pytorch/imagenet/ILSVRC2012
```

### Linear 
**Single-card Linear Examples**

To run linear evaluation on a single card, execute the following command:
```bash
$PYTHON eval_linear.py --pretrained_weights ./dino_vit_small/checkpoint.pth --data_path /data/pytorch/imagenet/ILSVRC2012 --output_dir ./dino_vit_small_eval_linear/
```

**Multi-card Linear Examples**

To run linear evaluation on 8 cards, execute the following command:
```bash
$PYTHON -m torch.distributed.launch --nproc_per_node=8 eval_linear.py --pretrained_weights ./dino_vit_small/checkpoint.pth --data_path /data/pytorch/imagenet/ILSVRC2012 --output_dir ./dino_vit_small_eval_linear/
```

### Copy Detection
**Single-card Copy Detection Examples**

To run copy detection on a single card, execute the following command:
```bash
$PYTHON eval_copy_detection.py --pretrained_weights ./dino_vit_small/checkpoint.pth --data_path /data/pytorch/copydays
```

**Multi-card Copy Detection Examples**

To run copy detection on 8 cards:
```bash
$PYTHON -m torch.distributed.launch --nproc_per_node=8 eval_copy_detection.py --pretrained_weights ./dino_vit_small/checkpoint.pth --data_path /data/pytorch/copydays
```

### Image Retrieval
**Single-card Image Retrieval Examples**

To run image retrieval on a single card, execute the following command:
```bash
$PYTHON eval_image_retrieval.py --pretrained_weights ./dino_vit_small/checkpoint.pth --data_path /data/pytorch/revisitop --dataset roxford5k
$PYTHON eval_image_retrieval.py --pretrained_weights ./dino_vit_small/checkpoint.pth --data_path /data/pytorch/revisitop --dataset rparis6k
```

**Multi-card Image Retrieval Examples**

To run image retrieval on 8 cards, execute the following command:
```bash
$PYTHON -m torch.distributed.launch --nproc_per_node=8 eval_image_retrieval.py --pretrained_weights ./dino_vit_small/checkpoint.pth --data_path /data/pytorch/revisitop --dataset roxford5k
$PYTHON -m torch.distributed.launch --nproc_per_node=8 eval_image_retrieval.py --pretrained_weights ./dino_vit_small/checkpoint.pth --data_path /data/pytorch/revisitop --dataset rparis6k
```

### Video Segmentation
To run video segmentation, execute the following command:
```bash
$PYTHON eval_video_segmentation.py --pretrained_weights ./dino_vit_small/checkpoint.pth --data_path /data/pytorch/davis-2017
```

### Other Tasks
**Visualizing Attention**

To visualize attention, execute the following command:
```bash
$PYTHON visualize_attention.py --pretrained_weights ./dino_vit_small/checkpoint.pth --image_path PATH_TO_SOURCE_IMAGE
```

**Video Generation**

To generate video with visualized attention, execute the following command:
```bash
$PYTHON video_generation.py --pretrained_weights ./dino_vit_small/checkpoint.pth --input_path PATH_TO_SOURCE_VIDEO
```

### Advanced
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

## Supported Configurations
| Validated on | SynapseAI Version | PyTorch Version | Mode |
|--------|-------------------|-----------------|----------------|
| Gaudi  | 1.12.0             | 2.0.1          | Training |

## Changelog
### 1.5.0
* Initial release.
### 1.6.0
* Enabled additional tasks (eval_copy_detection, eval_video_segmentation, eval_image_retrieval, visualize_attention, video_generation). 
* Removed workaround for index_copy_. 
* Removed workaround for bicubic interpolation mode. 
* Fixed OOM for batch_size=64 on FP32. 
### 1.9.0
* Added support for autocast on Gaudi
### 1.11.0
 - Dynamic Shapes will be enabled by default in future releases. It is currently enabled in the training script as a temporary solution.
### 1.12.0
 - Removed support for HMP (Habana Mixed Precision).
### Script Modifications 
Major changes done to original model from [facebookresearch/dino](https://github.com/facebookresearch/dino/tree/cb711401860da580817918b9167ed73e3eef3dcf) repository:
* Modified some scripts to run the model on Gaudi: 
  * Loaded Habana PyTorch module.
  * Changed tensors device assignment from `cuda` to `hpu`.
* Applied temporary workarounds scripts to enable the model on HPU: 
  * Changed the default batch_size_per_device to `32` for self-supervised part.
  * Avoided execution of torch.cat operator with empty tensors
  * Moved `dino_loss` to `cpu` device at the time of checkpoint saving due to a bug in PyTorch framework: https://github.com/pytorch/pytorch/issues/77533;
  * Increased the number of chunks in `knn_classifier` from `100` to `200`.
  * Moved `argsort` to `cpu`. 
* Improved performance of the model by limiting synchronization between CPU and the device within gradient clipping implementation.
* Additional functionalities like TensorBoard, throughput logging and limiting dataset size have been added.
