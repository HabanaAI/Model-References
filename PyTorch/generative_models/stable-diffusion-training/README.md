# Stable Diffusion for PyTorch

This directory provides scripts to train Stable Diffusion Model which is based on latent text-to-image diffusion model and is tested and maintained by Habana.
For more information on training and inference of deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

  - [Model-References](../../../README.md)
  - [Model Overview](#model-overview)
  - [Setup](#setup)
  - [Training and Examples](#training-and-examples)
  - [Supported Configuration](#supported-configuration)
  - [Known Issues](#known-issues)
  - [Changelog](#changelog)


## Model Overview
This implementation is based on the following paper - [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752).
This model uses a frozen CLIP ViT-L/14 text encoder to condition the model on text prompts. With its 860M UNet and 123M text encoder, the model is relatively lightweight and runs on a HPU.

### How to use
Users acknowledge and understand that the models referenced by Habana are mere examples for models that can be run on Gaudi.
Users bear sole liability and responsibility to follow and comply with any third party licenses pertaining to such models,
and Habana Labs disclaims and will bear no any warranty or liability with respect to users' use or compliance with such third party licenses.

## Setup
Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the environment including the `$PYTHON` environment variable.
This guide will walk you through the process of setting up your system to run the model on Gaudi.

### Clone Habana Model-References
In the docker container, clone this repository and switch to the branch that matches your SynapseAI version.
You can run the [`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the SynapseAI version.
```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
cd Model-References/PyTorch/generative_models/stable-diffusion-training
```

### Install Model Requirements
1. In the docker container, go to the model directory:
```bash
cd Model-References/PyTorch/generative_models/stable-diffusion-training
```

2. Install the required packages using pip:
```bash
pip install -r requirements.txt
```

### Model Checkpoint

Download the model checkpoint for `first_stage_config` by going to https://ommer-lab.com/files/latent-diffusion/
location, save kl-f8.zip to your local folder and unzip it. Make sure to point the `ckpt_path` in `hpu_config.yaml`
file to the full path of the checkpoint.

Users acknowledge and understand that by downloading the checkpoint referenced herein they will be required to comply
with third party licenses and rights pertaining to the checkpoint, and users will be solely liable and responsible
for complying with any applicable licenses. Habana Labs disclaims any warranty or liability with respect to users' use
or compliance with such third party licenses.

### Dataset Preparation

Users acknowledge and understand that by downloading the dataset referenced herein they will be required to comply
with third party licenses and usage rights to the data, and users will be solely liable and responsible for
complying with any applicable licenses. Habana Labs disclaims any warranty or liability with respect to users' use
or compliance with such third party licenses.

Training on stable-diffusion is performed using laion2B-en dataset: https://huggingface.co/datasets/laion/laion2B-en

However, to reach the first checkpoint, laion-high-resolution dataset
(https://huggingface.co/datasets/laion/laion-high-resolution) for training must be included.

1. To download Laion-2B-en dataset, run the following. The below method downloads the dataset locally when not using S3 bucket.
```bash
pip install img2dataset
mkdir laion2B-en && cd laion2B-en
for i in {00000..00127}; do wget https://huggingface.co/datasets/laion/laion2B-en/resolve/main/part-$i-5114fd87-297e-42b0-9d11-50f1df323dfa-c000.snappy.parquet; done
cd ..
```
2. Create a download.py file and add following:

```bash
from img2dataset import download
url_list ="laion2B-en/"
download(
    processes_count=16,
    thread_count=32,
    url_list=url_list,
    image_size=256,
    output_folder="output",
    output_format="files",
    input_format="parquet",
    url_col="URL",
    caption_col="TEXT",
    enable_wandb=False,
    number_sample_per_shard=1000,
    distributor="multiprocessing",
)
```
In addition,
"processes_count" and "thread_count" are tunable parameters based on the host machine where the dataset is downloaded.
"image_size" can be modified as per the requirement of the training checkpoint.

3. Download the dataset:
```bash
python download.py
```
Note: Generic data preparation can be found https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion5B.md

## Training and Examples
### Single Card and Multi-Card Training Examples
**Run training on 1 HPU:**
* HPU, Lazy mode with FP32 precision:
```
python main.py --base hpu_config.yaml --train --scale_lr False --seed 0 --hpus 1 --batch_size 4 --use_lazy_mode True --no-test True
```
* HPU, Lazy mode with BF16 mixed precision:
```
python main.py --base hpu_config.yaml --train --scale_lr False --seed 0 --hpus 1 --batch_size 8 --use_lazy_mode True --hmp --no-test True
```
**Run training on 8 HPU:**
* 8 HPUs, Lazy mode with FP32 precision:
```
python main.py --base hpu_config.yaml --train --scale_lr False --seed 0 --hpus 8 --batch_size 4 --use_lazy_mode True --no-test True
```
* 8 HPUs, Lazy mode with BF16 mixed precision:
```
python main.py --base hpu_config.yaml --train --scale_lr False --seed 0 --hpus 8 --batch_size 8 --use_lazy_mode True --hmp --no-test True
```
To run a specific number of iterations or epochs, use `--max_steps` and `--max_epochs` respectively

## Supported Configuration
| Validated on  | SynapseAI Version | PyTorch Version | Mode |
|---------|-------------------|-----------------|----------------|
| Gaudi   | 1.8.0             | 1.13.1          | Training |

## Known Issues
* Model was trained using "laion2B-en" dataset for limited number of steps with `batch_size: 8` and `accumulate_grad_batches: 16`
* ImageLogger callbacks were not executed as part of training
* Only scripts and configurations mentioned in this README are supported and verified.

## Changelog
### 1.8.0
Initial release.

### Script Modifications
Major changes done to the original model from [pesser/stable-diffusion](https://github.com/pesser/stable-diffusion/commit/693e713c3e72e72b8e2b97236eb21526217e83ad) repository:
* Changed README file content.
* environment.yaml is replaced from CompVis's stable-diffusion.
* Changed the basic implementation of dataset for reading files from directory in img2dataset's format(ldm/data/base.py).
* Changed default precision from autocast to full.
* PTL version changed from 1.4.2 to 1.7.7 with required modification in scripts and torchmetrics from 0.6 to 0.10.3.
* ImageLogger callback disabled while running on HPU.
* HPU support for single and multi card(using HPUParallelStrategy) is enabled.Also tuned parameter values for DDP.
* HMP support added for mixed precison training with required ops in fp32 and bf16.
* HPU and CPU config files were added.
* Introduced additional mark_steps as per need in the Model.
* Added FusedAdamw optimizer support nd made it enabled by default.
* Introduced the print frequency changes to extract the loss as per the user configured value from `refresh_rate`.
* Added changes in dataloader for multicard.
