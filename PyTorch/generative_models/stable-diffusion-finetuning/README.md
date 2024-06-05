# Stable Diffusion 2.1 FineTuning with Low-Rank Adaptation of Large Language Models for PyTorch
This directory provides scripts to fine-tune Stable Diffusion Model (2.1) which is based on latent text-to-image diffusion model and is tested and maintained by Intel® Gaudi®.
For more information on training and inference of deep learning models using Intel Gaudi AI accelerator, refer to [developer.habana.ai](https://developer.habana.ai/resources/). Before you get started, make sure to review the [Supported Configuration](#supported-configuration).

  - [Model-References](../../../README.md)
  - [Model Overview](#model-overview)
  - [Setup](#setup)
  - [Training and Examples](#training)
  - [Supported Configuration](#supported-configuration)
  - [Changelog](#changelog)

## Model Overview

This implementation is designed to fine-tune Stable Diffusion model (stabilityai/stable-diffusion-2-1-base) by Low-rank Adaptation which is considered to be very efficient. Here the pipeline is to fine-tune CLIP + Unet + token to gain better results.

  More details about LoRA and its usage with diffusion can be found at [Blog](https://huggingface.co/blog/lora) and [examples](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image#training-with-lora)

### How to Use
Users acknowledge and understand that the models referenced by Habana are mere examples for models that can be run on Gaudi.
Users bear sole liability and responsibility to follow and comply with any third party licenses pertaining to such models,
and Habana Labs disclaims and will bear no any warranty or liability with respect to users' use or compliance with such third party licenses.

## Setup
Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the environment including the `$PYTHON` environment variable. To achieve the best performance, please follow the methods outlined in the [Optimizing Training Platform guide](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html).
The guides will walk you through the process of setting up your system to run the model on Gaudi.

### Clone Intel Gaudi Model-References
In the docker container, clone this repository and switch to the branch that matches your Intel Gaudi software version.
You can run the [`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the Intel Gaudi software version.
```bash
git clone -b [Intel Gaudi software] https://github.com/HabanaAI/Model-References
cd Model-References/PyTorch/generative_models/stable-diffusion-finetuning
```
### Install Model Requirements
1. In the docker container, go to the model directory:
```bash
cd Model-References/PyTorch/generative_models/stable-diffusion-finetuning
```

2. Install the required packages using pip:
```bash
pip install -r requirements.txt
pip install .
```

## Training
### Model Checkpoint

The fine-tuning script internally will download checkpoints from https://huggingface.co/stabilityai/stable-diffusion-2-1-base.

Users acknowledge and understand that by downloading the checkpoint referenced herein they will be required to comply
with third party licenses and rights pertaining to the checkpoint, and users will be solely liable and responsible
for complying with any applicable licenses. Habana Labs disclaims any warranty or liability with respect to users' use
or compliance with such third party licenses.

### Dataset Preparation
For the finetuning we have used synthetic dataset.

1. In the docker container, go to the model directory:
```bash
cd Model-References/PyTorch/generative_models/stable-diffusion-finetuning
```

2. Generate synthetic dataset:
```bash
python data/scripts/gen_synth_data.py
```

### 1. Fine-tuning Stable diffusion with LoRA CLI

#### Single Card Training Examples
**Run training on 1 HPU:**

```bash
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR=<path-to-instances>
export OUTPUT_DIR=<path-to-output>

lora_pti \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --use_face_segmentation_condition \
  --resolution=512 \
  --train_batch_size=7 \
  --gradient_accumulation_steps=1 \
  --learning_rate_unet=5e-5 \
  --learning_rate_ti=2e-3 \
  --color_jitter \
  --lr_scheduler="linear" --lr_scheduler_lora="linear"\
  --lr_warmup_steps=0 \
  --placeholder_tokens="<s1>|<s2>" \
  --use_template="object"\
  --save_steps=50 \
  --max_train_steps_ti=500 \
  --max_train_steps_tuning=1000 \
  --perform_inversion=True \
  --clip_ti_decay \
  --weight_decay_ti=0.000 \
  --weight_decay_lora=0.001\
  --continue_inversion \
  --continue_inversion_lr=1e-3 \
  --device="hpu" \
  --lora_rank=16 \
  --use_lazy_mode=True \
  --use_fused_adamw=True \
  --print_freq=50 \
  --use_fused_clip_norm=True \
```

[Refer to reference model to see what these parameters mean](https://github.com/cloneofsimo/lora/discussions/121).


## Supported Configuration
| Validated on  | Intel Gaudi Software Version | PyTorch Version | Mode |
|---------|-------------------|-----------------|-----------------------|
| Gaudi   | 1.11.0             | 2.0.1          | Training |
| Gaudi 2 | 1.16.0             | 2.2.2          | Training |

## Changelog

### Script Modifications
### 1.15.0
* Added support for dynamic shapes in Stable Diffusion Finetuning.

### 1.13.0
* Modified training script to support diffusers version 0.21.4.

### 1.12.0
* Dynamic Shapes will be enabled by default in future releases. It is currently disabled in training script.

### 1.11.0
* Dynamic Shapes will be enabled by default in future releases. It is currently enabled in training script as a temporary solution.

### 1.10.0
* Modified README.
* Enabled PyTorch autocast on Gaudi.
* Added additional logging.
* Added support for HPU.
* Added FusedAdamW and FusedClipNorm.
* Added Tensorboard logging.
* Added device trace and memory stats reporting.
* Added print frequency change.
* Enabled HPU Graphs execution for host optimization.
