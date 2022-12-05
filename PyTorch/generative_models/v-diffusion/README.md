# V-diffusion for PyTorch

This directory provides scripts to perform inference on a V-diffusion model and is tested and maintained by Habana.

For more information on training and inference of deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents

* [Model-References](../../../README.md)
* [Model Overview](#model-overview)
* [Setup](#setup)
* [Model Checkpoint](#model-checkpoint)
* [Inference and Examples](#inference-and-examples)
* [Performance](#performance)
* [Supported Configuration](#supported-configuration)
* [Changelog](#changelog)

## Model Overview
[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) are trained to reverse a gradual noising process, in order to generate samples from the learned data distributions starting from random noise.
They use the 'v' objective from [Progressive Distillation for Fast Sampling of Diffusion Models](https://openreview.net/forum?id=TIdIXIpzhoI).
The scripts provided in this repository were tested with [PRK/PLMS sampling methods](https://openreview.net/forum?id=PlKWVd2yBkY).
The model generates images based on a textual prompt.

## Setup
Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the environment including the `$PYTHON` environment variable.
This guide will walk you through the process of setting up your system to run the model on Gaudi.

### Clone Habana Model-References
In the docker container, clone this repository and switch to the branch that matches your SynapseAI version.
You can run the [`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the SynapseAI version.
```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
cd Model-References/PyTorch/generative_models/v-diffusion
```

### Install Model Requirements
1. In the docker container, go to the model directory:
```bash
cd Model-References/PyTorch/generative_models/v-diffusion
```
2. Install the required packages using pip
```bash
$PYTHON -m pip install -r requirements.txt
```

## Model Checkpoint
Our example uses a [CC12M_1 CFG 256x256](https://the-eye.eu/public/AI/models/v-diffusion/cc12m_1_cfg.pth) model.
You can use the following set of commands to create a `checkpoints/` directory and download the model there.
```
cd Model-References/PyTorch/generative_models/v-diffusion
mkdir checkpoints
wget https://the-eye.eu/public/AI/models/v-diffusion/cc12m_1_cfg.pth && mv cc12m_1_cfg.pth checkpoints/
```

This is a 602M parameter CLIP conditioned model trained on [Conceptual 12M](https://github.com/google-research-datasets/conceptual-12m) for 3.1M steps and then fine-tuned for classifier-free guidance for 250K additional steps.

### Security
SHA-256 for the [CC12M_1 CFG 256x256](https://the-eye.eu/public/AI/models/v-diffusion/cc12m_1_cfg.pth) file: `4fc95ee1b3205a3f7422a07746383776e1dbc367eaf06a5b658ad351e77b7bda`,

## Inference and Examples

Consider the following command:
```
./cfg_sample.py "the rise of consciousness":5 -n 8 -bs 4 --seed 0 --device 'hpu' --hmp
```

It will generate 2 batches of 4 images (controlled by a `-bs` parameter) each for a total of 8 images (controlled by a `-n` parameter).
`:5` in the example above specifies a weight associated with the textual prompt.
A weight of 1 will sample images that match the prompt roughly as well as images that usually match prompts like that in the training set.
The default weight is 3.
For a more detailed description of parametrs, please see the help message:
```
./cfg_sample.py -h
```

## Performance
The first batch of images will generate a performance penalty.
All subsequent batches will be generated much faster.
For example, the following command will generate 4 batches of 4 images.
It will take significantly more time to generate the first set of 4 images than the remaining 3.
```
./cfg_sample.py "the rise of consciousness":5 -n 16 -bs 4 --seed 0 --device 'hpu' --hmp
```

## Supported Configuration
| Device | SynapseAI Version | PyTorch Version |
|--------|-------------------|-----------------|
| Gaudi  | 1.7.1             | 1.13.0          |

## Changelog
### 1.7.0
Removed PT_HPU_ENABLE_SPLIT_INFERENCE environment variable.
### 1.6.0
Initial release

### Script Modifications
Major changes done to original model from [crowsonkb/v-diffusion-pytorch](https://github.com/crowsonkb/v-diffusion-pytorch/tree/93b6a54986d8259837a100046777fba52d812554) repository:
* Changed README.
* Removed jupyter notebooks.
* Added HPU support.
* Added BF16 mixed precision logic.
* Replaced GroupNorm (with num_groups=1) with mathematically equivalent LayerNorm for better performance.
* Changed some code that was originally using variables in order to avoid graph recompilations.
* Added htcore.mark_step() in relevant places.
* Changed weights conversion in CLIP model so that weights are converted to bf16 on HPU and to fp16 otherwise.
* Moved randn operator execution to CPU.
* Changed the way script performance figures are reported.
* Replaced torch.atan2 with torch.atan in diffusion/utils.py.
* Changed repeat_interleave logic in cfg_model_fn.
* Set PT_HPU_ENABLE_SPLIT_INFERENCE environment variable.
* Removed torch.cuda.amp.autocast when running on HPU.