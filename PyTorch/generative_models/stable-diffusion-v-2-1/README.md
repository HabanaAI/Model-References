# Stable Diffusion 2.1 for PyTorch

This directory provides scripts to perform text-to-image inference on a stable diffusion 2.1 model and is tested and maintained by Habana.

For more information on training and inference of deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents

* [Model-References](../../../README.md)
* [Model Overview](#model-overview)
* [Setup](#setup)
* [Model Checkpoint](#model-checkpoint)
* [Inference and Examples](#inference-and-examples)
* [Supported Configuration](#supported-configuration)
* [Changelog](#changelog)
* [Known Issues](#known-issues)

## Model Overview
This implementation is based on the following paper - [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752).

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
```

### Install Model Requirements
1. In the docker container, go to the model directory:
```bash
cd Model-References/PyTorch/generative_models/stable-diffusion-v-2-1
```

2. Install the required packages using pip.
```bash
pip install -r requirements.txt
```

## Model Checkpoint
### Text-to-Image
Download the pre-trained weights (4.9GB).
```bash
wget https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.ckpt
```

## Inference and Examples
The following command generates a total of 9 images of size 768x768 and saves each sample individually as well as a grid of size `n_iter` x `n_samples` at the specified output location (default: `outputs/txt2img-samples`).

```bash
LOWER_LIST=ops_bf16.txt FP32_LIST=ops_fp32.txt $PYTHON scripts/txt2img.py --prompt "a professional photograph of an astronaut riding a horse" --ckpt v2-1_768-ema-pruned.ckpt --config configs/stable-diffusion/v2-inference-v.yaml --H 768 --W 768 --device hpu --n_samples 3 --n_iter 3
```

For a more detailed description of parameters, please use the following command to see a help message:
```bash
$PYTHON scripts/txt2img.py -h
```

## Performance
The first two batches of images generate a performance penalty.
All subsequent batches will be generated much faster.

## Supported Configuration
| Validated on  | SynapseAI Version | PyTorch Version | Mode |
|---------|-------------------|-----------------|----------------|
| Gaudi   | 1.8.0             | 1.13.1          | Inference |

## Changelog
### 1.8.0
Initial release.

### Script Modifications
Major changes done to the original model from [Stability-AI/stablediffusion](https://github.com/Stability-AI/stablediffusion/tree/d55bcd4d31d0316fcbdf552f2fd2628fdc812500) repository:
* Changed README.
* Added HPU support.
* Modified configs/stable-diffusion/v2-inference-v.yaml
* Changed logic in ddim sampler in order to avoid graph recompilations
* Changed code around einsum operation in ldm/modules/attention.py
* randn moved to cpu in scripts/txt2img.py and ldm/models/diffusion/ddim.py

## Known Issues
* Initial random noise generation has been moved to CPU.
Contrary to when noise is generated on Gaudi, CPU-generated random noise produces consistent output regardless of whether HPU Graph API is used or not.
