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
cd Model-References/PyTorch/generative_models/stable-diffusion-v-2-1
```

2. Install the required packages using pip.
```bash
pip install -r requirements.txt --user
```

## Model Checkpoint
### Text-to-Image
Download the pre-trained weights for 768x768 images (4.9GB)
```bash
wget https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.ckpt
```
and/or 512x512 images (4.9GB).
```bash
wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt
```

## Inference and Examples
The following command generates a total of 3 images of size 768x768 and saves each sample individually as well as a grid of size `n_iter` x `n_samples` at the specified output location (default: `outputs/txt2img-samples`).

```bash
$PYTHON scripts/txt2img.py --prompt "a professional photograph of an astronaut riding a horse" --ckpt v2-1_768-ema-pruned.ckpt --config configs/stable-diffusion/v2-inference-v.yaml --H 768 --W 768 --n_samples 1 --n_iter 3 --use_hpu_graph
```
To generate 3 images of a 512x512 size using a k-diffusion dpmpp_2m sampler with 35 steps, use the command:
```bash
$PYTHON scripts/txt2img.py --prompt "a professional photograph of an astronaut riding a horse" --ckpt v2-1_512-ema-pruned.ckpt --config configs/stable-diffusion/v2-inference.yaml --H 512 --W 512 --n_samples 1 --n_iter 3 --steps 35 --k_sampler dpmpp_2m --use_hpu_graph
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
| Gaudi   | 1.12.0             | 2.0.1          | Inference |
| Gaudi2   | 1.12.0             | 2.0.1          | Inference |

## Changelog
### 1.8.0
Initial release.

### 1.10.0
Decreased host overhead to minimum by rewriting samplers and the main sampling loop.

### Script Modifications
Major changes done to the original model from [Stability-AI/stablediffusion](https://github.com/Stability-AI/stablediffusion/tree/d55bcd4d31d0316fcbdf552f2fd2628fdc812500) repository:
* Changed README.
* Added HPU support.
* Modified configs/stable-diffusion/v2-inference-v.yaml and configs/stable-diffusion/v2-inference.yaml
* Changed code around einsum operation in ldm/modules/attention.py
* randn moved to cpu in scripts/txt2img.py and ldm/models/diffusion/ddim.py
* sampling is rewritten in an accelerator-friendly way

## Known Issues
* Initial random noise generation has been moved to CPU.
Contrary to when noise is generated on Gaudi, CPU-generated random noise produces consistent output regardless of whether HPU Graph API is used or not.
* The model supports batch sizes up to 16 on Gaudi and up to 8 on Gaudi2 for output images 512x512px, and batch size 1 for images 768x768px on Gaudi and Gaudi2.
