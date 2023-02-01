# Stable Diffusion for PyTorch

This directory provides scripts to perform text-to-image inference on a stable diffusion model and is tested and maintained by Habana.

For more information on training and inference of deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents

* [Model-References](../../../README.md)
* [Model Overview](#model-overview)
* [Setup](#setup)
* [Model Checkpoint](#model-checkpoint)
* [Inference and Examples](#inference-and-examples)
* [Supported Configuration](#supported-configuration)
* [Known Issues](#known-issues)
* [Changelog](#changelog)

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
cd Model-References/PyTorch/generative_models/stable-diffusion
```

### Install Model Requirements
1. In the docker container, go to the model directory:
```bash
cd Model-References/PyTorch/generative_models/stable-diffusion
```

2. Install the required packages using pip.
```bash
git config --global --add safe.directory `pwd`/src/taming-transformers && git config --global --add safe.directory `pwd`/src/clip && pip install -r requirements.txt
```

## Model Checkpoint
### Text-to-Image

Create a folder for checkpoint with the following command:
```bash
mkdir -p models/ldm/text2img-large/
```

Download the pre-trained weights (5.7GB) by going to https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/, and save
`model.ckpt` file to `models/ldm/text2img-large/` folder.

Users acknowledge and understand that by downloading the checkpoint referenced herein they will be required to comply with
third party licenses and rights pertaining to the checkpoint, and users will be solely liable and responsible for complying
with any applicable licenses. Habana Labs disclaims any warranty or liability with respect to users' use
or compliance with such third party licenses.

## Inference and Examples
Consider the following command:
```bash
$PYTHON scripts/txt2img.py --prompt "a virus monster is playing guitar, oil on canvas" --ddim_eta 0.0 --n_samples 16 --n_rows 4 --n_iter 1 --scale 5.0  --ddim_steps 50 --device 'hpu' --precision hmp --use_hpu_graph
```

This saves each sample individually as well as a grid of size `n_iter` x `n_samples` at the specified output location (default: `outputs/txt2img-samples`).
Quality, sampling speed and diversity are best controlled via the `scale`, `ddim_steps` and `ddim_eta` arguments.
As a rule of thumb, higher values of `scale` produce better samples at the cost of a reduced output diversity.
Furthermore, increasing `ddim_steps` generally also gives higher quality samples, but the returns diminish for values > 250.
Fast sampling (i.e. low values of `ddim_steps`) while retaining good quality can be achieved by using `--ddim_eta 0.0`.

For a more detailed description of parameters, please use the following command to see a help message:
```bash
$PYTHON scripts/txt2img.py -h
```

### Higher Output Resolution
The model has been trained on 256x256 images and usually provides the most semantically meaningful outputs in this native resolution.
However, the output resolution can be controlled with `-H` and `-W` parameters.
For example, this command produces four 512x512 outputs:
```bash
$PYTHON scripts/txt2img.py --prompt "a virus monster is playing guitar, oil on canvas" --ddim_eta 0.0 --n_samples 4 --n_iter 1 --scale 5.0  --ddim_steps 50 --device 'hpu' --precision hmp --H 512 --W 512 --use_hpu_graph
```

### HPU Graph API
In some scenarios, especially when working with lower batch sizes, kernel execution on HPU might be faster than op accumulation on CPU.
In such cases, the CPU becomes a bottleneck, and the benefits of using Gaudi accelerators are limited.
To combat this, Habana offers an HPU Graph API, which allows for one-time ops accumulation in a graph structure that is being reused over time.
To use this feature in stable-diffusion, add the `--use_hpu_graph` flag to your command, as instructed in the examples.
When this flag is not passed, inference on lower batch sizes might take more time.
On the other hand, this feature might introduce some overhead and degrade performance slightly for certain configurations, especially for higher output resolutions.

For more datails regarding inference with HPU Graphs, please refer to [the documentation](https://docs.habana.ai/en/latest/PyTorch/Inference_on_Gaudi/Inference_using_HPU_Graphs/Inference_using_HPU_Graphs.html).

## Performance
The first batch of images generates a performance penalty.
All subsequent batches will be generated much faster.
For example, the following command generates 4 batches of 4 images.
It will take more time to generate the first set of 4 images than the remaining 3.
```bash
$PYTHON scripts/txt2img.py --prompt "a virus monster is playing guitar, oil on canvas" --ddim_eta 0.0 --n_samples 4 --n_iter 4 --scale 5.0  --ddim_steps 50 --device 'hpu' --precision hmp --use_hpu_graph
```

## Supported Configuration
| Device  | SynapseAI Version | PyTorch Version |
|---------|-------------------|-----------------|
| Gaudi   | 1.7.1             | 1.13.0          |
| Gaudi2  | 1.7.1             | 1.13.0          |

## Known Issues
* When `--use_hpu_graph` flag is not passed to the script, the progress bar might be presenting misleading information about the execution status.
For example, it might be getting stuck at certain points during the process.
This is a result of the progress bar showing CPU op accumulation status and not the actual execution when HPU Graph API is not used.
* Initial random noise generation has been moved to CPU.
Contrary to when noise is generated on Gaudi, CPU-generated random noise produces consistent outputs regardless of whether HPU Graph API is used or not.

## Changelog
### 1.7.0
Initial release.

### Script Modifications
Major changes done to the original model from [pesser/stable-diffusion](https://github.com/pesser/stable-diffusion/commit/a166aa7fbf578f41f855efeab2e14001d6732563) repository:
* Changed README.
* Changed default config and ckpt in scripts/txt2img.py.
* Added HPU and Hmp (Habana mixed precision) support.
* Changed default precision from autocast to full.
* Changed logic in ldm/models/diffusion/ddim.py in order to avoid graph recompilations.
* Removed scripts/latent_imagenet_diffusion.ipynb.
* Added interactive mode for demonstrative purposes.
* Fixed python insecurity in requirements.txt (pytorch-lightning).
* Added an option to use HPU Graph API.
