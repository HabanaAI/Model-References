# Stable Diffusion 1.5 for PyTorch

This directory provides scripts to perform text-to-image inference on a stable diffusion 1.5 model and is tested and maintained by Habana.

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
cd Model-References/PyTorch/generative_models/stable-diffusion-v-1-5
```

2. Install the required packages using pip.
```bash
git config --global --add safe.directory `pwd`/src/taming-transformers && git config --global --add safe.directory `pwd`/src/clip && pip install -r requirements.txt --user
```

## Model Checkpoint
### Text-to-Image
Download the pre-trained weights (4GB). Make sure you have a Hugging Face account.
```bash
mkdir -p models/ldm/stable-diffusion-v1/
wget --user $YOUR_USERNAME --password $YOUR_PASSWORD -O models/ldm/stable-diffusion-v1/v1-5-pruned-emaonly.ckpt https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt
```

## Inference and Examples
The following command generates a total of 9 images and saves each sample individually as well as a grid of size `n_iter` x `n_samples` at the specified output location (default: `outputs/txt2img-samples`).

```bash
$PYTHON scripts/txt2img.py --prompt "a photograph of an astronaut riding a horse" --precision autocast --device hpu --n_iter 3 --n_samples 3 --use_hpu_graph
```

For a more detailed description of parameters, please use the following command to see a help message:
```bash
$PYTHON scripts/txt2img.py -h
```

### HPU Graph API
In some scenarios, especially when working with lower batch sizes, kernel execution on HPU might be faster than op accumulation on CPU.
In such cases, the CPU becomes a bottleneck, and the benefits of using Gaudi accelerators are limited.
To combat this, Habana offers an HPU Graph API which allows for one-time ops accumulation in a graph structure that is reused over time.
To use this feature in stable-diffusion, add the `--use_hpu_graph` flag to your command as instructed in the examples.
When this flag is not passed, inference on lower batch sizes might take more time.
On the other hand, this feature might introduce some overhead and descrease performance slightly for certain configurations, especially for higher output resolutions.

For more datails regarding inference with HPU Graphs, please refer to [the documentation](https://docs.habana.ai/en/latest/PyTorch/Inference_on_Gaudi/Inference_using_HPU_Graphs/Inference_using_HPU_Graphs.html).

## Performance
The first batch of images generates a performance penalty.
All subsequent batches will be generated much faster.

## Supported Configuration
| Validated on  | SynapseAI Version | PyTorch Version | Mode |
|---------|-------------------|-----------------|------------------|
| Gaudi   | 1.8.0             | 1.13.1          | Inference |
| Gaudi2  | 1.8.0             | 1.13.1          | Inference |

## Changelog
### 1.8.0
Initial release.

### Script Modifications
Major changes done to the original model from [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion/tree/69ae4b35e0a0f6ee1af8bb9a5d0016ccb27e36dc) repository:
* Changed README.
* Added HPU and HMP (Habana Mixed Precision) support.
* Changed default precision from autocast to full.
* Changed logic in ddim and plms samplers in order to avoid graph recompilations.
* Fixed python insecurity in requirements.txt (pytorch-lightning).
* Added an option to use HPU Graph API.
* Added interactive mode for demonstrative purposes.
* Changed verbosity level of transformers' logging to error to suppress insignificant warnings.

## Known Issues
* When `--use_hpu_graph` flag is not passed to the script, the progress bar might present misleading information about the execution status.
For example, it might get stuck at certain points during the process.
This is a result of the progress bar showing CPU op accumulation status and not the actual execution when HPU Graph API is not used.
* Initial random noise generation has been moved to CPU.
Contrary to when noise is generated on Gaudi, CPU-generated random noise produces consistent output regardless of whether HPU Graph API is used or not.
