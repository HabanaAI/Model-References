# Stable Diffusion for PyTorch with GPU migration

This directory provides scripts to perform text-to-image inference on a stable diffusion model and is tested and maintained by Habana.
The model has been enabled using an experimental feature called GPU migration.
*NOTE:* You can review the [Stable diffusion model](../../../../generative_models/stable-diffusion/README.md) enabled with a more traditional approach.

For more information on training and inference of deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).
To obtain model performance data, refer to the [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

## Table of Contents

* [Model-References](../../../../../README.md)
* [Model Overview](#model-overview)
* [Setup](#setup)
* [Model Checkpoint](#model-checkpoint)
* [Inference and Examples](#inference-and-examples)
* [Supported Configuration](#supported-configuration)
* [Known Issues](#known-issues)
* [Changelog](#changelog)
* [Enabling the Model from Scratch](#enabling-the-model-from-scratch)
* [GPU Migration Logs](#gpu-migration-logs)

## Model Overview
This implementation is based on [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752).

Enabling model functionality is made easy by GPU migration.
While some performance optimizations are usually still required, GPU migration handles several steps required. 

The following is a list of the different advantages of using GPU migration when compared to [the other model](../../../../generative_models/stable-diffusion/README.md):
* Modifying the default yaml config is not required.
* habana_compat.py helper file is not required.
* Re-checking the device type in multiple places is not required.
* Passing additional 'device' parameters to scripts and objects is not required.
* Adding a custom 'lazy_mode' parameter is not required.

For further details, refer to [Enabling the Model from Scratch](#enabling-the-model-from-scratch).

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

For convenience, export a MODEL_REFERENCES_PATH environment variable:
```bash
export MODEL_REFERENCES_PATH=/path/to/Model-References
```

### Install Model Requirements
1. In the docker container, go to the model directory:
```bash
cd $MODEL_REFERENCES_PATH/PyTorch/examples/gpu_migration/generative_models/stable-diffusion
```

2. Install the required packages using pip.
```bash
pip install -r requirements.txt
```

If you are getting errors, you might want to use the following command before trying to install the requirements again:
```bash
git config --global --add safe.directory `pwd`/src/taming-transformers && git config --global --add safe.directory `pwd`/src/clip
```

### Update PYTHONPATH
```bash
export PYTHONPATH=$MODEL_REFERENCES_PATH/PyTorch/examples/gpu_migration/generative_models/stable-diffusion/src/taming-transformers:$MODEL_REFERENCES_PATH/PyTorch/examples/gpu_migration/generative_models/stable-diffusion:$MODEL_REFERENCES_PATH/PyTorch/examples/gpu_migration/generative_models/stable-diffusion/src/clip:$PYTHONPATH
```

## Model Checkpoint
### Text-to-Image

1. Create a folder for checkpoint with the following command:
```bash
mkdir -p models/ldm/text2img-large/
```

2. Go to https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/, and save
`model.ckpt` file to `models/ldm/text2img-large/` folder to download the pre-trained weights (5.7GB)

Users acknowledge and understand that by downloading the checkpoint referenced herein they will be required to comply with
third party licenses and rights pertaining to the checkpoint, and users will be solely liable and responsible for complying
with any applicable licenses. Habana Labs disclaims any warranty or liability with respect to users' use
or compliance with such third party licenses.

## Inference and Examples
Consider the following command:
```bash
$PYTHON scripts/txt2img.py --prompt "a virus monster is playing guitar, oil on canvas" --ddim_eta 0.0 --n_samples 4 --n_iter 4 --scale 5.0  --ddim_steps 50
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

### Performance
The first two batches of images generate a performance penalty, while all subsequent batches will be generated much faster.
For example, the command from the top of the [Inference and Examples](#inference-and-examples) section generates 4 batches of 4 images.
It will take more time to generate the first two sets of 4 images than the remaining 2.

### Higher Output Resolution
The model has been trained on 256x256 images and usually provides the most semantically meaningful outputs in this native resolution.
However, the output resolution can be controlled with `-H` and `-W` parameters.
For example, this command produces four 512x512 outputs:
```bash
$PYTHON scripts/txt2img.py --prompt "a virus monster is playing guitar, oil on canvas" --ddim_eta 0.0 --n_samples 4 --n_iter 1 --scale 5.0  --ddim_steps 50 --H 512 --W 512
```

### HPU Graph API
In some scenarios, especially when working with lower batch sizes, kernel execution on HPU might be faster than Op accumulation on CPU.
In such cases, the CPU becomes a bottleneck, and the benefits of using Gaudi accelerators are limited.
To combat this, Habana offers an HPU Graph API, which allows for one-time Ops accumulation in a graph structure that is being reused over time.

For further details on running inference with HPU Graphs, refer to [Run Inference Using HPU Graphs](https://docs.habana.ai/en/latest/PyTorch/Inference_on_Gaudi/Inference_using_HPU_Graphs/Inference_using_HPU_Graphs.html) section.

## Supported Configuration
| Device  | SynapseAI Version | PyTorch Version | Mode |
|---------|-------------------|-----------------|-----------|
| Gaudi   | 1.12.0             | 2.0.1          | Inference |
| Gaudi2  | 1.12.0             | 2.0.1          | Inference |

## Known Issues
* Initial random noise generation has been moved to CPU.
Arguably, CPU-generated random noise produces better images.

## Changelog
### 1.10.0
* Removed PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES environment variable.
### 1.9.0
Major changes done to the original model from [pesser/stable-diffusion](https://github.com/pesser/stable-diffusion/commit/a166aa7fbf578f41f855efeab2e14001d6732563) repository:
* Added `import huda` and `htcore.mark_step()` to scripts/txt2img.py
* Changed README.
* Changed default config and ckpt in scripts/txt2img.py.
* Removed pytorch-lightning and torchmetrics from requirements.txt.
* Added HPU Graph support.

## Enabling the Model from Scratch
Habana provides scripts ready-to-use on Gaudi.
Listed below are the steps to enable the model from a reference source.

This section outlines the overall procedure for enabling any given model with GPU migration feature.
However, model-specific modifications will be required to enable the functionality and improve performance.

1. Clone the original GitHub repository and reset it to the commit this example is based on:
```bash
git clone https://github.com/pesser/stable-diffusion.git && cd stable-diffusion && git reset --hard a166aa7fbf578f41f855efeab2e14001d6732563
```
2. Install the requirements:
```bash
pip install -r requirements.txt
```

3. Apply a set of patches.
You can stop at any patch to see which steps have been performed to reach a particular level of functionality and performance.

The first patch adds the bare minimum to run the model on HPU. For purely functional changes (without performance optimization), run the following command:
```bash
git apply Model-References/PyTorch/examples/gpu_migration/generative_models/stable-diffusion/patches/minimal_changes.diff
```

4. To improve performance, apply the patch which adds HPU graph support:
```bash
git apply Model-References/PyTorch/examples/gpu_migration/generative_models/stable-diffusion/patches/hpu_graph.diff
```

5. As described in the [Known Issues](#known-issues) section, it is recommended to move the initial random noise generation to the CPU.
To apply this, run the following command:
```bash
git apply Model-References/PyTorch/examples/gpu_migration/generative_models/stable-diffusion/patches/randn_to_cpu.diff
```

To generate a grid of 4x4 samples with any of these patches, run the following command:
```bash
$PYTHON scripts/txt2img.py --prompt "a virus monster is playing guitar, oil on canvas" --ddim_eta 0.0 --n_samples 16 --n_rows 4 --n_iter 1 --scale 5.0  --ddim_steps 50 --ckpt /path/to/model.ckpt --config configs/latent-diffusion/txt2img-1p4B-eval.yaml
```

*NOTE:* When compared with the command presented in the [Inference and Examples](#inference-and-examples) section, this command has additional `--ckpt` and `--config` parameters.
If you followed the steps outlined in the [Model Checkpoint](#model-checkpoint) section, your `/path/to/model.ckpt` should be `Model-References/PyTorch/examples/gpu_migration/generative_models/stable-diffusion/models/ldm/text2img-large/model.ckpt`.
The default values of both --ckpt and --config parameters have been changed in the model code introduced in this directory for your convenience.
This change, though not essential for running the model, is not incorporated in any of the given patches.

## GPU Migration Logs
You can review GPU Migration logs under [gpu_migration_logs/gpu_migration_1762.log](gpu_migration_logs/gpu_migration_1762.log).
For further information, refer to [GPU Migration Toolkit documentation](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Model_Porting/GPU_Migration_Toolkit/GPU_Migration_Toolkit.html#enabling-logging-feature).