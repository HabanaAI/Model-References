# Bloom for PyTorch

This directory provide scripts to train the GPT based model called BLOOM 13B in the Megatron-DeepSpeed repository.

## Table of Contents
* [Model-References](../../../../README.md)
* [Model Overview](#model-overview)
* [Setup](#setup)
* [Training and Examples](#training-and-examples)
* [Supported Configuration](#supported-configuration)
* [Changelog](#changelog)
* [Known Issues](#known-issues)

## Model Overview
This implementation is based on https://github.com/microsoft/Megatron-DeepSpeed at 0c58dbb.
Megatron ([1](https://arxiv.org/pdf/1909.08053.pdf) and [2](https://arxiv.org/pdf/2104.04473.pdf)) is a large, powerful transformer developed by the Applied Deep Learning Research team at NVIDIA. This repository is for training large transformer language models such as Bloom at scale. Codebase is capable of efficiently training very large (hundreds of billions of parameters) language models with both model and data parallelism.

### How to use
Users bear sole liability and responsibility to follow and comply with any third party licenses, and Habana Labs disclaims and will bear no liability with respect to users’ use or compliance with third party licenses.


## Setup
Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) 
to set up the environment including the `$PYTHON` environment variable. To achieve the best performance, please follow the methods outlined in the [Optimizing Training Platform guide](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html).
The guides will walk you through the process of setting up your system to run the model on Gaudi2.  

### Install Habana DeepSpeed-fork
Please follow the instructions provided in the [DeepSpeed Installation Guide](https://docs.habana.ai/en/latest/PyTorch/DeepSpeed/Getting_Started_with_DeepSpeed/Getting_Started_with_DeepSpeed.html) to install deepspeed-fork.

### Clone Habana Model-References
In the docker container, clone this repository and switch to the branch that matches your SynapseAI version.
You can run the [`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the SynapseAI version.
```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```

```
export MODEL_REFERENCES_ROOT=/path/to/Model-References
export PYTHONPATH=/path/to/Model-References/PyTorch/common:$PYTHONPATH

```
### Install Model Requirements
* In the docker container, go to the model directory:
  ```bash
  cd Model-References/PyTorch/nlp/DeepSpeedExamples/Megatron-DeepSpeed/
  ```

* Install the required packages using pip:
  ```bash
  pip install -r requirements.txt
  ```

### Dataset Preparation
Follow the instructions in https://github.com/bigscience-workshop/bigscience/tree/master/data/oscar to download oscar-en full dataset. Note that the dataset takes around 550G of disk space.


## Training and Examples
Training of Bloom13B model is based on https://github.com/bigscience-workshop/bigscience/blob/master/train/tr1-13B-base/tr1-13B-round1.slurm

### Multi-Card Training Examples
* Update data root dir with the path of your choice:
  ```
  HL_DATA_DIR_ROOT=/data/bigscience/oscar-en
  ```

* Run 32 HPUs with BF16 precision: (Note: Make sure to change the IP addresses in hostsfile according to your setup)
  ```
  HL_HOSTSFILE=scripts/hostsfile HL_NUM_NODES=4 HL_PP=2 HL_TP=4 HL_DP=4 scripts/run_bloom13b.sh
  ```

* Run 64 HPUs with BF16 precision: (Note: Make sure to change the IP addresses in hostsfile according to your setup)
  ```
  HL_HOSTSFILE=scripts/hostsfile HL_NUM_NODES=8 HL_PP=2 HL_TP=4 HL_DP=8 scripts/run_bloom13b.sh
  ```


## Supported Configuration
| Validated on  | SynapseAI Version | PyTorch Version | Mode |
|---------|-------------------|-----------------|-------------|
| Gaudi2  | 1.10.0           | 2.0.1          | Training |


## Changelog
### 1.10.0
Updated the recommended 3D-parallelism configuration.
### 1.8.0
Initial release.

### Script Modifications
Major changes done to the original model from [microsoft/Megatron-DeepSpeed]( https://github.com/microsoft/Megatron-DeepSpeed/commit/0c58dbb3ad126ad0a58c7bd30944eee48b9249d0) repository:
* Changed README file content.
* Replaced CUDA specific API calls with generic ones.
* Switched GPT default optimizer from Adam to AdamW.
* Added support for universal checkpoint based on universal checkpoint support in Bloom.
* Added kill-switch mechanism to gracefully stop training based on support in Bloom.
* Added HPU memory logging.

## Known Issues
* Only scripts and configurations mentioned in this README are supported and verified.
