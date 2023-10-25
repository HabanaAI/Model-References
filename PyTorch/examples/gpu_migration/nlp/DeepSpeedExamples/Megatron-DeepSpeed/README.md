# Megatron-DeepSpeed BLOOM for GPU Migration Toolkit
This directory provides scripts for training large transformer language models such as Bloom at scale and is tested and maintained by Habana.

The model has been enabled using an experimental feature called GPU Migration Toolkit. For more details, refer to [GPU Migration Toolkit documentation](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Model_Porting/GPU_Migration_Toolkit/GPU_Migration_Toolkit.html). NOTE: You can review the [BLOOM model](https://github.com/HabanaAI/Model-References/tree/master/PyTorch/nlp/DeepSpeedExamples/Megatron-DeepSpeed) enabled with a more traditional approach.

For more information on training and inference of deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/). To obtain model performance data, refer to the [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-models-performance/#performance).

## Table of Contents
   * [Model References](https://github.com/HabanaAI/Model-References/blob/master/README.md)
   * [Model Overview](#model-overview)
   * [Setup](#setup)
   * [Training and Examples](#training-and-examples)
   * [Enabling the Model from scratch](#enabling-the-model-from-scratch)
   * [GPU Migration Toolkit Logs](#gpu-migration-logs)
   * [Supported Configuration](#supported-configuration)

## Model Overview
This implementation is based on https://github.com/microsoft/Megatron-DeepSpeed at 0c58dbb. Megatron ([1](https://arxiv.org/pdf/1909.08053.pdf) and [2](https://arxiv.org/pdf/2104.04473.pdf) is a large, powerful transformer developed by the Applied Deep Learning Research team at NVIDIA. Codebase is capable of efficiently training very large (hundreds of billions of parameters) language models with both model and data parallelism.

Enabling model functionality is made easy by GPU Migration Toolkit. While some performance optimizations are usually still required, GPU Migration Toolkit handles several steps required.

The following is a list of the different advantages of using GPU Migration Toolkit:
- Re-checking the device type in multiple places is not required.
- Passing additional ‘device’ parameters to scripts and objects is not required.

For further details, refer to [Enabling the Model from Scratch](#enabling-the-model-from-scratch).

### How to use
Users acknowledge and understand that the models referenced by Habana are mere examples for models that can be run on Gaudi. Users bear sole liability and responsibility to follow and comply with any third party licenses pertaining to such models, and Habana Labs disclaims and will bear no any warranty or liability with respect to users' use or compliance with such third party licenses.

## Setup
Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the environment including the $PYTHON environment variable.
To achieve the best performance, please follow the methods outlined in the Optimizing Training Platform guide. The guides will walk you through the process of setting up your system to run the model on Gaudi2.

### Clone Habana Model-References
In the docker container, clone this repository and switch to the branch that matches your SynapseAI version.
You can run the [`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the SynapseAI version.

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```

For convenience, export a MODEL_REFERENCES_PATH & PYTHONPATH environment variable:
```bash
export MODEL_REFERENCES_ROOT=/path/to/Model-References
```

### Install Model Requirements
- In the docker container, go to the model directory:
```bash
cd Model-References/PyTorch/examples/gpu_migration/nlp/DeepSpeedExamples/Megatron-DeepSpeed
```

- Install the required packages using pip:
```bash
pip install -r requirements.txt
```
### Install Habana DeepSpeed-fork
Please follow the instructions provided in the [DeepSpeed Installation Guide](https://docs.habana.ai/en/latest/PyTorch/DeepSpeed/Getting_Started_with_DeepSpeed/Getting_Started_with_DeepSpeed.html) to install deepspeed-fork.

### Install Apex
Please follow the instructions provided [here](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Model_Porting/GPU_Migration_Toolkit/GPU_Migration_Toolkit.html#limitations) to install Apex.

### Dataset Preparation
Follow the instructions in https://github.com/bigscience-workshop/bigscience/tree/master/data/oscar to download oscar-en full dataset. Note that the dataset takes around 550G of disk space.

## Training and Examples
Bloom13B model training is based on https://github.com/bigscience-workshop/bigscience/blob/master/train/tr1-13B-base/tr1-13B-round1.slurm.

### Multi-Card Training Examples
- Update data root directory with a path of your choice:
```bash
export HL_DATA_DIR_ROOT=/data/bigscience/oscar-en
```
- Run BLOOM on 8 HPUs with BF16 precision. Make sure to change the IP addresses in hostsfile according to your setup.
```bash
HL_HOSTSFILE=scripts/hostsfile HL_NUM_NODES=1 HL_PP=2 HL_TP=4 HL_DP=1 scripts/run_bloom13b.sh
```
## Enabling the Model from scratch
Habana provides scripts ready-to-use on Gaudi. Listed below are the steps to enable the model from a reference source.
This section outlines the overall procedure for enabling any given model with GPU Migration Toolkit feature. However, model-specific modifications will be required to enable the functionality and improve performance.

1. Clone the original GitHub repository and reset it to the commit this example is based on.
```bash
git clone https://github.com/microsoft/Megatron-DeepSpeed.git && git checkout 0c58dbb
```

2. Navigate to Megatron-Deepspeed subfolder and install requirements:
```bash
pip install -r requirements.txt
```

3. Apply a set of patches. You can stop at any patch to see which steps have been performed to reach a particular level of functionality and performance.
The first patch adds the bare minimum to run the model on HPU. For purely functional changes (without performance optimization), run the following command:
```bash
git apply Model-References/PyTorch/examples/gpu_migration/nlp/DeepSpeedExamples/Megatron-DeepSpeed/patches/functional_changes.diff
```
   First patch adds:
   - GPU Migration Toolkit package import in main script (pretrain_gpt.py).
   - Since HPU does not support CUDA kernels, there is no requirement to compile the kernels associated with CUDA (`megatron/initialize.py`).
   - Remove call to ds_report() which uses 3rd party calls to nvcc (`pretrain_gpt.py`).
   - HPU does not support fused_layer_norm_cuda (as explained above), therefore LayerNorm from Apex is used instead (It is eventually overwritten to torch.optim.LayerNorm by GPU Migration Toolkit) (`megatron/model/__init__.py`).
   - HPU supports BF16 data type (For this particular topology, mixed precision support directly comes from [Habana's DeepSpeed](https://docs.habana.ai/en/latest/PyTorch/DeepSpeed/DeepSpeed_User_Guide/DeepSpeed_User_Guide.html)). BF16 offers FP32-like dynamic range and loss scaling is not required in [BF16 mixed precision training](https://arxiv.org/pdf/1905.12322.pdf). Hence, cur_scale attribute is not available for BF16 Optimizer (`megatron/training.py`).
   - A script for running the Bloom model. Based on https://github.com/bigscience-workshop/bigscience/blob/master/train/tr1-13B-base/tr1-13B-round1.slurm (`scripts/run_bloom13b.sh`).

4. To improve the performance, apply the patch (which sets skip_bias_add argument to False for mpu.ColumnParallelLinear & mpu.RowParallelLinear)
```bash
git apply Model-References/PyTorch/examples/gpu_migration/nlp/DeepSpeedExamples/Megatron-DeepSpeed/patches/performance_patch_1.diff
```
## GPU Migration Toolkit Logs
You can review GPU Migration Toolkit logs under `gpu_migration_logs/gpu_migration_424488.log`. For further information, refer to [GPU Migration Toolkit documentation](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Model_Porting/GPU_Migration_Toolkit/GPU_Migration_Toolkit.html#enabling-logging-feature).

## Supported Configuration
| Device  | SynapseAI Version | PyTorch Version | Mode      |
|---------|-------------------|-----------------|-----------|
| Gaudi2  | 1.12.1            |  2.0.1          | Training  |