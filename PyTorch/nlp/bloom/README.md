# Inference of BLOOM using Pytorch

This directory provides scripts to run inference on the family of BLOOM models, developed and trained by Huggingface. These scripts are tested and maintained by Habana.

For more information on training and inference of deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

For model performance data, refer to the [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

## Table of Contents

* [Model-References](../../../../README.md)
* [Model Overview](#model-overview)
* [Inference and Examples](#inference-and-examples)
* [Single-card inference examples](#single-card-inference-examples)
* [Supported Configurations](#supported-configurations)
* [Changelog](#changelog)
* [Known Issues](#known-issues)

## Model Overview

Bloom is an autoregressive large language model. This repository is based on [Huggingface's Bigscience BLOOM model](https://bigscience.huggingface.co/blog/bloom)

BLOOM comes in various configurations, varying in the size of its hidden state and number of layers, and consequently the number of parameters.
We support models that fit the memory capacity of a single HPU, meaning up to BLOOM 7B1 on Gaudi2 and BLOOM 3B on first-gen Gaudi, assuming the FP32 data type.

Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the environment including the `$PYTHON` environment variable.
This guide will walk you through the process of setting up your system to run the model on Gaudi.

### How to use
Use of the pretrained model is subject to compliance with third party licenses, including the “BigScience Large Open-science Open-access Multilingual Language Model” (BLOOM). For guidance on the intended use of the BLOOM model, what will be considered misuse and out-of-scope uses, who are the intended users and additional terms please review and read the instructions in this [link](https://huggingface.co/bigscience/bloom#how-to-use). For the full license terms of BLOOM, please access this [link](https://huggingface.co/spaces/bigscience/license).

Users bear sole liability and responsibility to follow and comply with any third party licenses, and Habana Labs disclaims and will bear no liability with respect to users’ use or compliance with third party licenses.

### Clone Habana Model-References
In the docker container, clone this repository and switch to the branch that matches your SynapseAI version. You can run the
[`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the SynapseAI version.

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
cd Model-References/PyTorch/nlp/bloom
```

### Install Model Requirements
In the docker container, go to the model directory:
```bash
cd /root/Model-References/PyTorch/nlp/bloom
```
Install the required packages using pip:
```bash
$PYTHON -m pip install -r requirements.txt
```

### Model Checkpoint
Before running the model, the checkpoints need to be downloaded to a path by performing the following:
```
cd Model-References/PyTorch/nlp/bloom
mkdir checkpoints
$PYTHON utils/fetch_weights.py --weights ./checkpoints
```
You may also specify a `--model` parameter to the above script to fetch only the checkpoint for the model you wish to run.

## Inference and Examples

Consider the following command:
```
./bloom.py --weights ./checkpoints --max_length 32 "It was the best of times"
```

It will generate a continuation of the supplied prompt, up to a maximal length of 32 tokens (prompt included).
The default configuration would use the FP32 data type, and use a static-shape recipe with HPU graphs to minimize host overhead. It will utilize greedy search without an early stopping condition.
This configuration can be changed. For example, to re-enable the early stopping condition for greedy search, use `--ignore_eos f`.
For a more detailed description of parametrs, please see the help message:
```
./bloom.py -h
```

### Single-Card inference examples

Run BLOOM7B1 on Gaudi 2, using the FP32 data type, with a maximum length of 32 tokens:
```
./bloom.py --weights ./checkpoints --model bloom-7b1 --max_length 32 <Your prompt here>
```
Run BLOOM3B on Gaudi 1, using the FP32 data type, with a maximum length of 32 tokens:
```
./bloom.py --weights ./checkpoints --model bloom-3b --max_length 32 <Your prompt here>
```

Running the vanilla script would have incurred a one-time penalty whenever a new prompt length is encountered.
This could be mitigated in three ways:
* Warm-up before starting inference by running on prompts on all relevant lengths, discarding the results.
* Enable dynamic shape support by setting the environment variable `PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES` to 1.
* Use a static-shape recipe that pads sentences to the maximum length.

The static-shape recipe pads all shapes to the maximal length, as specified by the `--max_length` argument.
This causes redundant computation in the attention layers, which in turn causes a sub-linear slow-down as the maximum length scales and the actual length remains constant.
The model uses this static shape recipe by default, as the marginal increase in device time is offset by considerable improvement on the host-side, yielding overall higher throughput.

In addition, this model uses the ["HPU graph"](https://docs.habana.ai/en/latest/PyTorch/Inference_on_Gaudi/Gaudi_Inference.html#run-inference-using-hpu-graphs) feature by default to miminize the host time spent in the `forward()` call.
If HPU graphs are disabled, there could be noticeable host time spent in interpreting the lines in the `forward()` call, which can result in a latency increase.
To overcome this, the host time and device time can be overlapped by calling `htcore.mark_step()` after invoking BloomAttention and after invoking BloomMLP, or by setting the environment variable `PT_HPU_MAX_COMPOUND_OP_SIZE` to some value, like 100.

## Supported Configurations
| Device | SynapseAI Version | PyTorch Version |
|--------|-------------------|-----------------|
| Gaudi  | 1.7.0             | 1.12.0          |
| Gaudi2 | 1.7.0             | 1.12.0          |

## Changelog
### 1.7.0
Initial release

### Script Modifications
Major changes done to original model from [bigscience/bloom](https://huggingface.co/bigscience/bloom/tree/main) repository:
* Added HPU support.
* Moved constant computation, such as hidden dimension size, to occur once on CPU
* Used Torch GELU in lieu of re-implementing GELU in Python
* Added Habana-specific hardware optimizations:
  * Implemented a static shape model
  * Added HPU graph support
  * Moved beam-search logic to the CPU
  * Removed the early stopping condition from greedy search

## Known Issues

* Dynamic shapes are only supported on first-gen Gaudi.
* Downloaded checkpoint **bigscience/bloom** does not work on first-gen Gaudi and Gaudi2. All the remaining checkpoints work.
