# Inference of BLOOM using PyTorch

This directory provides scripts to run inference on the family of BLOOM models, developed and trained by Huggingface. These scripts are tested and maintained by Habana.

For more information on training and inference of deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

For model performance data, refer to the [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

## Table of Contents

* [Model-References](../../../../README.md)
* [Model Overview](#Model-Overview)
* [Setup](#Setup)
* [Inference and Examples](#Inference-and-Examples)
* [Static Shapes](#Static-Shapes)
* [Single-card inference examples](#Single_Card-Inference-Examples)
* [Multi-card inference examples](#Multi_Card-Inference-Examples)
* [Generation Modes](#Generation-Modes)
* [Supported Configurations](#Supported-Configurations)
* [Changelog](#Changelog)
* [Known Issues](#Known-Issues)

## Model Overview

BLOOM is an autoregressive large language model. This repository is based on [Huggingface's Bigscience BLOOM model](https://bigscience.huggingface.co/blog/bloom)

BLOOM comes in various configurations, varying in the size of its hidden state and number of layers, and consequently the number of parameters.
Habana supports all BLOOM models up to the largest 176B using DeepSpeed for distribution across multiple Gaudi cards.
BLOOM 176B in bfloat16 requires 8 Gaudi2 cards.

## Setup

Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) 
to set up the environment including the `$PYTHON` environment variable. To achieve the best performance, please follow the methods outlined in the [Optimizing Training Platform guide](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html).
The guides will walk you through the process of setting up your system to run the model on Gaudi.  

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

### Install DeepSpeed-fork
DeepSpeed-fork is required to run BLOOM on multiple cards. Install it using pip in the docker container:
```bash
pip install git+https://github.com/HabanaAI/DeepSpeed.git@RELEASE
```
For example:
```bash
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.8.0
```
For more details please refer to DeepSpeed-fork's [documentation](https://docs.habana.ai/en/latest/PyTorch/DeepSpeed/Getting_Started_with_DeepSpeed/Getting_Started_with_DeepSpeed.html).

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

## Static Shapes

Static Shapes support was added to the model in order to minimize graph recompilations and allow using HPU graphs. There are two main sources of dynamicity in Bloom:
* Varying length of input prompts.
* Updating KV-cache.

The first case can be mitigated by padding `input_token_ids` to `max_length` during the first run of the model, when the entire input prompt is presented at once to generate the initial KV-cache. It is essential to adjust the `attention_mask` accordingly, ensuring that the extra padding tokens are masked. Additionally, this also requires changes in the generation loop to accomodate that logits for next token are not always located at the end of the logits tensor.

The second case arises from using `torch.cat` to append the newly generated attention to the previous attention values. Using concatenation by itself can introduce dynamicity as one of the dimensions increases with each generated token. Furthermore, if the initial input prompt was padded, the newly created KV-cache would also be padded. Consequently, appending new attention values to the end of the cache becomes unfeasible. To solve this, it is necessary to keep track of the current token index and employ `torch.index_*` operations for in-place updates instead of concatenation.

### Single-Card Inference Examples

- Run BLOOM7B1 on Gaudi2, using the FP32 data type, with a maximum length of 32 tokens:
```
./bloom.py --weights ./checkpoints --model bloom-7b1 --max_length 32 <Your prompt here>
```
- Run BLOOM3B on first-gen Gaudi, using the FP32 data type, with a maximum length of 32 tokens:
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

### Multi-Card Inference Examples

- Run BLOOM 176B on 8 Gaudi2, using the BF16 data type, with a maximum length of 128 tokens:
```
deepspeed --num_gpus 8 ./bloom.py --weights ./checkpoints --model bloom --max_length 128 --dtype bf16 <Your prompt here>
```

## Generation Modes
The main BLOOM script (bloom.py) can be run in multiple generation modes:
* 'optimized' - Uses a custom HPU-optimized implementation of greedy-search, beam-search and sampling.
* 'vanilla' - Runs both model and generation loop on HPU using unmodified generation utils from the transformers library.

## Supported Configurations

**BLOOM 7B**
| Validated on | SynapseAI Version | PyTorch Version | Mode |
|--------|-------------------|-----------------|----------------|
| Gaudi  | 1.9.0             | 1.13.1          | Inference |
| Gaudi2 | 1.9.0             | 1.13.1          | Inference |

**BLOOM 176B**
| Validated on | SynapseAI Version | PyTorch Version | Mode |
|--------|-------------------|-----------------|----------------|
| Gaudi2  | 1.9.0             | 1.13.1          | Inference |

## Changelog

### 1.10.0
- Removed 'compatibility' mode.
- Moved generation options to a separate flag (`--options`).
- Added support for the following generation options: do_sample, temperature, top_k, top_p, repetition_penalty, length_penalty, no_repeat_ngram_size. For more details, run `--help_options`.
- Rebased the model code to transformers=4.27.3.

### 1.9.0
- Added support for generation modes.
- Added optimized beam-search and greedy-search implementations.

### 1.8.0
Added support for multi-card inference using DeepSpeed.

### 1.7.0
Initial release

### Script Modifications
Major changes done to original model from [bigscience/bloom](https://huggingface.co/bigscience/bloom/tree/main) repository:
* Added HPU support.
* Used Torch GELU in lieu of re-implementing GELU in Python
* Added Habana-specific hardware optimizations:
  * Implemented a static shape model
  * Added HPU graph support
  * Added custom HPU-optimized generation loop

## Known Issues
* Changing certain parameters, such as `--max_length`, `--use_kv_cache` or `--static_shapes`, can alter the shapes used in the model and in turn enable or disable certain numerical optimizations. This may affect the generated output.
* Unspecified order of reductions in DeepSpeed may change the output between executions. To circumvent this `HCL_USE_IN_ORDER_COLLECTIVE_GAUDI2` is set to `1` by default. Running with `HCL_USE_IN_ORDER_COLLECTIVE_GAUDI2=0` allows potential performance improvement at the cost of introducing output non-determinism.
* Current implementation of HPU graphs introduces big overhead in memory usage. Consider running without them in scenarios with large batch sizes.
* `torch.index_copy*` operations on HPU can be slow. A workaround for this has been introduced and it's enabled by default. To disable it run with `WA_INDEX_COPY=0`. Note that this workaround increases memory usage and may lead to OutOfMemory errors n bigger max_length/batch_size scenarios.
* Current implementation of `no_repeat_ngram_size` requires offloading certain operations to CPU. This has a negative impact on performance.
