# Inference of BLOOM using PyTorch

This directory provides scripts to run inference on the family of BLOOM models, developed and trained by Huggingface. These scripts are tested and maintained by Habana.

For more information on training and inference of deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

For model performance data, refer to the [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

## Table of Contents

* [Model-References](../../../README.md)
* [Model Overview](#Model-Overview)
* [Setup](#Setup)
* [Static Shapes](#Static-Shapes)
* [Memory Optimizations](#Memory-Optimizations)
* [Inference and Examples](#Inference-and-Examples)
* [Single-card inference examples](#Single_Card-Inference-Examples)
* [Multi-card inference examples](#Multi_Card-Inference-Examples)
* [Generation Modes](#Generation-Modes)
* [Language Model Evaluation Harness](#Language-Model-Evaluation-Harness)
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

## Static Shapes

Static Shapes support was added to the model in order to minimize graph recompilations and allow using HPU graphs. There are two main sources of dynamicity in Bloom:
* Varying length of input prompts.
* Updating KV-cache.

The first case can be mitigated by padding `input_token_ids` to `max_length` during the first run of the model, when the entire input prompt is presented at once to generate the initial KV-cache. It is essential to adjust the `attention_mask` accordingly, ensuring that the extra padding tokens are masked. Additionally, this also requires changes in the generation loop to accomodate that logits for next token are not always located at the end of the logits tensor.

The second case arises from using `torch.cat` to append the newly generated attention to the previous attention values. Using concatenation by itself can introduce dynamicity as one of the dimensions increases with each generated token. Furthermore, if the initial input prompt was padded, the newly created KV-cache would also be padded. Consequently, appending new attention values to the end of the cache becomes unfeasible. To solve this, it is necessary to keep track of the current token index and employ `torch.index_*` operations for in-place updates instead of concatenation.

### Input and Output Length

There are several ways to specify input and output lengths when using static_shapes:
* add `max_length=N` to generation options to specify maximum combined length (in tokens) of input and output.
* add `max_new_tokens=N` to generation options to specify maximum number of generated tokens. `max_new_tokens` takes precedence over `max_length` if both are provided.
* add `max_input_tokens=N` to generation options to specify maximum potential input length (in tokens). Value of this parameter is automatically calculated based on all input prompts, but it can be overriden by the user.

## Memory Optimizations

The following optimizations were introduced to reduce the model memory footprint:
* Logits trimming - when calculating the lm_head output, the model converts [bs, seq_len, hidden_dim] tensor to [bs, seq_len, vocab_size]. Only the logits corresponding to N+1 tokens are needed for output generation, and the rest can be discarded. Trimming the tensor before lm_head calculation saves computation time and memory.
* Reusing kv-cache - HPU graphs require static references to model inputs. To ensure that, the 'wrap_in_hpu_graph' utility uses a cache that handles storing and updating the data before calling `graph.replay()`. It affects all tensors that go through the model's `forward`. Tensors in kv-cache were initially passed through the model.forward(), resulting in an additional copy in the HPU graph cache. This led to significant memory overhead. Using an alternative approach, the kv-cache is preallocated and stored in the model.
* HPU graph bypass - due to the large intermediate tensor sizes, the first output generation step uses a lot of memory, which is then applied by the HPU graph overhead. Disabling HPU graphs for the first token saves memory at low performance cost.
* Splitting lm_head - by default, DeepSpeed does not shard the lm_head. Sharding and splitting it across cards saves a significant amount of memory.
* Specifying max_input_length - knowing the maximum possible input length enables an improved padding strategy. Inputs are padded only to max_input_length, followed by padding the subsequent tokens to the maximum length. This saves compute time and memory requirements, as the first step is usually the most memory-consuming.

## Inference and Examples

Consider the following command:
```
./bloom.py --weights ./checkpoints --options "max_length=32" "It was the best of times"
```

It will generate a continuation of the supplied prompt, up to a maximal length of 32 tokens (prompt included).
The default configuration uses the FP32 data type, and a static-shape recipe with HPU graphs to minimize host overhead. It will utilize greedy search without an early stopping condition.
This configuration can be changed. For example, to re-enable the early stopping condition for greedy search, use `--ignore_eos f`.
For a more detailed description of parametrs, please see the help message:
```
./bloom.py -h
```

### Single-Card Inference Examples

- Run BLOOM7B1 on Gaudi2, using the FP32 data type, with a maximum length of 32 tokens:
```
./bloom.py --weights ./checkpoints --model bloom-7b1 --options "max_length=32" <Your prompt here>
```
- Run BLOOM3B on first-gen Gaudi, using the FP32 data type, with a maximum length of 32 tokens:
```
./bloom.py --weights ./checkpoints --model bloom-3b --options "max_length=32" <Your prompt here>
```

Running the vanilla script would have incurred a one-time penalty whenever a new prompt length is encountered.
This could be mitigated in three ways:
* Warm-up before starting inference by running on prompts on all relevant lengths, discarding the results.
* Enable dynamic shape support by setting the environment variable `PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES` to 1.
* Use a static-shape recipe that pads sentences to the maximum length.

The static-shape recipe pads all shapes to the maximal length, as specified by the `max_length` or `max_new_tokens` generation options.
This causes redundant computation in the attention layers, which in turn causes a sub-linear slow-down as the maximum length scales and the actual length remains constant.
The model uses this static shape recipe by default, as the marginal increase in device time is offset by considerable improvement on the host-side, yielding overall higher throughput.
In addition, this model uses the ["HPU graph"](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html) feature by default to miminize the host time spent in the `forward()` call.
If HPU graphs are disabled, there could be noticeable host time spent in interpreting the lines in the `forward()` call, which can result in a latency increase.
To overcome this, the host time and device time can be overlapped by calling `htcore.mark_step()` after invoking BloomAttention and after invoking BloomMLP, or by setting the environment variable `PT_HPU_MAX_COMPOUND_OP_SIZE` to some value, like 100.

### FP8 Inference Support
- Run BLOOM7B1 in FP8 on Gaudi2 single card:

```
ENABLE_EXPERIMENTAL_FLAGS=true \
USE_DEFAULT_QUANT_PARAM=true \
UPDATE_GRAPH_OUTPUT_MME=false \
ENABLE_CALC_DYNAMIC_RANGE=false \
python3 bloom.py --weights ./checkpoints --batch_size 1 --model bloom-7b1 --dtype bf16 --options "max_length=32" -qf quantization/configuration/examples/quant_config.json <Your prompt here>
```
The -qf flag specifies the path to the quantization config file. The config file includes a single attribute for enabling / disabling quantization.

For more details on the above environment flags and the supported quantization see: [Run Inference Using FP8](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_FP8.html)

### Multi-Card Inference Examples

- Run BLOOM 176B on 8 Gaudi2, using the BF16 data type, with a maximum length of 128 tokens:
```
deepspeed --num_gpus 8 ./bloom.py --weights ./checkpoints --model bloom --options "max_length=128" --dtype bf16 <Your prompt here>
```

## Generation Modes
The main BLOOM script (bloom.py) can be run in multiple generation modes:
* 'optimized' (default) - Uses a custom HPU-optimized implementation of greedy-search, beam-search and sampling.
* 'vanilla' - Runs both model and generation loop on HPU using unmodified generation utils from the transformers library.

To run with optimized greedy-search:
```
deepspeed --num_gpus 8 ./bloom.py --weights ./checkpoints --model bloom --dtype bf16 --options "max_length=128,num_beams=1" <Your prompt here>
```
To run with beam-search:
```
deepspeed --num_gpus 8 ./bloom.py --weights ./checkpoints --model bloom --dtype bf16 --options "max_length=128,num_beams=8" <Your prompt here>
```
To run with sampling:
```
deepspeed --num_gpus 8 ./bloom.py --weights ./checkpoints --model bloom --dtype bf16 --options 'max_length=128,num_beams=1,do_sample=True,top_p=0.2,top_k=3,repetition_penalty=2.5,no_repeat_ngram_size=2' <Your prompt here>
```

### Supported Generation Options:
For a list of all supported generation options and their descriptions run:
```
./habana_generation_utils.py
```


## Language Model Evaluation Harness
The evaluation of BLOOM model can be done using bloom_eval.py script. It utilizes the [LM evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness)
 framework and provides the possibility to run one of four tasks: HellaSwag, LAMBADA, PiQA, WinoGrande.

 By default, it evalutes BLOOM 7B1 on all tasks using FP32 data type.
 ```
./bloom_eval.py --weights ./checkpoints --output_file eval.out
```

For a more detailed description of parameters, please see the help message:
```
./bloom_eval.py -h
```
### Examples
Evaluate BLOOM 7B1 on Gaudi on task PiQA, using the BF16 data type.
```
./bloom_eval.py --weights ./checkpoints --dtype bf16 --task piqa -o eval.json
```

Evaluate BLOOM 176B on 8 Gaudi2 on task WinoGrande, using the BF16 data type.
```
deepspeed --num_gpus 8 ./bloom_eval.py --weights ./checkpoints --model bloom --dtype bf16 --task winogrande -o eval.json
```

## Supported Configurations

**BLOOM 7B and BLOOM 176B**
| Validated on | SynapseAI Version | PyTorch Version | Mode |
|--------|-------------------|-----------------|----------------|
| Gaudi  | 1.12.1             | 2.0.1          | Inference |
| Gaudi2 | 1.12.1             | 2.0.1          | Inference |

## Changelog
### 1.11.0
- Removed index_copy_ workaround.
- Added max_new_tokens, min_new_tokens, max_input_tokens generation options.
- Enabled memory optimizations.

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
* Current implementation of `no_repeat_ngram_size` requires offloading certain operations to CPU. This has a negative impact on performance.
