# BERT-L Inference for PyTorch

This directory provides scripts to run inference on a BERT-Large model. These scripts are tested and maintained by Habana.

For more information on training and inference of deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

For model performance data, refer to the [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

## Table of Contents

* [Model-References](../../../../../../../../../README.md)
* [Model Overview](#model-overview)
* [Inference and Examples](#inference-and-examples)
* [Single-card inference examples](#single-card-inference-examples)
* [Supported Configurations](#supported-configurations)
* [Changelog](#changelog)
* [Known Issues](#known-issues)

## Model Overview

Bidirectional Encoder Representations from Transformers (BERT) is a technique for natural language processing (NLP) pre-training developed by Google.
The is Huggingface's implementation of the BERT Large model, a 24-layer, 1024-hidden, 16-heads, 340M parameter neural network architecture; It was trained on the BooksCorpus with 800M words, and a version of the English Wikipedia with 2,500M words.

### Setup
Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the environment including the `$PYTHON` environment variable.
This guide will walk you through the process of setting up your system to run the model on Gaudi.

### Clone Habana Model-References
In the docker container, clone this repository and switch to the branch that matches your SynapseAI version. You can run the
[`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the SynapseAI version.

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
cd Model-References/PyTorch/nlp/finetuning/huggingface/bert/transformers/examples/pytorch/bert-l-inference
```

### Install Model Requirements
In the docker container, go to the model directory:
```bash
cd /root/Model-References/PyTorch/nlp/finetuning/huggingface/bert/transformers/examples/pytorch/bert-l-inference
```
Install the required packages using pip:
```bash
$PYTHON -m pip install -r requirements.txt
```

## Inference and Examples

Consider the following command:
```
./bert-l-inference.py "Good morning BERT!"
```

The BERT model will be invoked on this input and the script will print its output, which is the final set of hidden states as well as some auxiliary data.
Note that this is merely a usage example. A typical use case would involve fine-tuning the model on some task. This script was deliberately left task-less to highlight common techniques for improving inference throughput, without burdening the reader with task-specific code, such as decoding the output. 

The default configuration uses FP32 data type, runs with a batch size of 1, and performs a single iteration on the input "Hello world!".
It will pad all inputs to the length of the maximal input to avoid dynamic shapes or ragged tensors.
For a more detailed description of the available parameters, please see the help message:
```
./bert-l-inference.py -h
```

### Single-Card Inference Examples

- Run using BF16 data type, repeating the input 100 times and performing performance measurements:
```
./bert-l-inference.py --dtype bf16 -i 100 --perf <Your prompt here>
```
- Run using FP32 data type, using HPU graphs and a batch size of 12, repeating the input 100 times and performing performance measurements:
```
./bert-l-inference.py --use_graphs -bs 12 -i 100 --perf <Your prompt here>
```

This model uses the ["HPU graph"](https://docs.habana.ai/en/latest/PyTorch/Inference_on_Gaudi/Inference_using_HPU_Graphs/Inference_using_HPU_Graphs.html) feature by default to miminize the host time spent in the `forward()` call.
If HPU graphs are disabled, there could be noticeable host time spent in interpreting the lines in the `forward()` call, which can result in a latency increase.

## Supported Configurations
| Device | SynapseAI Version | PyTorch Version |
|--------|-------------------|-----------------|
| Gaudi  | 1.7.1             | 1.13.0          |
| Gaudi2 | 1.7.1             | 1.13.0          |

## Changelog
### 1.7.1
Initial release

## Known Issues

* Autocast to BF16 was tested on transformers 4.20.1 only. Newer versions of the library may cause accuracy degradation. 
