# Inference of Wav2Vec2 using PyTorch

This directory provides scripts to run inference on Wav2Vec2ForCTC. These scripts are tested and maintained by Habana.

For more information on training and inference of deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

For model performance data, refer to the [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

## Table of Contents
   * [Model-References](../../../../README.md)
   * [Model Overview](#model-overview)
   * [Setup](#setup)
   * [Inference Examples](#inference-examples)
   * [Supported Configurations](#supported-configurations)
   * [Changelog](#changelog)

## Model Overview

This Wav2Vec2 model comes with a language modeling head on top for Connectionist Temporal Classification (CTC).

This model is based on [PreTrainedModel](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). For details on the generic methods the library implements for all its models (such as downloading or saving etc.), refer to [Wav2Vec2 for CTC](https://huggingface.co/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC) section. 

## Setup
Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) 
to set up the environment including the `$PYTHON` environment variable. To achieve the best performance, please follow the methods outlined in the [Optimizing Training Platform guide](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html).
The guides will walk you through the process of setting up your system to run the model on Gaudi.  

### Clone Habana Model-References
In the docker container, clone this repository and switch to the branch that matches your SynapseAI version. 
You can run the [`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the SynapseAI version.

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```

Note: If the repository is not in the PYTHONPATH, make sure to update by running the below.
```
export PYTHONPATH=/path/to/Model-References:$PYTHONPATH
```

### Install Model Requirements
In the docker container, go to the directory:
```bash
cd /root/Model-References/PyTorch/audio/wav2vec2/inference
```
Install the required packages using pip:
```bash
$PYTHON -m pip install -r requirements.txt
```

## Inference Examples

### Single-Card Inference Examples

- Run inference on 1 HPU, mixed precision BF16, test-clean dataset(2620 samples), base model:
```
$PYTHON wav2vec.py --dtype bf16 --buckets 5 --use_graphs --perf -a
```
- Run inference on 1 HPU, mixed precision BF16, test-clean dataset(2620 samples), large model:
```
$PYTHON wav2vec.py --dtype bf16 --buckets 5 --use_graphs --perf -a --large
```
- Run inference on 1 HPU, mixed precision BF16, dev-clean dataset(73 samples), base model:
```
$PYTHON wav2vec.py --dtype bf16 --buckets 5 --use_graphs --perf -a --dev_clean_ds --repeat 25
```
- Run inference on 1 HPU, precision FP32, test-clean dataset(2620 samples), base model:
```
$PYTHON wav2vec.py --dtype fp32 --buckets 5 --use_graphs --perf -a
```
- Run inference on 1 HPU, precision FP32, test-clean dataset(2620 samples), large model:
```
$PYTHON wav2vec.py --dtype fp32 --buckets 5 --use_graphs --perf -a --large
```
- Run inference on 1 HPU, precision FP32, dev-clean dataset(73 samples), base model:
```
$PYTHON wav2vec.py --dtype fp32 --buckets 5 --use_graphs --perf -a --dev_clean_ds --repeat 25
```

This model uses ["HPU Graphs"](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html) feature by default to minimize the host time spent in the `forward()` call.
If HPU graphs are disabled, there could be noticeable host time spent in interpreting the lines in
the `forward()` call, which can result in a latency increase.

## Supported Configurations
| Validated on | SynapseAI Version | PyTorch Version | Mode |
|--------|-------------------|-----------------|----------------|
| Gaudi  | 1.10.0             | 2.0.1          | Inference |
| Gaudi2 | 1.10.0             | 2.0.1          | Inference |

## Changelog
### 1.9.0
Peformance improvements.
### 1.8.0
Initial release.

### Script Modifications
The following lists the modifications applied to the script from [huggingface/wav2vec](https://huggingface.co/docs/transformers/main/model_doc/wav2vec2).

* Added support for Habana devices:

   - Added dtype support.
   - Added perf measurement flag.
   - Added "large" model flavor.
   - Added -a flag for measuring accuracy (WER).
   - Added test-clean dataset support with 2620 samples.

* To improve performance:

   - Added bucketing support.
   - Added HPU graph support.
   - Enabled async D2H copy using HPU streams.
   - Enabled async HPU Graph execution (HPU Graph is launched on a separate thread to free up main execution thread for CPU processing).

### Recommendations
For users who intend to modify this script, run new models or use new datasets, other than those used in this reference script, the following is recommended:
   - Periodically synchronize all active threads (e.g., every 2620 samples done in reference script). This allows freeing up of resources (e.g. Host pinned memory) and avoids failure due to resource exhaustion. This synchronization duration can be empirically determined for a given model & dataset.   
   - Ensure the number of streams created do not exceed 3000 (2620 streams created in reference script). Reuse streams if a number larger than this is required. 
