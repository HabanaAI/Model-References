# T5-base for TensorFlow

This directory provides a script to train T5-base model to achieve state-of-the-art accuracy, and is tested and maintained by Habana. For further information on performance, refer to [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

For more information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of contents

* [Model-References](../../../README.md)
* [Model Overview](#model-overview)
* [Setup](#setup)
* [Fine-tuning the Model](#fine-tuning-the-model)
* [Supported Configuration](#supported-configuration)
* [Changelog](#changelog)

## Model Overview

T5, or Text-to-Text Transfer Transformer, is a Transformer based architecture that uses a text-to-text approach.
Each task – including translation, question answering, and classification – is cast as feeding the model text
as input and training it to generate some target text. This allows using the same model,
loss function, hyperparameters, etc. across our diverse set of tasks. The changes compared to BERT include:

* Adding a causal decoder to the bidirectional architecture.
* Replacing the fill-in-the-blank cloze task with a mix of alternative pre-training tasks.

The T5-base model is adapted from [SnapThat's Github repo](https://github.com/snapthat/TF-T5-text-to-text/blob/master/snapthatT5/notebooks/TF-T5-%20Training.ipynb), and based on [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683).

### Model Changes

The following are the major changes that were implemented to the original model:

* Reorganized the script into multiple files.
* Enabled running the model offline and without any additional download after the setup.
* Added `bfloat16` support, and `compute_loss` is forced to work on FP32.
* Added additional data-gathering callbacks.
* Added `inference.py`

## Setup

Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) to set up the
environment including the `$PYTHON` environment variable. The guide will walk you through the process of setting up your system to run the model on Gaudi.

### Clone Habana Model-References

In the docker container, clone this repository and switch to the branch that matches your SynapseAI version. You can run the [`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the SynapseAI version.

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References /root/Model-References
```

**Note:** If Model-References repository path is not in the PYTHONPATH, make sure you update it:
```bash
export PYTHONPATH=$PYTHONPATH:/root/Model-References
```

### Install Model Requirements

1. In the docker container, go to the T5-base directory:

```bash
cd /root/Model-References/TensorFlow/nlp/T5-base
```

2. Install the required packages using pip:

```bash
$PYTHON -m pip install -r requirements.txt
```

### Download the Dataset

The topology uses SQuAD dataset and pre-trained weights provided by Huggingface. To download and pre-process the dataset, run the following command:

**NOTE:** This will download the pre-trained model as well.

```bash
cd /root/Model-References/TensorFlow/nlp/T5-base
$PYTHON prepare_data.py /data/huggingface
```

After downloading, `/data/huggingface` should contain `squad` and `t5_base` subdirectories. At this point, you are ready to fine-tune the model.

## Fine-tuning the Model

To fine-tune T5-base model on SQuAD dataset with BF16 precision and other default hyper-parameters, run:

```bash
cd /root/Model-References/TensorFlow/nlp/T5-base
$PYTHON train.py --dtype bf16 --data_dir /data/huggingface --model_dir ./model
```

**NOTE:** For further information on the possible arguments, run: `$PYTHON train.py --help`.

### Inference

Once the model is fine-tuned, it can be used for inference. To explore how the model behaves, run the following command:

```
$ $PYTHON inference.py --data_dir /data/huggingface --model_dir ./model
Provide context and ask model a question, for example:
Context: In 2019 Habana Labs announced its first AI accelerator. Gaudi, named after famous Catalan architect, was designed to accelerate training of deep neural networks in data centers.
Question: What is the name of the chip?
Answer:  <pad> Gaudi</s>
...
```

## Supported Configuration

| Device | SynapseAI Version | TensorFlow Version(s)  |
|:------:|:-----------------:|:-----:|
| Gaudi  | 1.6.1             | 2.9.1 |
| Gaudi  | 1.6.1             | 2.8.2 |

# Changelog

### 1.3.0

* Updated the transformers package to solve TF2.8 compatibility issues.

### 1.2.0

* Updated requirements.txt to improve performance.
