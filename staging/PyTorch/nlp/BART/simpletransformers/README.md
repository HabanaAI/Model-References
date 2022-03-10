# BART for PyTorch

This folder contains scripts to fine-tune BART model on Habana Gaudi<sup>TM</sup> device.
For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents
  * [Model-References/README.md](https://github.com/HabanaAI/Model-References/blob/master/README.md)
  * [Model Overview](#model-overview)
  * [Setup](#setup)
    - [Fine-tuning dataset preparation](#fine-tuning-dataset-preparation)
  * [Training the Model](#training-the-model)
  * [Model Changes](#model-changes)
  * [Known Issues](#known-issues)

## Model Overview

BART, Bidirectional and Auto-Regressive Transformers, is proposed in this paper: [Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://aclanthology.org/2020.acl-main.703/), ACL 2020. It is a denoising autoencoder that maps a corrupted document to the original document it was derived from. BART is implemented as a sequence-to-sequence model with a bidirectional encoder over corrupted text and a left-to-right autoregressive decoder. According to the paper, BART's architecture is related to that used in BERT, with these differences: (1) each layer of the decoder additionally performs cross-attention over the final hidden layer of the encoder; and (2) BERT uses an additional feed-forward network before wordprediction, which BART does not. BART contains roughly 10% more parameters than the equivalently sized BERT model.

### BART Fine-Tuning
- Suited for tasks:
  - Text paraphrasing: the model aims to generate paraphrases of the given input sentence.
  - Text summarization: the model aims to generate a summry of the given input sentence.
- Uses optimizer: FusedAdamW (AdamW: “ADAM with Weight Decay Regularization”).
- Based on model weights trained with pretraining.
- Light-weight: the training takes a few minutes.

The BART demo uses training scripts from simple transformers https://github.com/ThilinaRajapakse/simpletransformers.

## Setup

Please follow the instructions given in the following link for setting up the
environment including the `$PYTHON` environment variable: [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html).
This guide will walk you through the process of setting up your system to run
the model on Gaudi.

In the docker container, clone this repository and switch to the branch that matches your SynapseAI version. (Run the
[`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.) For example, if your SynapseAI version is 1.2.0

```bash
git clone -b 1.2.0 https://github.com/HabanaAI/Model-References
```

- Navigate to the BART model directory:
```bash
cd Model-References/staging/PyTorch/nlp/BART/simpletransformers
```

### Fine-tuning dataset preparation

- Public datasets can be downloaded with this script:
```bash
bash ./examples/seq2seq/paraphrasing/data_download.sh
```

Note: Going forward we assume that the dataset is located in `./data` directory.

## Training the Model

- Install the python packages needed for Fine Tuning:
```bash
cd Model-References/staging/PyTorch/nlp/BART/simpletransformers
pip install -e .
pip install bert_score
```

### Single card Training
i. Fine-tune BART (Eager mode)

- Run BART fine-tuning on the dataset using BF16 mixed precision:
```python
$PYTHON examples/seq2seq/paraphrasing/run_bart.py --use_habana --no_cuda --use_fused_adam --use_fused_clip_norm --max_seq_length 128 --train_batch_size 32 --num_train_epochs 5 --save_best_model --output_dir output --bf16
```

- Run BART fine-tuning on the dataset using FP32 data type:
```python
$PYTHON examples/seq2seq/paraphrasing/run_bart.py --use_habana --no_cuda --use_fused_adam --use_fused_clip_norm --max_seq_length 128 --train_batch_size 32 --num_train_epochs 5 --save_best_model --output_dir output
```

ii. Fine-tune BART (Lazy mode)

- Run BART fine-tuning on the dataset using BF16 mixed precision:
```python
$PYTHON examples/seq2seq/paraphrasing/run_bart.py --use_habana --lazy_mode --no_cuda --use_fused_adam --use_fused_clip_norm --max_seq_length 128 --train_batch_size 32 --num_train_epochs 5 --logging_steps 50 --save_best_model --output_dir output --bf16
```

- Run BART fine-tuning on the dataset using FP32 data type:
```python
$PYTHON examples/seq2seq/paraphrasing/run_bart.py --use_habana --lazy_mode --no_cuda --use_fused_adam --use_fused_clip_norm --max_seq_length 128 --train_batch_size 32 --num_train_epochs 5 --logging_steps 50 --save_best_model --output_dir output
```

### Multicard Training
To run multi-card demo, make sure the host machine has 512 GB of RAM installed. Modify the docker run command to pass 8 Gaudi cards to the docker container. This ensures the docker has access to all the 8 cards required for multi-card training.

- Run the multicard training on 8 cards (1 HLS) for BF16, BS32 Lazy mode:
```python
$PYTHON examples/seq2seq/paraphrasing/run_bart.py --use_habana --lazy_mode --no_cuda --use_fused_adam --use_fused_clip_norm --max_seq_length 128 --train_batch_size 32 --num_train_epochs 5 --logging_steps 50 --save_best_model --output_dir /tmp/multicards --bf16 --distributed
```

- Run the multicard training on 8 cards (1 HLS) for FP32, BS32 Lazy mode:
```python
$PYTHON examples/seq2seq/paraphrasing/run_bart.py --use_habana --lazy_mode --no_cuda --use_fused_adam --use_fused_clip_norm --max_seq_length 128 --train_batch_size 32 --num_train_epochs 5 --logging_steps 50 --save_best_model --output_dir /tmp/multicards --distributed
```

## Model Changes

The following changes have been added to scripts & source:
Modifications to the [simple transformer](https://github.com/ThilinaRajapakse/simpletransformers) source:

1. Added Habana Device support (seq2seq_model.py).
2. Modifications for saving checkpoint: Bring tensors to CPU and save (seq2seq_model.py).
3. Introduced Habana BF16 Mixed precision, adding ops lists for BF16 and FP32 (seq2seq_model.py, ops_bf16_bart.txt, ops_fp32_bart.txt).
4. Change for supporting HMP disable for optimizer.step (seq2seq_model.py).
5. Use fused AdamW optimizer on Habana device (seq2seq_model.py, train.py).
6. Use fused clip norm for grad clipping on Habana device (seq2seq_model.py, train.py).
7. Modified training script to use mpirun for distributed training (train.py, run_bart.py).
8. Gradients are used as views using gradient_as_bucket_view (seq2seq_model.py).
9. Default allreduce bucket size set to 200MB for better performance in distributed training (seq2seq_model.py).
10. Added changes to support Lazy mode with required mark_step (seq2seq_model.py).
11. Only print and save in the master process (seq2seq_model.py).
12. Added prediction (sentence generation) metrics (seq2seq_model.py).
13. Modified training script to use Habana TrainingRunner to launch training (train.py, run_bart.py).
14. Modified training script to use Habana data loader (seq2seq_model.py).
15. Add data_dir as an input arugument for data directory.
16. Added this README.

## Known Issues

1. Placing mark_step() arbitrarily may lead to undefined behaviour. Recommend to keep mark_step() as shown in provided scripts.
2. Sentence generation (prediction) is not enabled in this release. We plan to enable it in next release.
