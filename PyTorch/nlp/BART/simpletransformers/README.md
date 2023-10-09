# BART for PyTorch

This folder contains scripts to fine-tune BART model on Habana Gaudi device. To obtain model performance data, refer to the [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

For more information about training deep learning models using Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents
  * [Model-References](../../../../README.md)
  * [Model Overview](#model-overview)
  * [Setup](#setup)
  * [Training Examples ](#training-examples)
  * [Supported Configurations](#supported-configurations)
  * [Changelog](#changelog)
  * [Known Issues](#known-issues)

## Model Overview

BART, Bidirectional and Auto-Regressive Transformers, is proposed in this paper: [Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://aclanthology.org/2020.acl-main.703/), ACL 2020. It is a denoising autoencoder that maps a corrupted document to the original document it was derived from. BART is implemented as a sequence-to-sequence model with a bidirectional encoder over corrupted text and a left-to-right autoregressive decoder. According to the paper, BART's architecture is related to that used in BERT, with these differences: (1) each layer of the decoder additionally performs cross-attention over the final hidden layer of the encoder; and (2) BERT uses an additional feed-forward network before wordprediction, which BART does not. BART contains roughly 10% more parameters than the equivalently sized BERT model.

### BART Fine-Tuning
- Suited for tasks:
  - Text paraphrasing: The model aims to generate paraphrases of the given input sentence.
  - Text summarization: The model aims to generate a summary of the given input sentence.
- Uses optimizer: FusedAdamW (AdamW: “ADAM with Weight Decay Regularization”).
- Based on model weights trained with pre-training.
- Light-weight: The training takes a few minutes.

The BART demo uses training scripts from simple transformers https://github.com/ThilinaRajapakse/simpletransformers.

## Setup
Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) 
to set up the environment including the `$PYTHON` environment variable. To achieve the best performance, please follow the methods outlined in the [Optimizing Training Platform guide](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html).
The guides will walk you through the process of setting up your system to run the model on Gaudi.  

### Clone Habana Model-References
In the docker container, clone this repository and switch to the branch that
matches your SynapseAI version. You can run the
[`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.
```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```

Then, navigate to the BART model directory:
```bash
cd Model-References/PyTorch/nlp/BART/simpletransformers
```

### Install Model Requirements
Install the python packages required for fine-tuning:
```bash
cd Model-References/PyTorch/nlp/BART/simpletransformers
pip install -e .
pip install bert_score
```

### Fine-tuning Dataset Preparation

Public datasets can be downloaded with this script:
```bash
bash ./examples/seq2seq/paraphrasing/data_download.sh
```

**Note:** Going forward it is assumed that the dataset is located in `./data` directory.

## Training Examples

### Single Card and Multi-Card Training Examples

**Run training on 1 HPU - Lazy mode:**

- 1 HPU, BART fine-tuning on the dataset using BF16 mixed precision:
  ```python
  LOWER_LIST=ops_bf16_bart.txt FP32_LIST=ops_fp32_bart.txt $PYTHON examples/seq2seq/paraphrasing/train.py --use_habana --no_cuda --use_fused_adam --use_fused_clip_norm --max_seq_length 128 --train_batch_size 32 --num_train_epochs 5 --logging_steps 50 --save_best_model --output_dir output --bf16 autocast
  ```
- 1 HPU, BART fine-tuning on the dataset using FP32 data type:
  ```python
  $PYTHON examples/seq2seq/paraphrasing/train.py --use_habana --no_cuda --use_fused_adam --use_fused_clip_norm --max_seq_length 128 --train_batch_size 32 --num_train_epochs 5 --logging_steps 50 --save_best_model --output_dir output
  ```

**Run training on 8 HPUs:**

To run multi-card demo, make sure the host machine has 512 GB of RAM installed. Modify the docker run command to pass 8 Gaudi cards to the docker container. This ensures the docker has access to all the 8 cards required for multi-card training.

**NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Scaling_Guide/DDP_Based_Scaling.html#mpirun-configuration).

- 8 HPUs on a single server, BF16, batch size 32, Lazy mode:
  ```bash
  LOWER_LIST=ops_bf16_bart.txt FP32_LIST=ops_fp32_bart.txt mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root $PYTHON examples/seq2seq/paraphrasing/train.py --use_habana --no_cuda --use_fused_adam --use_fused_clip_norm --max_seq_length 128 --train_batch_size 32 --num_train_epochs 5 --logging_steps 50 --save_best_model --output_dir /tmp/multicards --bf16 autocast --distributed
  ```

- 8 HPUs on a single server, FP32, batch size 32, Lazy mode:
  ```bash
  mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root $PYTHON examples/seq2seq/paraphrasing/train.py --use_habana --no_cuda --use_fused_adam --use_fused_clip_norm --max_seq_length 128 --train_batch_size 32 --num_train_epochs 5 --logging_steps 50 --save_best_model --output_dir /tmp/multicards --distributed
  ```


## Supported Configurations

| Device | SynapseAI Version | PyTorch Version |
|-----|-----|-----|
| Gaudi | 1.12.0 | 2.0.1 |

## Changelog
### 1.12.0
 - Eager mode support is deprecated.
 - Removed PT_HPU_LAZY_MODE environment variable.
 - Removed flag lazy_mode.
 - Removed HMP; switched to Autocast.
 - Updated run commands.
### 1.9.0
 - Enabled PyTorch autocast on Gaudi
### 1.6.0
 - Changed BART distributed API to initialize_distributed_hpu.
### 1.5.0
 - Removed unnecessary mark_step.
### 1.4.0
 - Removed wrapper script run_bart.py.
 - Added support for reducing the print frequency of Running Loss to the frequency of logging_steps.

### Training Script Modifications

The following changes have been added to scripts & source:
modifications to the [simple transformer](https://github.com/ThilinaRajapakse/simpletransformers) source:

1. Added Habana Device support (seq2seq_model.py).
2. Modifications for saving checkpoint: Bring tensors to CPU and save (seq2seq_model.py).
3. Introduced Habana BF16 Mixed precision, adding ops lists for BF16 and FP32 (seq2seq_model.py, ops_bf16_bart.txt, ops_fp32_bart.txt).
4. Change for supporting HMP disable for optimizer.step (seq2seq_model.py).
5. Use fused AdamW optimizer on Habana device (seq2seq_model.py, train.py).
6. Use fused clip norm for grad clipping on Habana device (seq2seq_model.py, train.py).
7. Modified training script to use mpirun for distributed training (train.py).
8. Gradients are used as views using gradient_as_bucket_view (seq2seq_model.py).
9. Default allreduce bucket size set to 200MB for better performance in distributed training (seq2seq_model.py).
10. Added changes to support Lazy mode with required mark_step (seq2seq_model.py).
11. Only print and save in the master process (seq2seq_model.py).
12. Added prediction (sentence generation) metrics (seq2seq_model.py).
13. Modified training script to use Habana data loader (seq2seq_model.py).
14. Add data_dir as an input argument for data directory.
15. Added this README.

## Known Issues

1. Placing mark_step() arbitrarily may lead to undefined behavior. Recommend to keep mark_step() as shown in provided scripts.
2. Sentence generation (prediction) is not enabled in this release. We plan to enable it in next release.
