# Table of Contents
- [BERT, RoBERTa, ALBERT, and ELECTRA for PyTorch](#bert-roberta-albert-and-electra-for-pytorch)
  - [BERT Fine-Tuning](#bert-fine-tuning)
    - [SQuAD](#squad)
    - [MRPC](#mrpc)
  - [RoBERTa Fine-Tuning](#roberta-fine-tuning)
    - [SQuAD](#squad-1)
  - [ALBERT Fine-Tuning](#albert-fine-tuning)
    - [SQuAD](#squad-2)
  - [ELECTRA Fine-Tuning](#electra-fine-tuning)
    - [SQuAD](#squad-3)
- [Setup](#setup)
  - [Fine-Tuning Dataset Preparation](#fine-tuning-dataset-preparation)
    - [MRPC and SQuADv1.1 Dataset Preparation](#mrpc-and-squadv11-dataset-preparation)
  - [Model Overview](#model-overview)
- [Fine-Tuning](#fine-tuning)
  - [Reference Script](#reference-script)
  - [Training the Model](#training-the-model)
  - [Multicard Training](#multicard-training)
- [Changelog](#changelog)
  - [1.4.0](#140)
  - [1.2.0](#120)
- [Known Issues](#known-issues)
- [Training Script Modifications](#training-script-modifications)
  - [BERT Base and BERT Large Fine-Tuning](#bert-base-and-bert-large-fine-tuning)

# BERT, RoBERTa, ALBERT, and ELECTRA for PyTorch

This folder contains scripts to fine-tune language models on Habana Gaudi<sup>TM</sup> device to achieve state-of-the-art accuracy. Please visit [this page](https://developer.habana.ai/resources/habana-training-models/#performance) for performance information.

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

The demos included in this release are as follows:

## BERT Fine-Tuning
### SQuAD
- BERT Large fine-tuning for FP32 and BF16 Mixed precision for SQuADv1.1 dataset in Lazy mode.
- Multi card (1 server = 8 cards) support for BERT Large fine-tuning with FP32 and BF16 Mixed precision in Lazy mode.
- BERT Base fine-tuning for FP32 with SQuAD dataset in Eager mode.

### MRPC
- BERT Large fine-tuning with MRPC dataset for FP32 and BF16 Mixed precision in Lazy mode.
- BERT Base fine-tuning for FP32 with MRPC dataset in Eager mode.

## RoBERTa Fine-Tuning
### SQuAD
- RoBERTa Large fine-tuning for FP32 and BF16 Mixed precision for SQuADv1.1 dataset in Eager mode and Lazy mode.
- Multi card (1xHLS = 8 cards) support for RoBERTa Large fine-tuning with FP32 and BF16 Mixed precision in Lazy mode.
- RoBERTa Base fine-tuning for FP32 and BF16 Mixed precision with SQuAD dataset in Eager mode and lazy mode.

## ALBERT Fine-Tuning
### SQuAD
- ALBERT Large and XXLarge BF16 Mixed precision fine-tuning for SQuADv1.1 dataset in Eager and Lazy mode.
- ALBERT Large FP32 fine-tuning for SQuADv1.1 dataset in Lazy mode.
- Multi card (1xHLS = 8 cards) support for ALBERT Large and XXLarge BF16 Mixed precision fine tuning for SQuADv1.1 dataset in Lazy mode.

## ELECTRA Fine-Tuning
### SQuAD
- ELECTRA Large discriminator FP32 and BF16 Mixed precision for SQuADv1.1 dataset in Eager mode and Lazy mode.
- Multi card (1xHLS = 8 cards) support for ELECTRA Large discriminator fine-tuning with FP32 and BF16 Mixed precision in Lazy mode.


The demo_bert.py script is a wrapper for respective python training scripts. Additional environment variables are used in training scripts in order to achieve optimal results for each workload.

# Setup
Please follow the instructions given in the following link for setting up the
environment including the `$PYTHON` environment variable: [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html).
This guide will walk you through the process of setting up your system to run
the model on Gaudi.

In the docker container, clone this repository and switch to the branch that
matches your SynapseAI version. (Run the
[`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.)

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```

## Fine-Tuning Dataset Preparation

### MRPC and SQuADv1.1 Dataset Preparation
Public datasets available on the datasets hub at https://github.com/huggingface/datasets
Based on the finetuning task name the dataset will be downloaded automatically from the datasets Hub.

## Model Overview
Bidirectional Encoder Representations from Transformers (BERT) is a technique for natural language processing (NLP) pre-training developed by Google.
The original English-language BERT model comes with two pre-trained general types: (1) the BERT Base model, a 12-layer, 768-hidden, 12-heads, 110M parameter neural network architecture, and (2) the BERT Large model, a 24-layer, 1024-hidden, 16-heads, 340M parameter neural network architecture; both of which were trained on the BooksCorpus with 800M words, and a version of the English Wikipedia with 2,500M words.

# Fine-Tuning
- Located in: `Model-References/PyTorch/nlp/finetuning/huggingface/bert/`
- Suited for tasks:
  - `mrpc`: Microsoft Research Paraphrase Corpus (**MRPC**) is a paraphrase identification dataset, where systems aim to identify if two sentences are paraphrases of each other.
  - `squad`: Stanford Question Answering Dataset (**SQuAD**) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.
- Uses optimizer: **AdamW** ("ADAM with Weight Decay Regularization").
- Based on model weights trained with pretraining.
- Light-weight: the training takes a minute or so.
- Datasets for MRPC and SQuAD will be automatically downloaded the first time the model is run in the docker container.


## Reference Script
The demo uses training scripts and models from https://github.com/huggingface/transformers.git (tag v4.17.0). The training script fine-tunes BERT based language models on the [Microsoft Research Paraphrase Corpus](https://gluebenchmark.com/) (MRPC) and [Stanford Question Answering Dataset](https://rajpurkar.github.io/SQuAD-explorer/) (SQuADv1.1) dataset.


## Training the Model
- Install the python packages needed for fine-tuning
```
pip install -r Model-References/PyTorch/nlp/finetuning/huggingface/bert/transformers/examples/pytorch/question-answering/requirements.txt
pip install -r Model-References/PyTorch/nlp/finetuning/huggingface/bert/transformers/examples/pytorch/text-classification/requirements.txt
```
- Install the transformer
```
pip install Model-References/PyTorch/nlp/finetuning/huggingface/bert/transformers/.
```
i. Fine-tune BERT Base (Eager mode)

- Run BERT Base fine-tuning on the GLUE MRPC dataset using FP32 data type:
```
$PYTHON demo_bert.py finetuning --model_name_or_path base --mode eager --dataset_name mrpc --task_name mrpc --data_type fp32 --num_train_epochs 3 --batch_size 32 --max_seq_length 128 --learning_rate 2e-5 --do_eval
```
- Run BERT Base fine-tuning on the SQuAD dataset using FP32 data type:
```
$PYTHON demo_bert.py finetuning --model_name_or_path base --mode eager --dataset_name squad --task_name squad --data_type fp32 --num_train_epochs 2 --batch_size 12 --max_seq_length 384 --learning_rate 3e-5 --do_eval
```


ii. Fine-tune BERT Large (Eager mode)

- Run BERT Large fine-tuning on the MRPC dataset with FP32:
```
$PYTHON demo_bert.py finetuning --model_name_or_path large --mode eager --dataset_name mrpc --task_name mrpc --data_type fp32 --num_train_epochs 3 --batch_size 32 --max_seq_length 128 --learning_rate 3e-5 --do_eval
```
- Run BERT Large fine-tuning on the SQuAD dataset with FP32:
```
$PYTHON demo_bert.py finetuning --model_name_or_path large --mode eager --dataset_name squad --task_name squad --data_type fp32 --num_train_epochs 2 --batch_size 10 --max_seq_length 384 --learning_rate 3e-5 --do_eval
```

iii. Fine-tune BERT Large (Lazy mode) on the SQuAD dataset

- Run BERT Large fine-tuning on the SQuAD dataset using BF16 mixed precision:
```
$PYTHON demo_bert.py finetuning --model_name_or_path large --mode lazy --dataset_name squad --task_name squad --data_type bf16 --num_train_epochs 2 --batch_size 24 --max_seq_length 384 --learning_rate 3e-5 --do_eval --logging_steps=20

```
```
$PYTHON transformers/examples/pytorch/question-answering/run_qa.py --hmp --hmp_bf16=./ops_bf16_bert.txt --hmp_fp32=./ops_fp32_bert.txt --doc_stride=128 --use_lazy_mode --per_device_train_batch_size=24 --per_device_eval_batch_size=8 --dataset_name=squad --use_fused_adam --use_fused_clip_norm --save_steps=5000 --use_habana --max_seq_length=384 --learning_rate=3e-05 --num_train_epochs=2 --output_dir=/tmp/SQUAD/squad --logging_steps=20 --seed=42 --overwrite_output_dir --do_train --do_eval --model_name_or_path=bert-large-uncased-whole-word-masking

```
- Run BERT Large fine-tuning on the SQuAD dataset using FP32 data type:
```
$PYTHON demo_bert.py finetuning --model_name_or_path large --mode lazy --dataset_name squad --task_name squad --data_type fp32 --num_train_epochs 2 --batch_size 10 --max_seq_length 384 --learning_rate 3e-5 --do_eval
```


iv. Fine-tune BERT Large (Lazy mode) on the MRPC dataset
- Run BERT Large fine-tuning on the MRPC dataset using BF16 data type:
```
$PYTHON demo_bert.py finetuning --model_name_or_path large --mode lazy --dataset_name mrpc --task_name mrpc --data_type bf16 --num_train_epochs 3 --batch_size 32 --max_seq_length 128 --learning_rate 3e-5 --do_eval
```

- Run BERT Large fine-tuning on the MRPC dataset using FP32 data type:
```
$PYTHON demo_bert.py finetuning --model_name_or_path large --mode lazy --dataset_name mrpc --task_name mrpc --data_type fp32 --num_train_epochs 3 --batch_size 32 --max_seq_length 128 --learning_rate 3e-5 --do_eval
```

v. Fine-tune RoBERTa Base (Eager mode)

- Run RoBERTa Base fine-tuning on the SQuAD dataset using FP32 data type:
```
$PYTHON demo_bert.py finetuning --model_name_or_path roberta-base --mode eager --dataset_name squad --task_name squad --data_type fp32 --num_train_epochs 2 --batch_size 12 --max_seq_length 384 --learning_rate 3e-5 --do_eval
```
- Run RoBERTa Base fine-tuning on the SQuAD dataset using BF16 data type:
```
$PYTHON demo_bert.py finetuning --model_name_or_path roberta-base --mode eager --dataset_name squad --task_name squad --data_type bf16 --num_train_epochs 2 --batch_size 12 --max_seq_length 384 --learning_rate 3e-5 --do_eval
```

vi. Fine-tune RoBERTa Base (Lazy mode) on the SQuAD dataset

- Run RoBERTa Large fine-tuning on the SQuAD dataset using BF16 mixed precision:
```
$PYTHON demo_bert.py finetuning --model_name_or_path roberta-base --mode lazy --dataset_name squad --task_name squad --data_type bf16 --num_train_epochs 2 --batch_size 12 --max_seq_length 384 --learning_rate 3e-5 --do_eval

```
- Run RoBERTa Base fine-tuning on the SQuAD dataset using FP32 data type:
```
$PYTHON demo_bert.py finetuning --model_name_or_path roberta-base --mode lazy --dataset_name squad --task_name squad --data_type fp32 --num_train_epochs 2 --batch_size 12 --max_seq_length 384 --learning_rate 3e-5 --do_eval
```

vi. Fine-tune RoBERTa Large (Eager mode)

- Run RoBERTa Large fine-tuning on the SQuAD dataset with FP32:
```
$PYTHON demo_bert.py finetuning --model_name_or_path roberta-large --mode eager --dataset_name squad --task_name squad --data_type fp32 --num_train_epochs 2 --batch_size 12 --max_seq_length 384 --learning_rate 3e-5 --do_eval
```

- Run RoBERTa Large fine-tuning on the SQuAD dataset with BF16:
```
$PYTHON demo_bert.py finetuning --model_name_or_path roberta-large --mode eager --dataset_name squad --task_name squad --data_type bf16 --num_train_epochs 2 --batch_size 12 --max_seq_length 384 --learning_rate 3e-5 --do_eval
```

vii. Fine-tune RoBERTa Large (Lazy mode) on the SQuAD dataset

- Run RoBERTa Large fine-tuning on the SQuAD dataset using BF16 mixed precision:
```
$PYTHON demo_bert.py finetuning --model_name_or_path roberta-large --mode lazy --dataset_name squad --task_name squad --data_type bf16 --num_train_epochs 2 --batch_size 12 --max_seq_length 384 --learning_rate 3e-5 --do_eval

```
- Run RoBERTa Large fine-tuning on the SQuAD dataset using FP32 data type:
```
$PYTHON demo_bert.py finetuning --model_name_or_path roberta-large --mode lazy --dataset_name squad --task_name squad --data_type fp32 --num_train_epochs 2 --batch_size 12 --max_seq_length 384 --learning_rate 3e-5 --do_eval
```

viii. Fine-tune ALBERT Large (Eager mode)

- Run ALBERT Large fine-tuning on the SQuAD dataset using BF16 data type:
```
$PYTHON demo_bert.py finetuning --model_name_or_pat albert-large --dataset_name squad --task_name squad --do_eval --max_seq_length 384 --mode eager --batch_size 24 --learning_rate 5e-05 --num_train_epochs 2 --save_steps 50000 --seed 42 --doc_stride 128 --device hpu --data_type bf16 --per_device_eval_batch_size 24
```
- Run ALBERT XXLarge fine-tuning on the SQuAD dataset using BF16 data type:
```
$PYTHON demo_bert.py finetuning --model_name_or_pat albert-xxlarge --dataset_name squad --task_name squad --do_eval --max_seq_length 384 --mode eager --batch_size 2 --learning_rate 5e-06 --num_train_epochs 2 --save_steps 50000 --seed 42 --doc_stride 128 --device hpu --data_type bf16 --per_device_eval_batch_size 2
```

ix. Fine-tune ALBERT variants (Lazy mode) on the SQuAD dataset

- Run ALBERT Large fine-tuning on the SQuAD dataset using BF16 mixed precision:
```
$PYTHON demo_bert.py finetuning --model_name_or_path albert-large --dataset_name squad --task_name squad --do_eval --max_seq_length 384 --mode lazy --batch_size 32 --learning_rate 5e-05 --num_train_epochs 2 --save_steps 5000 --seed 42 --doc_stride 128 --device hpu --data_type bf16 --per_device_eval_batch_size 4
```

- Run ALBERT Large fine-tuning on the SQuAD dataset using FP32 data type:
```
$PYTHON demo_bert.py finetuning --model_name_or_path albert-large --dataset_name squad --task_name squad --do_eval --max_seq_length 384 --mode lazy --batch_size 12 --learning_rate 2.5e-05 --num_train_epochs 2 --save_steps 5000 --seed 42 --doc_stride 128 --device hpu --data_type fp32 --per_device_eval_batch_size 4
```

- Run ALBERT XXLarge fine-tuning on the SQuAD dataset using BF16 mixed precision:
```
$PYTHON demo_bert.py finetuning --model_name_or_path albert-xxlarge --dataset_name squad --task_name squad --do_eval --max_seq_length 384 --mode lazy --batch_size 12 --learning_rate 5e-06 --num_train_epochs 2 --save_steps 5000  --seed 42 --doc_stride 128 --device hpu --data_type bf16 --per_device_eval_batch_size 2
```

x. Fine-tune ELECTRA Large discriminator (Eager mode)
- Run ELECTRA Large fine-tuning on the SQuAD dataset using BF16 data type:
```
$PYTHON demo_bert.py finetuning --model_name_or_path electra-large-d --dataset_name squad --task_name squad --do_eval --max_seq_length 512 --mode eager --batch_size 12 --per_device_eval_batch_size 12 --learning_rate 1.66667e-05 --num_train_epochs 2  --save_steps 10000  --seed 42 --doc_stride 128 --device hpu --data_type bf16 --logging_steps=50
```

xi. Fine-tune ELECTRA Large discriminator (Lazy mode)
- Run ELECTRA Large fine-tuning on the SQuAD dataset using BF16 mixed precision:
```
$PYTHON demo_bert.py finetuning --model_name_or_path electra-large-d --dataset_name squad --task_name squad --do_eval --max_seq_length 512 --mode lazy --batch_size 12 --per_device_eval_batch_size 12 --learning_rate 1.66667E-05 --num_train_epochs 2  --save_steps 10000 --seed 42 --doc_stride 128 --device hpu --data_type bf16 --logging_steps=50
```

- Run ELECTRA Large fine-tuning on the SQuAD dataset using FP32 data type:
```
$PYTHON demo_bert.py finetuning --model_name_or_path electra-large-d --dataset_name squad --task_name squad --do_eval --max_seq_length 512 --mode lazy --batch_size 12 --per_device_eval_batch_size 2 --learning_rate 1.66667e-05 --num_train_epochs 2  --save_steps 10000 --seed 42 --doc_stride 128 --device hpu --data_type fp32
```


## Multicard Training
To run multi-card demo, make sure the host machine has 512 GB of RAM installed. Modify the docker run command to pass 8 Gaudi cards to the docker container. This ensures the docker has access to all the 8 cards required for multi-card demo. Number of cards can be configured using --world_size option in the demo script.

Use the following command to run the multicard BERT Large demo on 8 cards (1 server) for bf16, BS24 Lazy mode:

*mpirun map-by PE attribute value may vary on your setup and should be calculated as:<br>
socket:PE = floor((number of physical cores) / (number of gaudi devices per each node))*
```
$PYTHON demo_bert.py finetuning --model_name_or_path large --mode lazy --dataset_name squad --task_name squad --data_type bf16 --num_train_epochs 2 --batch_size 24 --max_seq_length 384 --learning_rate 3e-5 --do_eval --world_size 8
```
```
export MASTER_ADDR="localhost"
export MASTER_PORT="12345"
mpirun -n 8 --bind-to core --map-by socket:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON transformers/examples/pytorch/question-answering/run_qa.py --hmp --hmp_bf16=./ops_bf16_bert.txt --hmp_fp32=./ops_fp32_bert.txt --doc_stride=128 --use_lazy_mode --per_device_train_batch_size=24 --per_device_eval_batch_size=8 --dataset_name=squad --use_fused_adam --use_fused_clip_norm --save_steps=5000 --use_habana --max_seq_length=384 --learning_rate=3e-05 --num_train_epochs= --output_dir=/tmp/SQUAD/squad --logging_steps=1 --seed=42 --overwrite_output_dir --do_train --do_eval --model_name_or_path=bert-large-uncased-whole-word-masking
```
Use the following command to run the multicard BERT Large demo on 8 cards (1 server) for fp32, BS10 Lazy mode:
```
$PYTHON demo_bert.py finetuning --model_name_or_path large --mode lazy --dataset_name squad --task_name squad --data_type fp32 --num_train_epochs 2 --batch_size 10 --max_seq_length 384 --learning_rate 3e-5 --do_eval --world_size 8
```
Use the following command to run the multicard RoBERTa Large on 8 cards (1 server) for bf16, BS12 Lazy mode:
```
$PYTHON demo_bert.py finetuning --model_name_or_path roberta-large --mode lazy --dataset_name squad --task_name squad --data_type bf16 --num_train_epochs 2 --batch_size 12 --max_seq_length 384 --learning_rate 3e-5 --do_eval --world_size 8
```
Use the following command to run the multicard RoBERTa Large on 8 cards (1 server) for fp32, BS10 Lazy mode:
```
$PYTHON demo_bert.py finetuning --model_name_or_path roberta-large --mode lazy --dataset_name squad --task_name squad --data_type fp32 --num_train_epochs 2 --batch_size 10 --max_seq_length 384 --learning_rate 3e-5 --do_eval --world_size 8
```
Use the following command to run the multicard ALBERT Large on 8 cards (1 server) for bf16, BS32 Lazy mode:
```
$PYTHON demo_bert.py finetuning --model_name_or_path albert-large --mode lazy --dataset_name squad --task_name squad --data_type bf16 --num_train_epochs 2 --batch_size 32 --max_seq_length 384 --learning_rate 6e-5 --do_eval --device hpu --per_device_eval_batch_size 4 --world_size 8
```
Use the following command to run the multicard ALBERT XXLarge on 8 cards (1 server) for bf16, BS12 Lazy mode:
```
$PYTHON demo_bert.py finetuning --model_name_or_path albert-xxlarge --mode lazy --dataset_name squad --task_name squad --data_type bf16 --num_train_epochs 2 --batch_size 12 --max_seq_length 384 --learning_rate 5e-5 --do_eval --device hpu --per_device_eval_batch_size 2 --world_size 8
```
Use the following command to run the multicard ELECTRA Large discriminator on 8 cards (1 server) for bf16, BS12 Lazy mode:
```
$PYTHON demo_bert.py finetuning --model_name_or_path electra-large-d --dataset_name squad --task_name squad --do_eval --max_seq_length 512 --mode lazy --batch_size 12 --per_device_eval_batch_size 12 --learning_rate 1.66667E-05 --num_train_epochs 2  --save_steps 10000 --seed 42 --doc_stride 128 --device hpu --data_type bf16 --world_size 8 --logging_steps=50
```
Use the following command to run the multicard ELECTRA Large discriminator on 8 cards (1 server) for fp32, BS6 Lazy mode:
```
$PYTHON demo_bert.py finetuning --model_name_or_path electra-large-d --dataset_name squad --task_name squad --do_eval --max_seq_length 512 --mode lazy --batch_size 6 --per_device_eval_batch_size 2 --learning_rate 5e-05 --num_train_epochs 2  --save_steps 10000 --seed 42 --doc_stride 128 --device hpu --data_type fp32 --world_size 8 --logging_steps=50
```

# Supported Configurations

For BERT, RoBERTa, ALBERT

| Device | SynapseAI Version | PyTorch Version |
|-----|-----|-----|
| Gaudi | 1.4.1 | 1.10.2 |

For ELECTRA

| Device | SynapseAI Version | PyTorch Version |
|-----|-----|-----|
| Gaudi | 1.3.0 | 1.10.2 |

# Changelog
## 1.4.0
1. Lazy mode is set as default execution mode,for eager mode set --use-lazy-mode as False

## 1.2.0
1. Enabled HCCL flow for distributed training.

# Known Issues
1. Placing mark_step() arbitrarily may lead to undefined behaviour. Recommend to keep mark_step() as shown in provided scripts.
2. Only scripts & configurations mentioned in this README are supported and verified.
3. Sharded DDP and deepspeed are not supported.

# Training Script Modifications
This section lists the training script modifications for the BERT models.

## BERT Base and BERT Large Fine-Tuning
The following changes have been added to training  & modeling  scripts.

1. Added support for Habana devices:

    a. Load Habana specific library (training_args.py,trainer.py).

    b. Saving checkpoint: Bring tensors to CPU and save (trainer_pt_utils.py,trainer.py).

    c. Required environment variables are defined for habana device(trainer.py).

    d. Added Habana BF16 Mixed precision support and HMP disable for optimizer.step(training_args.py,trainer.py).

    e. Use fused AdamW optimizer and clip norm Habana device (training_args.py,trainer.py).

    f. Support for distributed training on Habana device(training_args.py).

    g. Added changes to support Lazy mode with required mark_step(trainer.py).

    h. Changes for dynamic loading of HCCL library(training_args.py).

    i. Cast losses to float32 for numpy ops in evaluation (trainer.py).


2. To improve performance:

    a. Changes to optimize grad accumulation and zeroing of grads during the backward pass(trainer.py).

    b. Gradients are used as views using gradient_as_bucket_view(trainer.py).

    c. Default allreduce bucket size set to 230MB for better performance in distributed training(trainer.py).

    d. Reducing the print frequency (ex: logging_steps=20).
