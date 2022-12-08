# BERT, RoBERTa, ALBERT, and ELECTRA for PyTorch

This folder contains scripts to fine-tune and run inference on language models on Habana Gaudi device to achieve state-of-the-art accuracy. To obtain model performance data, refer to the [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance)

For more information about training and inference deep learning models using Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

For details of inference script visit [BERT-L Inference for PyTorch](transformers/examples/pytorch/bert-l-inference/).

**Note**: BERT model is enabled on both first-gen Gaudi and Gaudi2.

## Table of Contents
  - [Model References](../../../../../README.md)
  - [Model Overview](#model-overview)
  - [Setup](#setup)
  - [Training Examples](#training-examples)
  - [Supported Configurations](#supported-configurations)
  - [Changelog](#changelog)
  - [Known Issues](#known-issues)
## Model Overview
Bidirectional Encoder Representations from Transformers (BERT) is a technique for natural language processing (NLP) pre-training developed by Google.
The original English-language BERT model comes with two pre-trained general types: (1) the BERT Base model, a 12-layer, 768-hidden, 12-heads, 110M parameter neural network architecture, and (2) the BERT Large model, a 24-layer, 1024-hidden, 16-heads, 340M parameter neural network architecture; both of which were trained on the BooksCorpus with 800M words, and a version of the English Wikipedia with 2,500M words.

### Fine-Tuning
- Located in: `Model-References/PyTorch/nlp/finetuning/huggingface/bert/`
- Suited for tasks:
  - `mrpc`: Microsoft Research Paraphrase Corpus (**MRPC**) is a paraphrase identification dataset, where systems aim to identify if two sentences are paraphrases of each other.
  - `squad`: Stanford Question Answering Dataset (**SQuAD**) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.
- Uses optimizer: **AdamW** ("ADAM with Weight Decay Regularization").
- Based on model weights trained with pre-training.
- Light-weight: the training takes a minute or so.
- Datasets for MRPC and SQuAD will be automatically downloaded the first time the model is run in the docker container.

### BERT Fine-Tuning
#### SQuAD
- BERT Large fine-tuning for FP32 and BF16 mixed precision for SQuADv1.1 dataset in Lazy mode.
- Multi-card (1 server = 8 cards) support for BERT Large fine-tuning with FP32 and BF16 mixed precision in Lazy mode.
- BERT Base fine-tuning for FP32 with SQuAD dataset in Eager mode.

#### MRPC
- BERT Large fine-tuning with MRPC dataset for FP32 and BF16 mixed precision in Lazy mode.
- BERT Base fine-tuning for FP32 with MRPC dataset in Eager mode.

### RoBERTa Fine-Tuning
#### SQuAD
- RoBERTa Large fine-tuning for FP32 and BF16 mixed precision for SQuADv1.1 dataset in Eager mode and Lazy mode.
- Multi-card (1 server = 8 cards) support for RoBERTa Large fine-tuning with FP32 and BF16 mixed precision in Lazy mode.
- RoBERTa Base fine-tuning for FP32 and BF16 mixed precision with SQuAD dataset in Eager mode and lazy mode.

### ALBERT Fine-Tuning
#### SQuAD
- ALBERT Large and XXLarge BF16 mixed precision fine-tuning for SQuADv1.1 dataset in Eager and Lazy mode.
- ALBERT Large FP32 fine-tuning for SQuADv1.1 dataset in Lazy mode.
- Multi card (1 server = 8 cards) support for ALBERT Large and XXLarge BF16 Mixed precision fine tuning for SQuADv1.1 dataset in Lazy mode.

### ELECTRA Fine-Tuning
#### SQuAD
- ELECTRA Large discriminator FP32 and BF16 mixed precision for SQuADv1.1 dataset in Eager mode and Lazy mode.
- Multi-card (1 server = 8 cards) support for ELECTRA Large discriminator fine-tuning with FP32 and BF16 mixed precision in Lazy mode.

### Reference Script

The demo uses training scripts and models from https://github.com/huggingface/transformers.git (tag v4.19.2). The training script fine-tunes BERT based language models on the [Microsoft Research Paraphrase Corpus](https://gluebenchmark.com/) (MRPC) and [Stanford Question Answering Dataset](https://rajpurkar.github.io/SQuAD-explorer/) (SQuADv1.1) dataset.

## Setup
Please follow the instructions provided in the [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the
environment including the `$PYTHON` environment variable.
The guide will walk you through the process of setting up your system to run the model on Gaudi.

### Clone Habana Model-References

In the docker container, clone this repository and switch to the branch that
matches your SynapseAI version. You can run the
[`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```
### Install Model Requirements
- Install the python packages needed for fine-tuning:
```
pip install -r Model-References/PyTorch/nlp/finetuning/huggingface/bert/transformers/examples/pytorch/question-answering/requirements.txt
pip install -r Model-References/PyTorch/nlp/finetuning/huggingface/bert/transformers/examples/pytorch/text-classification/requirements.txt
```
- Install the transformer:
```
pip install Model-References/PyTorch/nlp/finetuning/huggingface/bert/transformers/.
```
## Deepspeed Training
To run deepspeed multi-card demo, make sure the host machine has 512 GB of RAM installed. Modify the docker run command to pass 8 Gaudi cards to the docker container. This ensures the docker has access to all the 8 cards required for multi-card demo.


Use the following command to run the deepspeed multicard BERT Large demo on 8 cards (1 server) for bf16, BS24 Lazy mode:
```
USE_TORCH=true deepspeed transformers/examples/pytorch/question-answering/run_qa.py --bf16 --doc_stride=128 --use_lazy_mode   --per_device_train_batch_size=24 --per_device_eval_batch_size=8 --dataset_name=squad --use_fused_clip_norm --save_strategy="no" --use_hpu --max_seq_length=384   --learning_rate=3e-05 --num_train_epochs=2 --output_dir=~/tmp/squad_bf16_ds_zero1_8cards --logging_steps=10 --overwrite_output_dir --do_train --do_eval --model_name_or_path=bert-large-uncased-whole-word-masking \
--deepspeed deepspeed_bsz24_8cards_zero1_bf16_config.json
```

### MRPC and SQuADv1.1 Dataset Preparation

Public datasets are available on the datasets hub at https://github.com/huggingface/datasets.
Based on the fine-tuning task name, the dataset will be downloaded automatically from the datasets hub.


## Training Examples

### Single Card and Multi-Card Training Examples
**Run training on 1 HPU:**
- Run BERT Base fine-tuning on the GLUE MRPC dataset using FP32 data type, 1 HPU, Eager mode:
  ```
  $PYTHON transformers/examples/pytorch/text-classification/run_glue.py --task_name=MRPC --use_lazy_mode false --per_device_train_batch_size=32 --per_device_eval_batch_size=8 \
    --dataset_name=mrpc --use_fused_adam --use_fused_clip_norm --use_hpu --max_seq_length=128 --learning_rate=2e-05 --num_train_epochs=3 --output_dir=/tmp/mrpc --logging_steps=20 \
    --overwrite_output_dir --do_train --do_eval --save_steps 300 --model_name_or_path=bert-base-uncased
  ```
- Run BERT Base fine-tuning on the SQuAD dataset using FP32 data type, 1 HPU, Eager mode:
  ```
  $PYTHON transformers/examples/pytorch/question-answering/run_qa.py --doc_stride=128 --use_lazy_mode false --per_device_train_batch_size=12 --per_device_eval_batch_size=8 \
    --dataset_name=squad --use_fused_adam --use_fused_clip_norm --use_hpu --max_seq_length=384 --learning_rate=3e-05 --num_train_epochs=2 --output_dir=/tmp/squad --logging_steps=20 \
    --overwrite_output_dir --do_train --do_eval --save_steps=5000 --model_name_or_path=bert-base-uncased
  ```

- Run BERT Large fine-tuning on the MRPC dataset with FP32, 1 HPU, Eager mode:
  ```
  $PYTHON transformers/examples/pytorch/text-classification/run_glue.py --task_name=MRPC --use_lazy_mode false --per_device_train_batch_size=32 --per_device_eval_batch_size=8 \
    --dataset_name=mrpc --use_fused_adam --use_fused_clip_norm --use_hpu --max_seq_length=128 --learning_rate=2e-05 --num_train_epochs=3 --output_dir=/tmp/mrpc --logging_steps=20 \
    --overwrite_output_dir --do_train --do_eval --save_steps=300 --model_name_or_path=bert-large-uncased-whole-word-masking
  ```
- Run BERT Large fine-tuning on the SQuAD dataset with FP32, 1 HPU, Eager mode:
  ```
  $PYTHON transformers/examples/pytorch/question-answering/run_qa.py --doc_stride=128 --use_lazy_mode false --per_device_train_batch_size=10 --per_device_eval_batch_size=8 \
    --dataset_name=squad --use_fused_adam --use_fused_clip_norm --use_hpu --max_seq_length=384 --learning_rate=3e-05 --num_train_epochs=2 --output_dir=/tmp/squad --logging_steps=20 \
    --overwrite_output_dir --do_train --do_eval --save_steps=5000 --model_name_or_path=bert-large-uncased-whole-word-masking
  ```

- Run BERT Large fine-tuning on the SQuAD dataset using BF16 mixed precision, 1 HPU, Lazy mode:

  ```
  $PYTHON transformers/examples/pytorch/question-answering/run_qa.py --hmp --hmp_bf16=./ops_bf16_bert.txt --hmp_fp32=./ops_fp32_bert.txt --doc_stride=128 --use_lazy_mode \
    --per_device_train_batch_size=24 --per_device_eval_batch_size=8 --dataset_name=squad --use_fused_adam --use_fused_clip_norm --save_steps=5000 --use_hpu --max_seq_length=384 \
    --learning_rate=3e-05 --num_train_epochs=2 --output_dir=/tmp/SQUAD/squad --logging_steps=20 --overwrite_output_dir --do_train --do_eval \
    --model_name_or_path=bert-large-uncased-whole-word-masking
  ```
- Run BERT Large fine-tuning on the SQuAD dataset using FP32 data type, 1 HPU, Lazy mode:
  ```
  $PYTHON transformers/examples/pytorch/question-answering/run_qa.py --doc_stride=128 --per_device_train_batch_size=10 --per_device_eval_batch_size=8 --dataset_name=squad \
    --use_fused_adam --use_fused_clip_norm --use_hpu --max_seq_length=384 --learning_rate=3e-05 --num_train_epochs=2 --output_dir=/tmp/squad --logging_steps=20 \
    --overwrite_output_dir --do_train --do_eval --model_name_or_path=bert-large-uncased-whole-word-masking
  ```

- Run BERT Large fine-tuning on the MRPC dataset using BF16 data type, 1 HPU, Lazy mode:
  ```
  $PYTHON transformers/examples/pytorch/text-classification/run_glue.py --hmp --hmp_bf16=./ops_bf16_bert.txt --hmp_fp32=./ops_fp32_bert.txt --task_name=MRPC --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=8 --dataset_name=mrpc --use_fused_adam --use_fused_clip_norm --use_hpu --max_seq_length=128 --learning_rate=3e-05 --num_train_epochs=3 \
    --output_dir=/tmp/mrpc --logging_steps=20 --overwrite_output_dir --do_train --do_eval --model_name_or_path=bert-large-uncased-whole-word-masking
  ```

- Run BERT Large fine-tuning on the MRPC dataset using FP32 data type, 1 HPU, Lazy mode:
  ```
  $PYTHON transformers/examples/pytorch/text-classification/run_glue.py --task_name=MRPC --per_device_train_batch_size=32 --per_device_eval_batch_size=8 --dataset_name=mrpc \
    --use_fused_adam --use_fused_clip_norm --use_hpu --max_seq_length=128 --learning_rate=3e-05 --num_train_epochs=3 --output_dir=/tmp/mrpc --logging_steps=20 \
    --overwrite_output_dir --do_train --do_eval --model_name_or_path=bert-large-uncased-whole-word-masking
  ```

- Run RoBERTa Base fine-tuning on the SQuAD dataset using FP32 data type, 1 HPU, Eager mode:
  ```
  $PYTHON transformers/examples/pytorch/question-answering/run_qa.py --doc_stride=128 --use_lazy_mode false --per_device_train_batch_size=12 --per_device_eval_batch_size=8 \
    --dataset_name=squad --use_fused_adam --use_fused_clip_norm --use_hpu --max_seq_length=384 --learning_rate=3e-05 --num_train_epochs=2 --output_dir=/tmp/squad \
    --logging_steps=20 --overwrite_output_dir --do_train --do_eval --model_name_or_path=roberta-base
  ```
- Run RoBERTa Base fine-tuning on the SQuAD dataset using BF16 data type, 1 HPU, Eager mode:
  ```
  $PYTHON transformers/examples/pytorch/question-answering/run_qa.py --hmp --hmp_bf16=./ops_bf16_roberta.txt --hmp_fp32=./ops_fp32_roberta.txt --doc_stride=128 --use_lazy_mode false \
    --per_device_train_batch_size=12 --per_device_eval_batch_size=8 --dataset_name=squad --use_fused_adam --use_fused_clip_norm --use_hpu --max_seq_length=384 --learning_rate=3e-05 \
    --num_train_epochs=2 --output_dir=/tmp/squad --logging_steps=20 --overwrite_output_dir --do_train --do_eval --save_steps=50000 --model_name_or_path=roberta-base
  ```

- Run RoBERTa Base fine-tuning on the SQuAD dataset using BF16 mixed precision, 1 HPU, Lazy mode:
  ```
  $PYTHON transformers/examples/pytorch/question-answering/run_qa.py --hmp --hmp_bf16=./ops_bf16_roberta.txt --hmp_fp32=./ops_fp32_roberta.txt --doc_stride=128 \
    --per_device_train_batch_size=12 --per_device_eval_batch_size=8 --dataset_name=squad --use_fused_adam --use_fused_clip_norm --use_hpu --max_seq_length=384 --learning_rate=3e-05 \
    --num_train_epochs=2 --output_dir=/tmp/squad --logging_steps=20 --overwrite_output_dir --do_train --do_eval --save_steps=50000 --model_name_or_path=roberta-base
  ```
- Run RoBERTa Base fine-tuning on the SQuAD dataset using FP32 data type, 1 HPU, Lazy mode:
  ```
  $PYTHON transformers/examples/pytorch/question-answering/run_qa.py --doc_stride=128 --per_device_train_batch_size=12 --per_device_eval_batch_size=8 --dataset_name=squad \
    --use_fused_adam --use_fused_clip_norm --use_hpu --max_seq_length=384 --learning_rate=3e-05 --num_train_epochs=2 --output_dir=/tmp/squad --logging_steps=20 \
    --overwrite_output_dir --do_train --do_eval  --save_steps 50000 --model_name_or_path=roberta-base
  ```

- Run RoBERTa Large fine-tuning on the SQuAD dataset with FP32, 1 HPU, Eager mode:
  ```
  $PYTHON transformers/examples/pytorch/question-answering/run_qa.py --doc_stride=128 --use_lazy_mode false --per_device_train_batch_size=12 --per_device_eval_batch_size=8 \
    --dataset_name=squad --use_fused_adam --use_fused_clip_norm --use_hpu --max_seq_length=384 --learning_rate=3e-05 --num_train_epochs=2 --output_dir=/tmp/squad --logging_steps=20 \
    --overwrite_output_dir --do_train --do_eval --save_steps 50000 --model_name_or_path=roberta-large
  ```

- Run RoBERTa Large fine-tuning on the SQuAD dataset with BF16, 1 HPU, Eager mode:
  ```
  $PYTHON transformers/examples/pytorch/question-answering/run_qa.py --hmp --hmp_bf16=./ops_bf16_bert.txt --hmp_fp32=./ops_fp32_bert.txt --doc_stride=128 --use_lazy_mode false \
    --per_device_train_batch_size=12 --per_device_eval_batch_size=8 --dataset_name=squad --use_fused_adam --use_fused_clip_norm --use_hpu --max_seq_length=384 --learning_rate=3e-05 \
    --num_train_epochs=2 --output_dir=/tmp/squad --logging_steps=20 --overwrite_output_dir --do_train --do_eval --save_steps 50000 --model_name_or_path=roberta-large
  ```

- Run RoBERTa Large fine-tuning on the SQuAD dataset using BF16 mixed precision, 1 HPU, Lazy mode:
  ```
  $PYTHON transformers/examples/pytorch/question-answering/run_qa.py --hmp --hmp_bf16=./ops_bf16_bert.txt --hmp_fp32=./ops_fp32_bert.txt --doc_stride=128 --per_device_train_batch_size=12 \
    --per_device_eval_batch_size=8 --dataset_name=squad --use_fused_adam --use_fused_clip_norm --use_hpu --max_seq_length=384 --learning_rate=3e-05 --num_train_epochs=2 \
    --output_dir=/tmp/squad --logging_steps=20 --overwrite_output_dir --do_train --do_eval --save_steps 50000 --model_name_or_path=roberta-large
  ```
- Run RoBERTa Large fine-tuning on the SQuAD dataset using FP32 data type, 1 HPU, Lazy mode:
  ```
  $PYTHON transformers/examples/pytorch/question-answering/run_qa.py --doc_stride=128 --per_device_train_batch_size=12 --per_device_eval_batch_size=8 --dataset_name=squad \
    --use_fused_adam --use_fused_clip_norm --use_hpu --max_seq_length=384 --learning_rate=3e-05 --num_train_epochs=2 --output_dir=/tmp/squad --logging_steps=20 \
    --overwrite_output_dir --do_train --do_eval --model_name_or_path=roberta-large
  ```

- Run ALBERT Large fine-tuning on the SQuAD dataset using BF16 data type, 1 HPU, Eager mode:
  ```
  $PYTHON transformers/examples/pytorch/question-answering/run_qa.py --hmp --hmp_bf16=./ops_bf16_bert.txt --hmp_fp32=./ops_fp32_bert.txt --doc_stride=128 --use_lazy_mode false \
    --per_device_train_batch_size=24 --per_device_eval_batch_size=24 --dataset_name=squad --use_fused_adam --use_fused_clip_norm --save_steps=50000 --use_hpu --max_seq_length=384 \
    --learning_rate=5e-05 --num_train_epochs=2 --output_dir=/tmp/squad --logging_steps=20 --overwrite_output_dir --do_train --do_eval --model_name_or_path=albert-large-v2
  ```
- Run ALBERT XXLarge fine-tuning on the SQuAD dataset using BF16 data type, 1 HPU, Eager mode:
  ```
  $PYTHON transformers/examples/pytorch/question-answering/run_qa.py --hmp --hmp_bf16=./ops_bf16_bert.txt --hmp_fp32=./ops_fp32_bert.txt --doc_stride=128 --use_lazy_mode false \
    --per_device_train_batch_size=2 --per_device_eval_batch_size=2 --dataset_name=squad --use_fused_adam --use_fused_clip_norm --save_steps=50000 --use_hpu --max_seq_length=384 \
    --learning_rate=5e-06 --num_train_epochs=2 --output_dir=/tmp/squad --logging_steps=20 --overwrite_output_dir --do_train --do_eval --model_name_or_path=albert-xxlarge-v1
  ```

- Run ALBERT Large fine-tuning on the SQuAD dataset using BF16 mixed precision, 1 HPU, Lazy mode:
  ```
  $PYTHON transformers/examples/pytorch/question-answering/run_qa.py --hmp --hmp_bf16=./ops_bf16_bert.txt --hmp_fp32=./ops_fp32_bert.txt --doc_stride=128 --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=4 --dataset_name=squad --use_fused_adam --use_fused_clip_norm --save_steps=50000 --use_hpu --max_seq_length=384 --learning_rate=5e-05 --num_train_epochs=2 \
    --output_dir=/tmp/squad --logging_steps=20 --overwrite_output_dir --do_train --do_eval --model_name_or_path=albert-large-v2
  ```

- Run ALBERT Large fine-tuning on the SQuAD dataset using FP32 data type, 1 HPU, Lazy mode:
  ```
  $PYTHON transformers/examples/pytorch/question-answering/run_qa.py --doc_stride=128 --per_device_train_batch_size=12 --per_device_eval_batch_size=4 --dataset_name=squad --use_fused_adam \
    --use_fused_clip_norm --save_steps=50000 --use_hpu --max_seq_length=384 --learning_rate=2.5e-05 --num_train_epochs=2 --output_dir=/tmp/squad --logging_steps=20 \
    --overwrite_output_dir --do_train --do_eval --model_name_or_path=albert-large-v2
  ```

- Run ALBERT XXLarge fine-tuning on the SQuAD dataset using BF16 mixed precision, 1 HPU, Lazy mode:
  ```
  $PYTHON transformers/examples/pytorch/question-answering/run_qa.py --hmp --hmp_bf16=./ops_bf16_bert.txt --hmp_fp32=./ops_fp32_bert.txt --doc_stride=128 --per_device_train_batch_size=12 \
    --per_device_eval_batch_size=2 --dataset_name=squad --use_fused_adam --use_fused_clip_norm --save_steps=50000 --use_hpu --max_seq_length=384 --learning_rate=5e-06 --num_train_epochs=2 \
    --output_dir=/tmp/squad --logging_steps=20 --overwrite_output_dir --do_train --do_eval --model_name_or_path=albert-xxlarge-v1
  ```

- Run ELECTRA Large discriminator fine-tuning on the SQuAD dataset using BF16 data type, 1 HPU, Eager mode:
  ```
  $PYTHON transformers/examples/pytorch/question-answering/run_qa.py --hmp --hmp_bf16=./ops_bf16_electra.txt --hmp_fp32=./ops_fp32_electra.txt --doc_stride=128 --use_lazy_mode false \
    --per_device_train_batch_size=12 --per_device_eval_batch_size=8 --dataset_name=squad --use_fused_adam --use_fused_clip_norm --save_steps=10000 --use_hpu --max_seq_length=512 \
    --learning_rate=1.66667e-05 --num_train_epochs=2 --output_dir=/tmp/squad --logging_steps=20 --overwrite_output_dir --do_train --do_eval --model_name_or_path=google/electra-large-discriminator
  ```

- Run ELECTRA Large discriminator fine-tuning on the SQuAD dataset using BF16 mixed precision1 HPU, Lazy mode:
  ```
  $PYTHON transformers/examples/pytorch/question-answering/run_qa.py --hmp --hmp_bf16=./ops_bf16_electra.txt --hmp_fp32=./ops_fp32_electra.txt --doc_stride=128 --per_device_train_batch_size=12 \
    --per_device_eval_batch_size=8 --dataset_name=squad --use_fused_adam --use_fused_clip_norm --save_steps=10000 --use_hpu --max_seq_length=512 --learning_rate=1.66667e-05 \
    --num_train_epochs=2 --output_dir=/tmp/squad --logging_steps=20 --overwrite_output_dir --do_train --do_eval --model_name_or_path=google/electra-large-discriminator
  ```

- Run ELECTRA Large fine-tuning on the SQuAD dataset using FP32 data type, 1 HPU, Lazy mode::
  ```
  $PYTHON transformers/examples/pytorch/question-answering/run_qa.py --doc_stride=128 --per_device_train_batch_size=12 --per_device_eval_batch_size=8 --dataset_name=squad \
    --use_fused_adam --use_fused_clip_norm --save_steps=10000 --use_hpu --max_seq_length=512 --learning_rate=1.66667e-05 --num_train_epochs=2 --output_dir=/tmp/squad \
    --logging_steps=20 --overwrite_output_dir --do_train --do_eval --model_name_or_path=google/electra-large-discriminator
  ```

**Run training on 8 HPUs:**

To run multi-card demo, make sure the host machine has 512 GB of RAM installed. Modify the docker run command to pass 8 Gaudi cards to the docker container. This ensures the docker has access to all the 8 cards required for multi-card demo. Number of cards can be configured using `--world_size` option in the demo script.

**NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Scaling_Guide/DDP_Based_Scaling.html#mpirun-configuration).


- BERT Large, 8 HPUs (1 server), BF16, batch size 24, Lazy mode:
  ```bash
  export MASTER_ADDR="localhost"
  export MASTER_PORT="12345" 
  mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root $PYTHON transformers/examples/pytorch/question-answering/run_qa.py \
    --hmp --hmp_bf16=./ops_bf16_bert.txt --hmp_fp32=./ops_fp32_bert.txt --doc_stride=128 --per_device_train_batch_size=24 --per_device_eval_batch_size=8 --dataset_name=squad \
    --use_fused_adam --use_fused_clip_norm --use_hpu --max_seq_length=384 --learning_rate=3e-05 --num_train_epochs=2 --output_dir=/tmp/squad --logging_steps=20 \
    --overwrite_output_dir --do_train --do_eval --save_steps 5000 --model_name_or_path=bert-large-uncased-whole-word-masking
  ```

- BERT Large, 8 HPUs (1 server), FP32, batch size 10, Lazy mode:
  ```bash
  export MASTER_ADDR="localhost"
  export MASTER_PORT="12345" 
  mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root $PYTHON transformers/examples/pytorch/question-answering/run_qa.py \
    --doc_stride=128 --per_device_train_batch_size=10 --per_device_eval_batch_size=8 --dataset_name=squad --use_fused_adam --use_fused_clip_norm --use_hpu --max_seq_length=384 \
    --learning_rate=3e-05 --num_train_epochs=2 --output_dir=/tmp/squad --logging_steps=20 --overwrite_output_dir --do_train --do_eval --save_steps 5000 \
    --model_name_or_path=bert-large-uncased-whole-word-masking
  ```

- RoBERTa Large, 8 HPUs (1 server), BF16, batch size 12, Lazy mode:
  ```bash
  export MASTER_ADDR="localhost"
  export MASTER_PORT="12345" 
  mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root $PYTHON transformers/examples/pytorch/question-answering/run_qa.py \
    --hmp --hmp_bf16=./ops_bf16_bert.txt --hmp_fp32=./ops_fp32_bert.txt --doc_stride=128 --per_device_train_batch_size=12 --per_device_eval_batch_size=8 --dataset_name=squad \
    --use_fused_adam --use_fused_clip_norm --use_hpu --max_seq_length=384 --learning_rate=3e-05 --num_train_epochs=2 --output_dir=/tmp/squad --logging_steps=20 \
    --save_steps 5000 --overwrite_output_dir --do_train --do_eval --model_name_or_path=roberta-large
  ```
- RoBERTa Large, 8 HPUs (1 server), FP32, batch size 10, Lazy mode:
  ```bash
  export MASTER_ADDR="localhost"
  export MASTER_PORT="12345" 
  mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root $PYTHON transformers/examples/pytorch/question-answering/run_qa.py \
    --doc_stride=128 --per_device_train_batch_size=10 --per_device_eval_batch_size=8 --dataset_name=squad --use_fused_adam --use_fused_clip_norm --use_hpu --max_seq_length=384 \
    --learning_rate=3e-05 --num_train_epochs=2 --output_dir=/tmp/squad --logging_steps=20 --overwrite_output_dir --do_train --do_eval --model_name_or_path=roberta-large
  ```

- ALBERT Large, 8 HPUs (1 server) BF16, batch size 32, Lazy mode:
  ```bash
  export MASTER_ADDR="localhost"
  export MASTER_PORT="12345" 
  mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root $PYTHON transformers/examples/pytorch/question-answering/run_qa.py \
    --hmp --hmp_bf16=./ops_bf16_bert.txt --hmp_fp32=./ops_fp32_bert.txt --doc_stride=128 --per_device_train_batch_size=32 --per_device_eval_batch_size=2 --dataset_name=squad \
    --use_fused_adam --use_fused_clip_norm --use_hpu --max_seq_length=384 --learning_rate=6e-05 --num_train_epochs=2 --output_dir=/tmp/squad --logging_steps=20 \
    --save_steps 5000 --overwrite_output_dir --do_train --do_eval --model_name_or_path=albert-large-v2
  ```

- ALBERT XXLarge, 8 HPUs (1 server), BF16, batch size 12, Lazy mode:
  ```bash
  export MASTER_ADDR="localhost"
  export MASTER_PORT="12345" 
  mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root $PYTHON transformers/examples/pytorch/question-answering/run_qa.py \
    --hmp --hmp_bf16=./ops_bf16_bert.txt --hmp_fp32=./ops_fp32_bert.txt --doc_stride=128 --per_device_train_batch_size=12 --per_device_eval_batch_size=2 --dataset_name=squad \
    --use_fused_adam --use_fused_clip_norm --use_hpu --max_seq_length=384 --learning_rate=5e-05 --num_train_epochs=2 --output_dir=/tmp/squad --logging_steps=20 \
    --save_steps 5000 --overwrite_output_dir --do_train --do_eval --model_name_or_path=albert-xxlarge-v1
  ```

- ELECTRA Large discriminator, 8 HPUs (1 server), BF16, batch size 12, Lazy mode:
  ```bash
  export MASTER_ADDR="localhost"
  export MASTER_PORT="12345" 
  mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root $PYTHON transformers/examples/pytorch/question-answering/run_qa.py \
    --hmp --hmp_bf16=./ops_bf16_electra.txt --hmp_fp32=./ops_fp32_electra.txt --doc_stride=128 --per_device_train_batch_size=12 --per_device_eval_batch_size=8 --dataset_name=squad \
    --use_fused_adam --use_fused_clip_norm --save_steps=5000 --use_hpu --max_seq_length=512 --learning_rate=1.66667e-05 --num_train_epochs=2 --output_dir=/tmp/squad \
    --logging_steps=50 --overwrite_output_dir --do_train --do_eval --model_name_or_path=google/electra-large-discriminator
  ```

- ELECTRA Large discriminator, 8 HPUs (1 server), FP32, batch size 6, Lazy mode:
  ```bash
  export MASTER_ADDR="localhost"
  export MASTER_PORT="12345" 
  mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root $PYTHON transformers/examples/pytorch/question-answering/run_qa.py \
    --doc_stride=128 --per_device_train_batch_size=6 --per_device_eval_batch_size=2 --dataset_name=squad --use_fused_adam --use_fused_clip_norm --save_steps=5000 --use_hpu \
    --max_seq_length=512 --learning_rate=5e-05 --num_train_epochs=2 --output_dir=/tmp/squad --logging_steps=20  --overwrite_output_dir --do_train --do_eval \
    --model_name_or_path=google/electra-large-discriminator
  ```

## Supported Configurations

**BERT:**

| Device | SynapseAI Version | PyTorch Version |
|-----|-----|-----|
| Gaudi  | 1.7.1 | 1.13.0 |
| Gaudi2 | 1.7.1 | 1.13.0 |

**RoBERTa and ALBERT:**

| Device | SynapseAI Version | PyTorch Version |
|-----|-----|-----|
| Gaudi | 1.7.1 | 1.13.0 |

**ELECTRA:**

| Device | SynapseAI Version | PyTorch Version |
|-----|-----|-----|
| Gaudi | 1.5.0 | 1.11.0 |

## Changelog
### 1.5.0
1. Changes related to saving and loading checkpoint were removed.
2. Simplified the distributed initialization.
3. Removed unsupported models.
4. Removed dtype conversion changes in evaluation.
5. Added support for training on **Gaudi2** up to 8 cards.

### 1.4.0
Lazy mode is set as default execution mode. For eager mode set --use-lazy-mode as False.

### 1.2.0
Enabled HCCL flow for distributed training.


### Training Script Modifications
This section lists the training script modifications for the BERT models.

#### BERT Base and BERT Large Fine-Tuning
The following changes have been added to training and modeling  scripts.

1. Added support for Habana devices:

    a. Load Habana specific library (training_args.py,trainer.py).

    b. Required environment variables are defined for habana device(trainer.py).

    c. Added Habana BF16 Mixed precision support and HMP disable for optimizer.step(training_args.py,trainer.py).

    d. Use fused AdamW optimizer and clip norm Habana device (training_args.py,trainer.py).

    e. Support for distributed training on Habana device(training_args.py).

    f. Added changes to support Lazy mode with required mark_step(trainer.py).

    g. Changes for dynamic loading of HCCL library(training_args.py).


2. To improve performance:

    a. Changes to optimize grad accumulation and zeroing of grads during the backward pass(trainer.py).

    b. Gradients are used as views using gradient_as_bucket_view(trainer.py).

    c. Default allreduce bucket size set to 230MB for better performance in distributed training(trainer.py).

    d. Reducing the print frequency (ex: logging_steps=20).

## Known Issues
1. Placing mark_step() arbitrarily may lead to undefined behavior. Recommend to keep mark_step() as shown in provided scripts.
2. Only scripts & configurations mentioned in this README are supported and verified.
3. Sharded DDP and DeepSpeed are not supported.
