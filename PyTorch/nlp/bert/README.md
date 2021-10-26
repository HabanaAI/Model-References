# Table of Contents
- [BERT for PyTorch](#bert-for-pytorch)
  - [BERT Pre-Training](#bert-pre-training)
  - [BERT Fine-Tuning](#bert-fine-tuning)
    - [SQuAD](#squad)
    - [MRPC](#mrpc)
- [Setup](#setup)
  - [Pre-training dataset preparation](#pre-training-dataset-preparation)
  - [Fine-tuning dataset preparation](#fine-tuning-dataset-preparation)
    - [MRPC and SQuADv1.1 dataset preparation](#mrpc-and-squadv11-dataset-preparation)
  - [Model Overview](#model-overview)
- [Pre-Training](#pre-training)
  - [Reference Script](#reference-script)
  - [Training the Model](#training-the-model)
  - [Multicard Training](#multicard-training)
- [Fine Tuning](#fine-tuning)
  - [Reference Script](#reference-script-1)
  - [Training the Model](#training-the-model-1)
  - [Multicard Training](#multicard-training-1)
- [Training Script Modifications](#training-script-modifications)
  - [BERT Large Pre-training](#bert-large-pre-training)
    - [Known Issues](#known-issues)
  - [BERT Base and BERT Large Fine Tuning](#bert-base-and-bert-large-fine-tuning)
    - [Known Issues](#known-issues-1)

# BERT for PyTorch

This folder contains scripts to pre-train and fine-tune BERT model on Habana Gaudi<sup>TM</sup> device to achieve state-of-the-art accuracy. Please visit [this page](https://developer.habana.ai/resources/habana-training-models/#performance) for performance information.

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

The BERT demos included in this release are as follows:

## BERT Pre-Training
- BERT Large pre-training for BF16 mixed precision for Wikipedia BookCorpus and Wiki dataset in Lazy mode.
- Multi card (1 server = 8 cards) suuport for BERT Large Pretraining with BF16 Mixed precision in Lazy mode

## BERT Fine-Tuning
### SQuAD
- BERT Large fine-tuning for FP32 and BF16 Mixed precision for SQuADv1.1 dataset in Lazy mode.
- Multi card (1 server = 8 cards) support for BERT-Large Fine tuning with FP32 and BF16 Mixed precision in Lazy mode.
- BERT Base fine-tuning for FP32 with SQuAD dataset in Eager mode.

### MRPC
- BERT Large fine-tuning with MRPC dataset for FP32 and BF16 Mixed precision in Lazy mode.
- BERT Base fine-tuning for FP32 with MRPC dataset in Eager mode.


The Demo script is a wrapper for respective python training scripts. Additional environment variables are used in training scripts in order to achieve optimal results for each workload.

# Setup
Please follow the instructions given in the following link for setting up the
environment including the `$PYTHON` environment variable: [Gaudi Setup and
Installation Guide](https://github.com/HabanaAI/Setup_and_Install). Please
answer the questions in the guide according to your preferences. This guide will
walk you through the process of setting up your system to run the model on
Gaudi.

In the docker container, clone this repository and switch to the branch that
matches your SynapseAI version. (Run the
[`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.)

```
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```

## Pre-training dataset preparation

`Model-References/PyTorch/nlp/bert/pretraining/data` provides scripts to download, extract and preprocess [Wikipedia](https://dumps.wikimedia.org/) and [BookCorpus](http://yknzhu.wixsite.com/mbweb) datasets.
Install the required Python packages in the container:
```
 pip install -r Model-References/PyTorch/nlp/bert/pretraining/requirements.txt
```
Then, go to `data` folder and run the data preparation script.
```
cd Model-References/PyTorch/nlp/bert/pretraining/data
```
So it is recommended to download wiki data set alone using the following command.
```
bash create_datasets_from_start.sh
```
Wiki and BookCorpus data sets can be downloaded by runnining the script as follows.
```
bash create_datasets_from_start.sh wiki_books
```
Note that the pretraining dataset is huge and takes several hours to download. BookCorpus may have access and download constraints. The final accuracy may vary depending on the dataset and its size.
The script creates formatted dataset for the phase1 and 2 of pre-training.

## Fine-tuning dataset preparation

### MRPC and SQuADv1.1 dataset preparation
Public datasets available on the datasets hub at https://github.com/huggingface/datasets
Based on the finetuning task name the dataset will be downloaded automatically from the datasets Hub.

## Model Overview
Bidirectional Encoder Representations from Transformers (BERT) is a technique for natural language processing (NLP) pre-training developed by Google.
The original English-language BERT model comes with two pre-trained general types: (1) the BERTBASE model, a 12-layer, 768-hidden, 12-heads, 110M parameter neural network architecture, and (2) the BERTLARGE model, a 24-layer, 1024-hidden, 16-heads, 340M parameter neural network architecture; both of which were trained on the BooksCorpus with 800M words, and a version of the English Wikipedia with 2,500M words.
The Pretraining modeling scripts are derived from a clone of https://github.com/NVIDIA/DeepLearningExamples.git and the fine tuning is based on https://github.com/huggingface/transformers.git.


# Pre-Training
- Located in: `Model-References/PyTorch/nlp/bert/pretraining`
- Suited for datasets:
  - `wiki`, `bookswiki`(combination of BooksCorpus and Wiki datasets)
- Uses optimizer: **LAMB** ("Layer-wise Adaptive Moments optimizer for Batch training").
- Consists of 2 phases:
  - Task 1 - **Masked Language Model** - where given a sentence, a randomly chosen word is guessed.
  - Task 2 - **Next Sentence Prediction** - where the model guesses whether sentence B comes after sentence A
- The resulting (trained) model weights are language-specific (here: english) and has to be further "fitted" to do a specific task (with finetuning).
- Heavy-weight: the training takes several hours or days.

BERT training script supports pre-training of  dataset on BERT large for both FP32 and BF16 mixed precision data type using **Lazy mode**.

## Reference Script
The base training and modeling scripts for pretraining are based on a clone of
https://github.com/NVIDIA/DeepLearningExamples.

## Training the Model
Clone the Model-References git.
Set up the data set as mentioned in the section "Set up dataset".

```
cd Model-References/PyTorch/nlp/bert
```
Run `$PYTHON demo_bert.py pretraining -h` for command-line options.

You can use Python launcher of `habana_model_runner.py` located in `Model-References/central` folder to
launch the training for the specified model.


i. lazy mode, bf16 mixed precision, BS64 for phase1 and BS8 for phase2:
```
$PYTHON demo_bert.py pretraining --model_name_or_path large --mode lazy --data_type bf16 --batch_size 64 8 --accumulate_gradients --allreduce_post_accumulation --allreduce_post_accumulation_fp16 --data_dir <dataset_path_phase1> <dataset_path_phase2>
```

ii. lazy mode, fp32 precision, BS32 for phase1 and BS4 for phase2:
```
$PYTHON demo_bert.py pretraining --model_name_or_path large --mode lazy --data_type fp32 --batch_size 32 4 --accumulate_gradients --allreduce_post_accumulation --allreduce_post_accumulation_fp16 --data_dir <dataset_path_phase1> <dataset_path_phase2>
```

## Multicard Training
Follow the relevant steps under "Training the Model".
To run multi-card demo, make sure the host machine has 512 GB of RAM installed. Modify the docker run command to pass 8 Gaudi cards to the docker container. This ensures the docker has access to all the 8 cards required for multi-card demo.

Use the following commands to run multicard training on 8 cards:


i. lazy mode, bf16 mixed precision, per chip batch size of 64 for phase1 and 8 for phase2:
```
$PYTHON demo_bert.py pretraining --model_name_or_path large --mode lazy --data_type bf16 --world_size 8 --batch_size 64 8 --accumulate_gradients --allreduce_post_accumulation --allreduce_post_accumulation_fp16 --dist --data_dir <dataset_path_phase1> <data_path_phase2>
```

ii. lazy mode, fp32 precision, per chip batch size of 32 for phase1 and 4 for phase2:
```
$PYTHON demo_bert.py pretraining --model_name_or_path large --mode lazy --data_type fp32 --world_size 8 --batch_size 32 4  --accumulate_gradients --allreduce_post_accumulation --allreduce_post_accumulation_fp16 --dist --data_dir <dataset_path_phase1> <data_path_phase2>
```

# Fine Tuning
- Located in: `Model-References/PyTorch/nlp/bert/transformers/`
- Suited for tasks:
  - `mrpc`: Microsoft Research Paraphrase Corpus (**MRPC**) is a paraphrase identification dataset, where systems aim to identify if two sentences are paraphrases of each other.
  - `squad`: Stanford Question Answering Dataset (**SQuAD**) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.
- Uses optimizer: **AdamW** ("ADAM with Weight Decay Regularization").
- Based on model weights trained with pretraining.
- Light-weight: the training takes a minute or so.
- Datasets for MRPC and SQuAD will be automatically downloaded the first time the model is run in the docker container.

The BERT demo uses training scripts and models from https://github.com/huggingface/transformers.git (tag v4.8.2)

## Reference Script
The training script fine-tunes BERT base and large model on the [Microsoft Research Paraphrase Corpus](https://gluebenchmark.com/) (MRPC) and [Stanford Question Answering Dataset](https://rajpurkar.github.io/SQuAD-explorer/) (SQuADv1.1) dataset.


## Training the Model
i. Fine-tune BERT base (Eager mode)

- Run BERT base fine-tuning on the GLUE MRPC dataset using FP32 data type:
```
$PYTHON demo_bert.py finetuning --model_name_or_path base --mode eager --dataset_name mrpc --task_name mrpc --data_type fp32 --num_train_epochs 3 --batch_size 32 --max_seq_length 128 --learning_rate 2e-5 --do_eval
```
- Run BERT base fine-tuning on the SQuAD dataset using FP32 data type:
```
$PYTHON demo_bert.py finetuning --model_name_or_path base --mode eager --dataset_name squad --task_name squad --data_type fp32 --num_train_epochs 2 --batch_size 12 --max_seq_length 384 --learning_rate 3e-5 --do_eval
```


ii. Fine-tune BERT large (Eager mode)

- Run BERT Large fine-tuning on the MRPC dataset with FP32:
```
$PYTHON demo_bert.py finetuning --model_name_or_path large --mode eager --dataset_name mrpc --task_name mrpc --data_type fp32 --num_train_epochs 3 --batch_size 32 --max_seq_length 128 --learning_rate 3e-5 --do_eval
```
- Run BERT Large fine-tuning on the SQuAD dataset with FP32:
```
$PYTHON demo_bert.py finetuning --model_name_or_path large --mode eager --dataset_name squad --task_name squad --data_type fp32 --num_train_epochs 2 --batch_size 10 --max_seq_length 384 --learning_rate 3e-5 --do_eval
```

iii. Fine-tune BERT large- SQuAD (Lazy mode)

- Run BERT Large fine-tuning on the SQuAD dataset using BF16 mixed precision:
```
$PYTHON demo_bert.py finetuning --model_name_or_path large --mode lazy --dataset_name squad --task_name squad --data_type bf16 --num_train_epochs 2 --batch_size 24 --max_seq_length 384 --learning_rate 3e-5 --do_eval

```
- Run BERT Large fine-tuning on the SQuAD dataset using FP32 data type:
```
$PYTHON demo_bert.py finetuning --model_name_or_path large --mode lazy --dataset_name squad --task_name squad --data_type fp32 --num_train_epochs 2 --batch_size 10 --max_seq_length 384 --learning_rate 3e-5 --do_eval
```


iv. Fine-tune BERT large - MRPC (Lazy mode)
- Run BERT Large fine-tuning on the MRPC dataset using BF16 data type:
```
$PYTHON demo_bert.py finetuning --model_name_or_path large --mode lazy --dataset_name mrpc --task_name mrpc --data_type bf16 --num_train_epochs 3 --batch_size 32 --max_seq_length 128 --learning_rate 3e-5 --do_eval
```

- Run BERT Large fine-tuning on the MRPC dataset using FP32 data type:
```
$PYTHON demo_bert.py finetuning --model_name_or_path large --mode lazy --dataset_name mrpc --task_name mrpc --data_type fp32 --num_train_epochs 3 --batch_size 32 --max_seq_length 128 --learning_rate 3e-5 --do_eval
```


## Multicard Training
To run multi-card demo, make sure the host machine has 512 GB of RAM installed. Modify the docker run command to pass 8 Gaudi cards to the docker container. This ensures the docker has access to all the 8 cards required for multi-card demo. Number of cards can be configured using --world_size option in the demo script.

Use the following command to run the multicard demo on 8 cards (1 server) for bf16, BS24 Lazy mode:
```
$PYTHON demo_bert.py finetuning --model_name_or_path large --mode lazy --dataset_name squad --task_name squad --data_type bf16 --num_train_epochs 2 --batch_size 24 --max_seq_length 384 --learning_rate 3e-5 --do_eval --world_size 8
```
Use the following command to run the multicard demo on 8 cards (1 server) for fp32, BS10 Lazy mode:
```
$PYTHON demo_bert.py finetuning --model_name_or_path large --mode lazy --dataset_name squad --task_name squad --data_type fp32 --num_train_epochs 2 --batch_size 10 --max_seq_length 384 --learning_rate 3e-5 --do_eval --world_size 8
```


# Training Script Modifications
This section lists the training script modifications for the BERT models.

## BERT Large Pre-training
The following changes have been added to training & modeling scripts.

Modifications to the training script: (pretraining/run_pretraining.py)
1. Habana and CPU Device support.
2. Saving checkpoint: Bring tensors to CPU and save.
3. Pass position ids from training script to model.
4. int32 type instead of Long for input_ids, segment ids, position_ids and input_mask.
5. Habana BF16 Mixed precision support.
6. Use Python version of LAMB optimizer (from lamb.py).
7. Data loader changes include single worker, no pinned memory and skip last batch.
8. Conditional import of Apex modules.
9. Support for distributed training on Habana device.
10. Use Fused LAMB optimizer.
11. Loss computation brought outside of modeling script (pretraining/run_pretraining.py, pretraining/modeling.py).
12. Modified training script to use mpirun for distributed training. Introduced mpi barrier to sync the processes.
13. Default allreduce bucket size set to 230MB for better performance in distributed training.
14. Added changes to support Lazy mode with required mark_step().
15. Added support to use distributed all_reduce from training script instead of default Distributed Data Parallel.
16. All required enviornmental variables brought under training script for ease of usage.
17. Added changes to calculate the performance per step and report through dllogger.
18. Added support for lowering print frequency of loss and associated this with log_freq.
19. Changes for dynamic loading of HCL library.

Modifications to the modeling script: (pretraining/modeling.py)
1. On Non-Cuda devices, use the conventional linear and activation functions instead of combined linear activation.
2. On Non-Cuda devices, use conventional nn.Layernorm instead of fused layernorm or layernorm using discrete ops.
3. Set embedding padding index to 0 explicitly.
4. Take position ids from training script rather than creating in the model.
5. Alternate select op implementation using index select and squeeze.
6. Rewrote permute and view as flatten.

### Known Issues
1. Placing mark_step() arbitrarily may lead to undefined behaviour. Recommend to keep mark_step() as shown in provided scripts.



## BERT Base and BERT Large Fine Tuning
The following changes have been added to scripts & source:
Modifications for transformer source (transformers/src/transformers dir):

1. Added Habana Device support (training_args.py,trainer.py).
2. Modifications for saving checkpoint: Bring tensors to CPU and save (trainer_pt_utils.py,trainer.py).
3. Introduced Habana BF16 Mixed precision (training_args.py,trainer.py).
4. Change for supporting HMP disable for optimizer.step (trainer.py).
5. Use fused AdamW optimizer on Habana device (training_args.py,trainer.py).
6. Use fused clip norm for grad clipping on Habana device(training_args.py,trainer.py).
7. Modified training script to use mpirun for distributed training(training_args.py).
8. Gradients are used as views using gradient_as_bucket_view(trainer.py).
9. Changes to optimize grad accumulation and zeroing of grads during the backward pass(trainer.py).
10. Default allreduce bucket size set to 230MB for better performance in distributed training(trainer.py).
11. Added changes to support Lazy mode with required mark_step(trainer.py).
12. Changes for dynamic loading of HCL library(training_args.py).
13. All required enviornmental variables brought under training script for ease of usage(trainer.py).
14. Used div operator variant that takes both inputs as float (models/bert/modeling_bert.py).


### Known Issues
1. Placing mark_step() arbitrarily may lead to undefined behaviour. Recommend to keep mark_step() as shown in provided scripts.
