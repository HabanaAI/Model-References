# Table of Contents
- [BERT for PyTorch](#bert-for-pytorch)
  - [BERT Pre-Training](#bert-pre-training)
  - [BERT Fine-Tuning](#bert-fine-tuning)
    - [SQuAD](#squad)
    - [MRPC](#mrpc)
- [Setup](#setup)
  - [Pre-training dataset preparation](#pre-training-dataset-preparation)
  - [Fine-tuning dataset preparation](#fine-tuning-dataset-preparation)
    - [MRPC dataset preparation](#mrpc-dataset-preparation)
    - [SQuADv1.1 dataset preparation](#squadv11-dataset-preparation)
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
- BERT Large pre-training for BF16 mixed precision for Wikipedia BookCorpus and Wiki dataset in Graph and Lazy mode.
- Multi card (1 server = 8 cards) suuport for BERT Large Pretraining with BF16 Mixed precision in Graph and Lazy mode

## BERT Fine-Tuning
### SQuAD
- BERT Large fine-tuning for FP32 and BF16 Mixed precision for SQuADv1.1 dataset in Graph and Lazy mode.
- Multi card (1 server = 8 cards) support for BERT-Large Fine tuning with FP32 and BF16 Mixed precision in Graph and Lazy mode.
- BERT Base fine-tuning for FP32 with SQuAD dataset in Eager mode.

### MRPC
- BERT Large fine-tuning with MRPC dataset for FP32 and BF16 Mixed precision in Graph mode.
- BERT Base fine-tuning for FP32 with MRPC dataset in Eager mode.


Graph mode is supported using torch.jit.trace with check_trace=False.

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

### MRPC dataset preparation
MRPC dataset can be downloaded using download_glue_data.py script. Download python script from the following link and run the script to download data into `glue_data` directory.
```
mkdir -p $HOME/datasets/glue_data
cd $HOME/datasets/
wget https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py
$PYTHON download_glue_data.py --data_dir glue_data --tasks MRPC
```

### SQuADv1.1 dataset preparation
The data for SQuAD can be downloaded with the following links and should be saved into a directory. Specify path of the directory to demo script.
```
mkdir -p $HOME/datasets/Squad
cd $HOME/datasets/Squad
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
wget https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py
```
The pre-trained model will be downloaded the first time the demo is launched provided access to Internet is guaranteed.

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

BERT training script supports pre-training of  dataset on BERT large for both FP32 and BF16 mixed precision data type using **Graph and Lazy mode**.

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

i. graph mode, bf16 mixed precision, BS64 for phase1 and BS8 for phase2:
```
$PYTHON demo_bert.py pretraining --model_name_or_path large --mode graph --data_type bf16 --batch_size 64 8 --accumulate_gradients --allreduce_post_accumulation --allreduce_post_accumulation_fp16 --data_dir <dataset_path_phase1> <dataset_path_phase2>
```
ii. graph mode, fp32 precision, BS32 for phase1 and BS4 for phase2:
```
$PYTHON demo_bert.py pretraining --model_name_or_path large --mode graph --data_type fp32 --batch_size 32 4 --accumulate_gradients --allreduce_post_accumulation --allreduce_post_accumulation_fp16 --data_dir <dataset_path_phase1> <dataset_path_phase2>
```

iii. lazy mode, bf16 mixed precision, BS64 for phase1 and BS8 for phase2:
```
$PYTHON demo_bert.py pretraining --model_name_or_path large --mode lazy --data_type bf16 --batch_size 64 8 --accumulate_gradients --allreduce_post_accumulation --allreduce_post_accumulation_fp16 --data_dir <dataset_path_phase1> <dataset_path_phase2>
```

iv. lazy mode, fp32 precision, BS32 for phase1 and BS4 for phase2:
```
$PYTHON demo_bert.py pretraining --model_name_or_path large --mode graph --data_type fp32 --batch_size 32 4 --accumulate_gradients --allreduce_post_accumulation --allreduce_post_accumulation_fp16 --data_dir <dataset_path_phase1> <dataset_path_phase2>
```

## Multicard Training
Follow the relevant steps under "Training the Model".
To run multi-card demo, make sure the host machine has 512 GB of RAM installed. Modify the docker run command to pass 8 Gaudi cards to the docker container. This ensures the docker has access to all the 8 cards required for multi-card demo.

Use the following commands to run multicard training on 8 cards:

i. graph mode, bf16 mixed precision, per chip batch size of 64 for phase1 and 8 for phase2:
```
$PYTHON demo_bert.py pretraining --model_name_or_path large --mode graph --data_type bf16 --world_size 8 --batch_size 64 8 --accumulate_gradients --allreduce_post_accumulation --allreduce_post_accumulation_fp16 --dist --data_dir <dataset_path_phase1> <dataset_path_phase2>
```

ii. graph mode, fp32 precision, per chip batch size of 32 for phase1 and 4 for phase2:
```
$PYTHON demo_bert.py pretraining --model_name_or_path large --mode graph --data_type fp32 --world_size 8 --batch_size 32 4  --accumulate_gradients --allreduce_post_accumulation --allreduce_post_accumulation_fp16 --dist --data_dir <dataset_path_phase1> <dataset_path_phase2>
```

i. lazy mode, bf16 mixed precision, per chip batch size of 64 for phase1 and 8 for phase2:
```
$PYTHON demo_bert.py pretraining --model_name_or_path large --mode lazy --data_type bf16 --world_size 8 --batch_size 64 8 --accumulate_gradients --allreduce_post_accumulation --allreduce_post_accumulation_fp16 --dist --data_dir <dataset_path_phase1> <data_path_phase2>
```

ii. lazy mode, fp32 precision, per chip batch size of 32 for phase1 and 4 for phase2:
```
$PYTHON demo_bert.py pretraining --model_name_or_path large --mode lazy --data_type fp32 --world_size 8 --batch_size 32 4  --accumulate_gradients --allreduce_post_accumulation --allreduce_post_accumulation_fp16 --dist --data_dir <dataset_path_phase1> <data_path_phase2>
```

# Fine Tuning
- Located in: `Model-References/PyTorch/nlp/bert/finetuning`
- Suited for tasks:
  - `mrpc`: Microsoft Research Paraphrase Corpus (**MRPC**) is a paraphrase identification dataset, where systems aim to identify if two sentences are paraphrases of each other.
  - `squad`: Stanford Question Answering Dataset (**SQuAD**) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.
- Uses optimizer: **AdamW** ("ADAM with Weight Decay Regularization").
- Based on model weights trained with pretraining.
- Light-weight: the training takes a minute or so.
- Datasets for MRPC and SQuAD will be automatically downloaded the first time the model is run in the docker container.

The BERT demo uses training scripts and models from https://github.com/huggingface/transformers.git (tag v3.0.2)

## Reference Script
The training script fine-tunes BERT base and large model on the [Microsoft Research Paraphrase Corpus](https://gluebenchmark.com/) (MRPC) and [Stanford Question Answering Dataset](https://rajpurkar.github.io/SQuAD-explorer/) (SQuADv1.1) dataset.


## Training the Model
i. Fine-tune BERT base (Eager mode)

- Run BERT base fine-tuning on the GLUE MRPC dataset using FP32 data type:
```
$PYTHON demo_bert.py finetuning --model_name_or_path base --mode eager --task_name mrpc --data_type fp32 --num_train_epochs 3 --batch_size 32 --max_seq_length 128 --learning_rate 2e-5 --do_eval --data_dir <dataset_path>/MRPC
```
- Run BERT base fine-tuning on the SQuAD dataset using FP32 data type:
```
$PYTHON demo_bert.py finetuning --model_name_or_path base --mode eager --task_name squad --data_type fp32 --num_train_epochs 2 --batch_size 12 --max_seq_length 384 --learning_rate 3e-5 --do_eval --data_dir <dataset_path>/Squad
```


ii. Fine-tune BERT large (Eager mode)

- Run BERT Large fine-tuning on the MRPC dataset with FP32:
```
$PYTHON demo_bert.py finetuning --model_name_or_path large --mode eager --task_name mrpc --data_type fp32 --num_train_epochs 3 --batch_size 32 --max_seq_length 128 --learning_rate 3e-5 --do_eval --data_dir <dataset_path>/MRPC
```
- Run BERT Large fine-tuning on the SQuAD dataset with FP32:
```
$PYTHON demo_bert.py finetuning --model_name_or_path large --mode eager --task_name squad --data_type fp32 --num_train_epochs 2 --batch_size 10 --max_seq_length 384 --learning_rate 3e-5 --do_eval --data_dir <dataset_path>/Squad
```



iii. Fine-tune BERT large- SQuAD (Graph mode)

- Run BERT Large fine-tuning on the SQuAD dataset using BF16 mixed precision:
```
$PYTHON demo_bert.py finetuning --model_name_or_path large --mode graph --task_name squad --data_type bf16 --num_train_epochs 2 --batch_size 24 --max_seq_length 384 --learning_rate 3e-5 --do_eval --data_dir <dataset_path>/Squad
```
- Run BERT Large fine-tuning on the SQuAD dataset using FP32 data type:
```
$PYTHON demo_bert.py finetuning --model_name_or_path large --mode graph --task_name squad --data_type fp32 --num_train_epochs 2 --batch_size 10 --max_seq_length 384 --learning_rate 3e-5 --do_eval --data_dir <dataset_path>/Squad
```


iv. Fine-tune BERT large- SQuAD (Lazy mode)

- Run BERT Large fine-tuning on the SQuAD dataset using BF16 mixed precision:
```
$PYTHON demo_bert.py finetuning --model_name_or_path large --mode lazy --task_name squad --data_type bf16 --num_train_epochs 2 --batch_size 24 --max_seq_length 384 --learning_rate 3e-5 --do_eval --data_dir <dataset_path>/Squad
```
- Run BERT Large fine-tuning on the SQuAD dataset using FP32 data type:
```
$PYTHON demo_bert.py finetuning --model_name_or_path large --mode lazy --task_name squad --data_type fp32 --num_train_epochs 2 --batch_size 10 --max_seq_length 384 --learning_rate 3e-5 --do_eval --data_dir <dataset_path>/Squad
```


v. Fine-tune BERT large - MRPC (Graph mode)
- Run BERT Large fine-tuning on the MRPC dataset using FP32 data type:
```
$PYTHON demo_bert.py finetuning --model_name_or_path large --mode graph --task_name mrpc --data_type fp32 --num_train_epochs 3 --batch_size 32 --max_seq_length 128 --learning_rate 3e-5 --do_eval --data_dir <dataset_path>/MRPC
```


## Multicard Training
To run multi-card demo, make sure the host machine has 512 GB of RAM installed. Modify the docker run command to pass 8 Gaudi cards to the docker container. This ensures the docker has access to all the 8 cards required for multi-card demo. Number of cards can be configured using --world_size option in the demo script.

Use the following command to run the multicard demo on 8 cards (1 server) for bf16, BS24 Graph mode:
```
$PYTHON demo_bert.py finetuning --model_name_or_path large --task_name squad --mode graph --data_type bf16 --num_train_epochs 2 --world_size 8 --batch_size 24 --max_seq_length 384 --learning_rate 3e-5 --dist --do_eval --data_dir <dataset_path>/Squad
```
Use the following command to run the multicard demo on 8 cards (1 server) for fp32, BS10 Graph mode:
```
$PYTHON demo_bert.py finetuning --model_name_or_path large --task_name squad --mode graph --data_type fp32 --num_train_epochs 2 --world_size 8 --batch_size 10 --max_seq_length 384 --learning_rate 3e-5 --dist --do_eval --data_dir  <dataset_path>/Squad
```


Use the following command to run the multicard demo on 8 cards (1 server) for bf16, BS24 Lazy mode:
```
$PYTHON demo_bert.py finetuning --model_name_or_path large --task_name squad --mode lazy --data_type bf16 --num_train_epochs 2 --world_size 8 --batch_size 24 --max_seq_length 384 --learning_rate 3e-5 --dist --do_eval --data_dir <dataset_path>/Squad
```
Use the following command to run the multicard demo on 8 cards (1 server) for fp32, BS10 Lazy mode:
```
$PYTHON demo_bert.py finetuning --model_name_or_path large --task_name squad --mode lazy --data_type fp32 --num_train_epochs 2 --world_size 8 --batch_size 10 --max_seq_length 384 --learning_rate 3e-5 --dist --do_eval --data_dir  <dataset_path>/Squad
```


# Training Script Modifications
This section lists the training script modifications for the BERT models.

## BERT Large Pre-training
The following changes have been added to training & modeling scripts.

Modifications to the training script: (pretraining/run_pretraining.py)
1. Habana and CPU Device support.
2. Saving checkpoint: Bring tensors to CPU and save.
3. Torchscript jit trace mode support.
4. Pass position ids from training script to model.
5. int32 type instead of Long for input_ids, segment ids, position_ids and input_mask.
6. Habana BF16 Mixed precision support.
7. Use Python version of LAMB optimizer (from lamb.py).
8. Data loader changes include single worker, no pinned memory and skip last batch.
9. Conditional import of Apex modules.
10. Support for distributed training on Habana device.
11. Use Fused LAMB optimizer.
12. Loss computation brought outside of modeling script (pretraining/run_pretraining.py, pretraining/modeling.py).
13. Modified training script to use mpirun for distributed training. Introduced mpi barrier to sync the processes.
14. Default allreduce bucket size set to 230MB for better performance in distributed training.
15. Supports --tqdm_smoothing for controlling smoothing factor used for calculating iteration time.
16. Added changes to support Lazy mode with required mark_step().
17. Added support to use distributed all_reduce from training script instead of default Distributed Data Parallel.
18. All required enviornmental variables brought under training script for ease of usage.
19. Added changes to calculate the performance per step and report through dllogger.
20. Changes for dynamic loading of HCL library.

Modifications to the modeling script: (pretraining/modeling.py)
1. On Non-Cuda devices, use the conventional linear and activation functions instead of combined linear activation.
2. On Non-Cuda devices, use conventional nn.Layernorm instead of fused layernorm or layernorm using discrete ops.
3. Set embedding padding index to 0 explicitly.
4. Take position ids from training script rather than creating in the model.
5. Alternate select op implementation using index select and squeeze.
6. Rewrote permute and view as flatten to enable better fusion in Graph mode.
7. transpose_for_scores function modified to get a batchsize differently to enable the better fusion.

### Known Issues
1. Placing mark_step() arbitrarily may lead to undefined behaviour. Recommend to keep mark_step() as shown in provided scripts.
2. BERT Large Pre-training with FP32 data type is not fully tested for convergence in both Graph mode and Lazy Mode.



## BERT Base and BERT Large Fine Tuning
The following changes have been added to scripts & source:

Modifications to the example training scripts (finetuning/examples dir):

i. Modifications to the question-answering/run_squad.py:
1. Added Habana Device support.
2. Moved feature_index tensor to CPU.
3. Modifications for saving checkpoint: Bring tensors to CPU and save.
4. Modifications for adding support for Torchscript jit trace mode.
5. Used int32 type instead of Long for input_ids, position_ids attention_mask, start_positions and end_positions.
6. Distributed training : Use local barrier.
7. Introduced Habana BF16 Mixed precision to SQuAD script.
8. Use fused AdamW optimizer on Habana device.
9. Use fused clip norm for grad clipping on Habana device.
10. Modified training script to use mpirun for distributed training. Introduced mpi barrier to sync the processes.
11. Moved the params tensor list creation in clip norm wrapper to init func, so that list creation can be avoided in every iteration.
12. Gradients are used as views using gradient_as_bucket_view.
13. Changes for supporting HMP disable for optimizer.step().
14. Dropping the last batch if it is partial bacth size to effectively manage the memory.
15. Enabled the evaluation_during_training with fixes necessary.
16. Changes to optimize grad accumulation and zeroing of grads during the backward pass.
17. Default allreduce bucket size set to 230MB for better performance in distributed training.
18. Added changes to support Lazy mode with required mark_step().
19. Added support for lowering print frequency of loss and associated this with logging_steps.
20. All required enviornmental variables brought under training script for ease of usage.
21. Changes for dynamic loading of HCL library.

ii. Modifications to the text-classification/run_glue.py:
22. All required enviornmental variables brought under training script for ease of usage.


Modifications for transformer source (finetuning/src/transformers dir):

1. Added Habana Device support(training_args.py).
2. Introduced Habana BF16 Mixed precision to SQuAD script (training_args.py).
3. Distributed training : Use local barrier (trainer.py, training_args.py).
4. Distributed training : Use local barrier (trainer.py, training_args.py)
5. Checkpoint and tokenizer loading: Load optimizer from CPU; Load tokenizer from parent directory if available (tokenization_utils_base.py,trainer.py).
6. Distributed training : convert label_ids to int type (trainer.py).
7. Moved the params tensor list creation in clip norm wrapper to init func, so that list creation can be avoided in every iteration(trainer.py).
8. Used auto flush feature of summary writer instead explicit call (trainer.py).
9. Change for supporting HMP disable for optimizer.step (trainer.py).
10. Changes to optimize grad accumulation and zeroing of grads during the backward pass (trainer.py).
11. Modifications for saving checkpoint: Bring tensors to CPU and save (modeling_utils.py; trainer.py).
12. Alternate select op implementation using index select and squeeze (modeling_bert.py).
13. Used div operator variant that takes both inputs as float (modeling_bert.py).
14. Rewrote permute and view as flatten to enable better fusion in TorchScript trace mode (modeling_bert.py).
15. Used dummy tensor instead of ‘None’ for arguments like head_mask,inputs_embeds, start/end positions to be compatible with TorchScript trace mode (modeling_bert.py).
16. Alternate addcdiv implementation in AdamW using discrete ops to avoid scalar scale factor (optimization.py).



### Known Issues
1. MRPC finetuning: Final accuracy varies by 2% between different runs.
2. SQuAD finetuning: In Lazy mode logging_steps>1 is not fully functional. Will be fixed in next release.
3. Placing mark_step() arbitrarily may lead to undefined behaviour. Recommend to keep mark_step() as shown in provided scripts.
