# BERT for PyTorch with GPU Migration
This folder contains scripts to pre-train BERT model on Habana Gaudi device to achieve state-of-the-art accuracy. To obtain model performance data, refer to the [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).
 
The model has been enabled using an experimental feature called GPU Migration. 

*NOTE:* You can review [BERT model](../../../../nlp/bert/README.md) enabled with a more traditional approach.


For more information about training deep learning models using Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

**Note**: BERT is enabled on both Gaudi and Gaudi2.
## Table of Contents
- [Model References](../../../../README.md)
- [Model Overview](#model-overview)
- [Setup](#setup)
- [Training and Examples](#training-and-examples)
- [Pre-trained Model](#pre-trained-model)
- [Supported Configurations](#supported-configurations)
- [Changelog](#changelog)
- [Known Issues](#known-issues)
- [Enabling the Model from Scratch](#enabling-the-model-from-scratch)
- [GPU Migration Logs](#gpu-migration-logs)

## Model Overview
Bidirectional Encoder Representations from Transformers (BERT) is a technique for natural language processing (NLP) pre-training developed by Google. 

The original English-language BERT model comes with two pre-trained general types:

- The BERTBASE model, which is a 12-layer, 768-hidden, 12-heads, 110M parameter neural network architecture.
- The BERTLARGE model, which is a 24-layer, 1024-hidden, 16-heads, 340M parameter neural network architecture.
- Both models were trained on the BooksCorpus with 800M words, and a version of the English Wikipedia with 2,500M words. The base training and modeling scripts for pretraining are based on [NVIDIA Deep Learning Examples for Tensor Cores](https://github.com/NVIDIA/DeepLearningExamples.git).

The scripts included in this release are as follows:
- BERT Large pre-training for BF16 mixed precision for Wikipedia BookCorpus and Wiki dataset in Lazy mode.
- Multi-card (1 server = 8 cards) support for BERT Large pre-training with BF16 mixed precision in Lazy mode.

Enabling model functionality is made easy by GPU migration. While some performance optimizations are usually still required, GPU migration handles several steps required. 

The following is a list of the different advantages of using GPU migration when compared to [the other model](../../../../nlp/bert/README.md):

* Modyfing torch.cuda calls is not required.
* Changing FP16 to BF16 dtype is not required.
* Adding support for Habana Mixed Precision (hmp) is not required.
* Adding support for FusedLamb optimizer is not required.

For further details, refer to [Enabling the Model from Scratch](#enabling-the-model-from-scratch).

### Pre-Training
- Located in: `Model-References/PyTorch/examples/gpu_migration/nlp/bert/`
- Suited for datasets:
  - `wiki`, `bookswiki`(combination of BooksCorpus and Wiki datasets)
- Uses optimizer: **LAMB** ("Layer-wise Adaptive Moments optimizer for Batch training").
- Consists of two tasks:
  - **Masked Language Model** - where given a sentence, a randomly chosen word is guessed.
  - **Next Sentence Prediction** - where the model guesses whether sentence B comes after sentence A.
- The resulting (trained) model weights are language-specific (in this case, it is English) and has to be further "fitted" to perform a specific task with fine-tuning. However, fine-tuning has not been enabled with GPU Migration.
- Heavy-weight: the training takes several hours or days.

*NOTE:* BERT training script supports pre-training of dataset on BERT large for both FP32 and BF16 mixed precision data type using **Lazy mode**.

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
1. In the docker container, go to the BERT directory
```bash
cd Model-References/PyTorch/examples/gpu_migration/nlp/bert
```
2. Install the required packages using pip:
```bash
$PYTHON -m pip install -r requirements.txt
```
3. Install [apex package for PyTorch](https://github.com/NVIDIA/apex):
```bash
export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5;8.0;8.6;9.0"
pip3 install git+https://github.com/NVIDIA/apex.git@6816ef6467
```

### Pre-training Dataset Preparation

`Model-References/PyTorch/examples/gpu_migration/nlp/bert/data` provides scripts to download, extract and pre-process [Wikipedia](https://dumps.wikimedia.org/) and [BookCorpus](http://yknzhu.wixsite.com/mbweb) datasets.

Go to the `data` folder and run the data preparation script.
```
cd Model-References/PyTorch/nlp/bert/data
```
It is highly recommended to download Wiki dataset alone using the following command.
```
bash create_datasets_from_start.sh
```
Wiki and BookCorpus datasets can be downloaded by running the script as follows.
```
bash create_datasets_from_start.sh wiki_books
```
*NOTE:* The pre-training dataset is huge and takes several hours to download. BookCorpus may have access and download constraints. The final accuracy may vary depending on the dataset and its size.
The script creates a formatted dataset for Phase 1 and Phase 2 of the pre-training.

### Packing the Data
Habana supports using a [Data packing technique](https://github.com/HabanaAI/Gaudi-tutorials/blob/main/TensorFlow/DataPackingMLperfBERT/Data_Packing_Process_for_MLPERF_BERT.ipynb),
called Non-Negative Least Squares Histogram. Here, instead of padding with zero,
several short sequences are packed into one multi-sequence of size `max_seq_len`.
Thus, this removes most of the padding, which can lead to a speedup of up to 2&times;
in time-to-train (TTT). This packing technique can be applied on other datasets
with high variability in samples length.

*NOTE:* For each NLP dataset with sequential data samples, the speedup
with data packing is determined by the ratio of `max_seq_len` to
`average_seq_len` in that particular dataset. The larger the ratio, the higher
the speedup.

To pack the dataset, in docker run:
```bash
cd /root/Model-References/PyTorch/nlp/bert

$PYTHON pack_pretraining_data_pytorch.py --input_dir <dataset_path_phase1> --output-dir <packed_dataset_path_phase1> --max_sequence_length 128 --max_predictions_per_sequence 20

$PYTHON pack_pretraining_data_pytorch.py --input_dir <dataset_path_phase2> --output-dir <packed_dataset_path_phase2> --max_sequence_length 512 --max_predictions_per_sequence 80
```
**Note:** This will generate json at the path <output-dir>/../<tail_dir>_metadata.json with meta data info like: "avg_seq_per_sample" etc. This json will be
used as an input to run_pretraining.py to extract "avg_seq_per_sample" in case of packed dataset mode.


## Training and Examples

Please create a log directory to store `dllogger.json` and specify its location for `--json_summary` attribute.

### Single Card and Multi-Card Training Examples

**Run training on 8 HPUs:**

To run multi-card demo, make sure the host machine has 512 GB of RAM installed. Modify the docker run command to pass 8 Gaudi cards to the docker container. This ensures the docker has access to all the 8 cards required for multi-card demo.


- Lazy mode, 8 HPUs, BF16 mixed precision (through --fp16 flag), per chip batch size of 64 for Phase 1 and 8 for Phase 2:
```bash
$PYTHON -m torch.distributed.launch \
 --nproc_per_node=8 run_pretraining.py \
 --input_dir=/data/pytorch/bert_pretraining/hdf5_lower_case_1_seq_len_128/books_wiki_en_corpus \
 --output_dir=/tmp/results/checkpoints/ \
 --config_file=bert_config.json \
 --bert_model=bert-large-uncased \
 --train_batch_size=8192 --max_seq_length=128 \
 --max_predictions_per_seq=20 --max_steps=7038 \
 --warmup_proportion=0.2843 --num_steps_per_checkpoint=200 \
 --learning_rate=6e-3 --fp16 --gradient_accumulation_steps=128 \
 --do_train --json-summary /tmp/log_directory/dllogger.json \
 --allreduce_post_accumulation --allreduce_post_accumulation_fp16
```

```bash
$PYTHON-m torch.distributed.launch \
 --nproc_per_node=8 run_pretraining.py \
 --do_train --bert_model=bert-large-uncased \
 --config_file=./bert_config.json \
 --allreduce_post_accumulation \
 --allreduce_post_accumulation_fp16 \
 --json-summary=/tmp/log_directory/dllogger.json \
 --output_dir=/tmp/results/checkpoints/ \
 --input_dir=/data/pytorch/bert/pretraining/hdf5_lower_case_1_seq_len_512/books_wiki_en_corpus/ \
 --train_batch_size=4096 \
 --max_seq_length=512 --max_predictions_per_seq=80 \
 --warmup_proportion=0.128 --max_steps=1563 \
 --num_steps_per_checkpoint=200 --learning_rate=0.004 \
 --gradient_accumulation_steps=512 \
 --phase1_end_step=7038 --phase2 --fp16
```

- Lazy mode, 8 HPUs, FP32 precision, per chip batch size of 32 for Phase 1 and 4 for Phase 2:

```bash
$PYTHON -m torch.distributed.launch \
 --nproc_per_node=8 run_pretraining.py \
 --input_dir=/data/pytorch/bert_pretraining/hdf5_lower_case_1_seq_len_128/books_wiki_en_corpus \
 --output_dir=/tmp/results/checkpoints/ \
 --config_file=bert_config.json \
 --bert_model=bert-large-uncased \
 --train_batch_size=8192 --max_seq_length=128 \
 --max_predictions_per_seq=20 --max_steps=7038 \
 --warmup_proportion=0.2843 --num_steps_per_checkpoint=200 \
 --learning_rate=6e-3 --gradient_accumulation_steps=256 \
 --do_train --json-summary /tmp/log_directory/dllogger.json \
 --allreduce_post_accumulation --allreduce_post_accumulation_fp16
```

```bash
$PYTHON-m torch.distributed.launch \
 --nproc_per_node=8 run_pretraining.py \
 --do_train --bert_model=bert-large-uncased \
 --config_file=./bert_config.json \
 --allreduce_post_accumulation \
 --allreduce_post_accumulation_fp16 \
 --json-summary=/tmp/log_directory/dllogger.json \
 --output_dir=/tmp/results/checkpoints/ \
 --input_dir=/data/pytorch/bert/pretraining/hdf5_lower_case_1_seq_len_512/books_wiki_en_corpus/ \
 --train_batch_size=4096 \
 --max_seq_length=512 --max_predictions_per_seq=80 \
 --warmup_proportion=0.128 --max_steps=1563 \
 --num_steps_per_checkpoint=200 --learning_rate=0.004 \
 --gradient_accumulation_steps=512 \
 --phase1_end_step=7038 --phase2 --fp16
```

- Using packed data: lazy mode, 8 HPUs, BF16 mixed precision (through --fp16 flag), per chip batch size of 64 for Phase 1 and 8 for Phase 2:

```bash
$PYTHON -m torch.distributed.launch \
 --nproc_per_node=8 run_pretraining.py \
 --input_dir=/data/pytorch/bert_pretraining/hdf5_lower_case_1_seq_len_128/books_wiki_en_corpus_packed \
 --output_dir=/tmp/results/checkpoints/ \
 --config_file=bert_config.json \
 --bert_model=bert-large-uncased \
 --train_batch_size=8192 --max_seq_length=128 \
 --max_predictions_per_seq=20 --max_steps=7038 \
 --warmup_proportion=0.2843 --num_steps_per_checkpoint=200 \
 --learning_rate=6e-3 --fp16 --gradient_accumulation_steps=128 \
 --do_train --json-summary /tmp/log_directory/dllogger.json \
 --allreduce_post_accumulation --allreduce_post_accumulation_fp16
```

```bash
$PYTHON-m torch.distributed.launch \
 --nproc_per_node=8 run_pretraining.py \
 --do_train --bert_model=bert-large-uncased \
 --config_file=./bert_config.json \
 --allreduce_post_accumulation \
 --allreduce_post_accumulation_fp16 \
 --json-summary=/tmp/log_directory/dllogger.json \
 --output_dir=/tmp/results/checkpoints/ \
 --input_dir=/data/pytorch/bert/pretraining/hdf5_lower_case_1_seq_len_512/books_wiki_en_corpus_packed \
 --train_batch_size=4096 \
 --max_seq_length=512 --max_predictions_per_seq=80 \
 --warmup_proportion=0.128 --max_steps=1563 \
 --num_steps_per_checkpoint=200 --learning_rate=0.004 \
 --gradient_accumulation_steps=512 \
 --phase1_end_step=7038 --phase2 --fp16
```

- Using packed data: lazy mode, 8 HPUs, BF16 mixed precision (through --fp16 flag), per chip batch size of 64 for Phase 1 and 16 for Phase 2 on **Gaudi2**:

```bash
$PYTHON -m torch.distributed.launch \
 --nproc_per_node=8 run_pretraining.py \
 --input_dir=/data/pytorch/bert_pretraining/hdf5_lower_case_1_seq_len_128/books_wiki_en_corpus_packed \
 --output_dir=/tmp/results/checkpoints/ \
 --config_file=bert_config.json \
 --bert_model=bert-large-uncased \
 --train_batch_size=8192 --max_seq_length=128 \
 --max_predictions_per_seq=20 --max_steps=7038 \
 --warmup_proportion=0.2843 --num_steps_per_checkpoint=200 \
 --learning_rate=6e-3 --fp16 --gradient_accumulation_steps=128 \
 --do_train --json-summary /tmp/log_directory/dllogger.json \
 --allreduce_post_accumulation --allreduce_post_accumulation_fp16
```

```bash
$PYTHON-m torch.distributed.launch \
 --nproc_per_node=8 run_pretraining.py \
 --do_train --bert_model=bert-large-uncased \
 --config_file=./bert_config.json \
 --allreduce_post_accumulation \
 --allreduce_post_accumulation_fp16 \
 --json-summary=/tmp/log_directory/dllogger.json \
 --output_dir=/tmp/results/checkpoints/ \
 --input_dir=/data/pytorch/bert/pretraining/hdf5_lower_case_1_seq_len_512/books_wiki_en_corpus_packed \
 --train_batch_size=8192 \
 --max_seq_length=512 --max_predictions_per_seq=80 \
 --warmup_proportion=0.128 --max_steps=1563 \
 --num_steps_per_checkpoint=200 --learning_rate=0.004 \
 --gradient_accumulation_steps=512 \
 --phase1_end_step=7038 --phase2 --fp16
```

## Supported Configurations

| Device | SynapseAI Version | PyTorch Version | Mode |
|-----|-----|-----|------|
| Gaudi  | 1.9.0 | 1.13.1 | Training |
| Gaudi2 | 1.9.0 | 1.13.1 | Training |

## Changelog

### 1.9.0
- Added `import habana_frameworks.torch.gpu_migration` and `htcore.mark_step()` to run_pretraining.py
- Changed README.
- Added packed dataset support.
- Added performance optimizations:
  - Utilized fused F.gelu(x).
  - Overlap host execution with device.
  - Reduce host execution jiter.

## Enabling the Model from Scratch
Habana provides scripts ready-to-use on Gaudi. Listed below are the steps to enable the model from a reference source.

This section outlines the overall procedure for enabling any given model with GPU migration feature. However, model-specific modifications will be required to enable the functionality and improve performance

1. Clone the original GitHub repository and reset it to the commit this example is based on.
```bash
git clone https://github.com/NVIDIA/DeepLearningExamples.git && cd DeepLearningExamples && git reset --hard 7a4c42501ce05ac5b76999e3b9ddadeffd177b1d
```
2. Navigate to BERT subfolder and install requirements:
```bash
cd PyTorch/LanguageModeling/BERT/
pip install -r requirements.txt
export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5;8.0;8.6;9.0"
pip3 install git+https://github.com/NVIDIA/apex.git@6816ef6467
```
3.  Apply a set of patches. You can stop at any patch to see which steps have been performed to reach a particular level of functionality and performance.

The first patch adds the bare minimum to run the model on HPU. For purely functional changes (without performance optimization), run the following command:
```bash
git apply Model-References/PyTorch/examples/gpu_migration/nlp/bert/patches/minimal_changes.diff
```

4. To improve performance, apply the patch which adds packed dataset support.
```bash
git apply Model-References/PyTorch/examples/gpu_migration/nlp/bert/patches/use_packed_dataset.diff
```

5. You can further improve performance by applying the following set of optimizations:
```bash
git apply Model-References/PyTorch/examples/gpu_migration/nlp/bert/patches/performance_improvements.diff
```
At any of those stages, you can use the commands provided in the [examples section](#training-and-examples) to run the training.

## GPU Migration Logs
You can review GPU Migration logs under [gpu_migration_logs/gpu_migration_5494.log](gpu_migration_logs/gpu_migration_5494.log).
For further information, refer to [GPU Migration Toolkit documentation](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Model_Porting/GPU_Migration_Toolkit/GPU_Migration_Toolkit.html#enabling-logging-feature).