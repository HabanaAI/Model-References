# BERT for PyTorch

This folder contains scripts to pre-train , finetune BERT model and run inference on finetuned BERT model on Intel® Gaudi® AI Accelerator to achieve state-of-the-art accuracy. To obtain model performance data, refer to the [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance)

For more information about training deep learning models using Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

**Note**: BERT is enabled on both Gaudi and Gaudi2.
## Table of Contents
- [Model References](../../../README.md)
- [Model Overview](#model-overview)
- [Setup](#setup)
- [Training and Examples](#training-and-examples)
- [Inference and Examples](#inference-and-examples)
- [Pre-trained Model](#pre-trained-model)
- [Supported Configurations](#supported-configurations)
- [Changelog](#changelog)
- [Known Issues](#known-issues)

## Model Overview
Bidirectional Encoder Representations from Transformers (BERT) is a technique for natural language processing (NLP) pre-training developed by Google.
The original English-language BERT model comes with two pre-trained general types: (1) the BERTBASE model, a 12-layer, 768-hidden, 12-heads, 110M parameter neural network architecture, and (2) the BERTLARGE model, a 24-layer, 1024-hidden, 16-heads, 340M parameter neural network architecture; both of which were trained on the BooksCorpus with 800M words, and a version of the English Wikipedia with 2,500M words.
The base training and modeling scripts for pre-training are based on a clone of https://github.com/NVIDIA/DeepLearningExamples.git and fine-tuning is based on https://github.com/huggingface/transformers.git.

The scripts included in this release are as follows:
- BERT Large pre-training for BF16 mixed precision for Wikipedia BookCorpus and Wiki dataset in Lazy mode.
- BERT Large finetuning for BF16 mixed precision for Wikipedia BookCorpus and SQUAD dataset in Lazy mode.
- Multi-card (1 server = 8 cards) support for BERT Large pre-training and finetuning with BF16 mixed precision in Lazy mode.
- Multi-server (4 servers = 32 cards) support for BERT Large pre-training with BF16 mixed precision in Lazy mode.
- BERT pre-training 1.2B parameters using ZeroRedundancyOptimizer with BF16 mixed precision in Lazy mode.


Additional environment variables are used in training scripts in order to achieve optimal results for each workload.

### Pre-Training
- Located in: `Model-References/PyTorch/nlp/bert/`
- Suited for datasets:
  - `wiki`, `bookswiki`(combination of BooksCorpus and Wiki datasets)
- Uses optimizer: **LAMB** ("Layer-wise Adaptive Moments optimizer for Batch training").
- Consists of two tasks:
  - Task 1 - **Masked Language Model** - where given a sentence, a randomly chosen word is guessed.
  - Task 2 - **Next Sentence Prediction** - where the model guesses whether sentence B comes after sentence A.
- The resulting (trained) model weights are language-specific (here: english) and has to be further "fitted" to do a specific task (with fine-tuning).
- Heavy-weight: the training takes several hours or days.

BERT training script supports pre-training of dataset on BERT large for both FP32 and BF16 mixed precision data type using **Lazy mode**.

### Finetuning
- Located in: `Model-References/PyTorch/nlp/bert/`
- Suited for dataset:
  - `SQUAD`(Stanford Question Answering Dataset)
- Uses optimizer: **Fused ADAM**.
- Light-weight: the finetuning takes several minutes.

BERT finetuning script supports fine-tuning of SQUAD dataset on BERT large for both FP32 and BF16 mixed precision data type using **Lazy mode**.

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
cd Model-References/PyTorch/nlp/bert
```
2. Install the required packages using pip:
```bash
$PYTHON -m pip install -r requirements.txt
```
### Vocab File
Download the Vocab file located [here](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip).

### Download Dataset

#### Pre-Training:

`Model-References/PyTorch/nlp/bert/data` provides scripts to download, extract and pre-process [Wikipedia](https://dumps.wikimedia.org/) and [BookCorpus](http://yknzhu.wixsite.com/mbweb) datasets.

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
Note that the pre-training dataset is huge and takes several hours to download. BookCorpus may have access and download constraints. The final accuracy may vary depending on the dataset and its size.
The script creates formatted dataset for Phase 1 and Phase 2 of the pre-training.

#### Finetuning:
This section provides steps to extract and pre-process Squad Dataset(V1.1).

1. Go to `squad` folder.
```
cd Model-References/PyTorch/nlp/bert/data/squad
```
2. Download Squad dataset.
```
bash squad_download.sh
```

### Packing the Data
Habana supports using a [Data packing technique](https://github.com/HabanaAI/Gaudi-tutorials/blob/main/TensorFlow/DataPackingMLperfBERT/Data_Packing_Process_for_MLPERF_BERT.ipynb),
called Non-Negative Least Squares Histogram. Here, instead of padding with zero,
several short sequences are packed into one multi-sequence of size `max_seq_len`.
Thus, this removes most of the padding, which can lead to a speedup of up to 2&times;
in time-to-train (TTT). This packing technique can be applied on other datasets
with high variability in samples length.

Please note that for each NLP dataset with sequential data samples, the speedup
with data packing is determined by the ratio of `max_seq_len` to
`average_seq_len` in that particular dataset. The larger the ratio, the higher
the speedup.

To pack the dataset, in docker run:
```bash
cd /root/Model-References/PyTorch/nlp/bert

$PYTHON pack_pretraining_data_pytorch.py --input_dir <dataset_path_phase1> --output_dir <packed_dataset_path_phase1> --max_sequence_length 128 --max_predictions_per_sequence 20

$PYTHON pack_pretraining_data_pytorch.py --input_dir <dataset_path_phase2> --output_dir <packed_dataset_path_phase2> --max_sequence_length 512 --max_predictions_per_sequence 80
```
**Note:** This will generate json at the path <output_dir>/../<tail_dir>_metadata.json with meta data info like: "avg_seq_per_sample" etc. This json will be
used as an input to run_pretraining.py to extract "avg_seq_per_sample" in case of packed dataset mode.


## Training and Examples

Please create a log directory to store `dllogger.json` and specify its location for `--json_summary` attribute.

### Single Card and Multi-Card Pre-Training Examples
**Run training on 1 HPU:**

- Using packed data: lazy mode, 1 HPU, BF16 mixed precision, batch size 64 for Phase 1 and batch size 8 for Phase 2:

```bash
$PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased \
      --autocast --config_file=./bert_config.json \
      --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
      --json-summary=/tmp/log_directory/dllogger.json --output_dir=/tmp/results/checkpoints \
      --use_fused_lamb \
      --input_dir=/data/pytorch/bert_pretraining/packed_data/phase1/train_packed_new \
      --train_batch_size=8192 --max_seq_length=128 --max_predictions_per_seq=20 --max_steps=7038 \
      --warmup_proportion=0.2843 --num_steps_per_checkpoint=200 --learning_rate=0.006 --gradient_accumulation_steps=128
```

```bash
$PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased \
      --autocast --config_file=./bert_config.json \
      --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
      --json-summary=/tmp/log_directory/dllogger.json --output_dir=/tmp/results/checkpoints \
      --use_fused_lamb \
      --input_dir=/data/pytorch/bert_pretraining/packed_data/phase2/train_packed_new \
      --train_batch_size=4096 --max_seq_length=512 --max_predictions_per_seq=80 --max_steps=1563 \
      --warmup_proportion=0.128 --num_steps_per_checkpoint=200 --learning_rate=0.004 \
      --gradient_accumulation_steps=512 --resume_from_checkpoint --phase1_end_step=7038 --phase2
```

- Using packed data: Eager mode with torch.compile enabled, 1 HPU, BF16 mixed precision, batch size 64 for Phase 1 on **Gaudi2**::
```bash
export PT_HPU_LAZY_MODE=0
$PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased \
      --autocast --config_file=./bert_config.json \
      --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
      --json-summary=/tmp/log_directory/dllogger.json --output_dir=/tmp/results/checkpoints \
      --use_fused_lamb --use_torch_compile \
      --input_dir=/data/pytorch/bert_pretraining/packed_data/phase1/train_packed_new \
      --train_batch_size=8192 --max_seq_length=128 --max_predictions_per_seq=20 --max_steps=7038 \
      --warmup_proportion=0.2843 --num_steps_per_checkpoint=200 --learning_rate=0.006 --gradient_accumulation_steps=128
```


- Using packed data: lazy mode, 1 HPU, BF16 mixed precision, batch size 64 for Phase 1 and batch size 16 for Phase 2 on **Gaudi2**:

```bash
$PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased \
      --autocast --config_file=./bert_config.json \
      --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
      --json-summary=/tmp/log_directory/dllogger.json --output_dir=/tmp/results/checkpoints \
      --use_fused_lamb \
      --input_dir=/data/pytorch/bert_pretraining/packed_data/phase1/train_packed_new \
      --train_batch_size=8192 --max_seq_length=128 --max_predictions_per_seq=20 --max_steps=7038 \
      --warmup_proportion=0.2843 --num_steps_per_checkpoint=200 --learning_rate=0.006 --gradient_accumulation_steps=128
```

```bash
$PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased \
      --autocast --config_file=./bert_config.json \
      --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
      --json-summary=/tmp/log_directory/dllogger.json --output_dir=/tmp/results/checkpoints \
      --use_fused_lamb \
      --input_dir=/data/pytorch/bert_pretraining/packed_data/phase2/train_packed_new \
      --train_batch_size=4096 --max_seq_length=512 --max_predictions_per_seq=80 --max_steps=1563 \
      --warmup_proportion=0.128 --num_steps_per_checkpoint=200 --learning_rate=0.004 \
      --gradient_accumulation_steps=256 --resume_from_checkpoint --phase1_end_step=7038 --phase2
```

- Lazy mode, 1 HPU, unpacked data, BF16 mixed precision, batch size 64 for Phase1 and batch size 8 for Phase2:

```bash
$PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased \
      --autocast --config_file=./bert_config.json \
      --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
      --json-summary=/tmp/log_directory/dllogger.json --output_dir=/tmp/results/checkpoints --use_fused_lamb \
      --input_dir=/data/pytorch/bert_pretraining/hdf5_lower_case_1_seq_len_128/books_wiki_en_corpus \
      --train_batch_size=8192 --max_seq_length=128 --max_predictions_per_seq=20 --max_steps=7038 \
      --warmup_proportion=0.2843 --num_steps_per_checkpoint=200 --learning_rate=0.006 --gradient_accumulation_steps=128 \
      --enable_packed_data_mode False
```


```bash
$PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased \
      --autocast --config_file=./bert_config.json \
      --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
      --json-summary=/tmp/log_directory/dllogger.json --output_dir=/tmp/results/checkpoints --use_fused_lamb \
      --input_dir=/data/pytorch/bert_pretraining/hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/books_wiki_en_corpus \
      --train_batch_size=4096 --max_seq_length=512 --max_predictions_per_seq=80 --max_steps=1563 \
      --warmup_proportion=0.128 --num_steps_per_checkpoint=200 --learning_rate=0.004\
      --gradient_accumulation_steps=512 --resume_from_checkpoint --phase1_end_step=7038 --phase2 \
      --enable_packed_data_mode False
```

- Lazy mode, 1 HPU, unpacked data, FP32 precision, batch size 32 for Phase 1 and batch size 4 for Phase 2:

```bash
$PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased --config_file=./bert_config.json \
      --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
      --json-summary=/tmp/log_directory/dllogger.json --output_dir=/tmp/results/checkpoints --use_fused_lamb \
      --input_dir=/data/pytorch/bert_pretraining/hdf5_lower_case_1_seq_len_128/books_wiki_en_corpus \
      --train_batch_size=512 --max_seq_length=128 --max_predictions_per_seq=20 --max_steps=7038 \
      --warmup_proportion=0.2843 --num_steps_per_checkpoint=200 --learning_rate=0.006 --gradient_accumulation_steps=32 \
      --enable_packed_data_mode False
```

```bash
$PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased --config_file=./bert_config.json \
      --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
      --json-summary=/tmp/log_directory/dllogger.json --output_dir=/tmp/results/checkpoints --use_fused_lamb \
      --input_dir=/data/pytorch/bert_pretraining/hdf5_lower_case_1_seq_len_512/books_wiki_en_corpus \
      --train_batch_size=128 --max_seq_length=512 --max_predictions_per_seq=80 --max_steps=1563 \
      --warmup_proportion=0.128 --num_steps_per_checkpoint=200 --learning_rate=0.004 \
      --gradient_accumulation_steps=64 --resume_from_checkpoint --phase1_end_step=7038 --phase2 \
      --enable_packed_data_mode False
```

**Run training on 8 HPUs:**

To run multi-card demo, make sure the host machine has 512 GB of RAM installed. Modify the docker run command to pass 8 Gaudi cards to the docker container. This ensures the docker has access to all the 8 cards required for multi-card demo.

**NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Scaling_Guide/DDP_Based_Scaling.html#mpirun-configuration).

- Using packed data: lazy mode, 8 HPUs, BF16 mixed precision, per chip batch size of 64 for Phase 1 and 8 for Phase 2:

```bash
export MASTER_ADDR="localhost"
export MASTER_PORT="12345"
mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
$PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased --autocast --config_file=./bert_config.json --use_habana \
      --allreduce_post_accumulation --allreduce_post_accumulation_fp16 --json-summary=/tmp/log_directory/dllogger.json \
      --output_dir=/tmp/results/checkpoints --use_fused_lamb \
      --input_dir=/data/pytorch/bert_pretraining/packed_data/phase1/train_packed_new \
      --train_batch_size=8192 --max_seq_length=128 --max_predictions_per_seq=20 --max_steps=7038 \
      --warmup_proportion=0.2843 --num_steps_per_checkpoint=200 --learning_rate=0.006 --gradient_accumulation_steps=128
```

```bash
export MASTER_ADDR="localhost"
export MASTER_PORT="12345"
mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
$PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased --autocast --config_file=./bert_config.json --use_habana \
      --allreduce_post_accumulation --allreduce_post_accumulation_fp16 --json-summary=/tmp/log_directory/dllogger.json \
      --output_dir=/tmp/results/checkpoints --use_fused_lamb \
      --input_dir=/data/pytorch/bert_pretraining/packed_data/phase2/train_packed_new \
      --train_batch_size=4096 --max_seq_length=512 --max_predictions_per_seq=80 --max_steps=1563 \
      --warmup_proportion=0.128 --num_steps_per_checkpoint=200 --learning_rate=0.004 \
      --gradient_accumulation_steps=512 --resume_from_checkpoint --phase1_end_step=7038 --phase2
```

- Using packed data: lazy mode, 8 HPUs, BF16 mixed precision, per chip batch size of 64 for Phase 1 and 16 for Phase 2 on **Gaudi2**:

```bash
export MASTER_ADDR="localhost"
export MASTER_PORT="12345"
mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
$PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased --autocast --config_file=./bert_config.json --use_habana \
      --allreduce_post_accumulation --allreduce_post_accumulation_fp16 --json-summary=/tmp/log_directory/dllogger.json \
      --output_dir=/tmp/results/checkpoints --use_fused_lamb \
      --input_dir=/data/pytorch/bert_pretraining/packed_data/phase1/train_packed_new \
      --train_batch_size=8192 --max_seq_length=128 --max_predictions_per_seq=20 --max_steps=7038 \
      --warmup_proportion=0.2843 --num_steps_per_checkpoint=200 --learning_rate=0.006 --gradient_accumulation_steps=128
```

```bash
export MASTER_ADDR="localhost"
export MASTER_PORT="12345"
mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
$PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased --autocast --config_file=./bert_config.json --use_habana \
      --allreduce_post_accumulation --allreduce_post_accumulation_fp16 --json-summary=/tmp/log_directory/dllogger.json \
      --output_dir=/tmp/results/checkpoints --use_fused_lamb \
      --input_dir=/data/pytorch/bert_pretraining/packed_data/phase2/train_packed_new \
      --train_batch_size=4096 --max_seq_length=512 --max_predictions_per_seq=80 --max_steps=1563 \
      --warmup_proportion=0.128 --num_steps_per_checkpoint=200 --learning_rate=0.004 \
      --gradient_accumulation_steps=256 --resume_from_checkpoint --phase1_end_step=7038 --phase2
```

- Eager mode with torch.compile enabled, 8 HPUs, packed data, BF16 mixed precision, per chip batch size of 64 for Phase 1 on **Gaudi2**:

```bash
export PT_HPU_LAZY_MODE=0
export MASTER_ADDR="localhost"
export MASTER_PORT="12345"
mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
$PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased \
      --autocast --use_torch_compile \
      --config_file=./bert_config.json --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
      --json-summary=/tmp/log_directory/dllogger.json --output_dir=/tmp/BERT_PRETRAINING/results/checkpoints --use_fused_lamb \
      --input_dir=/data/pytorch/bert_pretraining/packed_data/phase1/train_packed_new \
      --train_batch_size=8192 --max_seq_length=128 --max_predictions_per_seq=20 --warmup_proportion=0.2843 \
      --max_steps=7038 --num_steps_per_checkpoint=200 --learning_rate=0.006 --gradient_accumulation_steps=128
```

- Lazy mode, 8 HPUs, unpacked data, BF16 mixed precision, per chip batch size of 64 for Phase 1 and 8 for Phase 2:
```bash
export MASTER_ADDR="localhost"
export MASTER_PORT="12345"
mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
$PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased \
      --autocast --use_lazy_mode=True \
      --config_file=./bert_config.json --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
      --json-summary=/tmp/log_directory/dllogger.json --output_dir=/tmp/BERT_PRETRAINING/results/checkpoints --use_fused_lamb \
      --input_dir=/data/pytorch/bert_pretraining/hdf5_lower_case_1_seq_len_128/books_wiki_en_corpus \
      --train_batch_size=8192 --max_seq_length=128 --max_predictions_per_seq=20 --warmup_proportion=0.2843 \
      --max_steps=7038 --num_steps_per_checkpoint=200 --learning_rate=0.006 --gradient_accumulation_steps=128 \
      --enable_packed_data_mode False
```

```bash
export MASTER_ADDR="localhost"
export MASTER_PORT="12345"
mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
$PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased \
      --autocast --use_lazy_mode=True \
      --config_file=./bert_config.json --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
      --json-summary=/tmp/log_directory/dllogger.json --output_dir=/tmp/BERT_PRETRAINING/results/checkpoints --use_fused_lamb \
      --input_dir=/data/pytorch/bert_pretraining/hdf5_lower_case_1_seq_len_512/books_wiki_en_corpus \
      --train_batch_size=4096 --max_seq_length=512 --max_predictions_per_seq=80 --warmup_proportion=0.128 \
      --max_steps=5 --num_steps_per_checkpoint=200 --learning_rate=0.004 --gradient_accumulation_steps=512 --resume_from_checkpoint --phase1_end_step=7038 --phase2 \
      --enable_packed_data_mode False
```

- Lazy mode, 8 HPUs, unpacked data, FP32 precision, per chip batch size of 32 for Phase 1 and 4 for Phase 2:

```bash
export MASTER_ADDR="localhost"
export MASTER_PORT="12345"
mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
$PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased --config_file=./bert_config.json \
      --use_habana --allreduce_post_accumulation  --allreduce_post_accumulation_fp16 \
      --json-summary=/tmp/log_directory/dllogger.json --output_dir=/tmp/results/checkpoints \
      --use_fused_lamb --input_dir=/data/pytorch/bert_pretraining/hdf5_lower_case_1_seq_len_128/books_wiki_en_corpus \
      --train_batch_size=8192 --max_seq_length=128 --max_predictions_per_seq=20 --max_steps=3 --warmup_proportion=0.2843 \
      --num_steps_per_checkpoint=200 --learning_rate=0.006 --gradient_accumulation_steps=256 \
      --enable_packed_data_mode False
```

```bash
export MASTER_ADDR="localhost"
export MASTER_PORT="12345"
mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root
$PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased --config_file=./bert_config.json \
      --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 --json-summary=/tmp/log_directory/dllogger.json \
      --output_dir=/tmp/results/checkpoints --use_fused_lamb \
      --input_dir=/data/pytorch/bert_pretraining/hdf5_lower_case_1_seq_len_512/books_wiki_en_corpus \
      --train_batch_size=4096 --max_seq_length=512 --max_predictions_per_seq=80 --max_steps=1563 --warmup_proportion=0.128 \
      --num_steps_per_checkpoint=200 --learning_rate=0.004 --gradient_accumulation_steps=512 \
      --resume_from_checkpoint --phase1_end_step=7038 --phase2 \
      --enable_packed_data_mode False
```


### Single Card and Multi-Card Finetuning Examples
**Run training on 1 HPU:**
- Lazy mode, 1 HPU, BF16 mixed precision, batch size 24 for train and batch size 8 for test:

```bash
$PYTHON run_squad.py --do_train --bert_model=bert-large-uncased \
      --config_file=./bert_config.json \
      --use_habana --use_fused_adam --do_lower_case --output_dir=/tmp/results/checkpoints \
      --json-summary=/tmp/log_directory/dllogger.json \
      --train_batch_size=24 --predict_batch_size=8 --seed=1 --max_seq_length=384 \
      --doc_stride=128 --max_steps=-1   --learning_rate=3e-5 --num_train_epochs=2 \
      --init_checkpoint=<path-to-checkpoint> \
      --vocab_file=<path-to-vocab> \
      --train_file=data/squad/v1.1/train-v1.1.json \
      --skip_cache --do_predict  \
      --predict_file=data/squad/v1.1/dev-v1.1.json \
      --do_eval --eval_script=data/squad/v1.1/evaluate-v1.1.py --log_freq 20 \
      --autocast
```

- Lazy mode, 1 HPU, FP32 precision, batch size 12 for train and batch size 8 for test:

```bash
$PYTHON run_squad.py --do_train --bert_model=bert-large-uncased --config_file=./bert_config.json \
      --use_habana --use_fused_adam --do_lower_case --output_dir=/tmp/results/checkpoints \
      --json-summary=/tmp/log_directory/dllogger.json \
      --train_batch_size=12 --predict_batch_size=8 --seed=1 --max_seq_length=384 \
      --doc_stride=128 --max_steps=-1   --learning_rate=3e-5 --num_train_epochs=2 \
      --init_checkpoint=<path-to-checkpoint> \
      --vocab_file=<path-to-vocab> \
      --train_file=data/squad/v1.1/train-v1.1.json \
      --skip_cache --do_predict  \
      --predict_file=data/squad/v1.1/dev-v1.1.json \
      --do_eval --eval_script=data/squad/v1.1/evaluate-v1.1.py --log_freq 20
```

- Eager mode with torch.compile enabled, 1 HPU, FP32 precision, batch size 12 for train and batch size 8 for test:

```bash
export PT_HPU_LAZY_MODE=0
$PYTHON run_squad.py --do_train --bert_model=bert-large-uncased --config_file=./bert_config.json \
      --use_habana --use_fused_adam --do_lower_case --output_dir=/tmp/results/checkpoints \
      --json-summary=/tmp/log_directory/dllogger.json --use_torch_compile \
      --train_batch_size=12 --predict_batch_size=8 --seed=1 --max_seq_length=384 \
      --doc_stride=128 --max_steps=-1   --learning_rate=3e-5 --num_train_epochs=2 \
      --init_checkpoint=<path-to-checkpoint> \
      --vocab_file=<path-to-vocab> \
      --train_file=data/squad/v1.1/train-v1.1.json \
      --skip_cache --do_predict  \
      --predict_file=data/squad/v1.1/dev-v1.1.json \
      --do_eval --eval_script=data/squad/v1.1/evaluate-v1.1.py --log_freq 20
```

**Run training on 8 HPUs:**

To run multi-card demo, make sure the host machine has 512 GB of RAM installed. Modify the docker run command to pass 8 Gaudi cards to the docker container. This ensures the docker has access to all the 8 cards required for multi-card demo.

**NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Scaling_Guide/DDP_Based_Scaling.html#mpirun-configuration).

- Lazy mode, 8 HPUs, BF16 mixed precision, per chip batch size of 24 for train and 8 for test:
```bash
export MASTER_ADDR="localhost"
export MASTER_PORT="12345"
mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
$PYTHON run_squad.py --do_train --bert_model=bert-large-uncased \
      --config_file=./bert_config.json \
      --use_habana --use_fused_adam --do_lower_case --output_dir=/tmp/results/checkpoints \
      --json-summary=/tmp/log_directory/dllogger.json \
      --train_batch_size=24 --predict_batch_size=8 --seed=1 --max_seq_length=384 \
      --doc_stride=128 --max_steps=-1   --learning_rate=3e-5 --num_train_epochs=2 \
      --init_checkpoint=<path-to-checkpoint> \
      --vocab_file=<path-to-vocab> \
      --train_file=data/squad/v1.1/train-v1.1.json \
      --skip_cache --do_predict  \
      --predict_file=data/squad/v1.1/dev-v1.1.json \
      --do_eval --eval_script=data/squad/v1.1/evaluate-v1.1.py --log_freq 20 \
      --autocast
```

- Lazy mode, 8 HPUs, FP32 precision, per chip batch size of 12 for train and 8 for test:

```bash
export MASTER_ADDR="localhost"
export MASTER_PORT="12345"
mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
$PYTHON run_squad.py --do_train --bert_model=bert-large-uncased --config_file=./bert_config.json \
      --use_habana --use_fused_adam --do_lower_case --output_dir=/tmp/results/checkpoints \
      --json-summary=/tmp/log_directory/dllogger.json \
      --train_batch_size=12 --predict_batch_size=8 --seed=1 --max_seq_length=384 \
      --doc_stride=128 --max_steps=-1   --learning_rate=3e-5 --num_train_epochs=2 \
      --init_checkpoint=<path-to-checkpoint> \
      --vocab_file=<path-to-vocab> \
      --train_file=data/squad/v1.1/train-v1.1.json \
      --skip_cache --do_predict  \
      --predict_file=data/squad/v1.1/dev-v1.1.json \
      --do_eval --eval_script=data/squad/v1.1/evaluate-v1.1.py --log_freq 20
```

- Eager mode with torch.compile enabled, 8 HPUs, BF16 mixed precision, per chip batch size of 24 for train and 8 for test:
```bash
export PT_HPU_LAZY_MODE=0
export MASTER_ADDR="localhost"
export MASTER_PORT="12345"
mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
$PYTHON run_squad.py --do_train --bert_model=bert-large-uncased \
      --config_file=./bert_config.json --use_torch_compile \
      --use_habana --use_fused_adam --do_lower_case --output_dir=/tmp/results/checkpoints \
      --json-summary=/tmp/log_directory/dllogger.json \
      --train_batch_size=24 --predict_batch_size=8 --seed=1 --max_seq_length=384 \
      --doc_stride=128 --max_steps=-1   --learning_rate=3e-5 --num_train_epochs=2 \
      --init_checkpoint=<path-to-checkpoint> \
      --vocab_file=<path-to-vocab> \
      --train_file=data/squad/v1.1/train-v1.1.json \
      --skip_cache --do_predict  \
      --predict_file=data/squad/v1.1/dev-v1.1.json \
      --do_eval --eval_script=data/squad/v1.1/evaluate-v1.1.py --log_freq 20 \
      --autocast
```

- Habana provides the pretraining checkpoints for most of the models. The user can simply feed the data from [BERT checkpoint](https://developer.habana.ai/catalog/bert-pretraining-for-pytorch/) to provide the path-to-checkpoint for  --init_checkpoint when you run the above model.

### Multi-Server Training Examples
To run multi-server demo, make sure the host machine has 512 GB of RAM installed.
Also ensure you followed the [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html)
to install and set up docker, so that the docker has access to all the 8 cards
required for multi-node demo. Multi-server configuration for BERT PT training up to
4 servers, each with 8 Gaudi cards, have been verified.

Before execution of the multi-server scripts, make sure all network interfaces are up. You can change the state of each network interface managed by the habanalabs driver using the following command:
```
sudo ip link set <interface_name> up
```
To identify if a specific network interface is managed by the habanalabs driver type, run:
```
sudo ethtool -i <interface_name>
```
#### Docker ssh Port Setup for Multi-Server Training

By default, the Habana docker uses `port 22` for ssh. The default port configured in the script is `port 3022`. Run the following commands to configure the selected port number , `port 3022` in example below.

```bash
sed -i 's/#Port 22/Port 3022/g' /etc/ssh/sshd_config
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
service ssh restart
```
#### Set up password-less ssh
To set up password-less ssh between all connected servers used in scale-out training, follow the below steps:

1. Run the following in all the nodes' docker sessions:
   ```bash
   mkdir ~/.ssh
   cd ~/.ssh
   ssh-keygen -t rsa -b 4096
   ```
   a. Copy id_rsa.pub contents from every node's docker to every other node's docker's ~/.ssh/authorized_keys (all public keys need to be in all hosts' authorized_keys):
   ```bash
   cat id_rsa.pub > authorized_keys
   vi authorized_keys
   ```
   b. Copy the contents from inside to other systems.

   c. Paste all hosts' public keys in all hosts' “authorized_keys” file.

2. On each system, add all hosts (including itself) to known_hosts. The IP addresses used below are just for illustration:
   ```bash
   ssh-keyscan -p 3022 -H 10.10.100.101 >> ~/.ssh/known_hosts
   ssh-keyscan -p 3022 -H 10.10.100.102 >> ~/.ssh/known_hosts
   ssh-keyscan -p 3022 -H 10.10.100.103 >> ~/.ssh/known_hosts
   ssh-keyscan -p 3022 -H 10.10.100.104 >> ~/.ssh/known_hosts
   ```
3. Install python packages required for BERT Pre-training model
   ```
   pip install -r Model-References/PyTorch/nlp/bert/requirements.txt
   ```

**Run training on 32 HPUs:**

**NOTE:**
- mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Scaling_Guide/DDP_Based_Scaling.html#mpirun-configuration).
- `$MPI_ROOT` environment variable is set automatically during Setup. See [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) for details.

- Using packed data: lazy mode, 32 HPUs, BF16 mixed precision, per chip batch size 64 for Phase 1 and batch size 8 for Phase 2:
```bash
export MASTER_ADDR="10.10.100.101"
export MASTER_PORT="12345"
mpirun --allow-run-as-root --mca plm_rsh_args "-p 3022" --bind-to core -n 32 --map-by ppr:4:socket:PE=6 \
--rank-by core --report-bindings --prefix --mca btl_tcp_if_include 10.10.100.101/16
      $MPI_ROOT -H 10.10.100.101:16,10.10.100.102:16,10.10.100.103:16,10.10.100.104:16 -x LD_LIBRARY_PATH \
      -x HABANA_LOGS -x PYTHONPATH -x MASTER_ADDR \
      -x MASTER_PORT \
      $PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased --autocast --config_file=./bert_config.json \
      --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
      --json-summary=/tmp/log_directory/dllogger.json --output_dir=/tmp/results/checkpoints \
      --use_fused_lamb --input_dir=/data/pytorch/bert_pretraining/packed_data/phase1/train_packed_new \
      --train_batch_size=2048 --max_seq_length=128 --max_predictions_per_seq=20 --max_steps=7038 \
      --warmup_proportion=0.2843 --num_steps_per_checkpoint=200 --learning_rate=0.006 \
      --gradient_accumulation_steps=32
```

```bash
export MASTER_ADDR="10.10.100.101"
export MASTER_PORT="12345"
mpirun --allow-run-as-root --mca plm_rsh_args "-p 3022" --bind-to core -n 32 --map-by ppr:4:socket:PE=6 \
--rank-by core --report-bindings --prefix --mca btl_tcp_if_include 10.10.100.101/16 \
      $MPI_ROOT -H 10.10.100.101:16,10.10.100.102:16,10.10.100.103:16,10.10.100.104:16 -x LD_LIBRARY_PATH \
      -x HABANA_LOGS -x PYTHONPATH -x MASTER_ADDR \
      -x MASTER_PORT \
      $PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased --autocast --config_file=./bert_config.json \
      --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
      --json-summary=/tmp/log_directory/dllogger.json --output_dir=/tmp/results/checkpoints \
      --use_fused_lamb --input_dir=/data/pytorch/bert_pretraining/packed_data/phase2/train_packed_new \
      --train_batch_size=1024 --max_seq_length=512 --max_predictions_per_seq=80 --max_steps=1563 --warmup_proportion=0.128 \ --num_steps_per_checkpoint=200 --learning_rate=0.004 --gradient_accumulation_steps=128 \
      --resume_from_checkpoint --phase1_end_step=7038 --phase2
```

- Lazy mode, 32 HPUs, unpacked data, BF16 mixed precision, batch size 64 for Phase 1 and batch size 8 for Phase 2:

```bash
export MASTER_ADDR="10.10.100.101"
export MASTER_PORT="12345"
mpirun --allow-run-as-root --mca plm_rsh_args -p3022 --bind-to core -n 32 --map-by ppr:4:socket:PE=6 \
--rank-by core --report-bindings --prefix --mca btl_tcp_if_include 10.10.100.101/16 \
$MPI_ROOT -H 10.10.100.101:16,10.10.100.102:16,10.10.100.103:16,10.10.100.104:16 \
      -x LD_LIBRARY_PATH -x HABANA_LOGS -x PYTHONPATH -x MASTER_ADDR -x MASTER_PORT -x https_proxy -x http_proxy \
$PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased \
      --autocast --config_file=./bert_config.json \
      --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
      --json-summary=/tmp/log_directory/dllogger.json --output_dir= /tmp/results/checkpoints \
      --use_fused_lamb --input_dir=/data/pytorch/bert_pretraining/hdf5_lower_case_1_seq_len_128/books_wiki_en_corpus \
      --train_batch_size=2048 --max_seq_length=128 --max_predictions_per_seq=20
      --max_steps=7038 --warmup_proportion=0.2843 \
      --num_steps_per_checkpoint=200 --learning_rate=0.006 --gradient_accumulation_steps=32 \
      --enable_packed_data_mode False
```

```bash
export MASTER_ADDR="10.10.100.101"
export MASTER_PORT="12345"
mpirun --allow-run-as-root --mca plm_rsh_args -p3022 --bind-to core -n 32 --map-by ppr:4:socket:PE=6 \
--rank-by core --report-bindings --prefix --mca btl_tcp_if_include 10.10.100.101/16 \
      $MPI_ROOT -H 10.10.100.101:16,10.10.100.102:16,10.10.100.103:16,10.10.100.104:16 -x LD_LIBRARY_PATH \
      -x HABANA_LOGS -x PYTHONPATH -x MASTER_ADDR -x MASTER_PORT \
      $PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased --autocast \
      --config_file=./bert_config.json --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
      --json-summary=/tmp/log_directory/dllogger.json --output_dir= /tmp/results/checkpoints \
      --use_fused_lamb --input_dir=/data/pytorch/bert_pretraining/hdf5_lower_case_1_seq_len_512/books_wiki_en_corpus \
      --train_batch_size=1024 --max_seq_length=512 --max_predictions_per_seq=80 --max_steps=1563 \
      --warmup_proportion=0.128 --num_steps_per_checkpoint=200 --learning_rate=0.004 \
      --gradient_accumulation_steps=128 --resume_from_checkpoint --phase1_end_step=7038 --phase2 \
      --enable_packed_data_mode False
```

### BERT Pre-Training with ZeroRedundancyOptimizer

BERT training script supports pre-training of BERT 1.2B parameters using ZeroRedundancyOptimizer with BF16 mixed precision data type in  **Lazy mode**.

- Lazy mode, 8 HPUs, BF16 mixed precision, per chip batch size 8 for Phase 1 and batch size 2 for Phase 2:

```bash
export MASTER_ADDR="localhost"
export MASTER_PORT="12345"
mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
$PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased --autocast --use_lazy_mode=True \
      --config_file=./bert_config_1.2B.json --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
      --json-summary=/tmp/log_directory/dllogger.json --output_dir=/tmp/BERT_PRETRAINING/results/checkpoints --use_fused_lamb \
      --input_dir=/data/pytorch/bert_pretraining/packed_data/phase1/train_packed_new \
      --train_batch_size=1024 --max_seq_length=128 --max_predictions_per_seq=20 --warmup_proportion=0.2843 \
      --max_steps=7038 --num_steps_per_checkpoint=200 --learning_rate=0.006 --gradient_accumulation_steps=128 \
      --use_zero_optimizer True

```

```bash
export MASTER_ADDR="localhost"
export MASTER_PORT="12345"
mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root \
$PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased --autocast --use_lazy_mode=True \
      --config_file=./bert_config_1.2B.json --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
      --json-summary=/tmp/log_directory/dllogger.json --output_dir=/tmp/BERT_PRETRAINING/results/checkpoints --use_fused_lamb \
      --input_dir=/data/pytorch/bert_pretraining/packed_data/phase2/train_packed_new \
      --train_batch_size=1024 --max_seq_length=512 --max_predictions_per_seq=80 --warmup_proportion=0.128 \
      --max_steps=1563 --num_steps_per_checkpoint=200 --learning_rate=0.004 --gradient_accumulation_steps=512 \
      --resume_from_checkpoint --phase1_end_step=7038 --phase2 --use_zero_optimizer True

```
## Inference and Examples
**Run inference on 1 HPU:**
- Lazy mode, 1 HPU, BF16 mixed precision, batch size 24:

```bash
$PYTHON run_squad.py --bert_model=bert-large-uncased --autocast \
      --config_file=./bert_config.json \
      --use_habana --do_lower_case --output_dir=/tmp/results/checkpoints \
      --json-summary=/tmp/log_directory/dllogger.json \
      --predict_batch_size=24 \
      --init_checkpoint=<path-to-checkpoint> \
      --vocab_file=<path-to-vocab> \
      --do_predict  \
      --predict_file=data/squad/v1.1/dev-v1.1.json \
      --do_eval --eval_script=data/squad/v1.1/evaluate-v1.1.py
```

- HPU graphs, 1 HPU, BF16 mixed precision, batch size 24:

```bash
$PYTHON run_squad.py --bert_model=bert-large-uncased --autocast --use_hpu_graphs \
      --config_file=./bert_config.json \
      --use_habana --do_lower_case --output_dir=/tmp/results/checkpoints \
      --json-summary=/tmp/log_directory/dllogger.json \
      --predict_batch_size=24 \
      --init_checkpoint=<path-to-checkpoint> \
      --vocab_file=<path-to-vocab> \
      --do_predict  \
      --predict_file=data/squad/v1.1/dev-v1.1.json \
      --do_eval --eval_script=data/squad/v1.1/evaluate-v1.1.py
```

- Lazy mode, 1 HPU, FP16 mixed precision, batch size 24:

```bash
$PYTHON run_squad.py --bert_model=bert-large-uncased --autocast \
      --config_file=./bert_config.json \
      --use_habana --do_lower_case --output_dir=/tmp/results/checkpoints \
      --json-summary=/tmp/log_directory/dllogger.json \
      --predict_batch_size=24 \
      --init_checkpoint=<path-to-checkpoint> \
      --vocab_file=<path-to-vocab> \
      --do_predict --fp16 \
      --predict_file=data/squad/v1.1/dev-v1.1.json \
      --do_eval --eval_script=data/squad/v1.1/evaluate-v1.1.py
```

- HPU graphs, 1 HPU, FP16 mixed precision, batch size 24:

```bash
$PYTHON run_squad.py --bert_model=bert-large-uncased --autocast --use_hpu_graphs \
      --config_file=./bert_config.json \
      --use_habana --do_lower_case --output_dir=/tmp/results/checkpoints \
      --json-summary=/tmp/log_directory/dllogger.json \
      --predict_batch_size=24 \
      --init_checkpoint=<path-to-checkpoint> \
      --vocab_file=<path-to-vocab> \
      --do_predict --fp16 \
      --predict_file=data/squad/v1.1/dev-v1.1.json \
      --do_eval --eval_script=data/squad/v1.1/evaluate-v1.1.py
```

**Run inference on 1 HPU with torch.compile:**
- 1 HPU, BF16 mixed precision, batch size 24:

```bash
$PYTHON run_squad.py --bert_model=bert-large-uncased --autocast \
      --config_file=./bert_config.json \
      --use_habana --do_lower_case --output_dir=/tmp/results/checkpoints \
      --json-summary=/tmp/log_directory/dllogger.json \
      --predict_batch_size=24 \
      --init_checkpoint=<path-to-checkpoint> \
      --vocab_file=<path-to-vocab> \
      --do_predict --use_torch_compile \
      --predict_file=data/squad/v1.1/dev-v1.1.json \
      --do_eval --eval_script=data/squad/v1.1/evaluate-v1.1.py
```

- 1 HPU, FP16 mixed precision, batch size 24:

```bash
$PYTHON run_squad.py --bert_model=bert-large-uncased --autocast \
      --config_file=./bert_config.json \
      --use_habana --do_lower_case --output_dir=/tmp/results/checkpoints \
      --json-summary=/tmp/log_directory/dllogger.json \
      --predict_batch_size=24 \
      --init_checkpoint=<path-to-checkpoint> \
      --vocab_file=<path-to-vocab> \
      --do_predict --use_torch_compile --fp16 \
      --predict_file=data/squad/v1.1/dev-v1.1.json \
      --do_eval --eval_script=data/squad/v1.1/evaluate-v1.1.py
```

When not using torch.compile this model recommends using the ["HPU graph"](https://docs.habana.ai/en/latest/PyTorch/Inference_on_Gaudi/Inference_using_HPU_Graphs/Inference_using_HPU_Graphs.html) model type to minimize the host time spent in the `forward()` call.

## Pre-trained Model and Checkpoint
PyTorch BERT is trained on Intel Gaudi AI Accelerators and the saved model & checkpoints are provided. You can use it for fine-tuning or transfer learning tasks with your own datasets. To download the saved model file, please refer to [Habana Catalog](https://developer.habana.ai/catalog/bert-pretraining-for-pytorch/) to obtain the URL.


## Supported Configurations

| Validated on | SynapseAI Version | PyTorch Version | Mode |
|--------|-------------------|-----------------|----------------|
| Gaudi   | 1.15.0             | 2.2.0          | Training |
| Gaudi   | 1.15.0             | 2.2.0          | Inference |
| Gaudi2  | 1.15.0             | 2.2.0          | Training |
| Gaudi2  | 1.15.0             | 2.2.0          | Inference |

## Changelog
### 1.15.0
1. Changed model configurations mentioned in this README:
- lazy mode, 1 HPU, BF16 mixed precision, batch size 64 for Phase 1 and batch size 16 for Phase 2 on **Gaudi2**
- lazy mode, 8 HPUs, BF16 mixed precision, per chip batch size of 64 for Phase 1 and 16 for Phase 2 on **Gaudi2**
### 1.14.0
1. Added support for dynamic shapes in BERT Pretraining

### 1.13.0
1. Added tensorboard logging.
2. Added support for torch.compile inference.
3. Added support for FP16 through autocast.
4. Aligned profiler invocation between training and inference loops.
5. Added support for dynamic shapes in BERT Finetuning
6. Added torch.compile support - performance improvement feature for PyTorch eager mode for
   BERT Pretraining. Supported only for phase1.
7. Added torch.compile support - performance improvement feature for PyTorch eager mode for
   BERT Finetuning.

### 1.12.0
1. Removed HMP; switched to Autocast.
2. Eager mode support is deprecated.

### 1.11.0
1. Dynamic Shapes will be enabled by default in future releases. It is currently enabled in BERT Pretraining Model
   training script as a temporary solution.

### 1.10.0
1. Support added for cached dataset for finetuning.

### 1.9.0
1. Enabled usage of PyTorch autocast
2. Enabled BERT finetuning(run_squad.py) with SQUAD dataset (training and inference).

### 1.6.0
1. ZeroReduancyOptimer is support is added and tested BERT 1.2B parameter config.

### 1.5.0
1. Packed dataset mode is set as default execution mode
2. Deprecated the flags `enable_packed_data_mode` and `avg_seq_per_pack` and added support for automatic detection of those parameters based on dataset metadata file.
3. Changes related to Saving and Loading checkpoint were removed.
4. Removed changes related to padding index and flatten.
5. Fixed throughput calculation for packed dataset.
6. Demo scripts were removed and references to custom demo script were replaced by community entry points in README
7. Reduced the number of distributed barrier calls to once per gradient accumulation steps
8. Simplified the distributed Initialization.
9. Added support for training on **Gaudi2** supporting up to 8 cards

### 1.4.0
1. Lazy mode is set as default execution mode,for eager mode set `use-lazy-mode` as False
2. Pretraining with packed dataset is supported


### 1.3.0
1. Single worker thread changes are removed.
2. Loss computation brought it back to training script.
3. Removed setting the embedding padding index as 0 explicitly.
4. Removed the select op implementation using index select and squeeze and retained the default code.
5. Permute and view is replaced as flatten.
6. Change `python` or `python3` to `$PYTHON` to execute correct version based on environment setup.

### 1.2.0
1. Enabled HCCL flow for distributed training.
2. Removed changes related to data type conversions for input_ids, segment ids, position_ids and input_mask.
3. Removed changes related to position ids from training script.
4. Removed changes related to no pinned memory and skip last batch.


### Training Script Modifications
The following changes have been added to training (run_pretraining.py and run_squad.py) and modeling (modeling.py) scripts.

1. Added support for Habana devices:

    a. Load Habana specific library.

    b. Support required for cpu to work.

    c. Required environment variables are defined for habana device.

    d. Added Habana BF16 Mixed precision support.

    e. Added python version of LAMB optimizer and will be used as default(from lamb.py).

    f. Support for distributed training on Habana device.

    g. Added changes to support Lazy mode with required mark_step().

    h. Added changes to calculate the performance per step and report through dllogger.

    i. Using conventional torch layernorm, linear and activation functions.

    j. Changes for dynamic loading of HCCL library.

    k. Added support for FusedAdamW and FusedClipNorm in run_squad.py.

    l. optimizer_grouped_parameters config has changed for weight_decay from 0.01 to 0.0.


2. To improve performance:

    a. Added support for Fused LAMB optimizer in run_pretraining.py.

    b. Bucket size set to 230MB for better performance in distributed training.

    c. Added support to use distributed all_reduce instead of default Distributed Data Parallel in pre-training.

    d. Added support for lowering print frequency of loss and associated this with log_freq.

    e. Added support for Fused ADAMW optimizer and FusedClipNorm in run_squad.py.


## Known Issues
1. Placing mark_step() arbitrarily may lead to undefined behaviour. Recommend to keep mark_step() as shown in provided scripts.
2. BERT 1.2B parameter model is restricted to showcase the PyTorch ZeroReduancyOptimer feature and not for Model convergence
3. Only scripts and configurations mentioned in this README are supported and verified.
