# BERT

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

**Note**: The model is enabled on both Gaudi and Gaudi2.

## Table of Contents
   * [Model-References/README.md](https://github.com/HabanaAI/Model-References/blob/master/README.md)
   * [Model Overview](#model-overview)
     * [BERT Pre-Training](#bert-pre-training)
     * [BERT Fine-Tuning](#bert-fine-tuning)
   * [Setup](#setup)
     * [Docker Setup and Dataset Generation](#docker-setup-and-dataset-generation)
     * [Run the docker container and clone the Model-References repository for non-K8s configurations only](#run-the-docker-container-and-clone-the-model-references-repository-for-non-k8s-configurations-only)
     * [Install Model Requirements](#install-model-requirements)
     * [Download and preprocess the datasets for Pretraining and Finetuning for non-K8s and K8s configurations](#download-and-preprocess-the-datasets-for-pretraining-and-finetuning-for-non-k8s-and-k8s-configurations)
       * [Pretraining datasets download instructions](#pretraining-datasets-download-instructions)
       * [Pretraining datasets packing instructions](#pretraining-datasets-packing-instructions)
       * [Finetuning datasets download instructions](#finetuning-datasets-download-instructions)
   * [TensorFlow BERT Training](#tensorflow-bert-training)
     * [Pre-requisites](#pre-requisites)
   * [Training BERT in non-Kubernetes environments](#training-bert-in-non-kubernetes-environments)
     * [Single-card training](#single-card-training)
     * [Multi-card/single-server Horovod-based distributed training](#multi-cardsingle-server-horovod-based-distributed-training)
     * [Multi-server Horovod-based scale-out distributed training](#multi-server-horovod-based-scale-out-distributed-training)
       * [Docker ssh port setup for Multi-server training](#docker-ssh-port-setup-for-multi-server-training)
       * [Setup password-less ssh between all connected servers used in the scale-out training](#setup-password-less-ssh-between-all-connected-servers-used-in-the-scale-out-training)
       * [Download pretrained model and MRPC dataset (if needed) on each node:](#download-pretrained-model-and-mrpc-dataset-if-needed-on-each-node)
       * [Run BERT training on multiple servers:](#run-bert-training-on-multiple-servers)
   * [Training BERT in Kubernetes Environment](#training-bert-in-kubernetes-environment)
     * [Single-card training on K8s](#single-card-training-on-k8s)
     * [Multi-card Horovod-based distributed training on K8s](#multi-card-horovod-based-distributed-training-on-k8s)
   * [Profile](#profile)
   * [Supported Configuration](#supported-configuration)
   * [Changelog](#changelog)
   * [Known Issues](#known-issues)

## Model Overview

Bidirectional Encoder Representations from Transformers (BERT) is a technique for natural language processing (NLP) pre-training developed by Google.
BERT was created and published in 2018 by Jacob Devlin and his colleagues from Google.
Google is leveraging BERT to better understand user searches.

The original English-language BERT model comes with two pre-trained general types: (1) the BERTBASE model, a 12-layer, 768-hidden, 12-heads, 110M parameter neural network architecture, and (2) the BERTLARGE model, a 24-layer, 1024-hidden, 16-heads, 340M parameter neural network architecture; both of which were trained on the BooksCorpus with 800M words, and a version of the English Wikipedia with 2,500M words.

The scripts are a mix of Habana modified pre-training scripts taken from [NVIDIA GitHub](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT) and Habana modified fine-tuning scripts taken from [Google GitHub](https://github.com/google-research/bert). We converted the training scripts to TensorFlow 2, added Habana device support and modified Horovod usage to Horovod function wrappers and global `hvd` object. For the details on changes, go to [CHANGES.md](./CHANGES.md).

Please visit [this page](../../../README.md#tensorflow-model-performance) for performance information.

### BERT Pre-Training
- Located in: `Model-References/TensorFlow/nlp/bert`
- The main training script is **`run_pretraining.py`**
- Suited for datasets:
  - `bookswiki`
  - `overfit`
- Uses optimizer: **LAMB** ("Layer-wise Adaptive Moments optimizer for Batch training").
- Consists of 2 phases:
  - Phase 1 - **Masked Language Model** - where given a sentence, a randomly chosen word is guessed.
  - Phase 2 - **Next Sentence Prediction** - where the model guesses whether sentence B comes after sentence A
- The resulting (trained) model weights are language-specific (here: English) and must be further "fitted" to do a specific task (with finetuning).
- Heavy-weight: the training takes several hours or days.

### BERT Fine-Tuning
- Located in: `Model-References/TensorFlow/nlp/bert`
- The main training script is **`run_classifier.py`** for MRPC and **`run_squad.py`** for SQuAD.
- Suited for tasks:
  - `mrpc`: Microsoft Research Paraphrase Corpus (**MRPC**) is a paraphrase identification dataset, where systems aim to identify if two sentences are paraphrases of each other.
  - `squad`: Stanford Question Answering Dataset (**SQuAD**) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.
- Uses optimizer: **AdamW** ("ADAM with Weight Decay Regularization").
- Based on model weights trained with pretraining.
- Light-weight: the training takes a minute or so.

## Setup

Please follow the instructions given in the following link for setting up the
environment including the `$PYTHON` and `$MPI_ROOT` environment variables:
[Gaudi Setup and Installation
Guide](https://github.com/HabanaAI/Setup_and_Install). Please answer the
questions in the guide according to your preferences. This guide will walk you
through the process of setting up your system to run the model on Gaudi.

### Docker Setup and Dataset Generation

In this section, we will first provide instructions to launch a Habana TensorFlow docker container and clone the [Model-References repository](https://github.com/HabanaAI/Model-References/). This is mainly applicable to non-Kubernetes configurations.

Next, we will provide instructions to download and preprocess the datasets, and copy them to locations that are on volumes mapped to the host (for persistence across container runs, since generating BERT datasets is a time-consuming process). This step is applicable to both non-K8s and K8s configurations.

### Run the docker container and clone the Model-References repository for non-K8s configurations only

We will assume there is a directory `$HOME/hlogs` on the host system which we will map as a container volume `<CONTAINER'S $HOME>/hlogs`. The BERT Python training examples given below re-direct stdout/stderr to a file in the container's `~/hlogs` directory. We will also assume that there is a directory `$HOME/tmp` on the host system, that contains sufficient disk space to hold the training output directories. We will map this directory as a container volume `<CONTAINER'S $HOME>/tmp`.

Please substitute `<CONTAINER'S $HOME>` with the path that running `echo $HOME` in the container returns, e.g. `/home/user1` or `/root`.

The following docker run command-line also assumes that the datasets that will be generated in the next sub-section titled [Download and preprocess the datasets for Pretraining and Finetuning for non-K8s and K8s configurations](#download-and-preprocess-the-datasets-for-pretraining-and-finetuning-for-non-k8s-and-k8s-configurations) will be manually copied to a directory `/data/tensorflow` on the host and mapped back to the container for subsequent training runs. This is because generating BERT datasets is a time-consuming process and we would like to generate the datasets once and reuse them for subsequent training runs in new docker sessions. Users can modify `/data/tensorflow` to a path of their choice.

```bash
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host -v $HOME/hlogs:<CONTAINER'S $HOME>/hlogs -v $HOME/tmp:<CONTAINER'S $HOME>/tmp -v /data/tensorflow:/data/tensorflow vault.habana.ai/gaudi-docker/${SYNAPSE_AI_VERSION}/${OS}/habanalabs/tensorflow-installer-tf-cpu-${TF_VERSION}:${SYNAPSE_AI_VERSION}-${SYNAPSE_BUILT_REVISION}
```

In the docker container, clone this repository and switch to the branch that
matches your SynapseAI version. (Run the
[`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.)

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```
Assuming the Model-References repository is located under `/root/`, set the PYTHONPATH and go to the BERT directory
```bash
export PYTHONPATH=/root/Model-References:/root/Model-References/TensorFlow/nlp/bert/:$PYTHONPATH

cd /root/Model-References/TensorFlow/nlp/bert/
```

### Install Model Requirements

In the docker container, go to the BERT directory
```bash
cd /root/Model-References/TensorFlow/nlp/bert
```
Install required packages using pip
```bash
$PYTHON -m pip install -r requirements.txt
```

### Download and preprocess the datasets for Pretraining and Finetuning for non-K8s and K8s configurations

#### Pretraining datasets download instructions

In `Model-References/TensorFlow/nlp/bert/data_preprocessing` folder, we provide scripts to download, extract and preprocess [Wikipedia](https://dumps.wikimedia.org/) and [BookCorpus](http://yknzhu.wixsite.com/mbweb) datasets.
To run the scripts, go to `data_preprocessing` folder and install required Python packages:

```bash
ln -s /usr/bin/python3.8 /usr/bin/python

cd /root/Model-References/TensorFlow/nlp/bert/data_preprocessing

pip install ipdb nltk progressbar html2text

apt-get update && apt-get install lbzip2
```
The pretraining dataset is 170GB+ and takes 15+ hours to download. The BookCorpus server gets overloaded most of the time and also contains broken links resulting in HTTP 403 and 503 errors. Hence, it is recommended to skip downloading BookCorpus with the script by running the following. By default, this script will download and preprocess the data in `/data/tensorflow/bert/books_wiki_en_corpus`.

```bash
export PYTHONPATH=/root/Model-References/TensorFlow/nlp/bert/:$PYTHONPATH
bash create_datasets_from_start.sh
```
Users are welcome to download BookCorpus from other sources to match our accuracy, or repeatedly try our script until the required number of files are downloaded by running the following:

```bash
bash create_datasets_from_start.sh wiki_books
```

#### Pretraining datasets packing instructions

We support the option to use a [data packing
technique](https://github.com/HabanaAI/Gaudi-tutorials/blob/main/TensorFlow/DataPackingMLperfBERT/Data_Packing_Process_for_MLPERF_BERT.ipynb),
called Non-Negative Least Squares Histogram. Here, instead of padding with zero,
we pack several short sequences into one multi-sequence of size `max_seq_len`.
Thus, we remove most of the padding, which can lead to a speedup of up to 2&times;
in time-to-train (TTT). This packing technique can be applied on other datasets
with high variability in samples length.

Please note that for each NLP dataset with sequential data samples, the speedup
with data packing is determined by the ratio of `max_seq_len` to
`average_seq_len` in that particular dataset. The larger the ratio, the higher
the speedup.

To pack the dataset, in docker run
```bash
cd /root/Model-References/TensorFlow/nlp/bert/data_preprocessing

$PYTHON pack_pretraining_data_tfrec.py --input-glob /data/tensorflow/bert/books_wiki_en_corpus/tfrecord/seq_len_128/books_wiki_en_corpus/training/ --output-dir /data/tensorflow/bert/books_wiki_en_corpus/tfrecord_packed/seq_len_128/books_wiki_en_corpus/training/ --max-sequence-length 128 --max-files 1472 --max-predictions-per-sequence 20

$PYTHON pack_pretraining_data_tfrec.py --input-glob /data/tensorflow/bert/books_wiki_en_corpus/tfrecord/seq_len_512/books_wiki_en_corpus/training/ --output-dir /data/tensorflow/bert/books_wiki_en_corpus/tfrecord_packed/seq_len_512/books_wiki_en_corpus/training/ --max-sequence-length 512 --max-files 1472 --max-predictions-per-sequence 80
```
#### Finetuning datasets download instructions

The MRPC dataset can be downloaded with `download/download_dataset.py` script. If `--dataset_path` option is not
specified for the download script, the MRPC dataset will be downloaded to `Model-References/TensorFlow/nlp/bert/dataset/MRPC`.
If needed, MRPC dataset can be moved to a shared directory, and this location can be provided as the `--dataset_path`
option to `run_classifier.py` during subsequent training runs.

The SQuAD dataset needs to be manually downloaded to a location of your choice, preferably a shared directory. This location should be provided as the `--dataset_path` option to `run_squad.py` when running BERT Finetuning with SQuAD. The examples that follow use `/data/tensorflow/bert/SQuAD` as this shared folder specified to `--dataset_path`.
The [SQuAD website](https://rajpurkar.github.io/SQuAD-explorer/) does not seem to link to the v1.1 datasets any longer,
but the necessary files can be found here:
- [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
- [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)

## TensorFlow BERT Training

In Model-References repository, we enabled TensorFlow BERT pretraining and finetuning on single-card, 8-cards and multi-server (16-cards, 32-cards, etc.)
The main training script for pretraining is **`run_pretraining.py`**. For finetuning with MRPC dataset, the main script is
`run_classifier.py`, and for finetuning with SQuAD dataset is `run_squad.py`.

For more details on the mixed precision training recipe customization via the `--bf16_config_path` option, please refer to the [TensorFlow Mixed Precision Training on Gaudi](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_User_Guide/TensorFlow_Mixed_Precision.html) documentation.

For more details on Horovod-based scaling of Gaudi on TensorFlow and using Host NICs vs. Gaudi NICs for multi-server scale-out training, please refer to the [Distributed Training with TensorFlow](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/index.html) documentation.

### Pre-requisites

Before start finetuning with `run_classifier.py` or `run_squad.py`, you need to download the pretrained
model for the appropriate BERT model variant as follows. For training with `run_pretraining.py`, you need to do the same
since `run_pretraining.py` requires `bert_config.json` which is in the downloaded package.

```bash
cd /root/Model-References/TensorFlow/nlp/bert/
export PYTHONPATH=./:../../common:../../:../../../central/:$PYTHONPATH
```

  For BERT Base:
```bash
  $PYTHON download/download_pretrained_model.py \
          "https://storage.googleapis.com/bert_models/2020_02_20/" \
          "uncased_L-12_H-768_A-12" \
          False
```

  For BERT Large:
```bash
  $PYTHON download/download_pretrained_model.py \
          "https://storage.googleapis.com/bert_models/2019_05_30/" \
          "wwm_uncased_L-24_H-1024_A-16" \
          True
```

## Training BERT in non-Kubernetes environments

### Single-card training

- Single-card pretraining Phase 1 of BERT Large in bfloat16 precision using BooksWiki dataset:

  ```bash
  cd /root/Model-References/TensorFlow/nlp/bert/

  TF_BF16_CONVERSION=/root/Model-References/TensorFlow/nlp/bert/bf16_config/bert.json \
  $PYTHON run_pretraining.py \
      --input_files_dir=/data/tensorflow/bert/books_wiki_en_corpus/tfrecord/seq_len_128/books_wiki_en_corpus/training \
      --init_checkpoint=wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt \
      --eval_files_dir=/data/tensorflow/bert/books_wiki_en_corpus/tfrecord/seq_len_128/books_wiki_en_corpus/test \
      --output_dir=/tmp/bert/phase_1 \
      --bert_config_file=wwm_uncased_L-24_H-1024_A-16/bert_config.json \
      --do_train=True \
      --do_eval=False \
      --train_batch_size=64 \
      --eval_batch_size=8 \
      --max_seq_length=128 \
      --max_predictions_per_seq=20 \
      --num_train_steps=7038 \
      --num_accumulation_steps=1024 \
      --num_warmup_steps=2000 \
      --save_checkpoints_steps=100 \
      --learning_rate=0.006  \
      --noamp --nouse_xla \
      --allreduce_post_accumulation=True \
      --dllog_path=/tmp/bert/phase_1/bert_dllog.json \
      --enable_scoped_allocator=False \
      --enable_packed_data_mode=False \
      --avg_seq_per_pack=1.0 \
      --resume=False \
  2>&1 | tee ~/hlogs/bert_large_pretraining_bf16_bookswiki_1card_phase1.txt
  ```

- Single-card pretraining Phase 2 of BERT Large in bfloat16 precision using BooksWiki dataset.
  - Initial checkpoint is from Phase 1
  - **For Gaudi2 use `--train_batch_size=16`**

  ```bash
  cd /root/Model-References/TensorFlow/nlp/bert/

  TF_BF16_CONVERSION=/root/Model-References/TensorFlow/nlp/bert/bf16_config/bert.json \
  $PYTHON run_pretraining.py \
      --input_files_dir=/data/tensorflow/bert/books_wiki_en_corpus/tfrecord/seq_len_512/books_wiki_en_corpus/training \
      --init_checkpoint=/tmp/bert/phase_1/model.ckpt-7038 \
      --eval_files_dir=/data/tensorflow/bert/books_wiki_en_corpus/tfrecord/seq_len_512/books_wiki_en_corpus/test \
      --output_dir=/tmp/bert/phase_2 \
      --bert_config_file=wwm_uncased_L-24_H-1024_A-16/bert_config.json \
      --do_train=True \
      --do_eval=False \
      --train_batch_size=8 \ # 16 for Gaudi2
      --eval_batch_size=8 \
      --max_seq_length=512 \
      --max_predictions_per_seq=80 \
      --num_train_steps=1564 \
      --num_accumulation_steps=4096 \
      --num_warmup_steps=200 \
      --save_checkpoints_steps=100 \
      --learning_rate=0.004  \
      --noamp \
      --nouse_xla \
      --allreduce_post_accumulation=True \
      --dllog_path=/tmp/bert/phase_2/bert_dllog.json \
      --enable_packed_data_mode=False \
      --avg_seq_per_pack=1.0 \
      --resume=False \
  2>&1 | tee ~/hlogs/bert_large_pretraining_bf16_bookswiki_1card_phase2.txt
  ```
  **Note:** Please make sure the folder for `bert_dllog.json ` exists. In the example above, make sure `/tmp/bert/phase_1/` and `/tmp/bert/phase_2/` exists.

- Single-card finetuning of BERT Large in bfloat16 precision using MRPC dataset:

  ```bash
  cd /root/Model-References/TensorFlow/nlp/bert/

  TF_BF16_CONVERSION=/root/Model-References/TensorFlow/nlp/bert/bf16_config/bert.json \
  $PYTHON run_classifier.py \
      --task_name=MRPC \
      --do_train=true \
      --do_eval=true \
      --data_dir=/data/tensorflow/bert/MRPC \
      --vocab_file=wwm_uncased_L-24_H-1024_A-16/vocab.txt \
      --bert_config_file=wwm_uncased_L-24_H-1024_A-16/bert_config.json \
      --init_checkpoint=wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt \
      --max_seq_length=128 \
      --train_batch_size=64 \
      --learning_rate=2e-05 \
      --num_train_epochs=3 \
      --output_dir=/tmp/bert \
      --use_horovod=false \
      --enable_scoped_allocator=False \
  2>&1 | tee ~/hlogs/bert_large_finetuning_bf16_mrpc_1_card.txt
  ```

- Single-card finetuning of BERT Large in bfloat16 precision using SQuAD dataset:
  ```bash
  cd /root/Model-References/TensorFlow/nlp/bert/

  TF_BF16_CONVERSION=/root/Model-References/TensorFlow/nlp/bert/bf16_config/bert.json \
  $PYTHON run_squad.py \
      --vocab_file=wwm_uncased_L-24_H-1024_A-16/vocab.txt \
      --bert_config_file=wwm_uncased_L-24_H-1024_A-16/bert_config.json \
      --init_checkpoint=wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt \
      --do_train=True \
      --train_file=/data/tensorflow/bert/SQuAD/train-v1.1.json \
      --do_predict=True \
      --predict_file=/data/tensorflow/bert/SQuAD/dev-v1.1.json \
      --do_eval=True \
      --train_batch_size=24 \
      --learning_rate=2e-05 \
      --num_train_epochs=2 \
      --max_seq_length=384 \
      --doc_stride=128 \
      --output_dir=/tmp/bert \
      --use_horovod=false \
      --enable_scoped_allocator=False \
  2>&1 | tee ~/hlogs/bert_large_finetuning_bf16_squad_1_card.txt
  ```

### Multi-card/single-server Horovod-based distributed training

Multi-card training has been enabled for TensorFlow BERT by using `mpirun` command with `--horovod` option
to call `run_pretraining.py`, `run_classifier.py` or `run_squad.py`.
Arguments for `mpirun` command in the subsequent examples are setup for best performance on a 56 core CPU host.
To run it on a system with lower core count, change the `--map-by` argument value.

*<br>mpirun map-by PE attribute value may vary on your setup and should be calculated as:<br>
socket:PE = floor((number of physical cores) / (number of gaudi devices per each node))*

```bash
cd /root/Model-References/TensorFlow/nlp/bert/

mpirun --allow-run-as-root \
       --tag-output \
       --merge-stderr-to-stdout \
       --output-filename /root/tmp/bert_log/ \
       --bind-to core \
       --map-by socket:PE=7 \
       -np 8 \
       $PYTHON <bert_script> --horovod ...
```

- 8 Gaudi cards pretraining Phase 1 of BERT Large in bfloat16 precision using BooksWiki dataset:
*<br>mpirun map-by PE attribute value may vary on your setup and should be calculated as:<br>
socket:PE = floor((number of physical cores) / (number of gaudi devices per each node))*

  ```bash
  cd /root/Model-References/TensorFlow/nlp/bert/

  mpirun --allow-run-as-root \
      --tag-output \
      --merge-stderr-to-stdout \
      --output-filename /root/tmp/bert_phase1_log \
      --bind-to core \
      --map-by socket:PE=7 \
      -np 8 \
      -x TF_BF16_CONVERSION=/root/Model-References/TensorFlow/nlp/bert/bf16_config/bert.json \
      $PYTHON run_pretraining.py \
          --input_files_dir=/data/tensorflow/bert/books_wiki_en_corpus/tfrecord/seq_len_128/books_wiki_en_corpus/training \
          --init_checkpoint=wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt \
          --eval_files_dir=/data/tensorflow/bert/books_wiki_en_corpus/tfrecord/seq_len_128/books_wiki_en_corpus/test \
          --output_dir=/root/tmp/bert/phase_1 \
          --bert_config_file=wwm_uncased_L-24_H-1024_A-16/bert_config.json \
          --do_train=True \
          --do_eval=False \
          --train_batch_size=64 \
          --eval_batch_size=8 \
          --max_seq_length=128 \
          --max_predictions_per_seq=20 \
          --num_train_steps=7038 \
          --num_accumulation_steps=128 \
          --num_warmup_steps=2000 \
          --save_checkpoints_steps=100 \
          --learning_rate=0.00075 \
          --horovod \
          --noamp \
          --nouse_xla \
          --allreduce_post_accumulation=True \
          --dllog_path=/root/dlllog/bert_dllog.json \
          --enable_scoped_allocator=False \
          --enable_packed_data_mode=False \
          --avg_seq_per_pack=1.0 \
          --resume=False \
  2>&1 | tee bert_large_pretraining_bf16_bookswiki_8_cards_phase1.log
  ```

- 8 Gaudi cards pretraining Phase 2 of BERT Large in bfloat16 precision using BooksWiki dataset.
  - Initial checkpoint is from Phase 1

*<br>mpirun map-by PE attribute value may vary on your setup and should be calculated as:<br>
socket:PE = floor((number of physical cores) / (number of gaudi devices per each node))*

```bash
  cd /root/Model-References/TensorFlow/nlp/bert/

  mpirun --allow-run-as-root \
      --tag-output \
      --merge-stderr-to-stdout \
      --output-filename /root/tmp/bert_phase2_log \
      --bind-to core \
      --map-by socket:PE=7 \
      -np 8 \
      -x TF_BF16_CONVERSION=/root/Model-References/TensorFlow/nlp/bert/bf16_config/bert.json \
      $PYTHON run_pretraining.py \
          --input_files_dir=/data/tensorflow/bert/books_wiki_en_corpus/tfrecord/seq_len_512/books_wiki_en_corpus/training \
          --init_checkpoint=/root/tmp/bert/phase_1/model.ckpt-7038 \
          --eval_files_dir=/data/tensorflow/bert/books_wiki_en_corpus/tfrecord/seq_len_512/books_wiki_en_corpus/test \
          --output_dir=/root/tmp/bert/phase_2 \
          --bert_config_file=wwm_uncased_L-24_H-1024_A-16/bert_config.json \
          --do_train=True \
          --do_eval=False \
          --train_batch_size=8 \
          --eval_batch_size=8 \
          --max_seq_length=512 \
          --max_predictions_per_seq=80 \
          --num_train_steps=1564 \
          --num_accumulation_steps=4096 \
          --num_warmup_steps=200 \
          --save_checkpoints_steps=100 \
          --learning_rate=0.004  \
          --horovod \
          --noamp \
          --nouse_xla \
          --allreduce_post_accumulation=True \
          --dllog_path=/root/dlllog/bert_dllog.json \
          --enable_packed_data_mode=False \
          --avg_seq_per_pack=1.0 \
          --resume=False \
  2>&1 | tee bert_large_pretraining_bf16_bookswiki_8_cards_phase2.log
  ```

- 8 Gaudi cards pretraining of BERT Large **with packed data** in bfloat16 precision using BooksWiki dataset on a single box (8 cards):

*<br>mpirun map-by PE attribute value may vary on your setup and should be calculated as:<br>
socket:PE = floor((number of physical cores) / (number of gaudi devices per each node))*

  TensorFlow BERT pretraining with packed dataset was enabled with `--enable_packed_data_mode=True` option.
  In phase 1, we use `--avg_seq_per_pack=1.2`. In phase 2, we use `--avg_seq_per_pack=2`.

  ```bash
  cd /root/Model-References/TensorFlow/nlp/bert/
  ```

  ```bash
  mpirun --allow-run-as-root \
      --tag-output \
      --merge-stderr-to-stdout \
      --output-filename /root/tmp/bert_phase1_log \
      --bind-to core \
      --map-by socket:PE=7 \
      -np 8 \
      -x TF_BF16_CONVERSION=/root/Model-References/TensorFlow/nlp/bert/bf16_config/bert.json \
      $PYTHON run_pretraining.py \
          --input_files_dir=/data/tensorflow/bert/books_wiki_en_corpus/tfrecord_packed/seq_len_128/books_wiki_en_corpus/training \
          --init_checkpoint=wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt \
          --eval_files_dir=/data/tensorflow/bert/books_wiki_en_corpus/tfrecord_packed/seq_len_128/books_wiki_en_corpus/test \
          --output_dir=/root/tmp/bert/phase_1 \
          --bert_config_file=wwm_uncased_L-24_H-1024_A-16/bert_config.json \
          --do_train=True \
          --do_eval=False \
          --train_batch_size=64 \
          --eval_batch_size=8 \
          --max_seq_length=128 \
          --max_predictions_per_seq=20 \
          --num_train_steps=7038 \
          --num_accumulation_steps=128 \
          --num_warmup_steps=2000 \
          --save_checkpoints_steps=100 \
          --learning_rate=0.00075 \
          --horovod \
          --noamp \
          --nouse_xla \
          --allreduce_post_accumulation=True \
          --dllog_path=/root/dlllog/bert_dllog.json \
          --enable_scoped_allocator=False \
          --enable_packed_data_mode=True \
          --avg_seq_per_pack=1.2 \
          --resume=False \
  2>&1 | tee bert_large_pretraining_bf16_bookswiki_8_cards_phase1.log
  ```

  For Phase 2:

  ```bash
  mpirun --allow-run-as-root \
      --tag-output \
      --merge-stderr-to-stdout \
      --output-filename /root/tmp/bert_phase2_log \
      --bind-to core \
      --map-by socket:PE=7 \
      -np 8 \
      -x TF_BF16_CONVERSION=/root/Model-References/TensorFlow/nlp/bert/bf16_config/bert.json \
      $PYTHON run_pretraining.py \
          --input_files_dir=/data/tensorflow/bert/books_wiki_en_corpus/tfrecord_packed/seq_len_512/books_wiki_en_corpus/training \
          --init_checkpoint=/root/tmp/bert/phase_1/model.ckpt-7038 \
          --eval_files_dir=/data/tensorflow/bert/books_wiki_en_corpus/tfrecord_packed/seq_len_512/books_wiki_en_corpus/test \
          --output_dir=/root/tmp/bert/phase_2 \
          --bert_config_file=wwm_uncased_L-24_H-1024_A-16/bert_config.json \
          --do_train=True \
          --do_eval=False \
          --train_batch_size=8 \
          --eval_batch_size=8 \
          --max_seq_length=512 \
          --max_predictions_per_seq=80 \
          --num_train_steps=1564 \
          --num_accumulation_steps=4096 \
          --num_warmup_steps=200 \
          --save_checkpoints_steps=100 \
          --learning_rate=0.004  \
          --horovod \
          --noamp \
          --nouse_xla \
          --allreduce_post_accumulation=True \
          --dllog_path=/root/dlllog/bert_dllog.json \
          --enable_packed_data_mode=True \
          --avg_seq_per_pack=2 \
          --resume=False \
  2>&1 | tee bert_large_pretraining_bf16_bookswiki_8_cards_phase2.log
  ```

- 8 Gaudi cards finetuning of BERT Large in bfloat16 precision using MRPC dataset on a single box (8 cards):

*<br>mpirun map-by PE attribute value may vary on your setup and should be calculated as:<br>
socket:PE = floor((number of physical cores) / (number of gaudi devices per each node))*

  ```bash
  cd /root/Model-References/TensorFlow/nlp/bert/

  mpirun --allow-run-as-root \
      --tag-output \
      --merge-stderr-to-stdout \
      --output-filename /root/tmp/bert_log \
      --bind-to core \
      --map-by socket:PE=7 \
      -np 8 \
      -x TF_BF16_CONVERSION=/root/Model-References/TensorFlow/nlp/bert/bf16_config/bert.json \
      $PYTHON run_classifier.py \
          --task_name=MRPC \
          --do_train=true \
          --do_eval=true \
          --data_dir=/data/tensorflow/bert/MRPC/ \
          --vocab_file=wwm_uncased_L-24_H-1024_A-16/vocab.txt \
          --bert_config_file=wwm_uncased_L-24_H-1024_A-16/bert_config.json \
          --init_checkpoint=wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt \
          --max_seq_length=128 \
          --train_batch_size=64 \
          --learning_rate=2e-05 \
          --num_train_epochs=3 \
          --output_dir=/tmp/bert \
          --use_horovod=true \
          --enable_scoped_allocator=False \
  2>&1 | tee bert_large_finetuning_bf16_mrpc_8_cards.txt
  ```

- 8 Gaudi cards finetuning of BERT Large in bfloat16 precision using SQuAD dataset on a single box (8 cards):

*<br>mpirun map-by PE attribute value may vary on your setup and should be calculated as:<br>
socket:PE = floor((number of physical cores) / (number of gaudi devices per each node))*

  ```bash
  cd /root/Model-References/TensorFlow/nlp/bert/

  mpirun --allow-run-as-root \
      --tag-output \
      --merge-stderr-to-stdout \
      --output-filename /root/tmp/bert_log \
      --bind-to core \
      --map-by socket:PE=7 \
      -np 8 \
      -x TF_BF16_CONVERSION=/root/Model-References/TensorFlow/nlp/bert/bf16_config/bert.json \
      $PYTHON run_squad.py \
          --vocab_file=wwm_uncased_L-24_H-1024_A-16/vocab.txt \
          --bert_config_file=wwm_uncased_L-24_H-1024_A-16/bert_config.json \
          --init_checkpoint=wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt \
          --do_train=True \
          --train_file=/data/tensorflow/bert/SQuAD/train-v1.1.json \
          --do_predict=True \
          --predict_file=/data/tensorflow/bert/SQuAD/dev-v1.1.json \
          --do_eval=True \
          --train_batch_size=24 \
          --learning_rate=2e-05 \
          --num_train_epochs=2 \
          --max_seq_length=384 \
          --doc_stride=128 \
          --output_dir=/tmp/bert \
          --use_horovod=true \
          --enable_scoped_allocator=False \
  2>&1 | tee bert_large_finetuning_bf16_squad_8_cards.log
  ```

### Multi-server Horovod-based scale-out distributed training

Multi-server support in the BERT Python scripts has been enabled using mpirun and Horovod, and has been tested with
2 boxes (16 Gaudi cards) and 4 boxes (32 Gaudi cards) configurations.

#### Docker ssh port setup for Multi-server training

Multi-server training works by setting these environment variables:

- **`MULTI_HLS_IPS`**: set this to a comma-separated list of host IP addresses
- `MPI_TCP_INCLUDE`: set this to a comma-separated list of interfaces or subnets. This variable will set the mpirun parameter: `--mca btl_tcp_if_include`. This parameter tells MPI which TCP interfaces to use for communication between hosts. You can specify interface names or subnets in the include list in CIDR notation e.g. MPI_TCP_INCLUDE=eno1. More details: [Open MPI documentation](https://www.open-mpi.org/faq/?category=tcp#tcp-selection). If you get mpirun `btl_tcp_if_include` errors, try un-setting this environment variable and let the training script automatically detect the network interface associated with the host IP address.
- `DOCKER_SSHD_PORT`: set this to the rsh port used by the sshd service in the docker container

This example shows how to setup for a 4 boxes training configuration. The IP addresses used are only examples:

```bash
# This environment variable is needed for multi-server training with Horovod.
# Set this to be a comma-separated string of host IP addresses, e.g.:
export MULTI_HLS_IPS="192.10.100.174,10.10.100.101,10.10.102.181,10.10.104.192"

# Set this to the network interface name for the ping-able IP address of the host on
# which the training script is run. This appears in the output of "ip addr".
# If you get mpirun btl_tcp_if_include errors, try un-setting this environment variable
# and let the training script automatically detect the network interface associated
# with the host IP address.
export MPI_TCP_INCLUDE="eno1"

# This is the port number used for rsh from the docker container, as configured
# in /etc/ssh/sshd_config
export DOCKER_SSHD_PORT=3022
```
By default, the Habana docker uses `port 3022` for ssh, and this is the default port configured in the training scripts.
Sometimes, mpirun can fail to establish the remote connection when there is more than one Habana docker session
running on the main server in which the Python training script is run. If this happens, you can set up a different ssh port as follows:

Follow [Setup](#setup) and [Docker setup and dataset generation](#docker-setup-and-dataset-generation) steps above on all server machines. In each server host's docker container:
```bash
vi /etc/ssh/sshd_config
```
Uncomment `#Port 22` and replace the port number with a different port number, example `Port 4022`. Next, restart the sshd service:
```bash
service ssh stop
service ssh start
```

Change the `DOCKER_SSHD_PORT` environment variable value to reflect this change into the Python scripts:
```bash
export DOCKER_SSHD_PORT=4022
```

#### Setup password-less ssh between all connected servers used in the scale-out training

1. Follow [Setup](#setup) and [Docker setup and dataset generation](#docker-setup-and-dataset-generation) steps above on all servers used in the scale-out training
2. Configure password-less ssh between all nodes:

   Do the following in all the nodes' docker sessions:
   ```bash
   mkdir ~/.ssh
   cd ~/.ssh
   ssh-keygen -t rsa -b 4096
   ```
   Copy id_rsa.pub contents from every node's docker to every other node's docker's ~/.ssh/authorized_keys (all public keys need to be in all hosts' authorized_keys):
   ```bash
   cat id_rsa.pub > authorized_keys
   vi authorized_keys
   ```
   Copy the contents from inside to other systems.
   Paste all hosts' public keys in all hosts' “authorized_keys” file.

3. On each system:
   Add all hosts (including itself) to known_hosts. If you configured a different docker sshd port, say `Port 4022`, in [Docker ssh port setup for multi-server training](#docker-ssh-port-setup-for-multi-hls-training), replace `-p 3022` with `-p 4022`. The IP addresses used are only examples:
   ```bash
   ssh-keyscan -p 3022 -H 192.10.100.174 >> ~/.ssh/known_hosts
   ssh-keyscan -p 3022 -H 10.10.100.101 >> ~/.ssh/known_hosts
   ssh-keyscan -p 3022 -H 10.10.102.181 >> ~/.ssh/known_hosts
   ssh-keyscan -p 3022 -H 10.10.104.192 >> ~/.ssh/known_hosts
   ```

#### Download pretrained model and MRPC dataset (if needed) on each node:

- Follow the instructions in [Pre-requisites](#Pre-requisites) section to download pretrained model for BERT Large or Base on each node.

- If finetuning for MRPC dataset, follow the instructions in the [Docker setup and dataset generation](#docker-setup-and-dataset-generation) section
  to download MRPC dataset on each node.

#### Run BERT training on multiple servers:

**Note that to run multi-server training over host NICs, environment variable `HOROVOD_HIERARCHICAL_ALLREDUCE=1` must be set.**

**Note that the total number of Gaudis used for training is determined by number of HLS servers specified by `MULTI_HLS_IPS` and number of workers per each HLS server.**

##### Examples

-  16 Gaudi cards pretraining Phase 1 of BERT Large in bfloat16 precision using BooksWiki dataset:
(The IP addresses in mpirun command are only examples.)

*<br>mpirun map-by PE attribute value may vary on your setup and should be calculated as:<br>
socket:PE = floor((number of physical cores) / (number of gaudi devices per each node))*

    ```bash
    cd /root/Model-References/TensorFlow/nlp/bert/

    mpirun --allow-run-as-root \
           --mca plm_rsh_args -p3022 \
           --bind-to core \
           --map-by socket:PE=7 \
           -np 16 \
           --mca btl_tcp_if_include 192.10.100.174/24 \
           --tag-output \
           --merge-stderr-to-stdout \
           --prefix $MPI_ROOT \
           -H 192.10.100.174:8,10.10.100.101:8 \
           -x GC_KERNEL_PATH \
           -x HABANA_LOGS \
           -x PYTHONPATH \
           -x TF_BF16_CONVERSION=/root/Model-References/TensorFlow/nlp/bert/bf16_config/bert.json \
           $PYTHON ./run_pretraining.py \
                    --input_files_dir=/data/tensorflow/bert/books_wiki_en_corpus/tfrecord/seq_len_128/books_wiki_en_corpus/training \
                    --eval_files_dir=/data/tensorflow/bert/books_wiki_en_corpus/tfrecord/seq_len_128/books_wiki_en_corpus/test \
                    --output_dir=/root/tmp/pretraining/phase_1 \
                    --bert_config_file=wwm_uncased_L-24_H-1024_A-16/bert_config.json \
                    --do_train=True \
                    --do_eval=False \
                    --train_batch_size=32 \
                    --eval_batch_size=8 \
                    --max_seq_length=128 \
                    --max_predictions_per_seq=20 \
                    --num_train_steps=100 \
                    --num_accumulation_steps=256 \
                    --num_warmup_steps=2000 \
                    --save_checkpoints_steps=100 \
                    --learning_rate=7.500000e-04 \
                    --horovod \
                    --noamp \
                    --nouse_xla \
                    --allreduce_post_accumulation=True \
                    --dllog_path=/root/tmp/pretraining/phase_1/bert_dllog.json
    ```
   -  16 Gaudi cards finetuning of BERT Large in bfloat16 precision using SQuAD dataset:
    (The IP addresses in mpirun command are only examples.)

   *<br>mpirun map-by PE attribute value may vary on your setup and should be calculated as:<br>
   socket:PE = floor((number of physical cores) / (number of gaudi devices per each node))*

    ```bash
    cd /root/Model-References/TensorFlow/nlp/bert/

    mpirun --allow-run-as-root \
           --mca plm_rsh_args -p3022 \
           --bind-to core \
           --map-by socket:PE=7 \
           -np 16 \
           --mca btl_tcp_if_include 10.211.162.97/24 \
           --tag-output \
           --merge-stderr-to-stdout \
           --prefix $MPI_ROOT \
           -H 10.211.162.97,10.211.160.140 \
           -x GC_KERNEL_PATH \
           -x HABANA_LOGS \
           -x PYTHONPATH \
           -x TF_BF16_CONVERSION=/root/Model-References/TensorFlow/nlp/bert/bf16_config/bert.json \
           $PYTHON ./run_squad.py \
                    --vocab_file=wwm_uncased_L-24_H-1024_A-16/vocab.txt \
                    --bert_config_file=wwm_uncased_L-24_H-1024_A-16/bert_config.json \
                    --init_checkpoint=wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt \
                    --do_train=True \
                    --train_file=/data/tensorflow/bert/SQuAD/train-v1.1.json \
                    --do_predict=True \
                    --predict_file=/data/tensorflow/bert/SQuAD/dev-v1.1.json \
                    --do_eval=True \
                    --train_batch_size=24 \
                    --learning_rate=3e-5 \
                    --num_train_epochs=0.5 \
                    --max_seq_length=384 \
                    --doc_stride=128 \
                    --output_dir=/root/tmp/squad_large/ \
                    --use_horovod=true
    ```

## Training BERT in Kubernetes Environment

- Set up the `PYTHONPATH` environment variable based on the file-system path to the Model-References repository.
For example :
    ```bash
    export PYTHONPATH=/root/Model-References:/usr/lib/habanalabs
    ```

- Follow the instructions in [Pre-requisites](#Pre-requisites) section to download the pretrained model
  for BERT Large or Base to a location that can be accessed by the workers, such as the folder for dataset.

- If finetuning for MRPC dataset, follow the instructions in the [Docker setup and dataset generation](#docker-setup-and-dataset-generation)
  section to download MRPC dataset to a folder that can be accessed by the workers.

- Make sure the Python packages in `requirements.txt` are installed in each worker node for pretraining.

- The command-line options for K8s environments are similar to those described earlier for non-K8s configurations, except for
multi-card distributed training, the mpirun command should be:

    ```bash
    mpirun ... -np <num_workers_total> $PYTHON run_pretraining.py --horovod  ...
    ```

    Or if for finetuning, the mpirun command should be:

     ```bash
     mpirun ... -np <num_workers_total> $PYTHON run_classifier.py --use_horovod=true  ...
     ```
*<br>mpirun map-by PE attribute value may vary on your setup and should be calculated as:<br>
socket:PE = floor((number of physical cores) / (number of gaudi devices per each node))*

### Single-card training on K8s

Commands for training TensorFlow BERT in K8S environment on single-card are same as in non-K8S environment.
Refer to [Training BERT in non-Kubernetes environments](#training-bert-in-non-kubernetes-environments) section for examples.


### Multi-card Horovod-based distributed training on K8s

**Note that to run multi-server training over host NICs, environment variable `HOROVOD_HIERARCHICAL_ALLREDUCE=1` must be set for each mpi process.**

The general command-line for multi-card Horovod-based distributed training over mpirun in K8s setups is as follows:
*<br>mpirun map-by PE attribute value may vary on your setup and should be calculated as:<br>
socket:PE = floor((number of physical cores) / (number of gaudi devices per each node))*

```bash
mpirun --allow-run-as-root \
       --bind-to core \
       --map-by socket:PE=6 \
       -np <num_workers_total> \
       --tag-output \
       --merge-stderr-to-stdout \
       bash -c "cd /root/Model-References/TensorFlow/nlp/bert;\
       $PYTHON ..."
```

#### Examples

- 8 Gaudi cards pretraining of BERT Large in bfloat16 precision using BooksWiki dataset on a K8s single box (8 cards):

  **Run multi-card pretraining with bookswiki, Phase 1:**

```bash
mpirun --allow-run-as-root \
   -np 8 \
   -x TF_BF16_CONVERSION=/root/Model-References/TensorFlow/nlp/bert/bf16_config/bert.json \
   $PYTHON /root/Model-References/TensorFlow/nlp/bert/run_pretraining.py \
       --input_files_dir=/data/tensorflow/bert/books_wiki_en_corpus/tfrecord/seq_len_128/books_wiki_en_corpus/training \
       --init_checkpoint=wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt \
       --eval_files_dir=/data/tensorflow/bert/books_wiki_en_corpus/tfrecord/seq_len_128/books_wiki_en_corpus/test \
       --output_dir=/output/tf/bert/phase_1 \
       --bert_config_file=wwm_uncased_L-24_H-1024_A-16/bert_config.json \
       --do_train=True \
       --do_eval=False \
       --train_batch_size=64 \
       --eval_batch_size=8 \
       --max_seq_length=128 \
       --max_predictions_per_seq=20 \
       --num_train_steps=7038 \
       --num_accumulation_steps=128 \
       --num_warmup_steps=2000 \
       --save_checkpoints_steps=100 \
       --learning_rate=0.00075 \
       --horovod \
       --noamp \
       --nouse_xla \
       --allreduce_post_accumulation=True \
       --dllog_path=/ouptput/bert/bert_dllog.json \
       --enable_scoped_allocator=False \
       --enable_packed_data_mode=False \
       --avg_seq_per_pack=1.0 \
       --resume=False \
2>&1 | tee 8card_pretrain_run_1.log
```

  **Run multi-card pretraining with bookswiki, Phase 2:**

```bash
 mpirun --allow-run-as-root \
     -np 8 \
     -x TF_BF16_CONVERSION=/root/Model-References/TensorFlow/nlp/bert/bf16_config/bert.json \
     $PYTHON /root/Model-References/TensorFlow/nlp/bert/run_pretraining.py \
         --input_files_dir=/data/tensorflow/bert/books_wiki_en_corpus/tfrecord/seq_len_128/books_wiki_en_corpus/training \
         --init_checkpoint=/output/tf/bert/phase_1/model.ckpt-7038 \
         --eval_files_dir=/data/tensorflow/bert/books_wiki_en_corpus/tfrecord/seq_len_128/books_wiki_en_corpus/test \
         --output_dir=/output/tf/bert/phase_2 \
         --bert_config_file=wwm_uncased_L-24_H-1024_A-16/bert_config.json \
         --do_train=True \
         --do_eval=False \
         --train_batch_size=8 \
         --eval_batch_size=8 \
         --max_seq_length=512 \
         --max_predictions_per_seq=80 \
         --num_train_steps=1564 \
         --num_accumulation_steps=4096 \
         --num_warmup_steps=200 \
         --save_checkpoints_steps=100 \
         --learning_rate=0.004  \
         --noamp \
         --nouse_xla \
         --allreduce_post_accumulation=True \
         --dllog_path=/root/dlllog/bert_dllog.json \
         --enable_packed_data_mode=False \
         --avg_seq_per_pack=1.0 \
         --resume=False \
 2>&1 | tee 8card_pretrain_run_2.log
```

- 8 Gaudi cards finetuning of BERT Large in bfloat16 precision using MRPC dataset on a K8s single box (8 cards):

*<br>mpirun map-by PE attribute value may vary on your setup and should be calculated as:<br>
socket:PE = floor((number of physical cores) / (number of gaudi devices per each node))*

```bash
mpirun --allow-run-as-root \
   -np 8 \
   -x TF_BF16_CONVERSION=/root/Model-References/TensorFlow/nlp/bert/bf16_config/bert.json \
   $PYTHON /root/Model-References/TensorFlow/nlp/bert/run_classifier.py \
       --task_name=MRPC \
       --do_train=true \
       --do_eval=true \
       --data_dir=/data/tensorflow/bert/MRPC/ \
       --vocab_file=wwm_uncased_L-24_H-1024_A-16/vocab.txt \
       --bert_config_file=wwm_uncased_L-24_H-1024_A-16/bert_config.json \
       --init_checkpoint=wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt \
       --max_seq_length=128 \
       --train_batch_size=64 \
       --learning_rate=2e-05 \
       --num_train_epochs=3 \
       --output_dir=/tmp/bert \
       --use_horovod=true \
       --enable_scoped_allocator=False \
2>&1 | tee bert_large_ft_mrpc_8cards.txt
```

- 8 Gaudi cards finetuning of BERT Large in bfloat16 precision using SQuAD dataset on a K8s single box (8 cards):

*<br>mpirun map-by PE attribute value may vary on your setup and should be calculated as:<br>
socket:PE = floor((number of physical cores) / (number of gaudi devices per each node))*

```bash
mpirun --allow-run-as-root \
   -np 8 \
   -x TF_BF16_CONVERSION=/root/Model-References/TensorFlow/nlp/bert/bf16_config/bert.json \
   bash -c "cd /root/Model-References/TensorFlow/nlp/bert; \
         $PYTHON run_squad.py \
         --vocab_file=wwm_uncased_L-24_H-1024_A-16/vocab.txt \
         --bert_config_file=wwm_uncased_L-24_H-1024_A-16/bert_config.json \
         --init_checkpoint=wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt \
         --do_train=True \
         --train_file=/data/tensorflow/bert/SQuAD/train-v1.1.json \
         --do_predict=True \
         --predict_file=/data/tensorflow/bert/SQuAD/dev-v1.1.json \
         --do_eval=True \
         --train_batch_size=24 \
         --learning_rate=2e-05 \
         --num_train_epochs=2 \
         --max_seq_length=384 \
         --doc_stride=128 \
         --output_dir=/tmp/bert/ \
         --use_horovod=true \
         --enable_scoped_allocator=False" \
2>&1 | tee bert_large_ft_squad_8cards.txt
```

## Profile

**Run training on 1 card with profiler**


- Single-card pretraining of BERT Large in bfloat16 precision using BooksWiki dataset:
    ```bash
        cd /root/Model-References/TensorFlow/nlp/bert/

        $PYTHON demo_bert.py \
            --command pretraining \
            --model_variant large \
            --data_type bf16 \
            --test_set bookswiki \
            --dataset_path /data/tensorflow/bert/books_wiki_en_corpus/tfrecord/ \
            --epochs 1
	    --no_steps_accumulation 1 \
            --iters 50,50 \
            --profile 30,31 \
        2>&1 | tee /tmp/hlogs/bert_large_pretraining_bf16_bookswiki_1_card.txt
    ```
    `--iter 50,50` - option will set 50 steps to phase one and 50 steps to phase two,
    both will be profiled. If you want to profile only the first phase please use as the following `--iter 50`.

- Single-card finetuning of BERT Large in bfloat16 precision using MRPC dataset:
    ```bash
        cd /root/Model-References/TensorFlow/nlp/bert/

        $PYTHON demo_bert.py \
            --command finetuning \
            --model_variant large \
            --data_type bf16 \
            --test_set mrpc \
            --dataset_path /data/tensorflow/bert/MRPC \
            --epochs 0.876 \
            --iterations_per_loop 1 \
            --profile 30,31 \
        2>&1 | tee ~/hlogs/bert_large_finetuning_bf16_mrpc_1_card.txt
    ```

- Single-card finetuning of BERT Large in bfloat16 precision using SQuAD dataset:
    ```bash
        cd /root/Model-References/TensorFlow/nlp/bert/

        $PYTHON demo_bert.py \
            --command finetuning \
            --model_variant large \
            --data_type bf16 \
            --test_set squad \
            --dataset_path /data/tensorflow/bert/SQuAD \
            --epochs 1 \
            --iterations_per_loop 1 \
            --iters 50 \
            --profile 30,31 \
        2>&1 | tee ~/hlogs/bert_large_finetuning_bf16_squad_1_card.txt
    ```

The above examples will produce profile trace for 2 steps (30,31)

## Supported Configuration

| Device | SynapseAI Version | TensorFlow Version(s)  |
|:------:|:-----------------:|:-----:|
| Gaudi  | 1.4.1             | 2.8.0 |
| Gaudi  | 1.4.1             | 2.7.1 |
| Gaudi2 | 1.4.1             | 2.8.0 |
| Gaudi2 | 1.4.1             | 2.7.1 |

## Changelog
### 1.2.0
* Cleanup script from deprecated habana_model_runner
* Added command-line flags: `num_train_steps`, `deterministic_run`, `deterministic_seed`
* Optimize dataset preparation code for BERT by replacing bzip2 with lbzip2 for faster decompression
* Added support for packed dataset, which can be controlled with flags: `enable_packed_data_mode` and `avg_seq_per_pack`.
  More information about data packing can be found here: https://developer.habana.ai/tutorials/tensorflow/data-packing-process-for-mlperf-bert/
* Updated requirements.txt
### 1.3.0
* Catch and propagate failed return codes in demo_bert.py
* Moved BF16 config json file from TensorFlow/common/ to model dir
* Modified steps in README to run topology on Kubernetes
* Removed redundant imports
* Added prefetch of dataset
* Change `python` or `python3` to `$PYTHON` to execute correct version based on environment setup.
### 1.4.0
* References to custom demo script were replaced by community entry points in README
* Unified references to bert directory in README
* Modified data preprocessing dir
* Import horovod-fork package directly instead of using Model-References' TensorFlow.common.horovod_helpers; wrapped horovod import with a try-catch block so that the user is not required to install this library when the model is being run on a single card
* Add BERT profiling support (command-line flag - `profile`)
### 1.4.1
* Added information to README about the instructions to run the model on Gaudi2
* Added information to README about Gaudi2 supported configurations.

## Known Issues

Running BERT Base and BERT Large Finetuning with SQuAD in 16-cards configuration, BS=24, Seq=384 raises a DataLossError "Data loss: corrupted record". Also, running BERT Base and BERT Large Pretraining in fp32 precision with BooksWiki and 16-cards configuration gives errors about "nodes in a cycle". These issues are being investigated.
