# BERT for TensorFlow

This directory provides a script to train a BERT model to achieve state-of-the-art accuracy, and is tested and maintained by Habana. To obtain model performance data, refer to the [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

For more information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

**Note**: The model is enabled on both first-gen Gaudi and Gaudi2.

## Table of Contents
* [Model-References](../../../README.md)
* [Model Overview](#model-overview)
* [Setup](#setup)
* [TensorFlow BERT Training](#tensorflow-bert-training)
* [Training BERT in non-Kubernetes environments](#training-bert-in-non-kubernetes-environments)
* [Training BERT in Kubernetes Environment](#training-bert-in-kubernetes-environment)
* [Pre-trained Model](#pre-trained-model)
* [Profile](#profile)
* [Supported Configuration](#supported-configuration)
* [Changelog](#changelog)
* [Known Issues](#known-issues)

## Model Overview

Bidirectional Encoder Representations from Transformers (BERT) is a technique for natural language processing (NLP) pre-training developed by Google.
BERT was created and published in 2018 by Jacob Devlin and his colleagues from Google.
Google is leveraging BERT to better understand user searches.

The original English-language BERT model comes with two pre-trained general types: (1) the BERTBASE model, a 12-layer, 768-hidden, 12-heads, 110M parameter neural network architecture, and (2) the BERTLARGE model, a 24-layer, 1024-hidden, 16-heads, 340M parameter neural network architecture; both of which were trained on the BooksCorpus with 800M words, and a version of the English Wikipedia with 2,500M words.

The scripts are a mix of Habana modified pre-training scripts taken from [NVIDIA GitHub](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT) and Habana modified fine-tuning scripts taken from [Google GitHub](https://github.com/google-research/bert). The training scripts were converted to TensorFlow 2, Habana device support was added and Horovod usage to Horovod function wrappers and global `hvd` object was modidied. For more details on these changes, go to [CHANGES.md](./CHANGES.md).


### BERT Pre-Training
-  The main training script is **`run_pretraining.py`**, located in: `Model-References/TensorFlow/nlp/bert`
- Suited for datasets:
  - `bookswiki`
  - `overfit`
- Uses optimizer: **LAMB** ("Layer-wise Adaptive Moments optimizer for Batch training").
- Consists of two tasks:
  - **Masked Language Model** - where when given a sentence, a randomly chosen word is guessed.
  - **Next Sentence Prediction** - where the model guesses whether sentence B comes after sentence A.
  Both tasks are predicted simultaneously in the training process.
- The resulting (trained) model weights are language-specific (here: English) and must be further "fitted" to do a specific task with fine-tuning.
- Heavy-weight: the training takes several hours or days.

### BERT Fine-Tuning
- The main training scripts are **`run_classifier.py`** for MRPC and **`run_squad.py`** for SQuAD, located in: `Model-References/TensorFlow/nlp/bert`.
- Suited for tasks:
  - `mrpc`: Microsoft Research Paraphrase Corpus (**MRPC**) is a paraphrase identification dataset, where systems aim to identify if two sentences are paraphrases of each other.
  - `squad`: Stanford Question Answering Dataset (**SQuAD**) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.
- Uses optimizer: **AdamW** ("ADAM with Weight Decay Regularization").
- Based on model weights trained with pre-training.
- Light-weight: The training takes a minute or so.

## Setup
Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) to set up the environment including the `$PYTHON` environment variable.  To achieve the best performance, please follow the methods outlined in the [Optimizing Training Platform guide](https://docs.habana.ai/en/latest/TensorFlow/Model_Optimization_TensorFlow/Optimization_Training_Platform.html).  
The guides will walk you through the process of setting up your system to run the model on Gaudi.  

### Docker Setup and Dataset Generation

This section first provides instructions to launch a Habana TensorFlow docker container and clone the [Model-References repository](https://github.com/HabanaAI/Model-References/). This is mainly applicable for non-Kubernetes (non-K8s) configurations.

This section also includes instructions to download and pre-process the datasets, and copy them to locations that are on volumes mapped to the host (for persistence across container runs, since generating BERT datasets is a time-consuming process). This step is applicable to both non-K8s and K8s configurations.

### Run the Docker Container

**Note:** Running the docker container is applicable for non-K8s configurations only.

For details on containers setup refer to: https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html#run-using-containers, and for mapping dataset to container refer to: https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html#map-dataset-to-docker

Further sections assumes that the directory, `$HOME/hlogs`, is on the host system and is mapped as a container volume `<CONTAINER'S $HOME>/hlogs`. The BERT Python training examples given below re-direct stdout/stderr to a file in the container's `~/hlogs` directory. Further sections also assumes that the directory, `$HOME/tmp`, is on the host system and contains sufficient disk space to hold the training output directories. This directory is mapped as a container volume, `<CONTAINER'S $HOME>/tmp`.

Please substitute `<CONTAINER'S $HOME>` with the path that running `echo $HOME` in the container returns, e.g. `/home/user1` or `/root`.

Next sections also asumes that the datasets generated in the sub-section titled [Download and Pre-process the Datasets for Pre-training and Fine-tuning](#download-and-preprocess-the-datasets-for-pretraining-and-finetuning) will be manually copied to the `/data/tensorflow` directory on the host and mapped back to the container for subsequent training runs. Since generating BERT datasets is a time-consuming process, it is more efficient to generate the datasets once and reuse them for subsequent training runs in new docker sessions. You can modify `/data/tensorflow` to a path of your choice.

### Clone Habana Model-References

**Note:** Cloning the Model-References repository is applicable for non-K8s configurations only.

In the docker container, clone this repository and switch to the branch that
matches your SynapseAI version. You can run the
[`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```
**Note:** Assuming the Model-References repository is located under `/root/` and is not in the PYTHONPATH, make sure to update by running the below and go to the BERT directory:
```bash
export PYTHONPATH=/root/Model-References:/root/Model-References/TensorFlow/nlp/bert/:$PYTHONPATH

cd /root/Model-References/TensorFlow/nlp/bert/
```

### Install Model Requirements

1. In the docker container, go to the BERT directory:
```bash
cd /root/Model-References/TensorFlow/nlp/bert
```
2. Install the required packages using pip:
```bash
$PYTHON -m pip install -r requirements.txt
```

### Download and Pre-process the Datasets for Pre-training and Fine-tuning

**Note:** The following sub-sections are applicable for both non-K8s and K8s configurations.

### Download Pre-training Datasets

The required scripts to download, extract and preprocess [Wikipedia](https://dumps.wikimedia.org/) and [BookCorpus](http://yknzhu.wixsite.com/mbweb) datasets are located in the `Model-References/TensorFlow/nlp/bert/data_preprocessing` folder. To run the scripts, go to `data_preprocessing` folder and install the required Python packages:

<!-- DATASET download_bert_pretraining_tensorflow -->
```bash
ln -s /usr/bin/python3.8 /usr/bin/python

cd /root/Model-References/TensorFlow/nlp/bert/data_preprocessing

pip install ipdb nltk progressbar html2text

apt-get update && apt-get install lbzip2
```
<!-- /DATASET download_bert_pretraining_tensorflow -->
The pre-training dataset is 170GB+ and takes 15+ hours to download. The BookCorpus server gets overloaded most of the time and also contains broken links resulting in HTTP 403 and 503 errors. Hence, it is recommended to skip downloading BookCorpus with the script by running the following. By default, this script will download and pre-process the data in `/data/tensorflow/bert/books_wiki_en_corpus`.

<!-- DATASET download_bert_pretraining_tensorflow -->
```bash
export PYTHONPATH=/root/Model-References/TensorFlow/nlp/bert/:$PYTHONPATH
bash create_datasets_from_start.sh
```
<!-- /DATASET download_bert_pretraining_tensorflow -->
You can download BookCorpus from other sources to match Habana's accuracy, or repeatedly try our script until the required number of files are downloaded by running the following:

<!-- DATASET download_bert_pretraining_tensorflow -->
```bash
bash create_datasets_from_start.sh wiki_books
```
<!-- /DATASET download_bert_pretraining_tensorflow -->

### Packing Pre-training Datasets

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

To pack the dataset, in the docker run:
<!-- DATASET process_bert_pretraining_tensorflow -->
```bash
cd /root/Model-References/TensorFlow/nlp/bert/data_preprocessing

$PYTHON pack_pretraining_data_tfrec.py --input-glob /data/tensorflow/bert/books_wiki_en_corpus/tfrecord/seq_len_128/books_wiki_en_corpus/training/ --output-dir /data/tensorflow/bert/books_wiki_en_corpus/tfrecord_packed/seq_len_128/books_wiki_en_corpus/training/ --max-sequence-length 128 --max-files 1472 --max-predictions-per-sequence 20

$PYTHON pack_pretraining_data_tfrec.py --input-glob /data/tensorflow/bert/books_wiki_en_corpus/tfrecord/seq_len_512/books_wiki_en_corpus/training/ --output-dir /data/tensorflow/bert/books_wiki_en_corpus/tfrecord_packed/seq_len_512/books_wiki_en_corpus/training/ --max-sequence-length 512 --max-files 1472 --max-predictions-per-sequence 80
```
<!-- /DATASET process_bert_pretraining_tensorflow -->
The script will output information about the packing procedure to the console. Parameters of the packing process will also be saved to a JSON file in the output directory
one level above the folder with packed data. Below is an example structure of an output directory:
<!-- DATASET process_bert_pretraining_tensorflow -->
```bash
cd /data/tensorflow/bert/books_wiki_en_corpus/tfrecord_packed/seq_len_128/books_wiki_en_corpus/training/
ls ..
```
<!-- /DATASET process_bert_pretraining_tensorflow -->
```bash
training  training_metadata.json
```
The main training script will use this metadata to set proper parameters for training with the packed dataset.
### Download Fine-tuning Datasets

The MRPC dataset can be downloaded with `download/download_dataset.py` script. If `--dataset_path` option is not
specified for the download script, the MRPC dataset will be downloaded to `Model-References/TensorFlow/nlp/bert/dataset/MRPC`.
If needed, MRPC dataset can be moved to a shared directory, and this location can be provided as the `--dataset_path`
option to `run_classifier.py` during subsequent training runs.

The SQuAD dataset needs to be manually downloaded to a location of your choice, preferably a shared directory. This location should be provided as the `--dataset_path` option to `run_squad.py` when running BERT Fine-tuning with SQuAD. The examples that follow use `/data/tensorflow/bert/SQuAD` as this shared folder specified to `--dataset_path`.
The [SQuAD website](https://rajpurkar.github.io/SQuAD-explorer/) does not seem to link to the v1.1 datasets any longer,
but the necessary files can be found here:
- [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
- [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)

## Training

In Model-References repository, TensorFlow BERT pre-training and fine-tuning is enabled on single-card (1 HPU), multi-card (8 HPUs) and multi-server (16 HPUs, 32 HPUs, etc.) configurations.
The main training script for pre-training is `run_pretraining.py`. For fine-tuning with MRPC dataset, the main script is
`run_classifier.py`, and for fine-tuning with SQuAD dataset is `run_squad.py`.

For more details on the mixed precision training recipe customization via the `--bf16_config_path` option, please refer to the [TensorFlow Mixed Precision Training on Gaudi](https://docs.habana.ai/en/latest/TensorFlow/TensorFlow_Mixed_Precision/TensorFlow_Mixed_Precision.html) documentation.

For more details on Horovod-based scaling of Gaudi on TensorFlow and using Host NICs vs. Gaudi NICs for multi-server scale-out training, please refer to the [Distributed Training with TensorFlow](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/index.html) documentation.

### Pre-requisites

Before using fine-tuning with `run_classifier.py` or `run_squad.py`, you need to download the pre-trained
model for the appropriate BERT model variant as follows. For training with `run_pretraining.py`, you need to do the same
since `run_pretraining.py` requires `bert_config.json` which is in the downloaded package.

```bash
cd /root/Model-References/TensorFlow/nlp/bert/
export PYTHONPATH=./:../../common:../../:../../../central/:$PYTHONPATH
```

- For BERT Base:
```bash
  $PYTHON download/download_pretrained_model.py \
          "https://storage.googleapis.com/bert_models/2020_02_20/" \
          "uncased_L-12_H-768_A-12" \
          False
```

- For BERT Large:
```bash
  $PYTHON download/download_pretrained_model.py \
          "https://storage.googleapis.com/bert_models/2019_05_30/" \
          "wwm_uncased_L-24_H-1024_A-16" \
          True
```

## Training in non-Kubernetes Environments

### Single Card and Multi-Card Training Examples

**Run training on 1 HPU:**
- Pre-training Phase 1 of BERT Large, 1 HPU, bfloat16 precision, BooksWiki dataset:

  ```bash
  cd /root/Model-References/TensorFlow/nlp/bert/
  ```
  <!-- SNIPPET bert_pretraining_phase1_1xcard_bf16_bookswiki -->
  ```bash
  TF_BF16_CONVERSION=/root/Model-References/TensorFlow/nlp/bert/bf16_config/bert.json \
  $PYTHON run_pretraining.py \
      --input_files_dir=/data/tensorflow/bert/books_wiki_en_corpus/tfrecord/seq_len_128/books_wiki_en_corpus/training \
      --init_checkpoint= \
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
      --noamp \
      --nouse_xla \
      --allreduce_post_accumulation=True \
      --dllog_path=/tmp/bert/phase_1/bert_dllog.json \
      --enable_scoped_allocator=False \
      --resume=False \
  2>&1 | tee ~/hlogs/bert_large_pretraining_bf16_bookswiki_1card_phase1.txt
  ```
  <!-- /SNIPPET -->

- Pre-training Phase 2 of BERT Large, 1 HPU, bfloat16 precision, BooksWiki dataset.
  - Initial checkpoint is from Phase 1
  - **For Gaudi2 use `--train_batch_size=16` and `--num_accumulation_steps=2048`**

  ```bash
  cd /root/Model-References/TensorFlow/nlp/bert/
  ```
  <!-- SNIPPET bert_pretraining_phase2_1xcard_bf16_bookswiki -->
  ```bash
  TF_BF16_CONVERSION=/root/Model-References/TensorFlow/nlp/bert/bf16_config/bert.json \
  $PYTHON run_pretraining.py \
      --input_files_dir=/data/tensorflow/bert/books_wiki_en_corpus/tfrecord/seq_len_512/books_wiki_en_corpus/training \
      --init_checkpoint=/tmp/bert/phase_1/model.ckpt-7038 \
      --eval_files_dir=/data/tensorflow/bert/books_wiki_en_corpus/tfrecord/seq_len_512/books_wiki_en_corpus/test \
      --output_dir=/tmp/bert/phase_2 \
      --bert_config_file=wwm_uncased_L-24_H-1024_A-16/bert_config.json \
      --do_train=True \
      --do_eval=False \
      --train_batch_size={Gaudi:8|Gaudi2:16} \
      --eval_batch_size=8 \
      --max_seq_length=512 \
      --max_predictions_per_seq=80 \
      --num_train_steps=1564 \
      --num_accumulation_steps={Gaudi:4096|Gaudi2:2048} \
      --num_warmup_steps=200 \
      --save_checkpoints_steps=100 \
      --learning_rate=0.004  \
      --noamp \
      --nouse_xla \
      --allreduce_post_accumulation=True \
      --dllog_path=/tmp/bert/phase_2/bert_dllog.json \
      --resume=False \
  2>&1 | tee ~/hlogs/bert_large_pretraining_bf16_bookswiki_1card_phase2.txt
  ```
  <!-- /SNIPPET -->
  **Note:** Please make sure the folder for `bert_dllog.json ` exists. In the example above, make sure `/tmp/bert/phase_1/` and `/tmp/bert/phase_2/` exists.

- Fine-tuning of BERT Large, 1 HPU, bfloat16 precision, MRPC dataset:
  ```bash
  cd /root/Model-References/TensorFlow/nlp/bert/
  ```
  <!-- SNIPPET bert_finetuning_1xcard_bf16_mrpc -->
  ```bash
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
  <!-- /SNIPPET -->

- Fine-tuning of BERT Large, 1 HPU, bfloat16 precision, SQuAD dataset:
  ```bash
  cd /root/Model-References/TensorFlow/nlp/bert/
  ```
  <!-- SNIPPET bert_finetuning_1xcard_bf16_squad -->
  ```bash
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
  <!-- /SNIPPET -->
**Run training on 8 HPUs - Horovod**

**NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration).

For multi-card, distributed training is Horovod-based only.

Multi-card training has been enabled for TensorFlow BERT by using `mpirun` command with `--horovod` option
to call `run_pretraining.py`, `run_classifier.py` or `run_squad.py`.
Arguments for `mpirun` command in the subsequent examples are set up for best performance on a 56 core CPU host.
To run it on a system with lower core count, change the `--map-by` argument value.

```bash
cd /root/Model-References/TensorFlow/nlp/bert/
mpirun --allow-run-as-root \
       --tag-output \
       --merge-stderr-to-stdout \
       --output-filename /root/tmp/bert_log/ \
       --bind-to core \
       --map-by socket:PE=6 \
       -np 8 \
       $PYTHON <bert_script> --horovod ...
```
- Pre-training of BERT Large **with packed data** Phase 1, 8 HPUs, bfloat16 precision, BooksWiki dataset on a single server (8 cards):

  - Running in this mode requires the [packing technique](#pretraining-datasets-packing-instructions) to be applied to the dataset. Generated packed data directory containing the required metadata file needs to be specified as `input_files_dir`.

  ```bash
  cd /root/Model-References/TensorFlow/nlp/bert/
  ```
  <!-- SNIPPET bert_pretraining_phase1_8xcard_bf16_bookswiki_packed -->
  ```bash
  mpirun --allow-run-as-root \
      --tag-output \
      --merge-stderr-to-stdout \
      --output-filename /root/tmp/bert_phase1_log \
      --bind-to core \
      --map-by socket:PE=6 \
      -np 8 \
      -x TF_BF16_CONVERSION=/root/Model-References/TensorFlow/nlp/bert/bf16_config/bert.json \
      $PYTHON run_pretraining.py \
          --input_files_dir=/data/tensorflow/bert/books_wiki_en_corpus/tfrecord_packed/seq_len_128/books_wiki_en_corpus/training \
          --init_checkpoint= \
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
          --resume=False \
  2>&1 | tee bert_large_pretraining_bf16_bookswiki_8_cards_phase1.log
  ```
  <!-- /SNIPPET -->
- Pre-training of BERT Large **with packed data** Phase 2, 8 HPUs, bfloat16 precision, BooksWiki dataset on a single server (8 cards):

  ```bash
    cd /root/Model-References/TensorFlow/nlp/bert/
  ```
  <!-- SNIPPET bert_pretraining_phase2_8xcard_bf16_bookswiki_packed -->
  ```bash
  mpirun --allow-run-as-root \
      --tag-output \
      --merge-stderr-to-stdout \
      --output-filename /root/tmp/bert_phase2_log \
      --bind-to core \
      --map-by socket:PE=6 \
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
          --train_batch_size={Gaudi:8|Gaudi2:16} \
          --eval_batch_size=8 \
          --max_seq_length=512 \
          --max_predictions_per_seq=80 \
          --num_train_steps=1564 \
          --num_accumulation_steps={Gaudi:512|Gaudi2:256} \
          --num_warmup_steps=200 \
          --save_checkpoints_steps=100 \
          --learning_rate=0.0005  \
          --horovod \
          --noamp \
          --nouse_xla \
          --allreduce_post_accumulation=True \
          --dllog_path=/root/dlllog/bert_dllog.json \
          --resume=False \
  2>&1 | tee bert_large_pretraining_bf16_bookswiki_8_cards_phase2.log
  ```
  <!-- /SNIPPET -->

- Pre-training Phase 1 of BERT Large, 8 HPUs, unpacked data, bfloat16 precision, BooksWiki dataset:

  ```bash
  cd /root/Model-References/TensorFlow/nlp/bert/
  ```
  <!-- SNIPPET bert_pretraining_phase1_8xcard_bf16_bookswiki -->
  ```bash
  mpirun --allow-run-as-root \
      --tag-output \
      --merge-stderr-to-stdout \
      --output-filename /root/tmp/bert_phase1_log \
      --bind-to core \
      --map-by socket:PE=6 \
      -np 8 \
      -x TF_BF16_CONVERSION=/root/Model-References/TensorFlow/nlp/bert/bf16_config/bert.json \
      $PYTHON run_pretraining.py \
          --input_files_dir=/data/tensorflow/bert/books_wiki_en_corpus/tfrecord/seq_len_128/books_wiki_en_corpus/training \
          --init_checkpoint= \
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
          --resume=False \
  2>&1 | tee bert_large_pretraining_bf16_bookswiki_8_cards_phase1.log
  ```
  <!-- /SNIPPET -->

- Pre-training Phase 2 of BERT Large, 8 HPUs, unpacked data, bfloat16 precision, BooksWiki dataset.
  - Initial checkpoint is from Phase 1
  - **For Gaudi2 use `--train_batch_size=16` and `--num_accumulation_steps=256`**

  ```bash
    cd /root/Model-References/TensorFlow/nlp/bert/
  ```
  <!-- SNIPPET bert_pretraining_phase2_8xcard_bf16_bookswiki -->
  ```bash
    mpirun --allow-run-as-root \
        --tag-output \
        --merge-stderr-to-stdout \
        --output-filename /root/tmp/bert_phase2_log \
        --bind-to core \
        --map-by socket:PE=6 \
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
            --train_batch_size={Gaudi:8|Gaudi2:16} \
            --eval_batch_size=8 \
            --max_seq_length=512 \
            --max_predictions_per_seq=80 \
            --num_train_steps=1564 \
            --num_accumulation_steps={Gaudi:512|Gaudi2:256} \
            --num_warmup_steps=200 \
            --save_checkpoints_steps=100 \
            --learning_rate=0.0005  \
            --horovod \
            --noamp \
            --nouse_xla \
            --allreduce_post_accumulation=True \
            --dllog_path=/root/dlllog/bert_dllog.json \
            --resume=False \
    2>&1 | tee bert_large_pretraining_bf16_bookswiki_8_cards_phase2.log
    ```
    <!-- /SNIPPET -->

- Fine-tuning of BERT Large, 8 HPUs, bfloat16 precision, MRPC dataset on a single server (8 cards):

  **NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration).

  ```bash
  cd /root/Model-References/TensorFlow/nlp/bert/
  ```
  <!-- SNIPPET bert_finetuning_8xcard_bf16_mrpc -->
  ```bash
  mpirun --allow-run-as-root \
      --tag-output \
      --merge-stderr-to-stdout \
      --output-filename /root/tmp/bert_log \
      --bind-to core \
      --map-by socket:PE=6 \
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
  <!-- /SNIPPET -->

- Fine-tuning of BERT Large, 8 HPUs, bfloat16 precision, SQuAD dataset on a single server (8 cards):

  **NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration).

  ```bash
  cd /root/Model-References/TensorFlow/nlp/bert/
  ```
  <!-- SNIPPET bert_finetuning_8xcard_bf16_squad -->
  ```bash
  mpirun --allow-run-as-root \
      --tag-output \
      --merge-stderr-to-stdout \
      --output-filename /root/tmp/bert_log \
      --bind-to core \
      --map-by socket:PE=6 \
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
  <!-- /SNIPPET -->

### Multi-server Training Examples

Multi-server training examples are Horovod-based scale-out distributed training.
Multi-server support in the BERT Python scripts has been enabled using `mpirun` and Horovod, and has been tested with
4 server (32 Gaudi cards) configurations.

Multi-server training works by setting these environment variables:

- `MULTI_HLS_IPS`: Set this to a comma-separated list of host IP addresses.
- `MPI_TCP_INCLUDE`: Set this to a comma-separated list of interfaces or subnets. This variable will set the mpirun parameter: `--mca btl_tcp_if_include`. This parameter tells MPI which TCP interfaces to use for communication between hosts. You can specify interface names or subnets in the include list in CIDR notation e.g. MPI_TCP_INCLUDE=eno1. More details: [Open MPI documentation](https://www.open-mpi.org/faq/?category=tcp#tcp-selection). If you get mpirun `btl_tcp_if_include` errors, try un-setting this environment variable and let the training script automatically detect the network interface associated with the host IP address.
- `DOCKER_SSHD_PORT`: Set this to the rsh port used by the sshd service in the docker container.

This example shows how to setup a 4 server training configuration. The IP addresses used are only examples:

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
####  Multi-server Training Docker ssh Setup
By default, the Habana docker uses `port 3022` for ssh, and this is the default port configured in the training scripts.
Sometimes, mpirun can fail to establish the remote connection when there is more than one Habana docker session
running on the main server in which the Python training script is run. If this happens, you can set up a different ssh port as follows:

1. Follow [Setup](#setup) and [Docker setup and dataset generation](#docker-setup-and-dataset-generation) steps above on all server machines.
2. In each server host's docker container, run:
```bash
vi /etc/ssh/sshd_config
```
3. Uncomment `#Port 22` and replace the port number with a different port number, example `Port 4022`. Then, restart the sshd service:
```bash
service ssh stop
service ssh start
```

4. Change the `DOCKER_SSHD_PORT` environment variable value to reflect this change into the Python scripts:
```bash
export DOCKER_SSHD_PORT=4022
```

#### Set up Password-less ssh
To set up password-less ssh between all connected servers used in scale-out training, follow the below steps:

1. Follow [Setup](#setup) and [Docker setup and dataset generation](#docker-setup-and-dataset-generation) steps above on all servers used in scale-out training.
2. Run the following in all the nodes' docker sessions:
   ```bash
   mkdir ~/.ssh
   cd ~/.ssh
   ssh-keygen -t rsa -b 4096
   ```
3. Copy id_rsa.pub contents from every node's docker to every other node's docker's ~/.ssh/authorized_keys (all public keys need to be in all hosts' authorized_keys):
   ```bash
   cat id_rsa.pub > authorized_keys
   vi authorized_keys
   ```
4. Copy the contents from inside to other systems.
5. Paste all hosts' public keys in all hosts' “authorized_keys” file.
6. On each system, add all hosts (including itself) to known_hosts. If you configured a different docker sshd port, say `Port 4022`, in [Docker ssh port setup for multi-server training](#docker-ssh-port-setup-for-multi-hls-training), replace `-p 3022` with `-p 4022`. The IP addresses used are only examples:
   ```bash
   ssh-keyscan -p 3022 -H 192.10.100.174 >> ~/.ssh/known_hosts
   ssh-keyscan -p 3022 -H 10.10.100.101 >> ~/.ssh/known_hosts
   ssh-keyscan -p 3022 -H 10.10.102.181 >> ~/.ssh/known_hosts
   ssh-keyscan -p 3022 -H 10.10.104.192 >> ~/.ssh/known_hosts
   ```


#### Download Pre-trained Model and MRPC Dataset

To download pre-trained model and MRPC dataset (if needed) on each node:
- Follow the instructions in [Prerequisites](#Pre-requisites) section to download pretrained model for BERT Large or Base on each node.

- If fine-tuning for MRPC dataset, follow the instructions in the [Docker setup and dataset generation](#docker-setup-and-dataset-generation) section
  to download MRPC dataset on each node.

**Run training on 16 HPUs:**

- The total number of Gaudis used for training is determined by number of servers specified by `MULTI_HLS_IPS` and number of workers per each server.


**NOTE:**
- mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration).
- `$MPI_ROOT` environment variable is set automatically during Setup. See [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) for details.

i. Pre-training Phase 1 of BERT Large, 32 HPUs, bfloat16 precision, BooksWiki dataset. The IP addresses in mpirun command are only examples:

  ```bash
  cd /root/Model-References/TensorFlow/nlp/bert/
  ```
  <!-- SNIPPET bert_pretraining_phase1_32xcard_bf16_bookswiki -->
  ```bash
  mpirun --allow-run-as-root \
          --mca plm_rsh_args -p3022 \
          --bind-to core \
          --map-by socket:PE=6 \
          -np 32 \
          --mca btl_tcp_if_include 192.10.100.174/24 \
          --tag-output \
          --merge-stderr-to-stdout \
          --prefix $MPI_ROOT \
          -H 192.10.100.174:8,10.10.100.101:8,10.10.102.181:8,10.10.104.192:8 \
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
                  --train_batch_size=64 \
                  --eval_batch_size=8 \
                  --max_seq_length=128 \
                  --max_predictions_per_seq=20 \
                  --num_train_steps=7038 \
                  --num_accumulation_steps=32 \
                  --num_warmup_steps=2000 \
                  --save_checkpoints_steps=100 \
                  --learning_rate=0.0001875 \
                  --horovod \
                  --noamp \
                  --nouse_xla \
                  --allreduce_post_accumulation=True \
                  --dllog_path=/root/tmp/pretraining/phase_1/bert_dllog.json
  ```
  <!-- /SNIPPET -->
ii. Fine-tuning of BERT Large, 32 HPUs, bfloat16 precision, SQuAD dataset. The IP addresses in mpirun command are only examples:

  ```bash
  cd /root/Model-References/TensorFlow/nlp/bert/
  ```
  <!-- SNIPPET bert_finetuning_32xcard_bf16_bookswiki -->
  ```bash
  mpirun --allow-run-as-root \
          --mca plm_rsh_args -p3022 \
          --bind-to core \
          --map-by socket:PE=6 \
          -np 32 \
          --mca btl_tcp_if_include 192.10.100.174/24 \
          --tag-output \
          --merge-stderr-to-stdout \
          --prefix $MPI_ROOT \
          -H 192.10.100.174:8,10.10.100.101:8,10.10.102.181:8,10.10.104.192:8 \
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
  <!-- /SNIPPET -->

## Training in Kubernetes Environment

- Set up the `PYTHONPATH` environment variable based on the file-system path to the Model-References repository.
For example :
    ```bash
    export PYTHONPATH=/root/Model-References:/usr/lib/habanalabs
    ```

- Follow the instructions in [Prerequisites](#Prerequisites) section to download the pre-trained model
  for BERT Large or BERT Base to a location that can be accessed by the workers, such as the folder for dataset.

- If fine-tuning for MRPC dataset, follow the instructions in the [Docker setup and dataset generation](#docker-setup-and-dataset-generation)
  section to download MRPC dataset to a folder that can be accessed by the workers.

- Make sure the Python packages in `requirements.txt` are installed in each worker node for pre-training.

- The command-line options for K8s environments are similar to those described earlier for non-K8s configurations, except for
multi-card distributed training, the mpirun command should be:

    ```bash
    mpirun ... -np <num_workers_total> $PYTHON run_pretraining.py --horovod  ...
    ```

- Or, if for finetuning, the mpirun command should be:

     ```bash
     mpirun ... -np <num_workers_total> $PYTHON run_classifier.py --use_horovod=true  ...
     ```

### Single Card and Multi-card Training on K8s Examples
For multi-card, distributed training on K8s is Horovod-based.

**NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration).

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

**Run training on 1 HPUs:**
Commands for training TensorFlow BERT in K8S environment on single-card are same as in non-K8S environment.
Refer to [Training BERT in non-Kubernetes environments](#training-bert-in-non-kubernetes-environments) section for examples.

**Run training on 8 HPUs:**

- Pre-training of BERT Large Phase 1, 8 HPUs, bfloat16 precision, BooksWiki dataset on a K8s single server (8 cards):

  **NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration).

  <!-- SNIPPET bert_pretraining_phase1_8xcard_bf16_bookswiki_k8s -->
  ```bash
  mpirun --allow-run-as-root \
    -np 8 \
    -x TF_BF16_CONVERSION=/root/Model-References/TensorFlow/nlp/bert/bf16_config/bert.json \
    $PYTHON /root/Model-References/TensorFlow/nlp/bert/run_pretraining.py \
        --input_files_dir=/data/tensorflow/bert/books_wiki_en_corpus/tfrecord/seq_len_128/books_wiki_en_corpus/training \
        --init_checkpoint= \
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
        --resume=False \
  2>&1 | tee 8card_pretrain_run_1.log
  ```
  <!-- /SNIPPET -->

- Pre-training of BERT Large Phase 2, 8 HPUs, bfloat16 precision, BooksWiki dataset on a K8s single server (8 cards):

  <!-- SNIPPET bert_pretraining_phase2_8xcard_bf16_bookswiki_k8s -->
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
          --num_accumulation_steps=512 \
          --num_warmup_steps=200 \
          --save_checkpoints_steps=100 \
          --learning_rate=0.004  \
          --noamp \
          --nouse_xla \
          --allreduce_post_accumulation=True \
          --dllog_path=/root/dlllog/bert_dllog.json \
          --resume=False \
  2>&1 | tee 8card_pretrain_run_2.log
  ```
  <!-- /SNIPPET -->

- Fine-tuning of BERT Large, 8 HPUs, bfloat16 precision, MRPC dataset on a K8s single server (8 cards):

  **NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration).

  <!-- SNIPPET bert_finetuning_8xcard_bf16_mrpc_k8s -->
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
  <!-- /SNIPPET -->

- Fine-tuning of BERT Large, 8 HPUs, bfloat16 precision, SQuAD dataset on a K8s single server (8 cards):

  **NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration)..

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

## Pre-trained Model
TensorFlow BERT is trained on Intel® Gaudi® AI Accelerators and the checkpoint files are created. You can use the checkpoints for fine-tuning or transfer learning tasks with your own datasets. To download the checkpoint files, please refer to [Habana Catalog](https://developer.habana.ai/catalog/bert-for-tensorflow/) to obtain the URL.

Once downloaded, you can use the fine-tuning example commands given in the sections below with one change. You will need to substitute this line:

```bash
--init_checkpoint=wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt
```
with this:
```bash
--init_checkpoint=Model-References/TensorFlow/nlp/bert/pretrained_checkpoint/model.ckpt-1564
```


## Profile

You can run profiling using a `--profile` flag. For example, to gather profiling traces for two steps, 30 and 31, add `--profile 30,31` to your command.
This feature is supported in both pretraining and finetuning scripts.

The above examples will produce profile trace for 2 steps (30,31)

## Supported Configuration

| Validated on | SynapseAI Version | TensorFlow Version(s) | Mode |
|:------:|:-----------------:|:-----:|:----------:|
| Gaudi   | 1.14.0             | 2.15.0         | Training |
| Gaudi2  | 1.14.0             | 2.15.0         | Training |

## Changelog

### 1.6.0
* Removed the flags `enable_packed_data_mode` and `avg_seq_per_pack`.
### 1.5.0
* Deprecated the flags `enable_packed_data_mode` and `avg_seq_per_pack` and added support for automatic detection of those parameters based on dataset metadata file.

### 1.4.1
* Added information to README about the instructions to run the model on Gaudi2
* Added information to README about Gaudi2 supported configurations.

### 1.4.0
* References to custom demo script were replaced by community entry points in README
* Unified references to bert directory in README
* Modified data preprocessing dir
* Import horovod-fork package directly instead of using Model-References' TensorFlow.common.horovod_helpers; wrapped horovod import with a try-catch block so that the user is not required to install this library when the model is being run on a single card.
* Added BERT profiling support (command-line flag - `profile`).

### 1.3.0
* Catch and propagate failed return codes in demo_bert.py.
* Moved BF16 config json file from TensorFlow/common/ to model dir.
* Modified steps in README to run topology on Kubernetes.
* Removed redundant imports.
* Added prefetch of dataset.
* Change `python` or `python3` to `$PYTHON` to execute correct version based on environment setup.

### 1.2.0
* Cleanup script from deprecated habana_model_runner.
* Added command-line flags: `num_train_steps`, `deterministic_run`, `deterministic_seed`
* Optimize dataset preparation code for BERT by replacing bzip2 with lbzip2 for faster decompression
* Added support for packed dataset, which can be controlled with flags: `enable_packed_data_mode` and `avg_seq_per_pack`.
  More information about data packing can be found here: https://developer.habana.ai/tutorials/tensorflow/data-packing-process-for-mlperf-bert/
* Updated requirements.txt.
## Known Issues

Running BERT Base and BERT Large Fine-tuning with SQuAD in 16-card configuration, BS=24, Seq=384 raises a DataLossError "Data loss: corrupted record". Also, running BERT Base and BERT Large Pretraining in fp32 precision with BooksWiki and 16-card configuration gives errors about "nodes in a cycle". These issues are being investigated.
