# DistilBERT

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents

   * [Model Overview](#model-overview)
   * [Setup](#setup)
   * [Fine-Tuning with SQuAD](#fine-tuning-with-squad)
   * [Change Log](#change-log)

## Model Overview

The [DistilBERT model](https://huggingface.co/distilbert-base-uncased) was proposed in the blog post [Smaller, faster, cheaper, lighter: Introducing DistilBERT, a distilled version of BERT](https://medium.com/huggingface/distilbert-8cf3380435b5), and the paper [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108) by HuggingFace. DistilBERT is a small, fast, cheap and light Transformer model trained by distilling BERT base.

The DistilBERT model is a distilled version of the BERT base model, a 12-layer, 768-hidden, 12-heads, 110M parameter neural network architecture, trained on the BooksCorpus with 800M words, and a version of the English Wikipedia with 2,500M words.

The DistilBERT scripts reuse part of [BERT scripts](../bert) with necessary changes, for the details on changes, go to [CHANGES.md](./CHANGES.md). For now, the scripts only contain fine-tuning scripts taken from BERT scripts and using the DistilBERT pre-trained model from Huggingface transformers. We converted the training scripts to TensorFlow 2, added Habana device support and modified Horovod usage to Horovod function wrappers and global `hvd` object.

Please visit [this page](../../../README.md#tensorflow-model-performance) for performance information.

## Setup

Please follow the instructions given in the following link for setting up the
environment including the `$PYTHON` environment variable: [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html).
This guide will walk you through the process of setting up your system to run
the model on Gaudi.

## Fine-Tuning with SQuAD

- Located in: `Model-References/TensorFlow/nlp/distilbert`
- The main training scripts are **`demo_distilbert.py`**, **`distilbert_squad_main`** and **`run_squad.py`**.
  - **`demo_distilbert.py`** provides a uniform command-line scale-out training user interface that runs on non-Kubernetes (non-K8s) and Kubernetes (K8s) platforms. It does all the necessary setup work, such as downloading the BERT base pre-trained model (DistilBERT will reuse the `vocab.txt` of BERT base), preparing output directories, generating HCL config JSON files for SynapseAI during Horovod multi-card runs, checking for required files and directories at relevant stages during training, etc. before calling the `run_squad.py` TensorFlow training scripts. `demo_distilbert.py` internally uses mpirun to invoke these TensorFlow scripts in Horovod-based training mode for multi-card runs in non-K8s setup. For usage in K8s clusters, the user is expected to launch `demo_distilbert.py` via `mpirun`, as subsequent sections will describe.
  - **`distilbert_squad_main.py`** generates `distilbert_config.json` based on `bert_config.json`, downloads the DistilBERT pre-trained model by Huggingface transformers and builds command.
  - **`run_squad.py`**: script implementing finetuning with SQuAD.
- Suited for tasks:
  - `squad`: Stanford Question Answering Dataset (**SQuAD**) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.
- Uses optimizer: **AdamW** ("ADAM with Weight Decay Regularization").
- Based on model weights trained with pretraining, the script will download the pretrained model from HuggingFace transformers.
- Light-weight: the training takes a minute or so.

### Install Model Requirements

In the docker container, go to the DistilBERT directory
```bash
cd /root/Model-References/TensorFlow/nlp/distilbert
```
Install required packages using pip
```bash
$PYTHON -m pip install -r requirements.txt
```

### Download the datasets for Fine-Tuning

The SQuAD dataset needs to be manually downloaded to a location of your choice, preferably a shared directory. This location should be provided as the `--dataset_path` option to `demo_distilbert.py` when running DistilBERT Finetuning with SQuAD. The examples that follow use `/software/data/tf/data/bert/SQuAD` as this shared folder specified to `--dataset_path`.
The [SQuAD website](https://rajpurkar.github.io/SQuAD-explorer/) does not seem to link to the v1.1 datasets any longer,
but the necessary files can be found here:
- [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
- [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)

### Fine-Tuning with run_squad.py

#### Pre-requisites

Please run the following additional setup steps before calling `run_squad.py`.

- Set the PYTHONPATH:
```bash
cd /path/to/Model-References/TensorFlow/nlp/distilbert/
export PYTHONPATH=../../../:$PYTHONPATH
```

- Download the pretrained model for the BERT Base and Huggingface DistilBERT model variant:
```bash
$PYTHON download/download_pretrained_model.py \
        "https://storage.googleapis.com/bert_models/2020_02_20/" \
        "uncased_L-12_H-768_A-12" \
        "distilbert-base-uncased" \
        False
```

#### Example for single card

```bash
cd /path/to/Model-References/TensorFlow/nlp/distilbert/

$PYTHON ./run_squad.py \
        --vocab_file=uncased_L-12_H-768_A-12/vocab.txt \
        --distilbert_config_file=uncased_L-12_H-768_A-12/distilbert_config.json \
        --init_checkpoint=uncased_L-12_H-768_A-12/distilbert-base-uncased.ckpt-1 \
        --bf16_config_path=distilbert.json \
        --do_train=True \
        --train_file=/data/tensorflow/bert/SQuAD/train-v1.1.json \
        --do_predict=True \
        --predict_file=/data/tensorflow/bert/SQuAD/dev-v1.1.json \
        --do_eval=True \
        --train_batch_size=32 \
        --learning_rate=5e-05 \
        --num_train_epochs=2 \
        --max_seq_length=384 \
        --doc_stride=128 \
        --output_dir=/root/tmp/squad_distilbert/ \
        --use_horovod=false
```

#### Example for multiple cards using mpirun

Users who prefer to run multi-card fine-tuning by directly calling `run_squad.py` can do so using the `mpirun` command and passing the `--use_horovod` flag to these scripts.

-  8 Gaudi cards finetuning of DistilBERT base in bfloat16 precision using SQuAD dataset:

```bash
cd /path/to/Model-References/TensorFlow/nlp/distilbert/

mpirun --allow-run-as-root \
       --tag-output \
       --merge-stderr-to-stdout \
       --output-filename /root/tmp//demo_distilbert_log/ \
       --bind-to core \
       --map-by socket:PE=7 \
       -np 8 \
       $PYTHON ./run_squad.py \
                --vocab_file=uncased_L-12_H-768_A-12/vocab.txt \
                --distilbert_config_file=uncased_L-12_H-768_A-12/distilbert_config.json \
                --init_checkpoint=uncased_L-12_H-768_A-12/distilbert-base-uncased.ckpt-1 \
                --bf16_config_path=distilbert.json \
                --do_train=True \
                --train_file=/data/tensorflow/bert/SQuAD/train-v1.1.json \
                --do_predict=True \
                --predict_file=/data/tensorflow/bert/SQuAD/dev-v1.1.json \
                --do_eval=True \
                --train_batch_size=32 \
                --learning_rate=2e-04 \
                --num_train_epochs=2 \
                --max_seq_length=384 \
                --doc_stride=128 \
                --output_dir=/root/tmp/squad_distilbert/ \
                --use_horovod=true
```

### Fine-Tuning with demo_distilbert.py (alternative)

#### Example for single card

```bash
cd /path/to/Model-References/TensorFlow/nlp/distilbert/

  $PYTHON demo_distilbert.py \
     --command finetuning \
     --model_variant base \
     --data_type bf16 \
     --test_set squad \
     --dataset_path /software/data/tf/data/bert/SQuAD \
     --batch_size 32 \
     --learning_rate 5e-05 \
     --output_dir /root/tmp/squad_distilbert/
```

#### Example for multiple cards

-  8 Gaudi cards finetuning of DistilBERT base in bfloat16 precision using SQuAD dataset:

```bash
cd /path/to/Model-References/TensorFlow/nlp/distilbert/

  $PYTHON demo_distilbert.py \
     --command finetuning \
     --model_variant base \
     --data_type bf16 \
     --test_set squad \
     --dataset_path /software/data/tf/data/bert/SQuAD \
     --batch_size 32 \
     --learning_rate 2e-04 \
     --output_dir /root/tmp/squad_distilbert/ \
     --use_horovod 8
```

## Change Log

### 1.3.0

* Change `python` or `python3` to `$PYTHON` to execute correct version based on environment setup.
