# DistilBERT for TensorFlow

This directory provides a script to train a DistilBERT model to achieve state-of-the-art accuracy, and is tested and maintained by Habana. To obtain model performance data, refer to the [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

For more information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents
   * [Model-References](../../../README.md)
   * [Model Overview](#model-overview)
   * [Setup](#setup)
   * [Fine-Tuning with SQuAD](#fine-tuning-with-squad)
   * [Supported Configuration](#supported-configuration)
   * [Changelog](#change-log)

## Model Overview

The [DistilBERT model](https://huggingface.co/distilbert-base-uncased) was proposed in the blog post [Smaller, faster, cheaper, lighter: Introducing DistilBERT, a distilled version of BERT](https://medium.com/huggingface/distilbert-8cf3380435b5), and the paper [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108) by HuggingFace. DistilBERT is a small, fast, cheap and light Transformer model trained by distilling BERT base.

The DistilBERT model is a distilled version of the BERT base model, a 12-layer, 768-hidden, 12-heads, 110M parameter neural network architecture, trained on the BooksCorpus with 800M words, and a version of the English Wikipedia with 2,500M words.

The DistilBERT scripts reuse part of [BERT scripts](../bert) with the necessary Habana modifications as detailed in [CHANGES.md](./CHANGES.md). For now, the scripts only contain fine-tuning scripts taken from BERT scripts and using the DistilBERT pre-trained model from Huggingface transformers. The training scripts were converted to TensorFlow 2, Habana device support was added and horovod import was wrapped with a try-catch block so you are not required to install this library when the model is being run on a single card.

### Fine-Tuning with SQuAD

- The main training script is **`run_squad.py`** which is located in: `Model-References/TensorFlow/nlp/distilbert`
  - **`run_squad.py`**: script implementing fine-tuning with SQuAD.
- Suited for tasks:
  - `squad`: Stanford Question Answering Dataset (**SQuAD**) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.
- Uses optimizer: **AdamW** ("ADAM with Weight Decay Regularization").
- Based on model weights trained with pre-training, the script will download the pre-trained model from HuggingFace transformers.
- Light-weight: the training takes a minute or so.



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
**Note:** If Model-References repository path is not in the PYTHONPATH, make sure you update it:
```bash
cd /path/to/Model-References/TensorFlow/nlp/distilbert/
export PYTHONPATH=../../../:$PYTHONPATH
```
### Install Model Requirements

1. In the docker container, go to the DistilBERT directory:
```bash
cd /root/Model-References/TensorFlow/nlp/distilbert
```
2. Install the required packages using pip:
```bash
$PYTHON -m pip install -r requirements.txt
```

### Download Datasets for Fine-Tuning

The SQuAD dataset needs to be manually downloaded to a location of your choice, preferably a shared directory.
The [SQuAD website](https://rajpurkar.github.io/SQuAD-explorer/) does not seem to link to the v1.1 datasets any longer,
but the necessary files can be found here:
- [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
- [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)

## Fine-Tuning with run_squad.py

### Prerequisites
Before running the training, download the pre-trained model for the BERT Base and Huggingface DistilBERT model variant:

```bash
$PYTHON download/download_pretrained_model.py \
        "https://storage.googleapis.com/bert_models/2020_02_20/" \
        "uncased_L-12_H-768_A-12" \
        "distilbert-base-uncased" \
        False
```

### Single Card and Multi-Card Examples

**Run training on 1 HPU:**

```bash
cd /path/to/Model-References/TensorFlow/nlp/distilbert/

$PYTHON ./run_squad.py \
        --vocab_file=uncased_L-12_H-768_A-12/vocab.txt \
        --distilbert_config_file=uncased_L-12_H-768_A-12/distilbert_config.json \
        --init_checkpoint=uncased_L-12_H-768_A-12/distilbert-base-uncased.ckpt-1 \
        --bf16_config_path=bf16_config/distilbert.json \
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

**Run training on 8 HPUs:**

If you prefer to run multi-card fine-tuning by directly calling `run_squad.py`, use the `mpirun` command and pass the `--use_horovod` flag to these scripts.

**NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration).

Fine-tuning of DistilBERT base, 8 HPUs, bfloat16 precision, SQuAD dataset:
```bash
cd /path/to/Model-References/TensorFlow/nlp/distilbert/

mpirun --allow-run-as-root \
       --tag-output \
       --merge-stderr-to-stdout \
       --output-filename /root/tmp/distilbert_log/ \
       --bind-to core \
       --map-by socket:PE=6 \
       -np 8 \
       $PYTHON ./run_squad.py \
                --vocab_file=uncased_L-12_H-768_A-12/vocab.txt \
                --distilbert_config_file=uncased_L-12_H-768_A-12/distilbert_config.json \
                --init_checkpoint=uncased_L-12_H-768_A-12/distilbert-base-uncased.ckpt-1 \
                --bf16_config_path=bf16_config/distilbert.json \
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

## Supported Configuration

| Validated on  | SynapseAI Version | TensorFlow Version(s)  | Mode |
|:------:|:-----------------:|:-----:|-------------|
| Gaudi  | 1.7.1             | 2.10.1 | Training |
| Gaudi  | 1.7.1             | 2.8.4 | Training |

## Changelog

### 1.4.0
* References to custom demo script were replaced by community entry points in README.
* Import horovod-fork package directly instead of using Model-References' TensorFlow.common.horovod_helpers; wrapped horovod import with a try-catch block so that the user is not required to install this library when the model is being run on a single card.
* Removed setup_jemalloc from demo_distilbert.py.
### 1.3.0
* Changed `python` or `python3` to `$PYTHON` to execute correct version based on environment setup.
