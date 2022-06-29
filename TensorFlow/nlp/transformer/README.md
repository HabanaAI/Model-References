# Transformer

This repository provides a script to train the Transformer model for Tensorflow on Habana
Gaudi<sup>TM</sup> device. Please visit [this page](../../../README.md#tensorflow-model-performance) for performance information. For more information, visit [developer.habana.ai](https://developer.habana.ai/resources/).

## Table Of Contents
* [Model Reference](../../../README.md)
* [Model Overview](#model-overview)
* [Setup](#setup)
* [Training the Model](#training-the-model)
* [Examples](#examples)
* [Changelog](#changelog)
* [Known Issues](#Known-Issues)

## Model Overview

The Transformer is a Neural Machine Translation (NMT) model which uses attention mechanism to boost training speed and overall accuracy.
The model was initially introduced in [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
This implementation is based on [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) implementation (authors: Google Inc., Artit Wangperawong).
Support for other models than Transformer was removed. Also, Horovod support was implemented, together with some adjustments in the topology script which allowed to simplify the computational graph.
Available model variants are tiny, base and big.


### Model architecture

The Transformer model uses standard NMT encoder-decoder architecture. Unlike other NMT models, it doesn't use recurrent connections and operates on fixed size context window.
The encoder stack is made up of N identical layers. Each layer is composed of the following sublayers:
- Self-attention layer
- Feedforward network (which is 2 fully-connected layers)

The decoder stack is also made up of N identical layers. Each layer is composed of the sublayers:
- Self-attention layer
- Multi-headed attention layer combining encoder outputs with results from the previous self-attention layer.
- Feedforward network (2 fully-connected layers)

The encoder uses self-attention to compute a representation of the input sequence. The decoder generates the output sequence one token at a time, taking the encoder output and previous decoder-outputted tokens as inputs.
The model also applies embeddings on the input and output tokens, and adds a constant positional encoding. The positional encoding adds information about the position of each token.

The complete description of the Transformer architecture can be found in [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper.

## Setup
Please follow the instructions given in the following link for setting up the
environment including the `$PYTHON` environment variable: [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html).
This guide will walk you through the process of setting up your system to run
the model on Gaudi.

### Download and generate the dataset

In the docker container, clone this repository and switch to the branch that
matches your SynapseAI version. (Run the
[`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.)

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```
Go to the transformer directory and generate the dataset. The following script will save the dataset to `/data/tensorflow/wmt32k_packed/train`:
```bash
cd Model-References/TensorFlow/nlp/transformer/
$PYTHON datagen.py \
    --data_dir=/data/tensorflow/wmt32k_packed/train \
    --tmp_dir=/tmp/transformer_datagen \
    --problem=translate_ende_wmt32k_packed \
    --random_seed=429459
```

### Install Model Requirements

In the docker container, go to the Transformer directory
```bash
cd /root/Model-References/TensorFlow/nlp/transformer
```
Install required packages using pip
```bash
$PYTHON -m pip install -r requirements.txt
```

## Training the Model

To train the model on 1 Gaudi card
```bash
$PYTHON trainer.py \
    --data_dir=<path_to_dataset>/train \
    --problem=translate_ende_wmt32k_packed \
    --model=transformer \
    --hparams_set=transformer_<model_size> \
    --hparams=batch_size=<batch_size> \
    --output_dir=<path_to_output_dir> \
    --local_eval_frequency=<eval_frequency> \
    --train_steps=<train_steps> \
    --schedule=train \
    --use_hpu=True \
    --use_bf16=<use_bf16>
```

### Run training on single card

Example parameters:
  * batch size 4096
  * float32
  * transformer_big,
  * 300k steps with a checkpoint saved every 10k steps

```bash
$PYTHON trainer.py \
    --data_dir=/data/tensorflow/wmt32k_packed/train/ \
    --problem=translate_ende_wmt32k_packed \
    --model=transformer \
    --hparams_set=transformer_big \
    --hparams=batch_size=4096 \
    --output_dir=./translate_ende_wmt32k_packed/transformer_big/bs4096 \
    --local_eval_frequency=10000 \
    --train_steps=300000 \
    --schedule=train \
    --use_hpu=True
```

### Run training on multiple cards
Example parameters:
  * 8 workers
  * global batch size 8 * 4096
  * bfloat16
  * transformer_big,
  * 300k steps with a checkpoint saved every 50k steps
  * learning rate constant 2.5

#### via mpirun
*<br>mpirun map-by PE attribute value may vary on your setup and should be calculated as:<br>
socket:PE = floor((number of physical cores) / (number of gaudi devices per each node))*
```bash
mpirun \
    --allow-run-as-root --bind-to core --map-by socket:PE=7 --np 8 \
    --tag-output --merge-stderr-to-stdout \
    $PYTHON trainer.py \
        --data_dir=/data/tensorflow/wmt32k_packed/train/ \
        --problem=translate_ende_wmt32k_packed \
        --model=transformer \
        --hparams_set=transformer_big \
        --hparams=batch_size=4096,learning_rate_constant=2.5 \
        --output_dir=./translate_ende_wmt32k_packed/transformer_big/bs4096 \
        --local_eval_frequency=50000 \
        --train_steps=300000 \
        --schedule=train \
        --use_horovod=True \
        --use_hpu=True \
        --use_bf16=True
```

### Run training on multiple HLS

*<br>mpirun map-by PE attribute value may vary on your setup and should be calculated as:<br>
socket:PE = floor((number of physical cores) / (number of gaudi devices per each node))*
To enable multi-HLS scenario you can run above commands, but you need to provide MULTI_HLS_IPS environment variable set to IPs of used HLS servers. For example for 16 Gaudi devices use:
```bash
export MULTI_HLS_IPS=192.10.100.174,10.10.100.101
mpirun \
    --allow-run-as-root --bind-to core --map-by socket:PE=7 --np 8 \
    --tag-output --merge-stderr-to-stdout \
    $PYTHON trainer.py \
        --data_dir=/data/tensorflow/wmt32k_packed/train/ \
        --problem=translate_ende_wmt32k_packed \
        --model=transformer \
        --hparams_set=transformer_big \
        --hparams=batch_size=4096,learning_rate_constant=3.0 \
        --output_dir=./translate_ende_wmt32k_packed/transformer_big/bs4096 \
        --local_eval_frequency=50000 \
        --train_steps=150000 \
        --schedule=train \
        --use_horovod=True \
        --use_hpu=True \
        --use_bf16=True
```

For setups with 32 Gaudi devices it is recommended to use `learning_rate_constant` 3.5 and `train_steps` 75000:

*<br>mpirun map-by PE attribute value may vary on your setup and should be calculated as:<br>
socket:PE = floor((number of physical cores) / (number of gaudi devices per each node))*
```bash
export MULTI_HLS_IPS=192.10.100.174,10.10.100.101,10.10.100.102,10.10.100.103
mpirun \
    --allow-run-as-root --bind-to core --map-by socket:PE=7 --np 8 \
    --tag-output --merge-stderr-to-stdout \
    $PYTHON trainer.py \
        --data_dir=/data/tensorflow/wmt32k_packed/train/ \
        --problem=translate_ende_wmt32k_packed \
        --model=transformer \
        --hparams_set=transformer_big \
        --hparams=batch_size=4096,learning_rate_constant=3.5 \
        --output_dir=./translate_ende_wmt32k_packed/transformer_big/bs4096 \
        --local_eval_frequency=50000 \
        --train_steps=75000 \
        --schedule=train \
        --use_horovod=True \
        --use_hpu=True \
        --use_bf16=True
```

## Evaluating BLEU Score

After training the model you can evaluate achieved BLEU score. First you need to download the validation file and tokenize it:
```bash
sacrebleu -t wmt14 -l en-de --echo src > wmt14.src
cat wmt14.src | sacremoses tokenize -l en > wmt14.src.tok
```

Then you can compute BLEU score of a single checkpoint by running the following command:
```bash
$PYTHON decoder.py \
    --problem=translate_ende_wmt32k_packed \
    --model=transformer \
    --hparams_set=transformer_big \
    --data_dir=<path_to_dataset>/train \
    --output_dir=<path_to_output_dir> \
    --checkpoint_path=<path_to_checkpoint> \
    --use_hpu=True \
    --decode_from_file=./wmt14.src.tok \
    --decode_to_file=./wmt14.tgt.tok
cat wmt14.tgt.tok | sacremoses detokenize -l de | sacrebleu -t wmt14 -l en-de
```
To split BLEU calculation to multiple cards run decoder.py through mpirun. For example:

*<br>mpirun map-by PE attribute value may vary on your setup and should be calculated as:<br>
socket:PE = floor((number of physical cores) / (number of gaudi devices per each node))*
```bash
mpirun \
    --allow-run-as-root --bind-to core --map-by socket:PE=7 --np 8 \
    --tag-output --merge-stderr-to-stdout \
    $PYTHON decoder.py \
        --problem=translate_ende_wmt32k_packed \
        --model=transformer \
        --hparams_set=transformer_big \
        --data_dir=<path_to_dataset>/train \
        --output_dir=<path_to_output_dir> \
        --checkpoint_path=<path_to_checkpoint> \
        --decode_from_file=./wmt14.src.tok \
        --decode_to_file=./wmt14.tgt.tok \
        --use_hpu=True \
        --use_horovod=True
cat wmt14.tgt.tok | sacremoses detokenize -l de | sacrebleu -t wmt14 -l en-de
```

## Advanced parameters

To get a list of all supported parameters and their default values run:
```bash
$PYTHON demo_transformer.py --help
```

## Supported Configuration

| xHPU | Device | SynapseAI Version | TensorFlow Version(s)  |
|:----:|:------:|:-----------------:|:-----:|
|1 HPU | Gaudi  | 1.3.0             | 2.7.1 |
|8 HPU | Gaudi  | 1.4.1             | 2.8.0 |
# Changelog
### 1.2.0
* Added support for recipe cache, see `TF_RECIPE_CACHE_PATH` in HabanaAI documentation for details
* Enabled multi-HLS training
### 1.3.0
* Enabled multinode BLEU calculation
* Update requirements.txt
### 1.4.0
* References to custom demo script were replaced by community entry points in README
* Import horovod-fork package directly instead of using Model-References' TensorFlow.common.horovod_helpers; wrapped horovod import with a try-catch block so that the user is not required to install this library when the model is being run on a single card
* Update requirements.txt
* Changed the default value of the log_step_count_steps flag

## Known Issues

* Only fp32 precision is supported when calculating BLEU on HPU
