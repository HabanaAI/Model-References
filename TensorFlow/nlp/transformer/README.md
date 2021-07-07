# Transformer

This repository provides a script to train the Transformer model for Tensorflow on Habana
Gaudi<sup>TM</sup> device. Please visit [this page](../../../README.md#tensorflow-model-performance) for performance information. For more information, visit [developer.habana.ai](https://developer.habana.ai/resources/).

## Table Of Contents
* [Model Overview](#model-overview)
* [Setup](#setup)
* [Training the Model](#training-the-model)
* [Examples](#examples)
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

Please follow the instructions given in the following link for setting up the environment: [Gaudi Setup and Installation Guide](https://github.com/HabanaAI/Setup_and_Install). Please answer the questions in the guide according to your preferences. This guide will walk you through the process of setting up your system to run the model on Gaudi.

### Download and generate the dataset

```bash
git clone https://github.com/HabanaAI/Model-References
cd Model-References/TensorFlow/nlp/transformer/
python3 datagen.py \
    --data_dir=$HOME/transformer_dataset/train \
    --tmp_dir=/tmp/transformer_datagen \
    --problem=translate_ende_wmt32k_packed \
    --random_seed=429459
```

## Training the Model

The easiest way to train the model is to use the supplied demo script, which is a simple wrapper around `trainer.py`:
```bash
python3 demo_transformer.py \
    --batch_size <batch_size> \
    --dtype <precision> \
    --data_dir <path_to_dataset>/train \
    --output_dir <path_to_output_dir> \
    --model <model_size> \
    --eval_freq <eval_frequency> \
    --train_steps <train_steps>
```

Alternatively, the `trainer.py` can be called directly:
```bash
python3 trainer.py \
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
python3 demo_transformer.py \
    --batch_size 4096 \
    --dtype fp32 \
    --data_dir ~/transformer_dataset/train \
    --model big \
    --eval_freq 10000 \
    --train_steps 300000
```
or
```bash
python3 trainer.py \
    --data_dir=~/transformer_dataset/train \
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

#### via demo script
```bash
python3 demo_transformer.py \
    --batch_size 4096 \
    --dtype bf16 \
    --data_dir ~/transformer_dataset/train \
    --model big \
    --eval_freq 50000 \
    --train_steps 300000 \
    --learning_rate_constant 2.5 \
    --hvd_workers 8
```
#### via mpirun
Note, it is required to set up the `HCL_CONFIG_PATH` environment variable to point to a valid HCL config JSON file for the HLS type being used. For documentation on creating an HCL config JSON file, please refer to [HCL JSON Config File Format](https://docs.habana.ai/en/latest/API_Reference_Guides/HCL_API_Reference.html#hcl-json-config-file-format).

```bash
HCL_CONFIG_PATH=hcl_config_8.json mpirun \
    --allow-run-as-root --bind-to core --map-by socket:PE=7 --np 8 \
    --tag-output --merge-stderr-to-stdout \
    python3 trainer.py \
        --data_dir=~/transformer_dataset/train \
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

## Evaluating BLEU Score

After training the model you can evaluate achieved BLEU score. First you need to download the validation file and tokenize it:
```bash
sacrebleu -t wmt14 -l en-de --echo src > wmt14.src
cat wmt14.src | sacremoses tokenize -l en > wmt14.src.tok
```

Then you can compute BLEU score of a single checkpoint by running the following command:
```bash
python3 demo_transformer.py \
    --dtype bf16 \
    --model big \
    --schedule calc_bleu \
    --data_dir <path_to_dataset>/train \
    --checkpoint_path <path_to_checkpoint> \
    --decode_from_file ./wmt14.src.tok \
    --no_hpu
```

To calculate BLEU score for all checkpoints under `path_to_output_dir`:
```bash
python3 demo_transformer.py \
    --dtype bf16 \
    --model big \
    --schedule calc_bleu \
    --data_dir <path_to_dataset>/train \
    --output_dir <path_to_output_dir> \
    --decode_from_file ./wmt14.src.tok \
    --no_hpu
```
After running this command maximum BLEU value will be stored as a tf event scalar - accuracy.

Alternatively you can call decoder.py directly:
```bash
python3 decoder.py \
    --problem=translate_ende_wmt32k_packed \
    --model=transformer \
    --hparams_set=transformer_big \
    --data_dir=<path_to_dataset>/train \
    --output_dir=<path_to_output_dir> \
    --checkpoint_path=<path_to_checkpoint> \
    --decode_from_file=./wmt14.src.tok \
    --decode_to_file=./wmt14.tgt.tok \
    --use_hpu=False
cat wmt14.tgt.tok | sacremoses detokenize -l de | sacrebleu -t wmt14 -l en-de
```

## Advanced parameters

To get a list of all supported parameters and their default values run:
```bash
python3 demo_transformer.py --help
```

## Known Issues

* Calculating BLEU works only on CPU
* Specifying random_seed breaks accuracy on HPU
