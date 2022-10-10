# Transformer for TensorFlow

This repository provides a script and recipe to train the Transformer model for Tensorflow on Habana Gaudi device. For further information on performance, refer to [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

For more information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents

  * [Model-References](../../../README.md)
  * [Model Overview](#model-overview)
  * [Setup](#setup)
  * [Training and Examples](#training-and-examples)
  * [Evaluating BLEU Score](#evaluating-bleu-score)
  * [Profile](#profile)
  * [Supported Configuration](#supported-configuration)
  * [Changelog](#changelog)
  * [Known Issues](#known-issues)

## Model Overview
The Transformer is a Neural Machine Translation (NMT) model which uses attention mechanism to boost training speed and overall accuracy.
The model was initially introduced in [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
This implementation is based on [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) implementation (authors: Google Inc., Artit Wangperawong).

There are three model variants available: tiny, base and big.

### Model Architecture
The Transformer model uses standard NMT encoder-decoder architecture. Unlike other NMT models, Transformer model does not use recurrent connections and operates on fixed size context window.
The encoder stack is made up of N identical layers. Each layer is composed of the following sub-layers:
- Self-attention layer
- Feedforward network (which is 2 fully-connected layers)

The decoder stack is also made up of N identical layers. Each layer is composed of the sub-layers:
- Self-attention layer
- Multi-headed attention layer combining encoder outputs with results from the previous self-attention layer.
- Feedforward network (2 fully-connected layers)

The encoder uses self-attention to compute a representation of the input sequence. The decoder generates the output sequence one token at a time, taking the encoder output and previous decoder-outputted tokens as inputs.
The model also applies embeddings on the input and output tokens, and adds a constant positional encoding. The positional encoding adds information about the position of each token.

The complete description of the Transformer architecture can be found in [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper.

## Setup
Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) to set up the
environment including the `$PYTHON` environment variable. The guide will walk you through the process of setting up your system to run the model on Gaudi.

### Clone Habana Model-References
In the docker container, clone this repository and switch to the branch that matches your SynapseAI version. You can run the [`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the SynapseAI version.
```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References /root/Model-References
```

**Note:** If Model-References repository path is not in the PYTHONPATH, make sure you update it:
```bash
export PYTHONPATH=$PYTHONPATH:/root/Model-References
```
### Download and Generate the Dataset

Go to the Transformer directory and generate the dataset. The following script will save the dataset to `/data/tensorflow/wmt32k_packed/train`:
```bash
cd Model-References/TensorFlow/nlp/transformer/
$PYTHON datagen.py \
    --data_dir=/data/tensorflow/wmt32k_packed/train \
    --tmp_dir=/tmp/transformer_datagen \
    --problem=translate_ende_wmt32k_packed \
    --random_seed=429459
```

### Install Model Requirements

1. In the docker container, go to the Transformer directory:
```bash
cd /root/Model-References/TensorFlow/nlp/transformer
```

2. Install the required packages using pip:
```bash
$PYTHON -m pip install -r requirements.txt
```

## Training and Examples

### Single card and Multi-Card Training Examples

**NOTE:** All training examples for 1 HPU and 8 HPUs are valid both for first-gen Gaudi and Gaudi2.

**Run training on 1 HPU:**

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

Run training on 1 HPU, batch size 4096, bfloat16, transformer_big, 300k steps with a checkpoint saved every 10k steps:

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
    --use_hpu=True \
    --use_bf16=True
```

**Run training on 8 HPUs:**

**NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration).

Run training on 8 HPUs, global batch size 8 * 4096, bfloat16, transformer_big, 300k steps with a checkpoint saved every 50k steps, learning rate constant 2.5:

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

### Multi-Server Training and Examples
To run training on multiple servers, make sure to set the `MULTI_HLS_IPS` environment 
variable with the IPs of the used servers.

**NOTE:** Multi-server training is supported only on first-gen Gaudi.

**Run training on 16 HPUs:**
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

**Run training on 32 HPUs:**

 **NOTE:** It is recommended to use `learning_rate_constant` 3.5 and `train_steps` 75000.

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
After training the model, you can evaluate the achieved BLEU score:
1. Download and tokenize the validation file:
```bash
sacrebleu -t wmt14 -l en-de --echo src > wmt14.src
cat wmt14.src | sacremoses tokenize -l en > wmt14.src.tok
```

2. Compute BLEU score of a single checkpoint:
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

3. Optional: To split BLEU calculation to multiple cards, run `decoder.py` through `mpirun`. For example:
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
**NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration).

## Profile
To run with profiling enabled, pass `--profile_steps` flag. It should be a comma separated pair of numbers - on which step to start and end profiling.

Profiler steps are counted individually for each run. Thus, if you run training for 100 steps, with `--profile_steps 99,100`, profiling will be always enabled for the last two steps, no matter the `global_step_count`.

**Run training on 1 HPU with profiler:**

```bash
 $PYTHON trainer.py \
    --data_dir=/data/tensorflow/wmt32k_packed/train/ \
    --problem=translate_ende_wmt32k_packed \
    --model=transformer \
    --hparams_set=transformer_big \
    --hparams=batch_size=4096 \
    --output_dir=./translate_ende_wmt32k_packed/transformer_big/bs4096 \
    --local_eval_frequency=10000 \
    --train_steps=100 \
    --schedule=train \
    --use_hpu=True \
--profile_steps 50,53
```
The above example will produce profile trace for 4 steps (50,51,52,53).

## Supported Configuration
| Device | SynapseAI Version | TensorFlow Version(s)  |
|:------:|:-----------------:|:-----:|
| Gaudi  | 1.6.1             | 2.9.1 |
| Gaudi  | 1.6.1             | 2.8.2 |
| Gaudi2 | 1.6.1             | 2.9.1 |
| Gaudi2 | 1.6.1             | 2.8.2 |

## Changelog
### 1.6.0
* Model enabled on Gaudi2, with the same config as first-gen Gaudi.
* Added profiling support.
* Enabled experimental variable clustering to improve performance.
* Removed advanced parameters section from README.

### 1.4.0
* Replaced references to custom demo script by community entry points in README.
* Added support to import horovod-fork package directly instead of using Model-References' TensorFlow.common.horovod_helpers; wrapped horovod import with a try-catch block so that the user is not required to install this library when the model is being run on a single card.
* Updated requirements.txt.
* Changed the default value of the log_step_count_steps flag.

### 1.3.0
* Enabled multi-HPU BLEU calculation.
* Updated requirements.txt.

### 1.2.0
* Added support for recipe cache, see `TF_RECIPE_CACHE_PATH` in HabanaAI documentation for details.
* Enabled multi-server training.

### Training Script Modifications
* Support for other models than Transformer was removed.
* Added support for Horovod together with some adjustments in the topology script to allow simplifying the computational graph.

## Known Issues

Only FP32 precision is supported when calculating BLEU on HPU.
