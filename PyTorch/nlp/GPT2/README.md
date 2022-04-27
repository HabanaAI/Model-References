# GPT2-small

## Table of Contents
  * [Model-References/README.md](https://github.com/HabanaAI/Model-References/blob/master/README.md)
  * [Model Overview](#model-overview)
  * [Setup](#setup)
  * [Training the model](#training-the-model)
  * [Model Changes](#model-changes)

## Model Overview

We use Fairseq modeling toolkit to train GPT2-small model on Habana's Gaudis.
For more information about Fairseq , https://fairseq.readthedocs.io/en/latest/

## Setup
Please follow the instructions given in the following link for setting up the
environment including the `$PYTHON` environment variable: [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html).
This guide will walk you through the process of setting up your system to run
the model on Gaudi.

The base training and modelling scripts for training are based on a clone of
https://github.com/pytorch/fairseq with certain changes in training scripts.
Please refer to later sections on training script and model modifications for a summary of
modifications to the original files.

In the docker container, clone this repository and switch to the branch that matches your SynapseAI version. (Run the
[`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.) For example, if your SynapseAI version is 1.4.0

```bash
git clone -b 1.4.0 https://github.com/HabanaAI/Model-References
```
- Navigate to the GPT2 model directory:
```bash
cd Model-References/PyTorch/nlp/GPT2
```

### Install the requirements:

* Python version >= 3.6
* Make sure to install python requirements in requirements.txt file

```bash
pip install -r requirements.txt
```

### Dataset download:

To obtain datasets, refer to [GettingTheDataset.md](GettingTheDataset.md) for instructions

### Installing GPT2-small:

```bash
pip install --editable ./
```
## Training the model:

- Example: 8 HPUs data-parallel:

    ```bash
    $PYTHON train.py \
    /data/path/to/OpenWebText_database/ \
    --distributed-world-size 8 \
    --hpu \
    --num-workers 2 \
    --task language_modeling \
    --arch transformer_lm_gpt \
    --share-decoder-input-output-embed \
    --dropout 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.95)' \
    --weight-decay 0.1 \
    --clip-norm 1.0 \
    --lr 0.0006 --lr-scheduler cosine --warmup-updates 1000 --warmup-init-lr 1e-07 \
    --tokens-per-sample 1024 \
    --sample-break-mode none \
    --zero-sharding os \
    --checkpoint-activations \
    --batch-size 8 --update-freq 8 --max-update 100000 \
    --save-dir checkpoints/openwebtext_gpt2_lr0.0006 \
    --save-interval-updates 2000 --keep-interval-updates 10 --log-interval 10 \
    --tensorboard-logdir tensorboard/openwebtext_gpt2_lr0.0006
    ```

## Model Changes:
* Compatability changes for HPU devices
* Run mark_step after each grad-accumulation iteration (update-freq > 1)
* Run mark_step after each optimizer step
* Disabling parameters_as_bucket_view parameter in zero1 optimization

## Supported Configurations

| Device | SynapseAI Version | PyTorch Version(s) |
|:-----:|:-----:|:-------------------------------:|
| Gaudi | 1.4.0 | 1.10.2                        |
