# Vision Transformer
For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

Please visit [this page](https://developer.habana.ai/resources/habana-training-models/#performance) for performance information.
## Table of contents

- [Model-References](../../../README.md)
- [Model overview](#model-overview)
- [Setup](#setup)
    - [Install model requirements](#install-model-requirements)
- [Training the model](#training-the-model)
    - [Run single-card training](#run-single-card-training)
    - [Run multi-card training](#run-multi-card-training)
    - [Parameters](#parameters)
- [Examples](#examples)
- [Supported Configuration](#supported-configuration)
- [Changelog](#changelog)

## Model Overview

This is a Keras implementation of the models described in [An Image is Worth 16x16 Words:
Transformes For Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf). It is based on an earlier implementation from [tuvovan](https://github.com/tuvovan/Vision_Transformer_Keras), modified to match the Flax implementation in the [official repository](https://github.com/google-research/vision_transformer).

The weights here are ported over from the weights provided in the official repository. See `utils.load_weights_numpy` to see how this is done (it's not pretty, but it does the job).

## Setup

Please follow the instructions given in the following link for setting up the
environment including the `$PYTHON` environment variable: [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html).
This guide will walk you through the process of setting up your system to run
the model on Gaudi.

In the docker container, clone this repository and switch to the branch that
matches your SynapseAI version. (Run the
[`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.)

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```
Go to the Vision Transformer directory:
```bash
cd Model-References/TensorFlow/computer_vision/VisionTransformer
```

### Install model requirements

In the docker container, go to the Vision Transformer directory
```bash
cd /root/Model-References/TensorFlow/computer_vision/VisionTransformer
```
Install required packages using pip
```bash
$PYTHON -m pip install -r requirements.txt
```

### Training Data

The script operates on ImageNet 1k, a widely popular image classification dataset from the ILSVRC challenge.
Post download of dataset, the preprocessing script is located in Resnet folder: [preprocess_imagenet.py](../Resnets/preprocess_imagenet.py)
In order to obtain the dataset, follow these steps:
1. Sign up with http://image-net.org/download-images and acquire the rights to download original images
2. Follow the link to the 2012 ILSVRC
 and download `ILSVRC2012_img_val.tar` and `ILSVRC2012_img_train.tar`.
3. If not present, then create directory path for `/data/tensorflow`. Use the commands below - they will prepare the dataset under `/data/tensorflow/imagenet/tf_records`. This is the default `--dataset_dir`
 for the training script.

```
export IMAGENET_HOME=/data/tensorflow
mkdir -p $IMAGENET_HOME/validation
mkdir -p $IMAGENET_HOME/train
tar xf ILSVRC2012_img_val.tar -C $IMAGENET_HOME/validation
tar xf ILSVRC2012_img_train.tar -C $IMAGENET_HOME/train
cd $IMAGENET_HOME/train
for f in *.tar; do
  d=`basename $f .tar`
  mkdir $d
  tar xf $f -C $d
done
cd $IMAGENET_HOME
rm $IMAGENET_HOME/train/*.tar # optional
wget -O synset_labels.txt https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt
cd /root/Model-References/TensorFlow/computer_vision/Resnets
$PYTHON preprocess_imagenet.py \
  --raw_data_dir=$IMAGENET_HOME \
  --local_scratch_dir=$IMAGENET_HOME/imagenet/tf_records
```

## Training the model

As a prerequisite, root of this repository must be added to `PYTHONPATH`.
For example:
```bash
export PYTHONPATH=$PYTHONPATH:$HOME/Model-References
```

### Run single-card training

Example of command line used for single-card training:
```bash
$PYTHON train.py --grad_accum_steps 2 --epochs 8
```

### Run multi-card training
**NOTE:** mpirun map-by PE attribute value may vary on your setup. Please refer to the instructions on [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration) for calculation.

Vision Transformer relies on mpi4py and tf.distribute to enable distributed training.
Since `batch_size` parameter is global, it must be scaled (BS of a single card times number of cards).
`distributed` flag must be used to ensure proper strategy is in use.

Example of command line used for distributed training:
```bash
mpirun -np 8 python train.py -d fp32 --batch_size 256 --warmup_steps 1000 --distributed
```

### Profile single-card training

Example of command line used for profiling single-card training:
```bash
$PYTHON train.py --epochs 1 --steps_per_epoch 20 --profile 12,15
```

### Parameters

You can modify the training behavior through various flags in `train.py` script.

- `dataset`, `dataset_dir`: Dataset directory.
- `optimizer`: Optimizer.
- `dtype`, `d`: Data type (fp32 or bf16).
- `batch_size`: Global batch size.
- `lr_sched`: Learning rate scheduler (linear, exp, steps, constant, WarmupCosine).
- `initial_lr`: Initial learning rate.
- `final_lr`: Final learning rate.
- `warmup_steps`: Warmup steps.
- `epochs`: Total number of epochs for training.
- `steps_per_epoch`: Number of steps for training per epoch, overrides default value.
- `validation_steps`: Number of steps for validation, overrides default value.
- `model`: Model (ViT-B_16, ViT-L_16, ViT-B_32, ViT-L_32).
- `train_subset`: Pattern to detect train subset in dataset directory.
- `val_subset`: Pattern to detect validation subset in dataset directory.
- `grad_accum_steps`: Gradient accumulation steps.
- `resume_from_checkpoint_path`: Path to checkpoint to start from.
- `resume_from_epoch`: Initial epoch index.
- `evaluate_checkpoint_path`: Checkpoint path for evaluating the model on --val_subset
- `weights_path`: Path to weights cache directory. ~/.keras is used if not set.
- `deterministic`: Enable deterministic behavior, this will also disable data augmentation. --seed must be set.
- `seed`: Seed to be used by random functions.
- `device`: Device type (CPU or HPU).
- `distributed`: Enable distributed training.
- `base_tf_server_port`: Rank 0 port used by tf.distribute.
- `save_summary_steps`: Steps between saving summaries to TensorBoard.
- `recipe_cache`: Path to recipe cache directory. Set to empty to disable recipe cache. Externally set 'TF_RECIPE_CACHE_PATH' will override this setting.
- `dump_config`: Side-by-side config file. Internal, do not use.

## Examples
**NOTE:** mpirun map-by PE attribute value may vary on your setup. Please refer to the instructions on [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration) for calculation.

| Command | Notes |
| ------- | ----- |
|`$PYTHON train.py --dataset /data -d bf16`| Single-card training in bf16 |
|`$PYTHON train.py --dataset /data -d fp32`| Single-card training in fp32 |
|`$PYTHON train.py --dataset /data --evaluate_checkpoint_path <PATH>`| Single-card evaluation |
|`mpirun -np 8 python train.py --dataset /data -d bf16 --batch_size 256 --warmup_steps 1000 --distributed`| 8-cards training in bf16 (with 1000 warmup steps and scaled batch size) |
|`mpirun -np 8 python train.py --dataset /data -d fp32 --distributed`| 8-cards training in fp32 (with unscaled batch size - not recommended) |
|`mpirun -np 8 python train.py --dataset /data --distributed --evaluate_checkpoint_path <PATH>`| 8-cards evaluation |

## Known issues

### Profiling in multi-card scenario

To profile in multi-card scenario, habanalabs driver must be loaded with increased `timeout_locked` parameter (eg. `timeout_locked=300`).

## Supported Configuration

| Device | SynapseAI Version | TensorFlow Version(s)  |
|:------:|:-----------------:|:-----:|
| Gaudi  | 1.5.0             | 2.9.1 |
| Gaudi  | 1.5.0             | 2.8.2 |

## Changelog
### 1.4.0
- updated dataset paths
- '--profile' parameter
- implementation override clean-up
### 1.3.0
- updated 'tensorflow_addons' and 'mpi4py' in requirements.txt
- added implementation override of Gelu
- improved robustness in multi-card scenarios

