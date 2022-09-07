# Vision Transformer for TensorFlow

This directory provides a script and recipe to train a Vision Transformer model to achieve state of the art accuracy, and is tested and maintained by Habana.
For further information on performance, refer to [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

For further information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents

* [Model-References](../../../README.md)
* [Model overview](#model-overview)
* [Setup](#setup)
* [Training and Examples](#training-the-model)
* [Profiling Example](#profiling-example)
* [Supported Configuration](#supported-configuration)
* [Changelog](#changelog)
* [Known Issues](#known-issues)

## Model Overview

This is a Keras implementation of the models described in [An Image is Worth 16x16 Words: Transformes For Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf).
It is based on an earlier implementation from [tuvovan](https://github.com/tuvovan/Vision_Transformer_Keras), modified to match the Flax implementation in the [official repository](https://github.com/google-research/vision_transformer).

The weights here are ported over from the weights provided in the official repository. For more details on implementation, refer to `utils.load_weights_numpy`.

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

### Install Model Requirements

1. In the docker container, go to the Vision Transformer directory:

```bash
cd /root/Model-References/TensorFlow/computer_vision/VisionTransformer
```

2. Install the required packages using pip:

```bash
$PYTHON -m pip install -r requirements.txt
```

### Training Data

The Vision Transformer script operates on ImageNet 1k, a widely popular image classification dataset from the ILSVRC challenge.
Post downloading the dataset, the pre-processing script will be located in ResNet folder: [preprocess_imagenet.py](../Resnets/preprocess_imagenet.py)
To obtain the dataset, perform the following steps:
1. Sign up with http://image-net.org/download-images and acquire the rights to download original images.
2. Follow the link to the 2012 ILSVRC and download `ILSVRC2012_img_val.tar` and `ILSVRC2012_img_train.tar`.
3. Use the below commands to prepare the dataset under `/data/tensorflow_datasets/imagenet/tf_records`. This is the default data directory for the training script.

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

## Training and Examples

### Single Card and Multi-Card Training Examples

**Run training on 1 HPU:**

Run training on 1 HPU with batch size 32, gradient accumulation, 2 steps and 8 epochs:

```bash
$PYTHON train.py --grad_accum_steps 2 --epochs 8
```

**Run training on 8 HPUs:**

**NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration).

Vision Transformer relies on mpi4py and tf.distribute to enable distributed training.
Since `batch_size` parameter is global, it must be scaled (BS of a single card times number of cards).
`distributed` flag must be used to ensure proper strategy is in use.

Run training on 8 HPUs with batch size 256, precision FP32 and 1000 warmup steps:

```bash
mpirun -np 8 python train.py -d fp32 --batch_size 256 --warmup_steps 1000 --distributed
```

## Profiling Example

### Single Card Profiling Training Example

**Run training on 1 HPU:**

Run training on 1 HPU with batch size 32, 1 epochs, 20 steps and 12-15 iterations:

```bash
$PYTHON train.py --epochs 1 --steps_per_epoch 20 --profile 12,15
```

### Parameters

You can modify the training behavior through various flags in `train.py` script.

- `dataset`, `dataset_dir`: Dataset directory.
- `optimizer`: Optimizer.
- `dtype`, `d`: Data type (FP32 or BF16).
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
- `evaluate_checkpoint_path`: Checkpoint path for evaluating the model on --val_subset.
- `weights_path`: Path to weights cache directory. ~/.keras is used if not set.
- `deterministic`: Enable deterministic behavior, this will also disable data augmentation. --seed must be set.
- `seed`: Seed to be used by random functions.
- `device`: Device type (CPU or HPU).
- `distributed`: Enable distributed training.
- `base_tf_server_port`: Rank 0 port used by tf.distribute.
- `save_summary_steps`: Steps between saving summaries to TensorBoard.
- `recipe_cache`: Path to recipe cache directory. Set to empty to disable recipe cache. Externally set 'TF_RECIPE_CACHE_PATH' will override this setting.
- `dump_config`: Side-by-side config file. Internal, do not use.

## Supported Configuration

| Device | SynapseAI Version | TensorFlow Version(s)  |
|:------:|:-----------------:|:-----:|
| Gaudi  | 1.6.0             | 2.9.1 |
| Gaudi  | 1.6.0             | 2.8.2 |


## Changelog

### 1.4.0

- Updated dataset paths.
- Added '--profile' parameter.
- implementation override clean-up.

### 1.3.0

- Updated 'tensorflow_addons' and 'mpi4py' in requirements.txt.
- Added implementation override of Gelu.
- Improved robustness in multi-card scenarios.

## Known Issues

### Profiling in Multi-card Scenario

To profile in multi-card scenario, habanalabs driver must be loaded with increased `timeout_locked` parameter (eg. `timeout_locked=300`).