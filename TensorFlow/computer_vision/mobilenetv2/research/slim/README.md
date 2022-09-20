# MobileNetV2 for TensorFlow

This directory provides a script to train a MobileNetV2 model to achieve state-of-the-art accuracy, and is tested and maintained by Habana. To obtain model performance data, refer to the [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

For more information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents

* [Model-References](../../../../../README.md)
* [Model Overview](#model-overview)
* [Setup](#setup)
* [Training and Examples](#training-and-examples)
* [Changelog](#changelog)
* [Known Issues](#known-issues)


## Model Overview

MobileNetV2 is a general mobile architecture based on an inverted residual structure and linear bottleneck, which improves the state of art accuracy and performance in mobile and embedded vision applications. It can be used as a very effective feature extractor for object detection (paired with SSDLite) and segmentation (paired with DeepLabv3. More details can be found [MobileNetV2](https://arxiv.org/pdf/1801.04381.pdf).

To see the changes implemented for this model, refer to [Training Script Modifications](#training-script-modifications).

### Hyperparameters - ImageNet Dataset

* Learning rate (0.045)
* Clone on cpu (False)
* Preprocessing name ("inception_v2")
* Log every n steps (20)
* Label smoothing (0.1)
* Moving average decay (0.9999)
* Batch size (96)
* Num clones (1)
* Learning rate decay factor (0.98)
* Num epochs per decay (2.5)

## Setup

Please follow the instructions provided in the [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the
environment including the `$PYTHON` environment variable.
The guide will walk you through the process of setting up your system to run the model on Gaudi.

### Training Data

MobileNetV2 uses ImageNet a widely popular image
classification dataset from the ILSVRC challenge as the training dataset.

1. Sign up with http://image-net.org/download-images and acquire the rights to download original images
2. Follow the link to the 2012 ILSVRC
 and download `ILSVRC2012_img_val.tar` and `ILSVRC2012_img_train.tar` to this directory.
3. Ensure Python3 and the following Python packages are installed: TensorFlow 2.2 and `absl-py`.

```
mkdir /path/to/imagenet_data
export IMAGENET_HOME=/path/to/imagenet_data
mkdir -p $IMAGENET_HOME/img_val
mkdir -p $IMAGENET_HOME/img_train
tar xf ILSVRC2012_img_val.tar -C $IMAGENET_HOME/img_val
tar xf ILSVRC2012_img_train.tar -C $IMAGENET_HOME/img_train
cd $IMAGENET_HOME/img_train
for f in *.tar; do
  d=`basename $f .tar`
  mkdir $d
  tar xf $f -C $d
done
cd $IMAGENET_HOME
rm $IMAGENET_HOME/img_train/*.tar # optional
wget -O synset_labels.txt https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt
cd Model-References/TensorFlow/computer_vision/Resnets
$PYTHON preprocess_imagenet.py \
  --raw_data_dir=$IMAGENET_HOME \
  --local_scratch_dir=$IMAGENET_HOME/tf_records \
  --nogcs_upload
```

When the script finishes, you will find 1024 and 128 training and validation files in the `DATA_DIR`. The files will match the patterns `train-????-of-1024` and `validation-?????-of-00128`, respectively.

The above commands create `tf_record` files in `/path/to/imagenet_data/tf_records`.

### Clone Habana Model-References
In the docker container, clone this repository and switch to the branch that
matches your SynapseAI version. You can run the
[`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```
Go to MobileNetV2 directory:

```bash
cd Model-References/TensorFlow/computer_vision/mobilenetv2/research/slim
```

**Note:** If the repository is not in the PYTHONPATH, make sure to update by running the below:
```bash
export PYTHONPATH=/path/to/Model-References:$PYTHONPATH
```
### Install Model Requirements
1. In the docker container, go to the MobileNetV2:
```bash
cd Model-References/TensorFlow/computer_vision/mobilenetv2/research/slim
```
2. Install the required packages using pip:
```bash
$PYTHON -m pip install -r requirements.txt
```
## Training and Examples

To train the model on a single Gaudi, see all command line options for training, run:

```bash
$PYTHON train_image_classifier.py --help
```

You can use the following commands to launch the training:

```bash
 $PYTHON train_image_classifier.py --dataset_dir <path to dataset>
```
```bash
$PYTHON train_image_classifier.py --dataset_dir <path to dataset> --dtype bf16
```
## Supported Configuration

| Device | SynapseAI Version | TensorFlow Version(s)  |
|:------:|:-----------------:|:-----:|
| Gaudi  | 1.5.0             | 2.9.1 |
| Gaudi  | 1.5.0             | 2.8.2 |

## Changelog
### 1.4.0
* Removed not used and obsolete files.
* References to custom demo script were replaced by community entry points in README.
### 1.3.0
* Moved logger.py and cloud_lib.py files to model's dir from TensorFlow/utils/logs.
* Set up BF16 conversion pass using a config from habana-tensorflow package instead of recipe json file.
* Changed `python` or `python3` to `$PYTHON` to execute correct version based on environment setup.

### Training Script Modifications

Our implementation is a fork of [Google Research MobileNet](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet) master branch.

Files changed:

-   `research/slim/download\_and\_convert\_data.py`
-   `research/slim/train\_image\_classifier.py`
-   `research/slim/eval\_image\_classifier.py`
-   `research/slim/nets/nasnet/nasnet.py`
-   `research/slim/nets/nasnet/pnasnet.py`

All of above files were converted to TF2.
The following are the changes specific to Gaudi that were made to the original scripts:

- `download\_and\_convert\_data.py` - Replaced `tf.app.flags.FLAGS` with `flags.FLAGS`.

- `train\_image\_classifier.py`:
  - Added HPU support.
  - Added Horovod support for multi-HPU.
  - Replaced `tf.app.flags.FLAGS` with `flags.FLAGS`.
  - Removed `contrib_quantize`.
  - Removed `contrib_quantize.create_training_graph`.
  - Added `tf.compat.v1.enable_resource_variables()`.

- `eval\_image\_classifier.py`:
  - Added HPU support.
  - Replaced `tf.app.flags.FLAGS` with `flags.FLAGS`.
  - Removed `contrib_quantize`.
  - Removed `contrib_quantize.create_training_graph`.
  - Added `tf.compat.v1.enable_resource_variables()`.

- `research/slim/nets/nasnet/nasnet.py`: Replaced `contrib_training` in `tensorflow.contrib` with `hp` in `tensorboard.plugins.hparams`.

- `research/slim/nets/nasnet/pnasnet.py`: Replaced `contrib_training` in `tensorflow.contrib` with `hp` in `tensorboard.plugins.hparams`.

- `research/slim/nets/deployment/model_deploy.py`: Changed the default device from CPU to HPU for `optimizer_device` and `variables_device` in DeploymentConfig.


## Known Issues

This model works fine on single Gaudi (good performance and convergence) but multi-card training does not converge for this model.
