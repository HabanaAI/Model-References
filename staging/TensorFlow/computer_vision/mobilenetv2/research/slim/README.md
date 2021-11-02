# MobileNetV2

## Table of Contents
* [Model Overview](#model-overview)
* [Setup](#Setup)
* [Training the Model](#training-the-model)

## Model Overview

MobileNetV2 is general mobile architecture based on an inverted residual structure and linear bottleneck, which improves the state of art accuracy and performance in mobile and embedded vision applications. It can be used as a very effective feature extractor for object detection (paired with SSDLite) and segmentation (paired with DeepLabv3). More reference can found [MobileNetV2](https://arxiv.org/pdf/1801.04381.pdf).

**THIS IS A PREVIEW OF OUR MOBILENETV2 IMPLEMENTATION. It is functional but improvement is still in progress.**

## Training Script Modifications

Our implementation is a fork of [Google Research MobileNet](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet) master branch.

Files changed:

-   research/slim/download\_and\_convert\_data.py
-   research/slim/train\_image\_classifier.py
-   research/slim/eval\_image\_classifier.py
-   research/slim/nets/nasnet/nasnet.py
-   research/slim/nets/nasnet/pnasnet.py

All of above files were converted to TF2.
The following are the changes specific to Gaudi that were made to the original scripts:

-  **download\_and\_convert\_data.py**

Replace **tf.app.flags.FLAGS** with **flags.FLAGS**

-  **train\_image\_classifier.py**

1. Added Habana HPU support
2. Added Horovod support for multinode
3. Replace **tf.app.flags.FLAGS** with **flags.FLAGS**
4. Remove **contrib_quantize**
5. Remove **contrib_quantize.create_training_graph**
6. Add tf.compat.v1.enable_resource_variables()

-  **eval\_image\_classifier.py**

1. Added Habana HPU support
2. Replace **tf.app.flags.FLAGS** with **flags.FLAGS**
3. Remove **contrib_quantize**
4. Remove **contrib_quantize.create_training_graph**
5. Add tf.compat.v1.enable_resource_variables()

-  **research/slim/nets/nasnet/nasnet.py**

Replace **contrib_training** in tensorflow.contrib with **hp** in tensorboard.plugins.hparams

-  **research/slim/nets/nasnet/pnasnet.py**

Replace **contrib_training** in tensorflow.contrib with **hp** in tensorboard.plugins.hparams

-  **research/slim/nets/deployment/model_deploy.py**

Change the default device from CPU to HPU for **optimizer_device** and **variables_device** in DeploymentConfig

### Hyperparameters

*ImageNet Dataset:*

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
### Install Drivers

Please follow the instructions given in the following link for setting up the
environment including the `$PYTHON` environment variable: [Gaudi Setup and
Installation Guide](https://github.com/HabanaAI/Setup_and_Install). Please
answer the questions in the guide according to your preferences. This guide will
walk you through the process of setting up your system to run the model on
Gaudi.

### Training Data

MobileNetV2 uses ImageNet as training dataset, a widely popular image
classification dataset from the ILSVRC challenge.

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

When the script finishes you will find 1024 and 128 training and validation files in the DATA_DIR. The files will match the patterns train-????-of-1024 and validation-?????-of-00128, respectively.

The above commands will create `tf_record` files in `/path/to/imagenet_data/tf_records`

## Training the Model

In the docker container, clone the Model-References repository and switch to the branch that
matches your SynapseAI version. (Run the
[`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.)

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```
Go to MobileNetV2 directory:

```bash
cd Model-References/staging/TensorFlow/computer_vision/mobilenetv2/research/slim
```

Note: If the repository is not in the PYTHONPATH, make sure you update it.
```bash
export PYTHONPATH=/path/to/Model-References:$PYTHONPATH
```

### Training the Model on single Gaudi
The script `demo_mobilenetv2.py` defines all the hyperparameters that will be used to launch the model script `train_image_classifier.py`

To see all the commandline options, run:

```bash
$PYTHON demo_mobilenetv2.py --help
```

**Examples:**

You can use the following commands to launch the training:

```bash
 $PYTHON demo_mobilenetv2.py --dataset_dir <path to dataset>
```
```bash
$PYTHON demo_mobilenetv2.py --dataset_dir <path to dataset> --dtype bf16
```

## Known Issues

This model works fine on single Gaudi (good performance and convergence) but multichip training does not converge for this model.
