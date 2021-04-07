# ResNet For TensorFlow

This repository provides a script and recipe to train the ResNet v1.5 models to achieve state-of-the-art accuracy, and is tested and maintained by Habana.

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

## Table Of Contents
* [Model Overview](#model-overview)
* [Training Script Modifications](#training-script-modifications)
* [Model Setup](#model-setup)
* [Training the Model](#training-the-model)
* [Preview Python Script](#Preview-Python-Script)


## Model Overview
Both ResNet v1.5 and ResNeXt model is a modified version of the original ResNet v1 model. It supports layers 50, 101, and 152.

## Training Script Modifications

Originally, scripts were taken from [Tensorflow Github](https://github.com/tensorflow/models.git), tag v1.13.0

Files used:

-   imagenet\_main.py
-   imagenet\_preprocessing.py
-   resnet\_model.py
-   resnet\_run\_loop.py

All of above files were converted to TF2 by using [tf\_upgrade\_v2](https://www.tensorflow.org/guide/upgrade?hl=en) tool.
Additionally, some other changes were committed for specific files.

The following are the changes specific to Gaudi that were made to the original scripts:

imagenet\_main.py
=================

1. Added Habana HPU support
2. Added Horovod support for multinode
3. Added mini_imagenet support
4. Changed the signature of input_fn with new parameters added, and num_parallel_batches removed
5. Changed parallel dataset deserialization to use dataset.interleave instead of dataset.apply
6. Added Resnext support
7. Added parameter to ImagenetModel::\_\_init\_\_ for selecting between resnet and resnext
8. Redefined learning_rate_fn to use warmup_epochs and use_cosine_lr from params
9. Added flags to specify the weight_decay and momentum
10. Added flag to enable horovod support
11. Added MLPerf support
12. Added calls to tf.compat.v1.disable_eager_execution() and tf.compat.v1.enable_resource_variables() in main code section

imagenet\_preprocessing.py
==========================

1. Changed the image decode, crop and flip function to take a seed to propagate to tf.image.sample_distorted_bounding_box
2. Changed the use of tf.expand_dims to tf.broadcast_to for better performance

resnet\_model.py
================

1. Added tf.bfloat16 to CASTABLE_TYPES
2. Changed calls to tf.<api> to tf.compat.v1.<api> for backward compatibility when running training scripts on TensorFlow v2
3. Deleted the call to pad the input along the spatial dimensions independently of input size for stride greater than 1
4. Added functionality for strided 2-D convolution with groups and explicit padding
5. Added functionality for a single block for ResNext, with a bottleneck
6. Added ResNeXt support

resnet\_run\_loop.py
====================

1. Changes for enabling Horovod, e.g. data sharding in multi-node, usage of Horovod's TF DistributedOptimizer, etc.
2. Added functionality to enable the use of LARS optimizer
3. Added MLPerf support, including in how the training schedule is defined, the usage of mlperf_logging, etc.
4. Added options to enable the use of tf.data's 'experimental_slack' and 'experimental.prefetch_to_device' options during the input processing
5. Added support for specific size thread pool for tf.data operations
6. TensorFlow v2 support: Changed dataset.apply(tf.contrib.data.map_and_batch(..)) to dataset.map(..., num_parallel_calls, ...) followed by dataset.batch()
7. Changed calls to tf.<api> to tf.compat.v1.<api> for backward compatibility when running training scripts on TensorFlow v2
8. Other TF v2 replacements for tf.contrib usages
9. Redefined learning_rate_with_decay to use warmup_epochs and use_cosine_lr
10. Adapt the warmup_steps based on MLPerf flag
11. Defined a new learning rate schedule for LARS optimizer, that handles linear scaling rule, gradual warmup, and LR decay
12. Added functionality for label smoothing
13. Commented out writing images to summary, for performance reasons
14. Added check for non-tf.bfloat16 input images having the same data type as the dtype that training is run with
15. Added functionality to define the cross-entropy function depending on whether label smoothing is enabled
16. Added support for loss scaling of gradients
17. Added flag for experimental_preloading that invokes the HabanaEstimator, besides other optimizations such as tf.data.experimental.prefetch_to_device
18. Added 'TF_DISABLE_SCOPED_ALLOCATOR' environment variable flag to disable Scoped Allocator Optimization (enabled by default) for Horovod runs
19. Added a flag to configure the save_checkpoint_steps
20. If the flag "use_train_and_evaluate" is set, or in multi-worker training scenarios, there is a one-shot call to tf.estimator.train_and_evaluate
21. resnet_main() returns a dictionary with keys 'eval_results' and 'train_hooks'
22. Added flags in 'define_resnet_flags' for flags_core.define_base, flags_core.define_performance, flags_core.define_distribution, flags_core.define_experimental, and many others (please refer to this function for all the flags that are available)


### Hyperparameters
**ResNet SGD:**

* Momentum (0.9),
* Learning rate (LR) = 0.128 (single node) or 0.1 (multinode) for 256 batch size. For other batch sizes we linearly scale the learning rate.
* Piecewise learning rate schedule.
* Linear warmup of the learning rate during the first 3 epochs.
* Weight decay: 0.0001.
* Label Smoothing: 0.1.

**ResNet LARS:**

* Momentum (0.9),
* Learning rate (LR) = 2.5 (single node) or 9.5 (1 HLS, i.e. 8 gaudis, global batch 2048 (8*256)) (1).
* Polynomial learning rate schedule.
* Linear warmup of the learning rate during the first 3 epochs (1).
* Weight decay: 0.0001.
* Label Smoothing: 0.1.

(1) These numbers apply for batch size lower than 8192. There are other configurations for higher global batch sizes. Note, however, that they haven't been tested yet:

* (8192 < batch size < 16384): LR = 10, warmup epochs = 5,
* (16384 < batch size < 32768): LR = 25, warmup epochs = 5,
* (bachsize > 32768): LR = 32, warmup epochs = 14.

**ResNeXt:**

* Momentum (0.875).
* Learning rate (LR) = 0.256 for 256 batch size, for other batch sizes we linearly scale the learning rate.
* Cosine learning rate schedule.
* Linear warmup of the learning rate during the first 8 epochs.
* Weight decay: 6.103515625e-05
* Label Smoothing: 0.1.

We do not apply Weight decay on batch norm trainable parameters (gamma/bias).
We train for:
  * 90 Epochs -> 90 epochs is a standard for ResNet family networks.
  * 250 Epochs -> best possible accuracy.
For 250 epoch training we also use [MixUp regularization](https://arxiv.org/pdf/1710.09412.pdf).

### Data Augmentation
This model uses the following data augmentation:

* For training:
    * Normalization.
    * Random resized crop to 224x224.
        * Scale from 8% to 100%.
        * Aspect ratio from 3/4 to 4/3.
    * Random horizontal flip.
* For inference:
    * Normalization.
    * Scale to 256x256.
    * Center crop to 224x224.

## Setup

### Install Drivers
Follow steps in the [Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) to install the driver.

### Training Data

The ResNet50 v1.5 script operates on ImageNet 1k, a widely popular image
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
python3 preprocess_imagenet.py \
  --raw_data_dir=$IMAGENET_HOME \
  --local_scratch_dir=$IMAGENET_HOME/tf_records \
  --nogcs_upload
```

The above commands will create `tf_record` files in `/path/to/imagenet_data/tf_records`

### Training the Model

1. Download docker
```
docker pull vault.habana.ai/gaudi-docker/0.13.0/ubuntu18.04/habanalabs/tensorflow-installer:0.13.0-380
```

2. Run docker

**NOTE:** This assumes the Imagenet dataset is under /opt/datasets/imagenet on the host. Modify accordingly.

```
docker run -it -v /dev:/dev --device=/dev:/dev -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice  -v /sys/kernel/debug:/sys/kernel/debug -v /opt/datasets/imagenet:/root/tensorflow_datasets/imagenet --net=host vault.habana.ai/gaudi-docker/0.13.0/ubuntu18.04/habanalabs/tensorflow-installer:0.13.0-380
```

OPTIONAL with mounted shared folder to transfer files out of docker:

```
docker run -it -v /dev:/dev --device=/dev:/dev -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice  -v /sys/kernel/debug:/sys/kernel/debug -v ~/shared:/root/shared -v /opt/datasets/imagenet:/root/tensorflow_datasets/imagenet --net=host vault.habana.ai/gaudi-docker/0.13.0/ubuntu18.04/habanalabs/tensorflow-installer:0.13.0-380
```

3. Clone the repository and go to resnet directory:

```
git clone https://github.com/HabanaAI/Model-References.git
cd Model-References/TensorFlow/computer_vision/Resnets
```

Note: If the repository is not in the PYTHONPATH, make sure you update it.
```
export PYTHONPATH=/path/to/Model-References:$PYTHONPATH
```

### Single-Card Training
#### The demo_resnet script

The script for training and evaluating the ResNet model has a variety of paramaters.
```
usage: ./demo_resnet [arguments]

mandatory arguments:
  -d <data_type>,    --dtype <data_type>             Data type, possible values: fp32, bf16

optional arguments:
  -rs <resnet_size>, --resnet-size <resnet_size>     ResNet size, default 50 (or 101, when --resnext flag is given),
                                                     possible values: 50, 101, 152
  -b <batch_size>,   --batch-size <batch_size>       Batch size, default 128 for fp32, 256 for bf16
  -e <epochs>,       --epochs <epochs>               Number of epochs, default to 1
  -a <data_dir>,     --data-dir <data_dir>           Data dir, defaults to /software/data/tf/data/imagenet/tf_records/
  -m <model_dir>,    --model-dir <model_dir>         Model dir, defaults to /home/user1/tmp/resnet50/
  -o,                --use-horovod                   Use horovod for training
  -s <steps>,        --steps <steps>                 Max train steps
  -l <steps>,        --eval-steps <steps>            Max evaluation steps
  -n,                --no-eval                       Don't do evaluation
  -v <steps>,        --display-steps <steps>         How often display step status
  -c <steps>,        --checkpoint-steps <steps>      How often save checkpoint
  -r                 --recover                       If crashed restart training from last checkpoint. Requires -s to be set
  -k                 --no-experimental-preloading    Disables support for 'data.experimental.prefetch_to_device' TensorFlow operator. If not set:
                                                     - loads extension dynpatch_prf_remote_call.so (via LD_PRELOAD)
                                                     - sets environment variable HBN_TF_REGISTER_DATASETOPS to 1
                                                     - this feature is experimental and works only with single node
                     --use-train-and-evaluate        If set, uses tf.estimator.train_and_evaluate for the training and evaluation
                     --epochs-between-evals <epochs> Number of training epochs between evaluations, default 1
                     --enable-lars-optimizer         If set uses LARSOptimizer instead of default one
                     --stop_threshold <accuracy>     Threshold accuracy which should trigger the end of training.
                     --resnext                       Run resnext
```

These are example run commands:
```
examples:
  ./demo_resnet -d bf16
  ./demo_resnet -d fp32
  ./demo_resnet -d fp32 -rs 101
  ./demo_resnet -d bf16 -s 1000
  ./demo_resnet -d bf16 -s 1000 -l 50
  ./demo_resnet -d bf16 -e 9
  ./demo_resnet -d fp32 -e 9 -b 128 -a /root/tensorflow_datasets/imagenet/tf_records/ -m /root/tensorflow-training/demo/ck_81_080_450_bs128
```

#### The demo_keras_resnet script
This script uses Keras, a high-level neural networks library running on top of Tensorflow for training and evaluating the ResNet model:
```
usage: ./demo_keras_resnet [arguments]
mandatory arguments:
  -d <data_type>,    --dtype <data_type>          Data type, possible values: fp32, bf16
optional arguments:
  -b <batch_size>,   --batch-size <batch_size>    Batch size, default 128 for fp32, 256 for bf16"
  -e <epochs>,       --epochs <epochs>            Number of epochs, default to 1"
  -a <data_dir>,     --data-dir <data_dir>        Data dir, defaults to /software/data/tf/data/imagenet/tf_records/
  -m <model_dir>,    --model-dir <model_dir>      Model dir, defaults to $HOME/tmp/resnet/
  -o,                --use-horovod                Use horovod for training
  -s <steps>,        --steps <steps>              Max train steps
  -n,                --no-eval                    Don't do evaluation
  -v <steps>,        --display-steps <steps>      How often display step status
  -l <steps>,        --steps-per-loop             Number of steps per training loop. Will be capped at steps per epoch.
  -c,                --enable-checkpoint          Whether to enable a checkpoint callback and export the savedmodel.
  -r                 --recover                    If crashed restart training from last checkpoint. Requires -s to be set
  -k                 --no-experimental-preloading Disables support for 'data.experimental.prefetch_to_device' TensorFlow operator. If not set:
                                                  - loads extension dynpatch_prf_remote_call.so (via LD_PRELOAD)
                                                  - sets environment variable HBN_TF_REGISTER_DATASETOPS to 1
                                                  - this feature is experimental and works only with single node
example:
  ./demo_keras_resnet -d bf16
  ./demo_keras_resnet -d fp32
  ./demo_keras_resnet -d fp -rs 101
  ./demo_keras_resnet -d bf16 -s 1000
  ./demo_keras_resnet -d bf16 -s 1000 -l 50
  ./demo_keras_resnet -d bf16 -e 9"
  ./demo_keras_resnet -d fp32 -e 9 -b 128 -a home/user1/tensorflow_datasets/imagenet/tf_records/ -m /home/user1/tensorflow-training/demo/ck_81_080_450_bs128

```

### Multi-Card Training

#### The demo_resnet_hvd.sh script

The script requires additional parameters.

```
export IMAGENET_DIR=/path/to/tensorflow_datasets/imagenet/
export RESNET_SIZE=<resnet-size>
./demo_resnet_hvd.sh
```

### Example Commands
To benchmark the training performance on a specific batch size, run:

**For single Gaudi**

ResNet 50 (FP32)

```
./demo_resnet -rs 50 -d fp32 -b 128 -a /path/to/tensorflow_datasets/imagenet/tf_records/ -m /root/tensorflow-training/demo/ck_81_080_450_bs128
```

ResNet 50 (BF16)

```
./demo_resnet -rs 50 -d bf16 -b 256 -a /path/to/tensorflow_datasets/imagenet/tf_records/ -m /root/tensorflow-training/demo/resnet50
```

ResNet 101 (BF16)

```
./demo_resnet -rs 101 -d bf16 -b 256 -a /path/to/tensorflow_datasets/imagenet/tf_records/ -m /root/tensorflow-training/demo/resnet101
```

ResNeXt 101 (BF16)

```
./demo_resnet -rs 101 --resnext -d bf16 -b 256 -a /path/to/tensorflow_datasets/imagenet/tf_records/ -m /root/tensorflow-training/demo/resnext101
```

**For multiple Gaudi cards**

ResNet 50 (BF16)
```
export IMAGENET_DIR=/path/to/tensorflow_datasets/imagenet/
export RESNET_SIZE=50
./demo_resnet_hvd.sh
```
ResNet 101 (BF16)
```
export IMAGENET_DIR=/path/to/tensorflow_datasets/imagenet/
export RESNET_SIZE=101
./demo_resnet_hvd.sh
```
ResNeXt 101 (BF16)
```
export IMAGENET_DIR=/path/to/tensorflow_datasets/imagenet/
export RESNET_SIZE=101
./demo_resnet_hvd.sh --resnext
```

## Preview Python Script
This is the preview of TensorFlow ResNet Python scripts with yaml configuration of parameters
For single card (in the future multi-node workloads will be supported as well) you can use model runners that are written in Python as opposed to bash.

You can run the following script: **Model-References/TensorFlow/habana_model_runner.py** which accepts two arguments:
- --model *model_name*
- --hb_config *path_to_yaml_config_file*

Example of config files can be found in the **Model-References/TensorFlow/computer_vision/Resnets/resnet_estimator_default.yaml** (for resnet_estimator model) as well as **Model-References/TensorFlow/computer_vision/Resnets/resnet_keras/resnet_keras_default.yaml** (for resnet_keras model).

You can use these scripts as such:
> cd Model-References/TensorFlow/computer_vision/Resnets
  >
  > python3 ../../habana_model_runner.py --model resnet_estimator --hb_config *path_to_yaml_config_file*
  >
  > **Example:**
  >
  > python3 ../../habana_model_runner.py --model resnet_estimator --hb_config resnet_estimator_default.yaml




