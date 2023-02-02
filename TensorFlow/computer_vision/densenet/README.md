# DenseNet Keras for TensorFlow

This directory provides a script and recipe to train a DenseNet121 model to achieve state of the art accuracy, and is tested and maintained by Habana.
For further information on performance, refer to [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

For further information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

## Table Of Contents

* [Model-References](../../../README.md)
* [Model Overview](#model-overview)
* [Setup](#setup)
* [Training Examples](#training-examples)
* [Advanced](#advanced)
* [Supported Configuration](#supported-configuration)
* [Changelog](#changelog)

## Model Overview

For further information on DenseNet Keras model, refer to [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993).

### Model Changes

The DenseNet Keras model is a modified version of the original model located in [Fine-tune Convolutional Neural Network in Keras](https://github.com/flyyufelix/cnn_finetune).

The below lists the major changes applied to the model:

* Changed some scripts to run the model on Gaudi. This includes loading Habana TensorFlow modules and using  multiple Gaudi cards helpers.
* Added support for distributed training using HPUStrategy.
* Optimized data pipeline.
* Removed color distortion and random image rotation from training data augmentation.
* Added learning rate scheduler with warmup and set as default.
* Added kernel and bias regularization including Conv2D and Dense layers.
* Added further TensorBoard and performance logging options.
* Added further synthetic data and tensor dumping options.


## Setup

Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) to set up the
environment including the `$PYTHON` environment variable.
The guide will walk you through the process of setting up your system to run the model on Gaudi.

### Training Data

The DenseNet121 script operates on ImageNet 1k, a widely popular image classification dataset from the ILSVRC challenge.
To obtain the dataset, perform the following steps:
1. Sign up with http://image-net.org/download-images and acquire the rights to download original images.
2. Follow the link to the 2012 ILSVRC and download `ILSVRC2012_img_val.tar` and `ILSVRC2012_img_train.tar`.
3. Use the below commands to prepare the dataset under `/data/tensorflow_datasets/imagenet/tf_records`. This is the default data_dir for the training script.

```bash
export IMAGENET_HOME=/data/tensorflow_datasets/imagenet
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

cd Model-References/TensorFlow/computer_vision/Resnets
$PYTHON preprocess_imagenet.py \
  --raw_data_dir=$IMAGENET_HOME \
  --local_scratch_dir=$IMAGENET_HOME/tf_records
```

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

1. In the docker container, go to the DenseNet directory:

```bash
cd /root/Model-References/TensorFlow/computer_vision/densenet
```

2. Install the required packages using pip:

```bash
$PYTHON -m pip install -r requirements.txt
```

## Training Examples

### Single Card and Multi-Card Training Examples

**Run training on 1 HPU:**

- 1 HPU, batch 256, 90 epochs, BF16 precision, SGD:

    ```bash
    $PYTHON train.py \
        --dataset_dir /data/tensorflow_datasets/imagenet/tf_records \
        --dtype bf16 \
        --model densenet121 \
        --optimizer    sgd \
        --batch_size   256 \
        --epochs       90 \
        --run_on_hpu
    ```
- 1 HPU, batch 128, 1 epoch, FP32 precision, SGD:

    ```bash
    $PYTHON train.py \
        --dataset_dir /data/tensorflow_datasets/imagenet/tf_records \
        --dtype fp32 \
        --model densenet121 \
        --optimizer    sgd \
        --batch_size   128 \
        --epochs       1 \
        --run_on_hpu
    ```

**Run training on 8 HPUs - tf.distribute:**

**NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration).

8 HPUs on 1 server, batch 1024 (128 per HPU), 90 epochs, BF16 precision, SGD:

    ```bash
    mpirun --allow-run-as-root --bind-to core --map-by socket:PE=6 -np 8 \
        $PYTHON train.py \
        --dataset_dir /data/tensorflow_datasets/imagenet/tf_records \
        --dtype bf16 \
        --model densenet121 \
        --optimizer sgd \
        --batch_size 1024 \
        --initial_lr 0.28 \
        --epochs 90 \
        --run_on_hpu \
        --use_hpu_strategy
    ```

## Advanced

The following sections provide further information on the dataset, running training and inference, and the training results.

### Scripts and Sample Code

The following lists the critical files in the root directory:

* `train.py` - Serves as the entry point to the application, and encapsulates the training routine.
* `densenet.py`- Defines the model architecture.

The `utils/` folder contains the following required utilities to train a DenseNet model:
* `arguments.py` - Defines the command line argument parser.
* `dataset.py`- Implements the functions defining the data pipeline.
* `image_processing.py` - Implements the data preprocessing and augmentation.

The `models/` folder contains the following required utilities to allow training:
* `models.py`: Defines utilities necessary for training.
* `optimizers.py`: Defines optimizer utilities.

### Parameters

The list of the available parameters for the `train.py` script contains the following:

```
Optional arguments:
  Name                          Type        Description
  --dataset_dir                 string      Path to dataset (default: /data/imagenet/tf_records)
  --model_dir                   string      Directory for storing saved model and logs (default:./)
  --dtype                       string      Training precision {fp32,bf16} (default: bf16)
  --dropout_rate                float       Dropout rate (default: 0.000000)
  --optimizer                   string      Optimizer used for training {sgd,adam,rmsprop}
  --batch_size                  int         Training batch size (default: 256)
  --initial_lr                  float       Initial lerning rate (default: 0.100000)
  --weight_decay                float       Weight decay (default: 0.000100)
  --epochs                      int         Total number of epochs for training (default: 90)
  --steps_per_epoch             int         Number of steps per epoch (default: None)
  --validation_steps            int         Number of validation steps, set to 0 to disable validation (default: None)
  --model                       string      Model size {densenet121,densenet161,densenet169}
  --train_subset                string      Name of training subset from dataset_dir
  --val_subset                  string      Name of validation subset from dataset_dir
  --resume_from_checkpoint_path string      Path to checkpoint from which to resume training (default: None)
  --resume_from_epoch           int         Epoch from which to resume training (used in conjunction with resume_from_checkpoint_path argument) (default: 0)
  --evaluate_checkpoint_path    string      Checkpoint path for evaluating the model on --val_subset (default: None)
  --seed                        int         Random seed (default: None)
  --warmup_epochs               int         Number of epochs with learning rate warmup (default: 5)
  --save_summary_steps          int         Steps between saving summaries to TensorBoard; when None, logging to TensorBoard is disabled. (enabling this option might affect the performance) (default: None)

Optional switches:
  Name                          Description
  --run_on_hpu                  Whether to use HPU for training (default: False)
  --use_hpu_strategy            Enables HPU strategy for distributed training
  -h, --help                    Show this help message and exit
```
## Supported Configuration

| Validated on | SynapseAI Version | TensorFlow Version(s) | Mode |
|:------:|:-----------------:|:-----:|:----------:|
| Gaudi   | 1.8.0             | 2.11.0         | Training |
| Gaudi   | 1.8.0             | 2.8.4          | Training |

## Changelog

### 1.4.0

* Introduced the `--dataset_num_parallel_calls` script parameter to manipulate a number of threads used to pre-process the dataset in parallel.
* Replaced references to the custom demo script by community entry points in the README.

### 1.3.0

* Aligned the script with new imagenet naming conventions `img_train->train` and `img_val->validation`.
* Added multiple servers configuration.
* Fixed for incorrect LR when checkpoint is loaded.
* Updated the requirements.txt.

### 1.2.0

* Decreased the recommended batch size in the README to 128 per HPU for 1 server scenario to reach accuracy target after 90 epochs.
* Set `--initial_lr` to 0.28 in the README for 128 batch size on 1 server.
  * Removed in-code rescaling of the learning rate provided as `--initial_lr`

