# DenseNet Keras model

This repository provides a script and recipe to train DenseNet121 to achieve state of the art accuracy, and is tested and maintained by Habana Labs, Ltd. an Intel Company.

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

## Table Of Contents
  * [Model Overview](#model-overview)
  * [Setup](#setup)
  * [Training](#training)
  * [Advanced](#advanced)
  * [Changelog](#changelog)

## Model Overview

Paper: [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
Original script: https://github.com/flyyufelix/cnn_finetune


### Model changes
Major changes done to original model from [https://github.com/flyyufelix/cnn_finetune](https://github.com/flyyufelix/cnn_finetune):
* Some scripts were changed in order to run the model on Gaudi. It includes loading habana tensorflow modules and using multi Gaudi card helpers;
* Support for distributed training using HPUStrategy was addedd;
* Data pipeline was optimized;
* Color distortion and random image rotation was removed from training data augmentation;
* Learning rate scheduler with warmup was added and set as default;
* Kernel and bias regularization was added Conv2D and Dense layers;
* Additional tensorboard and performance logging options were added;
* Additional synthetic data and tensor dumping options were added.


## Setup

Please follow the instructions given in the following link for setting up the
environment including the `$PYTHON` environment variable: [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html).
This guide will walk you through the process of setting up your system to run
the model on Gaudi.

### Training Data

The script operates on ImageNet 1k, a widely popular image classification dataset from the ILSVRC challenge.
In order to obtain the dataset, follow these steps:
1. Sign up with http://image-net.org/download-images and acquire the rights to download original images
2. Follow the link to the 2012 ILSVRC
 and download `ILSVRC2012_img_val.tar` and `ILSVRC2012_img_train.tar`.
3. Use the commands below - they will prepare the dataset under `/data/tensorflow_datasets/imagenet/tf_records`. This is the default data_dir for the training script.

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

## Training

In the docker container, clone this repository and switch to the branch that
matches your SynapseAI version. (Run the
[`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.)

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```
Go to the Densenet directory:
```bash
cd Model-References/TensorFlow/computer_vision/densenet
```

### Install Model Requirements

In the docker container, go to the DenseNet directory
```bash
cd /root/Model-References/TensorFlow/computer_vision/densenet
```
Install required packages using pip
```bash
$PYTHON -m pip install -r requirements.txt
```

### Training Examples
**Run training on 1 HPU**

- DenseNet121, 1 HPU, batch 256, 90 epochs, bf16 precision, SGD
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
- DenseNet121, 1 HPU, batch 128, 1 epoch, fp32 precision, SGD
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

**Run training on 8 HPU - tf.distribute**

- DenseNet121, 8 HPUs on 1 server, batch 1024 (128 per HPU), 90 epochs, bf16 precision, SGD

  *<br>mpirun map-by PE attribute value may vary on your setup and should be calculated as:<br>
  socket:PE = floor((number of physical cores) / (number of gaudi devices per each node))*

    ```bash
    mpirun --allow-run-as-root --bind-to core --map-by socket:PE=4 -np 8 \
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

The following sections provide greater details of the dataset, running training and inference, and the training results.

### Scripts and sample code

In the root directory, the most important files are:

* `train.py`: Serves as the entry point to the application. Encapsulates the training routine.
* `densenet.py`: Defines the model architecture.

The `utils/` folder contains the necessary utilities to train DenseNet model. Its main components are:
* `arguments.py`: Defines command line argumnet parser.
* `dataset.py`: Implements the functions defining the data pipeline.
* `image_processing.py`: Implements the data preprocessing and augmentation.

The `models/` folder contains utilities allowing for training
* `models.py`: Defines utilities necessary for training.
* `optimizers.py`: Defines optimizer utilities.

### Parameters

The list of the available parameters for the `train.py` script contains:
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

| Device | SynapseAI Version | TensorFlow Version(s)  |
|:------:|:-----------------:|:-----:|
| Gaudi  | 1.4.0             | 2.8.0 |
| Gaudi  | 1.4.0             | 2.7.1 |

## Changelog
### 1.2.0
* decrease recommended batch size in README to 128 per node for single-HLS scenario to reach accuracy target after 90 epochs
* set `--initial_lr` to 0.28 in README for above batch size for single-HLS
  * removed in-code rescaling of learning rate provided as `--initial_lr`
### 1.3.0
* align script with new imagenet naming convention (img_train->train, img_val->validation)
* add multiHLS config
* fix for incorrect LR when checkpoint is loaded
* update requirements.txt
### 1.4.0
* introduced `--dataset_num_parallel_calls` script parameter to manipulate number of threads used to preprocess dataset in parallel
* References to custom demo script were replaced by community entry points in README.