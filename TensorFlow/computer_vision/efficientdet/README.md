# EfficientDet for TensorFlow

This directory provides a script and recipe to train an EfficientDet model to achieve state of the art accuracy, and is tested and maintained by Habana. For further information on performance, refer to [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

For further information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents

* [Model-References](../../../README.md)
* [Model Overview](#model-overview)
* [Setup](#setup)
* [Training and Examples](#training-and-examples)
* [Supported Configuration](#supported-configuration)
* [Changelog](#changelog)
* [Known Issues](#known-issues)

## Model Overview

This directory describes how to train an EfficientDet model on multiple HPUs by using Horovod supporting synchronized batch norm.
For the full list of AutoML related models and libraries, refer to [Brain AutoML](https://github.com/itsliupeng/automl).

For further information on EfficientDet, Mingxing Tan, Ruoming Pang and Quoc V. Le., refer to [Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070).

## Setup

Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) to set up the environment including the `$PYTHON` environment variable.  To achieve the best performance, please follow the methods outlined in the [Optimizing Training Platform guide](https://docs.habana.ai/en/latest/TensorFlow/Model_Optimization_TensorFlow/Optimization_Training_Platform.html).  
The guides will walk you through the process of setting up your system to run the model on Gaudi.

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

1. In the docker container, go to the EfficientDet directory:

```bash
cd /root/Model-References/TensorFlow/computer_vision/efficientdet
```

2. Install the required packages using pip:

```bash
$PYTHON -m pip install -r requirements.txt
```

### Download COCO 2017 Dataset

1. Download COCO 2017 dataset, by running the following command:

```
cd dataset
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip && rm train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip && rm val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
unzip test2017.zip && rm test2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip && rm annotations_trainval2017.zip
```

2. Convert COCO 2017 data to tfrecord, by running the following command::

```
mkdir tfrecord
# training data
PYTHONPATH=".:$PYTHONPATH" $PYTHON ./dataset/create_coco_tfrecord.py \
  --image_dir=./dataset/train2017 \
  --image_info_file=./dataset/annotations/instances_train2017.json \
  --object_annotations_file=./dataset/annotations/instances_train2017.json \
  --caption_annotations_file=./dataset/annotations/captions_train2017.json \
  --output_file_prefix=tfrecord/train \
  --num_shards=32
# validation data
PYTHONPATH=".:$PYTHONPATH" $PYTHON ./dataset/create_coco_tfrecord.py \
  --image_dir=./dataset/val2017 \
  --image_info_file=./dataset/annotations/instances_val2017.json \
  --object_annotations_file=./dataset/annotations/instances_val2017.json \
  --caption_annotations_file=./dataset/annotations/captions_val2017.json \
  --output_file_prefix=tfrecord/val \
  --num_shards=32
```

### Download the Backbone Checkpoint

To download the backbone checkpoint, run the following command:

```
cd backbone
wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckpts/efficientnet-b0.tar.gz
tar -xvf efficientnet-b0.tar.gz
```

## Training and Examples

### Single Card and Multi-Card Training Examples

**Run training on 1 HPU:**

```
$PYTHON main.py [options]
```

Run training on 1 HPU with batch size 8, full training of 300 epochs, on COCO datase:

```bash
$PYTHON main.py --mode=train --train_batch_size 8 --num_epochs 300 --training_file_pattern "/data/tensorflow/coco2017/tf_records/train-*" --backbone_ckpt "/data/tensorflow/efficientdet/backbones/efficientnet-b0"
```

**Run training on 8 HPUs:**

**NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration).

```bash
mpirun --allow-run-as-root --bind-to core --map-by socket:PE=6 -np <num_workers> $PYTHON main.py --use_horovod <num_workers>
```

Run training on 8 HPUs with batch size 8, full training of 300 epochs, on COCO dataset:

```bash
mpirun --allow-run-as-root --bind-to core --map-by socket:PE=6 -np 8 $PYTHON main.py --backbone_ckpt=/data/tensorflow/efficientdet/backbones/efficientnet-b0/ --training_file_pattern=/data/tensorflow/coco2017/tf_records/train-* --model_dir /tmp/efficientdet --use_horovod 8 --keep_checkpoint_max=300
```

The following are the available options for training:

main.py
  - `--backbone_ckpt`: Location of the EfficientNet checkpoint to use for the model initialization.
  - `--ckpt`: Start training from this EfficientDet checkpoint.
  - `--cp_every_n_steps`: Number of iterations after which checkpoint is saved.
  - `--deterministic`: Deterministic input data.
  - `--eval_after_training`: Run one evaluation after the training finishes.
  - `--eval_batch_size`: Evaluation batch size.
  - `--eval_master`: GRPC URL of the evaluation master. Set to an appropriate value when running on CPU/GPU.
  - `--eval_samples`: The number of samples for evaluation.
  - `--eval_timeout`: Maximum seconds between checkpoints before evaluation terminates.
  - `--gcp_project`: Project name for the Cloud TPU-enabled project. If not specified, we will attempt to automatically detect the GCE project from metadata.
  - `--hparams`: Comma separated k=v pairs of hyperparameters.
  - `--input_partition_dims`: A list that describes the partition dims for all the tensors.
  - `--iterations_per_loop`: Number of iterations per TPU training loop.
  - `--log_every_n_steps`: Number of iterations after which training parameters are logged.
  - `--min_eval_interval`: Minimum seconds between evaluations.
  - `--mode`: Mode to run: train or eval (default: train).
  - `--model_dir`: Location of model_dir.
  - `--model_name`: Model name: retinanet or efficientdet.
  - `--no_hpu`: Do not load Habana modules = train on CPU/GPU.
  - `--num_cores`: Number of TPU cores for training.
  - `--num_cores_per_replica`: Number of TPU cores perreplica when using spatial partition.
  - `--num_epochs`: Number of epochs for training.
  - `--num_examples_per_epoch`: Number of examples in one epoch.
  - `--testdev_dir`: COCO testdev dir. If true, ignore val_json_file.
  - `--tpu`: The Cloud TPU to use for training. This should be either the name used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.
  - `--tpu_zone`: GCE zone where the Cloud TPU is located in. If not specified, we will attempt to automatically detect the GCE project from metadata.
  - `--train_batch_size`: Training batch size.
  - `--training_file_pattern`: Glob for training data files (e.g., COCO train - minival set).
  - `--use_amp`: Use AMP.
  - `--use_fake_data`: Use fake input.
  - `--use_horovod`: Use Horovod for distributed training
  - `--use_spatial_partition`: Use spatial partition.
  - `--use_tpu`: Use TPUs rather than CPUs/GPUs.
  - `--use_xla`: Use XLA.
  - `--val_json_file`: COCO validation JSON containing golden bounding boxes.
  - `--validation_file_pattern`: Glob for evaluation tfrecords (e.g., COCO val2017 set).


## Supported Configuration

| Validated on | SynapseAI Version | TensorFlow Version(s) | Mode |
|:------:|:-----------------:|:-----:|:----------:|
| Gaudi   | 1.8.0             | 2.11.0         | Training |
| Gaudi   | 1.8.0             | 2.8.4          | Training |

## Changelog

### 1.4.0

* Added support to import horovod-fork package directly instead of using Model-References' TensorFlow.common.horovod_helpers. Wrapped horovod import with a try-catch block so that installing the library is not required when the model is running on a single card.
* Replaced references to custom demo script by community entry points in the README.

### 1.2.0

* Removed the workaround for 6D tensors which brings back 6D tensors into the script.

## Known Issues

* Currently, the TPC fuser is disabled.
* Currently, the maximum batch size is 32. Batch size 64 will be available in the future.
* Only d0 variant is avalable. Variants d1 to d7 will be available in the future.
