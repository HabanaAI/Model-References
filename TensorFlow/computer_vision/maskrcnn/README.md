# Mask R-CNN for TensorFlow

## Table of Contents

- [Model Overview](#model-overview)
  - [Model architecture](#model-architecture)
- [Setup](#setup)
  - [Install Model Requirements](#install-model-requirements)
  - [Setting up COCO 2017 dataset](#setting-up-coco-2017-dataset)
  - [Download the pre-trained weights](#download-the-pre-trained-weights)
- [Training the model](#training-the-model)
  - [Run single-card training](#run-single-card-training)
  - [Run 8-cards training](#run-8-cards-training)
  - [Parameters](#parameters)
- [Examples](#examples)
- [Changelog](#changelog)
  - [1.2.0](#120)

## Model Overview

Mask R-CNN is a convolution-based neural network for the task of object instance segmentation. The paper describing the model can be found [here](https://arxiv.org/abs/1703.06870). This repository provides a script to train Mask R-CNN for Tensorflow on Habana
Gaudi, and is an optimized version of the implementation in [NVIDIA's Mask R-CNN](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/MaskRCNN) for Gaudi.

Changes in the model:

- Support for Habana device was added
- Horovod support

Please visit [this page](https://developer.habana.ai/resources/habana-training-models/#performance) for performance information. For more information about training deep learning models on Gaudi, visit [developer.habana.ai](developer.habana.ai).

### Model architecture

Mask R-CNN builds on top of Faster R-CNN adding an additional mask head for the task of image segmentation.

The architecture consists of the following:

- ResNet-50 backbone with Feature Pyramid Network (FPN)
- Region proposal network (RPN) head
- RoI Align
- Bounding and classification box head
- Mask head

## Setup
Please follow the instructions given in the following link for setting up the
environment including the `$PYTHON` environment variable: [Gaudi Setup and
Installation Guide](https://github.com/HabanaAI/Setup_and_Install). Please
answer the questions in the guide according to your preferences. This guide will
walk you through the process of setting up your system to run the model on
Gaudi.

In the docker container, clone this repository and switch to the branch that
matches your SynapseAI version. (Run the
[`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.)

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```
Go to the Mask RCNN directory:
```bash
cd Model-References/TensorFlow/computer_vision/maskrcnn
```

### Install Model Requirements

In the docker container, go to the Mask R-CNN directory
```bash
cd /root/Model-References/TensorFlow/computer_vision/maskrcnn
```
Install required packages using pip
```bash
$PYTHON -m pip install -r requirements.txt
```

### Setting up COCO 2017 dataset

This repository provides scripts to download and preprocess the [COCO 2017 dataset](http://cocodataset.org/#download). If you already have the data then you do not need to run the following script, proceed to [Download the pre-trained weights](#download-the-pre-trained-weights).

The following script will save TFRecords files to the data directory, `/data`.
```bash
cd dataset
bash download_and_preprocess_coco.sh /data
```

By default, the data directory is organized into the following structure:
```bash
<data_directory>
    raw-data/
        train2017/
        train2017.zip
        test2017/
        test2017.zip
        val2017/
        val2017.zip
        annotations_trainval2017.zip
        image_info_test2017.zip
    annotations/
        instances_train2017.json
        person_keypoints_train2017.json
        captions_train2017.json
        instances_val2017.json
        person_keypoints_val2017.json
        image_info_test2017.json
        image_info_test-dev2017.json
        captions_val2017.json
    train-*.tfrecord
    val-*.tfrecord
```

### Download the pre-trained weights
This repository also provides scripts to download the pre-trained weights of ResNet-50 backbone.
The script will make a new directory with the name `weights` in the current directory and download the pre-trained weights in it.

```bash
./download_and_process_pretrained_weights.sh
```

Ensure that the `weights` folder created has a `resnet` folder in it.
Inside the `resnet` folder there should be 3 folders for checkpoints and weights: `extracted_from_maskrcnn`, `resnet-nhwc-2018-02-07` and `resnet-nhwc-2018-10-14`.
Before moving to the next step, ensure `resnet-nhwc-2018-02-07` is not empty.

## Training the model

As a prerequisite, root of this repository must be added to `PYTHONPATH`. For example:
```bash
export PYTHONPATH=$PYTHONPATH:$HOME/Model-References
```

### Run single-card training

Both `demo_mask_rcnn.py` and `mask_rcnn_main.py` script can be used to run training. `demo_mask_rcnn.py` is a thin wrapper for `mask_rcnn_main.py`, that reduces boilerplate.

Using `demo_mask_rcnn.py` with all default parameters:
```bash
$PYTHON demo_mask_rcnn.py
```

With exemplary parameters:
```bash
$PYTHON demo_mask_rcnn.py train --dataset /data -d bf16
```

Equivalent command using `mask_rcnn_main.py`:
```bash
TF_BF16_CONVERSION=/path/to/Model-References/TensorFlow/common/bf16_config/full.json $PYTHON mask_rcnn_main.py --mode=train --checkpoint="weights/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603" --eval_samples=5000 --init_learning_rate=0.005 --learning_rate_steps=240000,320000 --model_dir="results" --num_steps_per_eval=29568 --total_steps=360000 --training_file_pattern="/data/train-*.tfrecord" --validation_file_pattern="/data/val-*.tfrecord" --val_json_file="/data/annotations/instances_val2017.json"
```

### Run 8-cards training

When `demo_mask_rcnn.py` is used, then `hvd_workers` parameter can be set. HCL configuration and hyperparameters adjustment will be performed automatically.
```bash
$PYTHON demo_mask_rcnn.py train --dataset /data -d bf16 --hvd_workers 8
```

It's possible to invoke `mpirun` directly, both using `demo_mask_rcnn.py` and `mask_rcnn_main.py`. Please note that if you are training the model in a Kubernetes environment you need to use the following mpirun command. In this case, create a valid HCL configuration file and provide the file path using the `HCL_CONFIG_PATH` environment variable.
For the HCL configuration file format, read [the API Reference Guides page](https://docs.habana.ai/en/latest/API_Reference_Guides/HCL_API_Reference.html#hcl-json-config-file-format).
Don't use `hvd_workers` parameter and remember to adjust hyperparameters (typically by multiplying `init_learning_rate` and dividing `learning_rate_steps`, `num_steps_per_eval` and `total_steps` by a number of workers).

Multi-card training in `bf16` over mpirun using `demo_mask_rcnn.py`:
```bash
mpirun --allow-run-as-root --np 8 $PYTHON demo_mask_rcnn.py train --dataset /data -d bf16 --init_learning_rate 0.04 --learning_rate_steps 30000,40000 --num_steps_per_eval 3696 --total_steps 45000
```
Equivalent command using `mask_rcnn_main.py`:
```bash
TF_BF16_CONVERSION=/path/to/Model-References/TensorFlow/common/bf16_config/full.json mpirun --allow-run-as-root --np 8 $PYTHON mask_rcnn_main.py --mode=train --training_file_pattern="/data/train-*.tfrecord" --validation_file_pattern="/data/val-*.tfrecord" --val_json_file="/data/annotations/instances_val2017.json" --init_learning_rate=0.04 --learning_rate_steps=30000,40000 --num_steps_per_eval=3696 --total_steps=45000 --checkpoint="weights/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603" --model_dir="results"
```

You can train the model in different data type by setting the `TF_BF16_CONVERSION` environment variable. For more details on the mixed precision training JSON recipe files, please refer to the [TensorFlow Mixed Precision Training on Gaudi](https://docs.habana.ai/en/latest/Tensorflow_User_Guide/Tensorflow_User_Guide.html#tensorflow-mixed-precision-training-on-gaudi) documentation.
For multi-card training in `bf16-basic` over mpirun using `mask_rcnn_main.py`:
```bash
TF_BF16_CONVERSION=/path/to/Model-References/TensorFlow/common/bf16_config/basic.json mpirun --allow-run-as-root --np 8 $PYTHON mask_rcnn_main.py --mode=train --training_file_pattern="/data/train-*.tfrecord" --validation_file_pattern="/data/val-*.tfrecord" --val_json_file="/data/annotations/instances_val2017.json" --init_learning_rate=0.04 --learning_rate_steps=30000,40000 --num_steps_per_eval=3696 --total_steps=45000 --checkpoint="weights/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603" --model_dir="results"
```

For multi-card training in `fp32` over mpirun using `mask_rcnn_main.py`:
```bash
TF_BF16_CONVERSION=0 mpirun --allow-run-as-root --np 8 $PYTHON mask_rcnn_main.py --mode=train --training_file_pattern="/data/train-*.tfrecord" --validation_file_pattern="/data/val-*.tfrecord" --val_json_file="/data/annotations/instances_val2017.json" --init_learning_rate=0.04 --learning_rate_steps=30000,40000 --num_steps_per_eval=3696 --total_steps=45000 --checkpoint="weights/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603" --model_dir="results"
```

### Parameters

You can modify the training behavior through the various flags in the `demo_mask_rcnn.py` script and the `mask_rcnn_main.py`.
Flags in the `demo_mask_rcnn.py` script are as follows:

- `command`: Run `train`, `train_and_eval` or `eval` on MS COCO. `train_and_eval` by default.
- `dataset`: Dataset directory.
- `checkpoint`: Path to model checkpoint.
- `model_dir`: Model directory.
- `total_steps`, `s`: The number of steps to use for training, should be adjusted according to the `train_batch_size` flag. Note that for first 100 steps performance won't be reported by the script (-1 will be shown).
- `dtype`, `d`: Data type, `fp32`, `bf16` or `bf16-basic`.
`bf16` and `bf16-basic` automatically converts the appropriate ops to the bfloat16 format. This approach is similar to Automatic Mixed Precision of TensorFlow, which can reduce memory requirements and speed up training. `bf16-basic` allows only matrix multiplications and convolutions to be converted.
- `hvd_workers`: Number of Horovod workers, disabled by default.
- `train_batch_size`, `bs`, `b`: Batch size for training.
- `no_eval_after_training`, `t`: Disable single evaluation steps after training when `train` command is used.
- `clean_model_dir`, `c`: Clean model directory before execution.
- `use_fake_data`, `f`: Use fake input.
- `pyramid_roi_impl`: Implementation to use for PyramidRoiAlign. `habana` (default), `habana_fp32` and `gather` can be used.
- `eval_samples`: Number of eval samples. Number of steps will be divided by `eval_batch_size`.
- `eval_batch_size`: Batch size for evaluation.
- `num_steps_per_eval`: Number of steps used for evaluation.
- `deterministic`: Enable deterministic behavior.
- `save_summary_steps`: Steps between saving summaries to TensorBoard.
- `map_by_socket`: MPI maps processes to sockets. Used only when `hvd_workers` is set.
- `device`: Device type. `CPU` and `HPU` can be used.
- `profile`: Gather TensorBoard profiling data.
- `init_learning_rate`: Initial learning rate.
- `learning_rate_steps`: Warmup learning rate decay factor. Expected format: "first_value,second_value".

## Examples

| Command | Notes |
| ------- | ----- |
|`$PYTHON demo_mask_rcnn.py train --dataset /data -d bf16`| Single-card training in bf16 |
|`$PYTHON demo_mask_rcnn.py train --dataset /data -d fp32`| Single-card training in fp32 |
|`$PYTHON demo_mask_rcnn.py eval --dataset /data`| Single-card evaluation in bf16 |
|`$PYTHON demo_mask_rcnn.py train --dataset /data -d bf16 --hvd_workers 8`| 8-cards training in bf16 |
|`$PYTHON demo_mask_rcnn.py train --dataset /data -d fp32 --hvd_workers 8`| 8-cards training in fp32 |
|`$PYTHON demo_mask_rcnn.py eval --dataset /data --hvd_workers 8`| 8-cards evaluation in bf16 |
|`mpirun --allow-run-as-root --np 8 $PYTHON demo_mask_rcnn.py --dataset /data --init_learning_rate 0.04 --learning_rate_steps 30000,40000 --num_steps_per_eval 3696 --total_steps 45000`| 8-cards training and evaluation in bf16 |


## Changelog
### 1.2.0
- remove workaround for Multiclass NMS
