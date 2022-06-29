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

## Model Overview

Mask R-CNN is a convolution-based neural network for the task of object instance segmentation. The paper describing the model can be found [here](https://arxiv.org/abs/1703.06870). This repository provides a script to train Mask R-CNN for Tensorflow on Habana
Gaudi, and is an optimized version of the implementation in [NVIDIA's Mask R-CNN](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/MaskRCNN) for Gaudi.

Changes in the model:

- Support for Habana device was added
- Horovod support

Please visit [this page](https://developer.habana.ai/resources/habana-training-models/#performance) for performance information. For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

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
Go to the Mask R-CNN directory:
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

The following script will save TFRecords files to the data directory, `/data/tensorflow/coco2017/tf_records`.
```bash
cd dataset
bash download_and_preprocess_coco.sh /data/tensorflow/coco2017/tf_records
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

`mask_rcnn_main.py` script can be used to run training.

Using `mask_rcnn_main.py` with all default parameters:
```bash
$PYTHON mask_rcnn_main.py
```

With exemplary parameters:
```bash
TF_BF16_CONVERSION=full $PYTHON mask_rcnn_main.py --mode=train --checkpoint="weights/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603" --eval_samples=5000 --init_learning_rate=0.005 --learning_rate_steps=240000,320000 --model_dir="results" --num_steps_per_eval=29568 --total_steps=360000 --training_file_pattern="/data/tensorflow/coco2017/tf_records/train-*.tfrecord" --validation_file_pattern="/data/tensorflow/coco2017/tf_records/val-*.tfrecord" --val_json_file="/data/tensorflow/coco2017/tf_records/annotations/instances_val2017.json"
```

You can train the model in different data type by setting the `TF_BF16_CONVERSION` environment variable. `TF_BF16_CONVERSION=full` and `TF_BF16_CONVERSION=basic` automatically converts the appropriate ops to the bfloat16 format. This approach is similar to Automatic Mixed Precision of TensorFlow, which can reduce memory requirements and speed up training. `basic` allows only matrix multiplications and convolutions to be converted. For more details on the mixed precision training JSON recipe files, please refer to the [TensorFlow Mixed Precision Training on Gaudi](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_User_Guide/TensorFlow_Mixed_Precision.html
) documentation.

### Run 8-cards training

Multi-card training in `bf16` over mpirun using `mask_rcnn_main.py`:

*<br>mpirun map-by PE attribute value may vary on your setup and should be calculated as:<br>
socket:PE = floor((number of physical cores) / (number of gaudi devices per each node))*

```bash
export TF_BF16_CONVERSION=full; mpirun --allow-run-as-root --bind-to core --map-by socket:PE=7 --np 8 -x TF_BF16_CONVERSION $PYTHON mask_rcnn_main.py --mode=train --training_file_pattern="/data/tensorflow/coco2017/tf_records/train-*.tfrecord" --validation_file_pattern="/data/tensorflow/coco2017/tf_records/val-*.tfrecord" --val_json_file="/data/tensorflow/coco2017/tf_records/annotations/instances_val2017.json" --init_learning_rate=0.04 --learning_rate_steps=30000,40000 --num_steps_per_eval=3696 --total_steps=45000 --checkpoint="weights/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603" --model_dir="results"
```


For multi-cards training, remember to adjust hyperparameters (typically by multiplying `init_learning_rate` and dividing `learning_rate_steps`, `num_steps_per_eval` and `total_steps` by a number of workers).

For multi-card training in `bf16-basic` over mpirun using `mask_rcnn_main.py`:

*<br>mpirun map-by PE attribute value may vary on your setup and should be calculated as:<br>
socket:PE = floor((number of physical cores) / (number of gaudi devices per each node))*

```bash
export TF_BF16_CONVERSION=basic; mpirun --allow-run-as-root --bind-to core --map-by socket:PE=7 --np 8 -x TF_BF16_CONVERSION $PYTHON mask_rcnn_main.py --mode=train --training_file_pattern="/data/tensorflow/coco2017/tf_records/train-*.tfrecord" --validation_file_pattern="/data/tensorflow/coco2017/tf_records/val-*.tfrecord" --val_json_file="/data/tensorflow/coco2017/tf_records/annotations/instances_val2017.json" --init_learning_rate=0.04 --learning_rate_steps=30000,40000 --num_steps_per_eval=3696 --total_steps=45000 --checkpoint="weights/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603" --model_dir="results"
```

For multi-card training in `fp32` over mpirun using `mask_rcnn_main.py`:

*<br>mpirun map-by PE attribute value may vary on your setup and should be calculated as:<br>
socket:PE = floor((number of physical cores) / (number of gaudi devices per each node))*
```bash
export TF_BF16_CONVERSION=0; mpirun --allow-run-as-root --bind-to core --map-by socket:PE=7 --np 8 -x TF_BF16_CONVERSION $PYTHON mask_rcnn_main.py --mode=train --training_file_pattern="/data/tensorflow/coco2017/tf_records/train-*.tfrecord" --validation_file_pattern="/data/tensorflow/coco2017/tf_records/val-*.tfrecord" --val_json_file="/data/tensorflow/coco2017/tf_records/annotations/instances_val2017.json" --init_learning_rate=0.04 --learning_rate_steps=30000,40000 --num_steps_per_eval=3696 --total_steps=45000 --checkpoint="weights/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603" --model_dir="results"
```

### Parameters

You can modify the training behavior through the various flags in the `mask_rcnn_main.py`. Flags in the `mask_rcnn_main.py` script are as follows:

- `mode`: Run `train`, `train_and_eval` or `eval` on MS COCO. `train_and_eval` by default.
- `training_file_pattern`: TFRecords file pattern for the training files
- `validation_file_pattern`: TFRecords file pattern for the validation files
- `val_json_file`: Filepath for the validation json file
- `checkpoint`: Path to model checkpoint.
- `model_dir`: Model directory.
*<br> WARNING: Unexpected errors can occur if data from previous runs remains in the model directory. In particular, the model will not continue training if it has previously executed more total_steps than the current run, and saved data to the model directory.*<br>
- `total_steps`: The number of steps to use for training, should be adjusted according to the `train_batch_size` flag. Note that for first 100 steps performance won't be reported by the script (-1 will be shown).
- `train_batch_size`: Batch size for training.
- `noeval_after_training`: Disable single evaluation steps after training when `train` command is used.
- `use_fake_data`: Use fake input.
- `pyramid_roi_impl`: Implementation to use for PyramidRoiAlign. `habana` (default), `habana_fp32` and `gather` can be used.
- `eval_samples`: Number of eval samples. Number of steps will be divided by `eval_batch_size`.
- `eval_batch_size`: Batch size for evaluation.
- `num_steps_per_eval`: Number of steps used for evaluation.
- `deterministic`: Enable deterministic behavior.
- `save_summary_steps`: Steps between saving summaries to TensorBoard.
- `device`: Device type. `CPU` and `HPU` can be used.
- `profile`: Gather TensorBoard profiling data.
- `init_learning_rate`: Initial learning rate.
- `learning_rate_steps`: Warmup learning rate decay factor. Expected format: "first_value,second_value".

## Examples

*<br>mpirun map-by PE attribute value may vary on your setup and should be calculated as:<br>
socket:PE = floor((number of physical cores) / (number of gaudi devices per each node))*

| Command | Notes |
| ------- | ----- |
| `TF_BF16_CONVERSION=full $PYTHON mask_rcnn_main.py --mode=train --training_file_pattern="/data/tensorflow/coco2017/tf_records/train-*.tfrecord" --validation_file_pattern="/data/tensorflow/coco2017/tf_records/val-*.tfrecord" --val_json_file="/data/tensorflow/coco2017/tf_records/annotations/instances_val2017.json"` | Single-card training in bf16 |
| `TF_BF16_CONVERSION=0 $PYTHON mask_rcnn_main.py --mode=train --training_file_pattern="/data/tensorflow/coco2017/tf_records/train-*.tfrecord" --validation_file_pattern="/data/tensorflow/coco2017/tf_records/val-*.tfrecord" --val_json_file="/data/tensorflow/coco2017/tf_records/annotations/instances_val2017.json"`| Single-card training in fp32 |
| `TF_BF16_CONVERSION=full $PYTHON mask_rcnn_main.py --mode=eval --validation_file_pattern="/data/tensorflow/coco2017/tf_records/val-*.tfrecord" --val_json_file="/data/tensorflow/coco2017/tf_records/annotations/instances_val2017.json"` | Single-card evaluation in bf16 |
| `export TF_BF16_CONVERSION=full; mpirun --allow-run-as-root --bind-to core --map-by socket:PE=7 --np 8 -x TF_BF16_CONVERSION $PYTHON mask_rcnn_main.py --mode=train --training_file_pattern="/data/tensorflow/coco2017/tf_records/train-*.tfrecord" --validation_file_pattern="/data/tensorflow/coco2017/tf_records/val-*.tfrecord" --val_json_file="/data/tensorflow/coco2017/tf_records/annotations/instances_val2017.json" --init_learning_rate=0.04 --learning_rate_steps=30000,40000 --num_steps_per_eval=3696 --total_steps=45000`| 8-cards training in bf16 |
| `export TF_BF16_CONVERSION=0; mpirun --allow-run-as-root --bind-to core --map-by socket:PE=7 --np 8 -x TF_BF16_CONVERSION $PYTHON mask_rcnn_main.py --mode=train --training_file_pattern="/data/tensorflow/coco2017/tf_records/train-*.tfrecord" --validation_file_pattern="/data/tensorflow/coco2017/tf_records/val-*.tfrecord" --val_json_file="/data/tensorflow/coco2017/tf_records/annotations/instances_val2017.json" --init_learning_rate=0.04 --learning_rate_steps=30000,40000 --num_steps_per_eval=3696 --total_steps=45000` | 8-cards training in fp32 |
| `export TF_BF16_CONVERSION=full; mpirun --allow-run-as-root --bind-to core --map-by socket:PE=7 --np 8 -x TF_BF16_CONVERSION $PYTHON mask_rcnn_main.py --mode=eval --validation_file_pattern="/data/tensorflow/coco2017/tf_records/val-*.tfrecord" --val_json_file="/data/tensorflow/coco2017/tf_records/annotations/instances_val2017.json"` | 8-cards evaluation in bf16 |
| `export TF_BF16_CONVERSION=full; mpirun --allow-run-as-root --bind-to core --map-by socket:PE=7 --np 8 -x TF_BF16_CONVERSION $PYTHON mask_rcnn_main.py --training_file_pattern="/data/tensorflow/coco2017/tf_records/train-*.tfrecord" --validation_file_pattern="/data/tensorflow/coco2017/tf_records/val-*.tfrecord" --val_json_file="/data/tensorflow/coco2017/tf_records/annotations/instances_val2017.json" --init_learning_rate=0.04 --learning_rate_steps=30000,40000 --num_steps_per_eval=3696 --total_steps=45000`| 8-cards training and evaluation in bf16 |

## Supported Configuration

| Device | SynapseAI Version | TensorFlow Version(s)  |
|:------:|:-----------------:|:-----:|
| Gaudi  | 1.4.1             | 2.8.0 |
| Gaudi  | 1.4.1             | 2.7.1 |

## Changelog
### 1.2.0
- remove workaround for Multiclass NMS
### 1.3.0
- setup BF16 conversion pass using a config from habana-tensorflow package instead of recipe json file
- fixed freeze in multi-card scenario when checkpoint already exist
- update requirements.txt
- Change `python` or `python3` to `$PYTHON` to execute correct version based on environment setup.
### 1.4.0
- References to custom demo script were replaced by community entry points in README.
- Default values of some parameters were changed
- add flag to specify maximum number of cpus to be used
- disable cache and prefetch when not enough RAM
