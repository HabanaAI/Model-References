# Mask R-CNN for TensorFlow

This directory provides a script to train a Mask R-CNN model to achieve state-of-the-art accuracy, and is tested and maintained by Habana. To obtain model performance data, refer to the [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

For more information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents

- [Model-References](../../../README.md)
- [Model Overview](#model-overview)
- [Setup](#setup)
- [Training Examples](#training-examples)
- [Supported Configuration](#supported-configuration)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Model Overview

Mask R-CNN is a convolution-based neural network for the task of object instance segmentation. The paper describing the model can be found [here](https://arxiv.org/abs/1703.06870). This model is an optimized version of the implementation in [NVIDIA's Mask R-CNN](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/MaskRCNN) for Gaudi.

### Model Changes

- Support for Habana device was added
- Horovod support


### Model Architecture

Mask R-CNN builds on top of Faster R-CNN adding an additional mask head for the task of image segmentation.

The architecture consists of the following:

- ResNet-50 backbone with Feature Pyramid Network (FPN)
- Region proposal network (RPN) head
- RoI Align
- Bounding and classification box head
- Mask head

## Setup
Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) to set up the environment including the `$PYTHON` environment variable.  To achieve the best performance, please follow the methods outlined in the [Optimizing Training Platform guide](https://docs.habana.ai/en/latest/TensorFlow/Model_Optimization_TensorFlow/Optimization_Training_Platform.html).  
The guides will walk you through the process of setting up your system to run the model on Gaudi.

### Clone Habana Model-References
In the docker container, clone this repository and switch to the branch that
matches your SynapseAI version. You can run the
[`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```
Go to the Mask R-CNN directory:
```bash
cd Model-References/TensorFlow/computer_vision/maskrcnn
```
**Note:** If the repository is not in the PYTHONPATH, make sure to update by running the below:
```bash
export PYTHONPATH=$PYTHONPATH:root/Model-References
```

### Install Model Requirements

1. In the docker container, go to the Mask R-CNN directory:
```bash
cd /root/Model-References/TensorFlow/computer_vision/maskrcnn
```
2. Install the required packages using pip:
```bash
$PYTHON -m pip install -r requirements.txt
```

### Setting up COCO 2017 Dataset

This repository provides scripts to download and pre-process the [COCO 2017 dataset](http://cocodataset.org/#download). If you already have the data then you do not need to run the following script. Proceed to [Download the pre-trained weights](#download-the-pre-trained-weights).

The following script saves TFRecords files to the data directory, `/data/tensorflow/coco2017/tf_records`.
```bash
cd dataset
bash download_and_preprocess_coco.sh /data/tensorflow/coco2017/tf_records
```

By default, the data directory is organized in the following structure:
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

### Download Pre-trained Weights
This directory also provides scripts to download the pre-trained weights of ResNet-50 backbone.
The script creates a new directory with the name `weights` in the current directory and downloads the pre-trained weights in it.

```bash
./download_and_process_pretrained_weights.sh
```

Ensure that the `weights` folder created has a `resnet` folder in it.
Inside the `resnet` folder there should be three folders for checkpoints and weights: `extracted_from_maskrcnn`, `resnet-nhwc-2018-02-07` and `resnet-nhwc-2018-10-14`.
Before moving to the next step, ensure `resnet-nhwc-2018-02-07` is not empty.

## Training Examples

### Single Card and Multi-Card Training Examples

**Run training on 1 HPU:**

- Using `mask_rcnn_main.py` with all default parameters:
  ```bash
  $PYTHON mask_rcnn_main.py
  ```

- Using `mask_rcnn_main.py` with exemplary parameters:
  ```bash
  TF_BF16_CONVERSION=full $PYTHON mask_rcnn_main.py --mode=train --checkpoint="weights/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603" --init_learning_rate=0.005 --learning_rate_steps=240000,320000 --model_dir="results" --total_steps=360000 --training_file_pattern="/data/tensorflow/coco2017/tf_records/train-*.tfrecord" --validation_file_pattern="/data/tensorflow/coco2017/tf_records/val-*.tfrecord" --val_json_file="/data/tensorflow/coco2017/tf_records/annotations/instances_val2017.json"
  ```

- For Gaudi2, training batch size can be increased for better performance:
  ```bash
  TF_BF16_CONVERSION=full $PYTHON mask_rcnn_main.py --mode=train --checkpoint="weights/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603" --init_learning_rate=0.015 --train_batch_size=12 --learning_rate_steps=80000,106667 --model_dir="results" --total_steps=120000 --training_file_pattern="/data/tensorflow/coco2017/tf_records/train-*.tfrecord" --validation_file_pattern="/data/tensorflow/coco2017/tf_records/val-*.tfrecord" --val_json_file="/data/tensorflow/coco2017/tf_records/annotations/instances_val2017.json"
  ```

You can train the model in mixed precision by setting the `TF_BF16_CONVERSION=full` environment variable. This approach is similar to Automatic Mixed Precision of TensorFlow, which can reduce memory requirements and speed up training. For more details on the mixed precision training JSON recipe files, refer to the [TensorFlow Mixed Precision Training on Gaudi](https://docs.habana.ai/en/latest/TensorFlow/TensorFlow_Mixed_Precision/TensorFlow_Mixed_Precision.html) documentation.

**Run training on 8 HPUs:**

**NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration).

**NOTE:** Due to high memory requirements, training on 8 HPUs may require increasing the number of huge pages in your system. The recommended number of huge pages is 200000.

For multi-card training, remember to adjust hyperparameters (typically by multiplying `init_learning_rate` and dividing `learning_rate_steps`, `num_steps_per_eval` and `total_steps` by a number of workers).

- 8 HPUs `bf16-full` over mpirun using `mask_rcnn_main.py`:
  ```bash
  export TF_BF16_CONVERSION=full
  mpirun --allow-run-as-root --bind-to core --map-by socket:PE=6 --np 8 -x TF_BF16_CONVERSION $PYTHON mask_rcnn_main.py --mode=train --training_file_pattern="/data/tensorflow/coco2017/tf_records/train-*.tfrecord" --validation_file_pattern="/data/tensorflow/coco2017/tf_records/val-*.tfrecord" --val_json_file="/data/tensorflow/coco2017/tf_records/annotations/instances_val2017.json" --init_learning_rate=0.04 --learning_rate_steps=30000,40000 --total_steps=45000 --checkpoint="weights/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603" --model_dir="results"
  ```

- For Gaudi2, training batch size can be increased for better performance:
  ```bash
  export TF_BF16_CONVERSION=full;
  mpirun --allow-run-as-root --bind-to core --map-by socket:PE=6 --np 8 -x TF_BF16_CONVERSION $PYTHON mask_rcnn_main.py --mode=train --training_file_pattern="/data/tensorflow/coco2017/tf_records/train-*.tfrecord" --validation_file_pattern="/data/tensorflow/coco2017/tf_records/val-*.tfrecord" --val_json_file="/data/tensorflow/coco2017/tf_records/annotations/instances_val2017.json" --init_learning_rate=0.12 --train_batch_size=12 --learning_rate_steps=10000,13334 --total_steps=15000 --checkpoint="weights/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603" --model_dir="results"
  ```

- 8 HPUs, `FP32` over mpirun using `mask_rcnn_main.py`:
  ```bash
  export TF_BF16_CONVERSION=0
  mpirun --allow-run-as-root --bind-to core --map-by socket:PE=6 --np 8 -x TF_BF16_CONVERSION $PYTHON mask_rcnn_main.py --mode=train --training_file_pattern="/data/tensorflow/coco2017/tf_records/train-*.tfrecord" --validation_file_pattern="/data/tensorflow/coco2017/tf_records/val-*.tfrecord" --val_json_file="/data/tensorflow/coco2017/tf_records/annotations/instances_val2017.json" --init_learning_rate=0.04 --learning_rate_steps=30000,40000 --num_steps_per_eval=3696 --total_steps=45000 --checkpoint="weights/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603" --model_dir="results"
  ```

### Parameters

You can modify the training behavior through the various flags in the `mask_rcnn_main.py`. Flags in the `mask_rcnn_main.py` script are as follows:

- `mode`: Run `train`, `train_and_eval` or `eval` on MS COCO. `train_and_eval` by default.
- `training_file_pattern`: TFRecords file pattern for the training files.
- `validation_file_pattern`: TFRecords file pattern for the validation files.
- `val_json_file`: Filepath for the validation json file
- `checkpoint`: Path to model checkpoint.
- `model_dir`: Model directory.
*<br> WARNING: Unexpected errors can occur if data from previous runs remains in the model directory. In particular, the model will not continue training if it has previously executed more total_steps than the current run, and saved data to the model directory.*<br>
- `total_steps`: The number of steps to use for training should be adjusted according to the `train_batch_size` flag. Note that for first 100 steps performance is not reported by the script (-1 will be shown).
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


## Supported Configuration

| Validated on | SynapseAI Version | TensorFlow Version(s) | Mode |
|:------:|:-----------------:|:-----:|:----------:|
| Gaudi   | 1.13.0             | 2.13.1         | Training |
| Gaudi2  | 1.13.0             | 2.13.1         | Training |

## Changelog

### 1.7.0

- Fixed printing evaluation time

### 1.5.0

- Added note about Gaudi2.
- `profile` now accept steps.
- Removed demo script.

### 1.4.0

- References to custom demo script were replaced by community entry points in README.
- Default values of some parameters were changed.
- Added flag to specify maximum number of CPUss to be used.
- Disabled cache and prefetch when not enough RAM.

### 1.3.0

- Set up BF16 conversion pass using a config from habana-tensorflow package instead of recipe json file.
- Fixed freeze in multi-card scenario when checkpoint already exist.
- Updated requirements.txt.
- Changed `python` or `python3` to `$PYTHON` to execute correct version based on environment setup.

### 1.2.0

- Removed workaround for Multiclass NMS.

## Known issues

For Gaudi2, default eval_batch_size=4 should be used to ensure proper accuracy.