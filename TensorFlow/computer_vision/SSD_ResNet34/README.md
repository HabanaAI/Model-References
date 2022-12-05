# SSD ResNet-34 for TensorFlow

This directory provides a script to train a SSD ResNet-34 model to achieve state-of-the-art accuracy, and is tested and maintained by Habana. To obtain model performance data, refer to the [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

For more information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

* [Model-References](../../../README.md)
* [Model Overview](#model-overview)
* [Setup](#setup)
* [Training and Examples](#training-and-examples)
* [Advanced](#advanced)
* [Profile](#profile)
* [Supported Configuration](#supported-configuration)
* [Changelog](#changelog)

## Model Overview

This directory provides a script to train the Single Shot Detection (SSD) (Liu et al., 2016) with backbone ResNet-34 trained with COCO2017 dataset on Habana Gaudi (HPU). It is based on the MLPerf training 0.6 implementation by Google. The model provides output as bounding boxes.

To see the changes implemented for this model, refer to [Training Script Modifications](#training-script-modifications).

## Setup
Please follow the instructions provided in the [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the
environment including the `$PYTHON` environment variable.
The guide will walk you through the process of setting up your system to run the model on Gaudi.
### Clone Habana Model-References

In the docker container, clone this repository and switch to the branch that
matches your SynapseAI version. You can run the
[`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References /root/Model-References
```
**Note:** If the repository is not in the PYTHONPATH, make sure to update by running the below:

```bash
export PYTHONPATH=/root/Model-References:$PYTHONPATH
```

### Install Model Requirements

1. In the docker container, go to the SSD_ResNet34 directory:
```bash
cd /root/Model-References/TensorFlow/computer_vision/SSD_ResNet34
```
2. Install the required packages using pip:
```bash
$PYTHON -m pip install -r requirements.txt
```

### Download and Pre-process COCO2017 Dataset

The topology script is already configured for COCO2017 (117266 training images, 5000 validation images).

Only images with any bounding box annotations are used for training.
The dataset directory should be mounted to `/data/tensorflow/coco2017/ssd_tf_records`.

The topology uses tf-records which can be prepared as follows:
```bash
cd /root/Model-References/TensorFlow/computer_vision/SSD_ResNet34
export TMP_DIR=$(mktemp -d)
export SSD_PATH=$(pwd)
pushd $TMP_DIR
git clone https://github.com/tensorflow/tpu
cd tpu
git checkout 0ffe1274745806c411ed3dda7e84f692e00df8af
git apply ${SSD_PATH}/coco-tf-records.patch
cd tools/datasets
bash download_and_preprocess_coco.sh /data/tensorflow/coco2017/ssd_tf_records
popd
rm -rf $TMP_DIR
```
### Download Pre-trained ResNet-34 Weights

The topology uses pre-trained ResNet-34 weights.
They should be mounted to `/data/tensorflow/ssd-r34/tensorflow_datasets/ssd/ssd_r34-mlperf/mlperf_artifact/`.

```bash
mkdir /data/tensorflow/ssd-r34/tensorflow_datasets/ssd/ssd_r34-mlperf/mlperf_artifact
cd /data/tensorflow/ssd-r34/tensorflow_datasets/ssd/ssd_r34-mlperf/mlperf_artifact
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/ssd-backbone/checkpoint
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/ssd-backbone/model.ckpt-28152.data-00000-of-00001
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/ssd-backbone/model.ckpt-28152.index
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/ssd-backbone/model.ckpt-28152.meta
```

## Training and Examples

Go to SSD ResNet-34 directory:
```bash
cd /root/Model-References/TensorFlow/computer_vision/SSD_ResNet34
```
### Prerequisites

During single card training, the model requires over 90GB of host memory for
its dataset due to caching. To handle this on the Gaudi device, it is necessary
to allocate huge pages in the system configuration, otherwise `Error mapping
memory` may occur. This configuration can be done as follows:

1. Back up `/etc/sysctl.conf`:
```bash
cp /etc/sysctl.conf /etc/sysctl.conf.bak
```

2. Ensure the following line is in `/etc/sysctl.conf`, either by adding it or by editing
   the setting:
```bash
vm.nr_hugepages = 153600
```
3. Run `sysctl -p`. If for some reason, the setting cannot be applied, you will have to remove
`cache()` from `SSDInputReader` in `dataloader.py`.

4. After completing training of this model, restore `/etc/sysctl.conf` to its
   original state:
```bash
mv /etc/sysctl.conf.bak /etc/sysctl.conf
```

### Single Card and Multi-Card Training Examples

```bash
$PYTHON ssd.py -e <epoch> -b <batch_size> -d <precision> --model_dir <path/to/model_dir>
```
**Run training on 1 HPU:**

- 1 HPU (Gaudi), batch 128, 64 epochs, precision BF16, and remaining default hyperparameters and save summary data every 10 steps:
  ```bash
  $PYTHON ssd.py -e 64 -b 128 -d bf16 --model_dir /tmp/ssd_1_hpu --save_summary_steps 10 --data_dir /data/tensorflow/coco2017/ssd_tf_records --resnet_checkpoint /data/tensorflow/ssd-r34/tensorflow_datasets/ssd/ssd_r34-mlperf/mlperf_artifact
  ```
Each epoch will take ceil(117266 / 128) = 917 steps so the whole training will take 917 * 64 = 58688 steps.
Checkpoints will be saved to `/tmp/ssd_1_hpu`.

- 1 HPU (Gaudi), batch size 128, 64 epochs, precision fp32 and remaining default hyperparameters:
  ```bash
  $PYTHON ssd.py -e 64 -b 128 -d fp32 --model_dir /tmp/ssd_1_hpu --data_dir /data/tensorflow/coco2017/ssd_tf_records --resnet_checkpoint /data/tensorflow/ssd-r34/tensorflow_datasets/ssd/ssd_r34-mlperf/mlperf_artifact
  ```
Each epoch will take ceil(117266 / 128) = 917 steps so the whole training will take 917 * 64 = 58688 steps.
Checkpoints will be saved to `/tmp/ssd_1_hpu`.

- 1 HPU (Gaudi2), batch 240, 64 epochs, precision BF16 and remaining default hyperparameters. Note that on Gaudi2 you can get better performance when choosing a batch size divisible by 24 (amount of TPC engines):
  ```bash
  $PYTHON ssd.py -e 64 -b 240 -d bf16 --model_dir /tmp/ssd_1_hpu --data_dir /data/tensorflow/coco2017/ssd_tf_records --resnet_checkpoint /data/tensorflow/ssd-r34/tensorflow_datasets/ssd/ssd_r34-mlperf/mlperf_artifact
  ```
Each epoch will take ceil(117266 / 240) = 489 steps so the whole training will take 489 * 64 = 31296 steps.
Checkpoints will be saved to `/tmp/ssd_1_hpu`.

**Run training on 8 HPUs:**

**NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration).

- 8 HPUs, batch 128, 64 epochs, precision BF16, and other default hyperparameters, saving summary data every 10 steps and checkpoints after 64 epochs as follows. Note that it is required to provide `--use_horovod` argument.
 ```bash
 mpirun --allow-run-as-root --bind-to core --map-by socket:PE=6 --np 8 \
  $PYTHON /root/Model-References/TensorFlow/computer_vision/SSD_ResNet34/ssd.py --use_horovod --epochs 64 --batch_size 128 --dtype bf16 --model_dir /tmp/ssd_8_hpus --save_summary_steps 10 --save_checkpoints_epochs 64 \
   --data_dir /data/tensorflow/coco2017/ssd_tf_records --resnet_checkpoint /data/tensorflow/ssd-r34/tensorflow_datasets/ssd/ssd_r34-mlperf/mlperf_artifact
 ```
Each epoch will take ceil(117266 / (128 * 8)) = 115 steps so the whole training will take 115 * 64 = 7360 steps.
Checkpoints will be saved in `/tmp/ssd_8_hpus`.


### Run Evaluation

To calculate mAP for the saved checkpoints in `/tmp/ssd_1_hpu`, run:
```bash
$PYTHON ssd.py --mode eval --model_dir /tmp/ssd_1_hpu --data_dir /data/tensorflow/coco2017/ssd_tf_records --resnet_checkpoint /data/tensorflow/ssd-r34/tensorflow_datasets/ssd/ssd_r34-mlperf/mlperf_artifact
```

## Advanced

To see all possible arguments, run:
```bash
$PYTHON ssd.py --help
```

## Profile

Run training on 1 HPU with profiler:

```bash
$PYTHON ssd.py --save_summary_steps 10 --data_dir /data/tensorflow/coco2017/ssd_tf_records --resnet_checkpoint /data/tensorflow/ssd-r34/tensorflow_datasets/ssd/ssd_r34-mlperf/mlperf_artifact --profile 50,53 --steps 100
```
The above example will produce profile trace for 4 steps (50,51,52,53).

## Supported Configuration

| Device | SynapseAI Version | TensorFlow Version(s)  |
|:------:|:-----------------:|:-----:|
| Gaudi  | 1.7.1             | 2.8.4 |
| Gaudi  | 1.7.1             | 2.10.1 |
| Gaudi2 | 1.7.1             | 2.10.1 |
| Gaudi2 | 1.7.1             | 2.8.4 |

## Changelog
### 1.7.0
* Improved learning rate schedule.
* Froze first two layers.
* Fixed weight decay (do not decay batch norm).
### 1.4.0
* remove tf.Estimator
* improve profiling experience
* optimize `_localization_loss`
* import horovod-fork package directly instead of using Model-References' TensorFlow.common.horovod_helpers; wrapped horovod import with a try-catch block so that the user is not required to install this library when the model is being run on a single card
* references to custom demo script were replaced by community entry points in README.
### 1.3.0
* add profiling support
* add prefetching to device (improves performance)

### Training Script Modifications
The SSD ResNet-34 model is based on https://github.com/mlperf/training_results_v0.6/ (Google/benchmarks/ssd/implementations/tpu-v3-32-ssd).

The functional changes are:
*  Migrated from TF 1.15 to TF 2.2.
*  Enabled resource variables
*  TPU-specific topk_mask implementation is replaced with implementation from the reference.
*  Removed TPU and GCP related code.
*  Used argparse instead of tf.flags.
*  Removed mlperf 0.6 logger.
*  Added flags: `num_steps`, `log_step_count_steps`, and `save_checkpoints_steps`.
*  Used `dataset.interleave` instead of `tf.data.experimental.parallel_interleave` in `dataloader.py`.
*  Name scopes `concat_cls_outputs` and `concat_box_outputs` added to `concat_outputs`.
*  Fixed issues with COCO2017 dataset - provided script for generating correct dataset, 117266 training examples.
*  Removed weight_decay loss from total loss calculation (fwd only).
*  Updated input normalization formula to use multiplication by reciprocal instead of division.
*  Added logging hook that calculates IPS and total training time.
*  Disabled Eager mode.
*  Enabled resource variables.
*  Added Horovod for distributed training.
*  Added support for HPU (load_habana_modules and Habana Horovod).
*  Turned off dataset caching when RAM size is not sufficient.
*  Added support for TF profiling.
*  Added inference mode.
*  Added support for distributed batch normalization.
*  Improved learning rate schedule.
*  Froze first two layers of ResNet34 backbone.
*  Fixed weight decay.

The performance changes are:
*  Boxes and classes are transposed in dataloder instead of the in model to improve performance.
*  Introduced custom `softmax_cross_entropy_mme` loss function that better utilizes HPU hardware (by implementing reduce_sum through conv2d which is computed on MME in parallel with other TPC operations and transposing tensors for reduce_max).


