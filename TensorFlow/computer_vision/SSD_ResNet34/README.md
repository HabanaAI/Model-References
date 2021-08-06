# SSD_ResNet34

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

* [Model-References](../../../README.md)
* [Model overview](#model-overview)
* [Setup](#setup)
* [Training](#training)
* [Advanced](#advanced)
* [Known issues](#known-issues)

## Model Overview

This repository provides a script to train the Single Shot Detection (SSD) (Liu et al., 2016) with backbone ResNet-34 trained with COCO2017 dataset on Habana Gaudi (HPU). It is based on the MLPerf training 0.6 implementation by Google. The model provides output as bounding boxes. Please visit [this page](../../../README.md#tensorflow-model-performance) for performance information.

### SSD-ResNet34 model changes
The SSD ResNet34 model is based on https://github.com/mlperf/training_results_v0.6/tree/master/Google/benchmarks/ssd/implementations/tpu-v3-32-ssd.

The functional changes are:
*  Migrated from TF 1.15 to TF 2.2
*  Enabled resource variables
*  TPU-specific topk_mask implementation is replaced with implementation from the reference
*  Removed TPU and GCP related code
*  Used argparse instead of tf.flags
*  Removed mlperf 0.6 logger
*  Added flags: num_steps, log_step_count_steps, save_checkpoints_steps
*  Used dataset.interleave instead of tf.data.experimental.parallel_interleave in dataloader.py
*  Name scopes 'concat_cls_outputs' and 'concat_box_outputs' added to concat_outputs
*  Fixed issues with COCO2017 dataset - provided script for generating correct dataset, 117266 training examples
*  Removed weight_decay loss from total loss calculation (fwd only)
*  Updated input normalization formula to use multiplication by reciprocal instead of division
*  Added logging hook that calculates IPS and total training time
*  Disabled Eager mode
*  Enabled resource variables
*  Added Horovod for distibuted training
*  Added support for HPU (load_habana_modules and Habana Horovod)
*  Added demo_ssd allowing to run multinode training with OpenMPI
*  Turned off dataset caching when RAM size is not sufficient
*  Added support for TF profiling
*  Added inference mode
*  Added support for distributed batch normalization

The performance changes are:
*  Boxes and classes are transposed in dataloder not in model to improve performance
*  Introduced custom softmax_cross_entropy_mme loss function that better utilizes HPU hardware (by implementing reduce_sum through conv2d which is computed on MME in parallel with other TPC operations and transposing tensors for reduce_max)

### Default configuration

- Learning rate base = 3e-3
- Weight decay = 5e-4
- Epochs for learning rate Warm-up  = 5
- Batch size = 128
- Epochs for training = 50
- Data type = bf16
- Loss calculation = False
- Mode = train
- Epochs at which learning rate decays = 0
- Number of samples for evaluation = 5000
- Number of example in one epoch = 117266
- Number of training steps = 0
- Frequency of printing loss = 1
- Frequency of saving checkpoints = 5
- Maximum number of checkpoints stored = 20

## Setup

Please follow the instructions given in the following link for setting up the environment: [Gaudi Setup and Installation Guide](https://github.com/HabanaAI/Setup_and_Install). Please answer the questions in the guide according to your preferences. This guide will walk you through the process of setting up your system to run the model on Gaudi.

### Clone Habana Model-References
```bash
git clone https://github.com/HabanaAI/Model-References /root/Model-References
```

Add Model-References to PYTHONPATH
```bash
export PYTHONPATH=/root/Model-References:$PYTHONPATH
```

### Download and preprocess COCO2017 dataset

The topology script is already configured for COCO2017 (117266 training images, 5000 validation images).

Only images with any bounding box annotations are used for training.
The dataset directory should be mounted to `/data/coco2017/ssd_tf_records`.

The topology uses tf-records which can be prepared in the following way:
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
bash download_and_preprocess_coco.sh /data/coco2017/ssd_tf_records
popd
rm -rf $TMP_DIR
```
### Download pretrained ResNet34 weights

The topology uses pretrained ResNet34 weights.
They should be mounted to `/data/ssd_r34-mlperf/mlperf_artifact`.

```bash
mkdir /data/ssd_r34-mlperf/mlperf_artifact
cd /data/ssd_r34-mlperf/mlperf_artifact
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/ssd-backbone/checkpoint
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/ssd-backbone/model.ckpt-28152.data-00000-of-00001
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/ssd-backbone/model.ckpt-28152.index
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/ssd-backbone/model.ckpt-28152.meta
```

## Training

Go to SSD ResNet34 directory:
```bash
cd /root/Model-References/TensorFlow/computer_vision/SSD_ResNet34
```

### Run training on single Gaudi
```bash
python3 demo_ssd.py -e <epoch> -b <batch_size> -d <precison> --model_dir <path/to/model_dir>
```
For example:

- The following command will train the topology on single Gaudi, batch size 128, 50 epochs, precision bf16 and remaining default hyperparameters.
    ```python
    python3 demo_ssd.py -e 50 -b 128 -d bf16 --model_dir /tmp/ssd_1_hpu
    ```
    Each epoch will take ceil(117266 / 128) = 917 steps so the whole training will take 917 * 50 = 45850 steps.
    Checkpoints will be saved to `/tmp/ssd_1_hpu`.

- The following command will train the topology on single Gaudi, batch size 128, 50 epochs, precision fp32 and remaining default hyperparameters.
    ```python
    python3 demo_ssd.py -e 50 -b 128 -d fp32 --model_dir /tmp/ssd_1_hpu
    ```
    Each epoch will take ceil(117266 / 128) = 917 steps so the whole training will take 917 * 50 = 45850 steps.
    Checkpoints will be saved to `/tmp/ssd_1_hpu`.

### Run training on 8 Gaudi cards

```bash
python3 demo_ssd.py -e <epoch> -b <batch_size> -d <precison> --model_dir <path/to/model_dir> --hvd_workers 8
```
For example:

- In order to train the topology on 8 Gaudi cards, batch size 128, 50 epochs, precison bf16 and other default hyperparameters.
    ```bash
    python3 demo_ssd.py -e 50 -b 128 -d bf16 --model_dir /tmp/ssd_8_hpus --hvd_workers 8
    ```
    Each epoch will take ceil(117266 / (128 * 8)) = 115 steps so the whole training will take 115 * 50 = 5750 steps.
    Checkpoints will be saved in `/tmp/ssd_8_hpus`.

- In order to train the topology on 8 Gaudi cards, batch size 256, 50 epochs, precison bf16 and other default hyperparameters.
    ```bash
    python3 demo_ssd.py -e 50 -b 256 -d bf16 --model_dir /tmp/ssd_8_hpus --hvd_workers 8
    ```
    Each epoch will take ceil(117266 / (256 * 8)) = 58 steps so the whole training will take 58 * 50 = 2900 steps.
    Checkpoints will be saved in `/tmp/ssd_8_hpus`.

### Run training on 8 Gaudi cards via mpirun

For example to run 8 Gaudi cards training via mpirun with batch size 128, 50 epochs, precision bf16 and other default hyperparameters.
- Command using `demo_ssd.py`:
   ```bash
   mpirun --allow-run-as-root --bind-to core --map-by socket:PE=7 --np 8 \
     python3 /root/Model-References/TensorFlow/computer_vision/SSD_ResNet34/demo_ssd.py --use_horovod --hvd_workers=1 --epochs=50 --batch_size=128 --dtype=bf16 --model_dir=/tmp/ssd_8_hpus
   ```

- Equivalent command using `ssd.py`:
   ```bash
   mpirun --allow-run-as-root --bind-to core --map-by socket:PE=7 --np 8 \
     python3 /root/Model-References/TensorFlow/computer_vision/SSD_ResNet34/ssd.py --use_horovod --epochs 50 --batch_size 128 --dtype bf16 --model_dir /tmp/ssd_8_hpus
   ```

Note that it is required to provide `--use_horovod` argument.

### Run evaluation

In order to calculate mAP for the saved checkpoints in `/tmp/ssd_1_hpu`:
```bash
python3 ssd.py --mode eval --model_dir /tmp/ssd_1_hpu
```

## Advanced

The following section provides details of running training.

### Scripts and sample code

In the `root/Model-References/TensorFlow/computer_vision/SSD_ResNet34` directory, the most important files are:

* `demo_ssd.py`: Serves as a wrapper script for the training file `ssd.py`. Also preloads libjemalloc and allows to run distributed training as it contains the `--hvd_workers` argument.
* `ssd.py`: The training script of the SSD model. Contains all the other arguments.

### Parameters

Modify the training behavior through the various flags present in the `ssd.py` file. Some of the important parameters in the
`ssd.py` script are as follows:

-  `-d` or `--dtype`            : Data type, `fp32` or `bf16`
-  `-b` or `--batch_size`       : Batch size
-  `-e`, `--epochs`             : Epochs
-  `--mode`                     : Different types: train,eval
-  `--no_hpu`                   : Do not load habana module. Hence train of CPU/GPU
-  `--use_horovod`              : Use Horovod for distributed training
-  `--inference`                : Path to image for inference (if set then mode is ignored)
-  `-f`, `--use_fake_data`      : Use fake data to reduce the input preprocessing overhead
-  `--val_json_file`            : COCO validation JSON containing golden bounding boxes (default: `DEFAULT_VAL_JSON_PATH`)
-  `--resnet_checkpoint`        : Location of the ResNet ckpt to use for model (default: `DEFAULT_RN34_CKPT_PATH`)
-  `--training_file_pattern`    : Prefix for training data files  (default: `DEFAULT_TRAINING_FILE_PATTERN`)
-  `--val_file_pattern`         : Prefix for evaluation tfrecords (default: `DEFAULT_VAL_FILE_PATTERN`)
-  `--num_examples_per_epoch`   : Number of examples in one epoch (default: 117266)
-  `-s`                         : Number of training steps (`epochs` and `num_examples_per_epoch` are ignored when set)
-  `-v`                         : How often print `global_step/sec` and loss
-  `-c`                         : How often save checkpoints (default: 5)
-  `--keep_ckpt_max`            : Maximum number of checkpoints to keep (default: 20)

## Known issues
* sporadic NaNs and low accuracy in multinode trainings

* `Error mapping memory` is known to occur during single-card training. The solution is as follows:

1. Edit `/etc/sysctl.conf` and add following line:
```
vm.nr_hugepages = 153600
```
2. Run `sysctl -p`