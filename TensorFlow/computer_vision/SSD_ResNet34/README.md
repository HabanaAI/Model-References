# SSD_ResNet34

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

* [Model overview](#model-overview)
* [Setup](#setup)
* [Advanced](#advanced)
* [Known issues](#known-issues)

## Model Overview

This repository provides a script to train the Single Shot Detection (SSD) (Liu et al., 2016) with backbone ResNet-34 trained with COCO2017 dataset on Habana Gaudi (HPU). It is based on the MLPerf training 0.6 implementation by Google. The model provides output as bounding boxes.

### SSD-ResNet34 model changes
The SSD ResNet34 model is based on https://github.com/mlperf/training_results_v0.6/ (Google/benchmarks/ssd/implementations/tpu-v3-32-ssd)

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

### Requirements

Habana drivers installed along with its dependencies which can be found [here](https://docs.habana.ai/Installation_Guide/GAUDI_Installation_Guide.html).

#### Get the Habana Tensorflow docker image
```bash
docker pull vault.habana.ai/gaudi-docker/0.13.0/ubuntu18.04/habanalabs/tensorflow-installer:0.13.0-380
```

### Launch the Docker container
```bash
docker run -it -v /data:/data -v /tmp:/tmp --device=/dev:/dev -v /dev:/dev -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
   --cap-add=sys_nice -e DISPLAY=$DISPLAY -e PYTHONPATH=/usr/lib/habanalabs/:/root:/root/Model-References \
   -v /sys/kernel/debug:/sys/kernel/debug --user=root --workdir=/root --net=host \
   vault.habana.ai/gaudi-docker/0.13.0/ubuntu18.04/habanalabs/tensorflow-installer:0.13.0-380
```

### Clone Habana Model Garden
```bash
git clone https://github.com/HabanaAI/Model-References /root/Model-References
cd /root/Model-References/TensorFlow/computer_vision/SSD_ResNet34
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

Weights used in MLPerf benchmark can be downloaded from [here](https://console.cloud.google.com/storage/browser/mlperf_artifcats/v0.6_training).

Habana will be providing distinct pretrained ResNet34 weights soon.
```bash
$ md5sum /data/ssd_r34-mlperf/mlperf_artifact/*
7e9f4ba94c150b5c0c073cb738e7ad39  /data/ssd_r34-mlperf/mlperf_artifact/checkpoint
1b7fabd0872b765da2fc60af46fe6390  /data/ssd_r34-mlperf/mlperf_artifact/model.ckpt-28152.data-00000-of-00001
96080eb2d106f7bd4a682bde003b1436  /data/ssd_r34-mlperf/mlperf_artifact/model.ckpt-28152.index
6b31dd1a042cdbbb2fd7fcd6f2c5e843  /data/ssd_r34-mlperf/mlperf_artifact/model.ckpt-28152.meta
```

### Run training on single node

The following command will train the topology on single node, batch size 128, 50 epochs, default hyperparameters.

```bash
./demo_ssd -e 50 -b 128 --model_dir /tmp/ssd_single_node
```
Each epoch will take ceil(117266 / 128) = 917 steps so the whole training will take 917 * 50 = 45850 steps.
Checkpoints will be saved to `/tmp/ssd_single_node`.

### Run training on multinode (8 nodes)

In order to train the topology on multinode (8 nodes), batch size 128, 50 epochs, default hyperparameters.

```bash
./demo_ssd -e 50 -b 128 --model_dir /tmp/ssd_8_nodes --multinode 8
```
Each epoch will take ceil(117266 / (128 * 8)) = 115 steps so the whole training will take 115 * 50 = 5750 steps.
Checkpoints will be saved in `/tmp/ssd_8_nodes`.

### Training Examples

|  Gaudi nodes, Batch Size, epochs, Precision       |                              Command Line                                           |
| ------------------------------------------------- | ----------------------------------------------------------------------------------- |
| 1-node, BS=128, e=50, Precision= bf16             | `./demo_ssd -e 50 -b 128 -d bf16 --model_dir /tmp/ssd_single_node`                  |
| 1-node, BS=256, e=50, Precision= bf16             | `./demo_ssd -e 50 -b 256 -d bf16 --model_dir /tmp/ssd_single_node`                  |
| 1-node, BS=128, e=50, Precision= fp32             | `./demo_ssd -e 50 -b 128 -d fp32 --model_dir /tmp/ssd_single_node`                  |
| 8-nodes, BS=128, e=50, Precision= bf16            | `./demo_ssd -e 50 -b 128 -d bf16 --model_dir /tmp/ssd_8_nodes --multinode 8`        |
| 8-nodes, BS=256, e=50, Precision= bf16            | `./demo_ssd -e 50 -b 256 -d bf16 --model_dir /tmp/ssd_8_nodes --multinode 8`        |

### Run evaluation

In order to calculate mAP for the saved checkpoints in `/tmp/ssd_single_node`:
```bash
./demo_ssd --mode eval --model_dir /tmp/ssd_single_node
```

## Advanced

The following section provides details of running training.

### Scripts and sample code

In the `root/Model-References/TensorFlow/computer_vision/SSD-ResNet34` directory, the most important files are:

* `demo_ssd`: Serves as a wrapper script for the training file `ssd.py`. Also preloads libjemalloc and allows to run distributed training as it contains the --multinode argument.
* `ssd.py`: The training script of the SSD model. Contains all the other arguments.

#### Preview of TensorFlow Resnet Python scripts with yaml configuration of parameters

During single card (in the future multi-node workloads will be supported as well) training you can use model runners that are written in Python as opposed to bash.

You can run the following script: `/root/Model-References/TensorFlow/habana_model_runner.py` which accepts two arguments:
```
- --model *model_name*
- --hb_config *path_to_yaml_config_file*
```
Example of config files can be found in the `/root/Model-References/TensorFlow/computer_vision/SSD_ResNet34/ssd_resnet34_default.yaml`.

You can use these scripts as such:
```bash
cd Model-References/TensorFlow/computer_vision/SSD_ResNet34
python3 ../../habana_model_runner.py --model ssd_resnet34 --hb_config ssd_resnet34_default.yaml
```
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
-  `--calculate_loss`           : By default loss is not calculated in order to improve performance (loss=99999.9)
-  `--val_json_file`            : COCO validation JSON containing golden bounding boxes (default: `DEFAULT_VAL_JSON_PATH`)
-  `--resnet_checkpoint`        : Location of the ResNet ckpt to use for model (default: `DEFAULT_RN34_CKPT_PATH`)
-  `--training_file_pattern`    : Prefix for training data files  (default: `DEFAULT_TRAINING_FILE_PATTERN`)
-  `--val_file_pattern`         : Prefix for evaluation tfrecords (default: `DEFAULT_VAL_FILE_PATTERN`)
-  `-v`                         : How often print `global_step/sec` and loss
-  `-c`                         : How often save checkpoints
-  `-p`                         : Run TF ProfilerHook

Note: Please find the exhaustive list of all modifiable parameters in `ssd.py` file or use `./demo_ssd -h`

## Known issues
* dataloader does not use [`tf.data.Dataset.cache()`]("https://www.tensorflow.org/versions/r2.2/api_docs/python/tf/data/Dataset#cache") during single node training
* sporadic NaNs and low accuracy in multinode trainings
