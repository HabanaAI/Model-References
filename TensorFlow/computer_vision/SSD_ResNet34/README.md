# SSD_ResNet34

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

* [Model-References](../../../README.md)
* [Model overview](#model-overview)
* [Setup](#setup)
* [Training](#training)
* [Advanced](#advanced)

## Model Overview

This repository provides a script to train the Single Shot Detection (SSD) (Liu et al., 2016) with backbone ResNet-34 trained with COCO2017 dataset on Habana Gaudi (HPU). It is based on the MLPerf training 0.6 implementation by Google. The model provides output as bounding boxes. Please visit [this page](https://developer.habana.ai/resources/habana-training-models/#performance) for performance information.

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
Please follow the instructions given in the following link for setting up the
environment including the `$PYTHON` environment variable: [Gaudi Setup and
Installation Guide](https://github.com/HabanaAI/Setup_and_Install). Please
answer the questions in the guide according to your preferences. This guide will
walk you through the process of setting up your system to run the model on
Gaudi.

Set the `MPI_ROOT` environment variable to the directory where OpenMPI is installed.

For example, in Habana containers, use

```bash
export MPI_ROOT=/usr/local/openmpi/
```

### Clone Habana Model-References

In the docker container, clone this repository and switch to the branch that
matches your SynapseAI version. (Run the
[`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.)

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References /root/Model-References
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
They should be mounted to `/data/ssd_r34-mlperf`.

```bash
mkdir /data/ssd_r34-mlperf
cd /data/ssd_r34-mlperf
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
#### Prerequisites

During single card trainings, the model requires over 90GB of host memory for
its dataset due to caching. To handle this on the Gaudi device, it is necessary
to allocate huge pages in the system configuration, otherwise `Error mapping
memory` may occur. This configuration can be done as follows:

1. Back up `/etc/sysctl.conf` as follows:
```bash
cp /etc/sysctl.conf /etc/sysctl.conf.bak
```

2. Ensure the following line is in `/etc/sysctl.conf`, either by adding it or by editing
   an setting:
```bash
vm.nr_hugepages = 153600
```
3. Run `sysctl -p`

If for some reason, the setting cannot be applied, you will have to remove
`cache()` from `SSDInputReader` in `dataloader.py`.

4. After completing training of this model, restore `/etc/sysctl.conf` to its
   original state:
```bash
mv /etc/sysctl.conf.bak /etc/sysctl.conf
```

#### Run training
```bash
$PYTHON demo_ssd.py -e <epoch> -b <batch_size> -d <precision> --model_dir <path/to/model_dir>
```
For example:

- The following command will train the topology on single Gaudi using batch size 128, 50 epochs, precision bf16, and remaining default hyperparameters and save summary data every 10 steps.
    ```python
    $PYTHON demo_ssd.py -e 50 -b 128 -d bf16 --model_dir /tmp/ssd_1_hpu --save_summary_steps 10
    ```
    Each epoch will take ceil(117266 / 128) = 917 steps so the whole training will take 917 * 50 = 45850 steps.
    Checkpoints will be saved to `/tmp/ssd_1_hpu`.

- The following command will train the topology on single Gaudi, batch size 128, 50 epochs, precision fp32 and remaining default hyperparameters.
    ```python
    $PYTHON demo_ssd.py -e 50 -b 128 -d fp32 --model_dir /tmp/ssd_1_hpu
    ```
    Each epoch will take ceil(117266 / 128) = 917 steps so the whole training will take 917 * 50 = 45850 steps.
    Checkpoints will be saved to `/tmp/ssd_1_hpu`.

### Run training on 8 Gaudi cards

```bash
$PYTHON demo_ssd.py -e <epoch> -b <batch_size> -d <precision> --model_dir <path/to/model_dir> --hvd_workers 8
```
For example:

- In order to train the topology on 8 Gaudi cards, batch size 128, 50 epochs, precision bf16 and other default hyperparameters.
    ```bash
    $PYTHON demo_ssd.py -e 50 -b 128 -d bf16 --model_dir /tmp/ssd_8_hpus --hvd_workers 8
    ```
    Each epoch will take ceil(117266 / (128 * 8)) = 115 steps so the whole training will take 115 * 50 = 5750 steps.
    Checkpoints will be saved in `/tmp/ssd_8_hpus`.

### Run training on 8 Gaudi cards via mpirun

For example, the mpirun command to run 8 Gaudi cards training with batch size 128, 50 epochs, precision bf16, and other default hyperparameters, saving summary data every 10 steps and checkpoints every 50 epochs is as follows.
*<br>mpirun map-by PE attribute value may vary on your setup and should be calculated as:<br>
socket:PE = floor((number of physical cores) / (number of gaudi devices per each node))*
```bash
mpirun --allow-run-as-root --bind-to core --map-by socket:PE=7 --np 8 \
  $PYTHON /root/Model-References/TensorFlow/computer_vision/SSD_ResNet34/ssd.py --use_horovod --epochs 50 --batch_size 128 --dtype bf16 --model_dir /tmp/ssd_8_hpus --save_summary_steps 10 --save_checkpoints_epochs 50
```

Note that it is required to provide `--use_horovod` argument.

### Run evaluation

In order to calculate mAP for the saved checkpoints in `/tmp/ssd_1_hpu`:
```bash
$PYTHON ssd.py --mode eval --model_dir /tmp/ssd_1_hpu
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

-  `-d bf16/fp32` or `--dtype bf16/fp32`             Data type: fp32 or bf16 (default: bf16)
-  `-b N` or `--batch_size N`                        Batch size (default: 128)
-  `-e N` or `--epochs N`                            Number of epochs for training (default: 50)
-  `--mode train/eval`                               Mode: `train` or `eval` (default: train)
-  `--use_horovod`                                   Use Horovod for distributed training (default: False)
-  `--inference IMAGE`                               Path to image for inference (if set then mode is ignored) (default: None)
-  `--no_hpu`                                        Do not load Habana modules = train on CPU/GPU (default: False)
-  `--distributed_bn`                                Use distributed batch norm (default: False)
-  `--vis_dataloader`                                Visualize dataloader (default: False)
-  `--profiling`                                     Turn on profiling (generate json every 20 steps) (default: False)
-  `--use_cocoeval_cc`                               Use cocoeval cc (default: False)
-  `-f` or `--use_fake_data`                         Use fake data to reduce the input preprocessing overhead (for unit tests) (default: False)
-  `--lr_warmup_epoch N`                             Number of epochs for learning rate warmup (default: 5.0)
-  `--base_lr BASE_LR`                               base learning rate (default: 0.003)
-  `--weight_decay WD`                               L2 wight decay (default: 0.0005)
-  `--k K`                                           k is an integer defining at which epochs the learning rate decays: [40, 50] * (1 + k/10) (default: 0)
-  `--model_dir <dir>`                               Location of model_dir (default: /tmp/ssd)
-  `--resnet_checkpoint <PATH>`                      Location of the ResNet ckpt to use for model init. (default: /data/ssd_r34-mlperf)
-  `--data_dir <PATH>`                               Path to dataset (default: /data/coco2017/ssd_tf_records)
-  `--eval_samples N`                                Number of samples for evaluation. (default: 5000)
-  `--num_examples_per_epoch N`                      Number of examples in one epoch (default: 117266)
-  `-s STEPS` or `--steps STEPS`                     Number of training steps (epochs and num_examples_per_epoch are ignored when set) (default: 0)
-  `-v STEPS` or ` --log_step_count_steps STEPS`     How often print global_step/sec and loss (default: 1)
-  `-c EPOCHS` or `--save_checkpoints_epochs EPOCHS` How often save checkpoints (default: 5.0)
-  `--keep_ckpt_max N`                               Maximum number of checkpoints to keep (default: 20)
-  `--save_summary_steps SAVE_SUMMARY_STEPS`         How often save summary (default: 1)
-  `--static`                                        Enables use of static dataloader (default: False)
-  `--recipe_cache RECIPE_CACHE`                     Path to recipe cache directory. Set to empty to disable recipe cache.
                                                     Externally set "TF_RECIPE_CACHE_PATH" will override this settings. (default: /tmp/ssd_recipe_cache/)
