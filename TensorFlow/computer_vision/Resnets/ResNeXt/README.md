# ResNeXt for TensorFlow

This directory provides a script and recipe to train the ResNet v1.5 models to achieve state-of-the-art accuracy and is tested and maintained by Habana. To obtain model performance data, refer to the [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

For more information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents

* [Model-References](../../../../README.md)
* [Model Overview](#model-overview)
* [Setup](#setup)
* [Media Loading Acceleration](#media-loading-acceleration)
* [Training and Examples](#training-and-examples)
* [Advanced](#advanced)
* [Pre-trained Model](#pre-trained-model)
* [Profile](#profile)
* [Supported Configuration](#supported-configuration)
* [Changelog](#changelog)
* [Known Issues](#known-issues)


## Model Overview
ResNeXt is a modified version of the original ResNet v1 model. This implementation defines ResNeXt101 which features 101 layers.

To see the changes implemented for this model, refer to [Training Script Modifications](#training-script-modifications).

## Setup
Please follow the instructions provided in the [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the
environment including the `$PYTHON` environment variable.
The guide will walk you through the process of setting up your system to run the model on Gaudi.

### Training Data

The script operates on ImageNet 1k, a widely popular image classification dataset from the ILSVRC challenge.
In order to obtain the dataset, follow these steps:
1. Sign up with http://image-net.org/download-images and acquire the rights to download the original images.
2. Follow the link to the 2012 ILSVRC and download `ILSVRC2012_img_val.tar` and `ILSVRC2012_img_train.tar`.
3. Use the below commands to prepare the dataset under `/data/tensorflow/imagenet/tf_records`.
 This is the default data_dir for the training script.
 In `/data/tensorflow/imagenet/train` and `/data/tensorflow/imagenet/val` directories original JPEG files will stay
 and can be used for Media Loading Acceleration on Gaudi2.
 See examples with `--data_dir` and `--jpeg_data_dir` parameters for details.

```
export IMAGENET_HOME=/data/tensorflow/imagenet
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
mv $IMAGENET_HOME/validation $IMAGENET_HOME/val
cd $IMAGENET_HOME/val
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```

### Clone Habana Model-References

In the docker container, clone this repository and switch to the branch that
matches your SynapseAI version. You can run the
[`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.
```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```
**Note:** If the repository is not in the PYTHONPATH,  make sure to update by running the below:
```bash
export PYTHONPATH=/path/to/Model-References:$PYTHONPATH
```

### Install Model Requirements

1. In the docker container, go to the ResNeXt directory:
```bash
cd /root/Model-References/TensorFlow/computer_vision/Resnets/ResNeXt
```
2. Install the required packages using pip:
```bash
$PYTHON -m pip install -r requirements.txt
```

### Set up jemalloc

There are instances where the performance improvement is noticed if jemalloc is set up before running the below examples. Generally, it is recommended to set up jemalloc before training ResNeXt model.
To set up jemalloc, export the `LD_PRELOAD` with the path to `libjemalloc.so`
Only one of the `libjemalloc.so.1` or `libjemalloc.so.2` will be present.
To locate the files, search inside the following directories:
* /usr/lib/x86_64-linux-gnu/
* /usr/lib64/

Once `libjemalloc.so.1` or `libjemalloc.so.2` is detected, export it using command `export LD_PRELOAD=/path/to/libjemalloc.so`. For example:
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
```

## Media Loading Acceleration
Gaudi2 offers a dedicated hardware engine for Media Loading operations, such as JPEG decoding and data augmentation. This can be leveraged in ResNet models to decrease CPU usage and exceed performance limitation when data processing on CPU is a bottleneck.
Currently, the supported file format is JPEG only.

**Note:** TFRecord will be supported in a future release.

ResNeXt automatically uses hardware Media Loading Acceleration unless:
1. Training is done on first-gen Gaudi processors. First-gen Gaudi does not have a dedicated hardware for Media Loading Acceleration.
2. `hpu_media_loader` Python package is not installed.
3. `FORCE_HABANA_IMAGENET_LOADER_FALLBACK` environment variable is set.
4. A location of the ImageNet dataset containing JPEGs (--jpeg_data_dir parameter) is not provided.

In the above cases, media processing will be done on CPU.


## Training

**Note**: Due to performance and accuracy issues, it is recommended to always use `data_loader_image_type fp32` (`dlit fp32`) when training from scratch.
However, `data_loader_image_type bf16` (`dlit bf16`) needs to be passed when saving and using pre-trained models (see the [Pre-trained Model section](#pre-trained-model)).

### Single Card and Multi-Card Training Examples
**Run training on 1 HPU:**

- 1 HPU, BF16, batch 128, 90 epochs:
  ```bash
  $PYTHON imagenet_main.py -dt bf16 -dlit fp32 -bs 128 -te 90 -ebe 90 --data_dir /data/tensorflow/imagenet/tf_records/
  ```

- 1 HPU, BF16, batch 256, 90 epochs, **Gaudi2 with media acceleration**:
  ```bash
  $PYTHON imagenet_main.py -dt bf16 -dlit fp32 -bs 256 -te 90 -ebe 90 --jpeg_data_dir /data/tensorflow/imagenet
  ```

**Run training on 8 HPUs - Horovod:**

**NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration).

- 8 HPUs, BF16, batch 128, 90 epochs:

  ```bash
  mpirun --allow-run-as-root --tag-output --merge-stderr-to-stdout --output-filename /root/tmp/resnext_log --bind-to core --map-by socket:PE=7 -np 8 \
    $PYTHON imagenet_main.py --use_horovod -dt bf16 -dlit fp32 -bs 128 -te 90 -ebe 90 --data_dir /data/tensorflow/imagenet/tf_records/
  ```

- 8 HPUs, BF16, batch 256, 90 epochs, **Gaudi2 with media acceleration**:

  ```bash
  mpirun --allow-run-as-root --tag-output --merge-stderr-to-stdout --output-filename /root/tmp/resnext_log --bind-to core --map-by socket:PE=7 -np 8 \
    $PYTHON imagenet_main.py --use_horovod -dt bf16 -dlit fp32 -bs 256 -te 90 -ebe 90 --jpeg_data_dir /data/tensorflow/imagenet
  ```

### Multi-Server Training Examples

**Run training on 64 HPUs - Horovod:**

Multi-server training works by setting these environment variables:
  - `-H`: Set this to a comma-separated list of host IP addresses. Make sure to modify IP addresses below to match your system.
  - `--mca btl_tcp_if_include`: Provide network interface associated with IP address. More details can be found in the [Open MPI documentation](https://www.open-mpi.org/faq/?category=tcp#tcp-selection). If you get mpirun `btl_tcp_if_include` errors, try un-setting this environment variable and let the training script automatically detect the network interface associated with the host IP address.
  - `HCCL_SOCKET_IFNAME`: Defines the prefix of the network interface name that is used for HCCL sideband TCP communication. If not set, the first network interface with a name that does not start with lo or docker will be used.

**NOTE:**
- mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration).
- `$MPI_ROOT` environment variable is set automatically during Setup. See [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) for details.
- `HABANA_VISIBLE_MODULES` environment variable describes which Gaudi modules are available for training. On 1.5.0, 1.6.0 and 1.6.1 for multi-server training, it should be set to `0,1,2,3,4,5,6,7`.

```bash
mpirun \
 --allow-run-as-root --mca plm_rsh_args -p3022 \
 --bind-to core \
 --map-by socket:PE=7 -np 64 \
 --mca btl_tcp_if_include 192.10.100.174/24 \
 --tag-output --merge-stderr-to-stdout \
 --output-filename /root/tmp/resnext_log --prefix $MPI_ROOT \
 -H 192.10.100.174:8,10.10.100.101:8,10.10.100.102:8,10.10.100.203:8,10.10.100.104:8,10.10.100.205:8,10.10.100.106:8,10.10.100.207:8 \
 -x GC_KERNEL_PATH -x HABANA_LOGS \
 -x PYTHONPATH -x HCCL_SOCKET_IFNAME=<interface_name> \
 -x HABANA_VISIBLE_MODULES=0,1,2,3,4,5,6,7 \
   $PYTHON imagenet_main.py \
     --use_horovod -dt bf16 \
     -dlit fp32 \
     -bs 128 \
     -te 90 \
     -ebe 90 \
     --weight_decay 7.93E-05 \
     --data_dir /data/tensorflow/imagenet/tf_records/
```

**Note:** On a large scale, saving checkpoints has a significant impact on scalability. To disable checkpoints and improve performance during training, use `--disable_checkpoints`.

## Advanced

To see all possible arguments for imagenet_main.py, run:
```bash
$PYTHON imagenet_main.py --helpfull
```

## Pre-trained Model
TensorFlow ResNeXt is trained on Habana Gaudi cards and the checkpoint and saved model files are created. You can use them for fine-tuning or transfer learning tasks with your own datasets. To download the checkpoint or saved model files, please refer to [Habana Catalog](https://developer.habana.ai/catalog/resnext-for-tensorflow/) to obtain the URL.


## Profile

To run training on 1 HPU with profiler:

```bash
$PYTHON imagenet_main.py -dt bf16 -dlit fp32 -bs 128 -te 90 -ebe 90 --data_dir /data/tensorflow/imagenet/tf_records/ \
  --hooks ProfilerHook --profile_steps 5,8 --max_train_steps 10
```
The above example will produce profile trace for 4 steps (5,6,7,8).


## Supported Configuration

| Device | SynapseAI Version | TensorFlow Version(s)  |
|:------:|:-----------------:|:-----:|
| Gaudi  | 1.6.1             | 2.9.1 |
| Gaudi  | 1.6.1             | 2.8.2 |
| Gaudi2 | 1.6.1             | 2.9.1 |
| Gaudi2 | 1.6.1             | 2.8.2 |

## Changelog
### 1.6.0
- Add logging of global_examples/sec and examples/sec
- Switch from depracated function export_savedmodel() to export_saved_model()
- Enable exporting of SavedModels for training
### 1.5.0
- Added examples to run training with batch size 256 on Gaudi2.
- Added support for image processing acceleration on Gaudi2 (JPEG format only).
### 1.4.0
- Import horovod-fork package directly instead of using Model-References' TensorFlow.common.horovod_helpers; wrapped horovod import with a try-catch block so that the user is not required to install this library when the model is being run on a single card.
- Replaced references to custom demo script by community entry points in README.
- Added profile steps range support to ResNeXt model.
- Removed `setup_jemalloc()` from demo_resnext.
- Reduced frequency of logging.
- Switched from depracated flag TF_ENABLE_BF16_CONVERSION to TF_BF16_CONVERSION.
### 1.3.0
- Moved files from TensorFlow/common/ and TensorFlow/utils/ to model dir, align imports.
- Set up BF16 conversion pass using a config from habana-tensorflow package instead of recipe json file.
- Removed usage of HBN_TF_REGISTER_DATASETOPS as prerequisite for experimental preloading.
- Updated imagenet and align to new naming convention (img_train->train, img_val->validation).
- Added flag to specify maximum number of CPUs to be used.
- Postponed image transfer to device in order to leave DMA free for more critical data.
- Added support for TF profiling.
- Updated requirements.txt.
### 1.2.0
- Removed support for LARS optimizer.
- Moved distribution_utils from main dir `/TensorFlow/utils/` to ResNeXt script dir.

### Training Script Modifications

Originally, scripts were taken from [Tensorflow GitHub](https://github.com/tensorflow/models.git), tag v1.13.0.
Files used:

- `imagenet\_main.py`
- `imagenet\_preprocessing.py`
- `resnet\_model.py`
- `resnet\_run\_loop.py`

All of above files were converted to TF2 by using the [tf\_upgrade\_v2](https://www.tensorflow.org/guide/upgrade?hl=en) tool. Additionally, some other changes were committed for specific files.

The following are the changes specific to Gaudi that were made to the original scripts:

- `imagenet\_main.py`:

  - Added Habana Gaudi support.
  - Added Horovod support for multi-node.
  - Added mini_imagenet support.
  - Changed the signature of input_fn with new parameters added, and num_parallel_batches removed.
  - Changed parallel dataset deserialization to use dataset.interleave instead of dataset.apply.
  - Added ResNeXt support.
  - Added parameter to ImagenetModel::__init__ for selecting between ResNet and ResNeXt.
  - Redefined learning_rate_fn to use warmup_epochs and use_cosine_lr from params.
  - Added flags to specify the weight_decay and momentum.
  - Added flag to enable horovod support.
  - Added calls to tf.compat.v1.disable_eager_execution() and tf.compat.v1.enable_resource_variables() in main code section.
  - Added flag to specify maximum number of CPUs to be used.

- `imagenet\_preprocessing.py`:

  - Changed the image decode, crop and flip function to take a seed to propagate to tf.image.sample_distorted_bounding_box.
  - Changed the use of tf.expand_dims to tf.broadcast_to for better performance.

- `resnet\_model.py`:

  - Added tf.bfloat16 to CASTABLE_TYPES.
  - Changed calls to tf.<api> to tf.compat.v1.<api> for backward compatibility when running training scripts on TensorFlow v2.
  - Deleted the call to pad the input along the spatial dimensions independently of input size for stride greater than 1.
  - Added functionality for strided 2-D convolution with groups and explicit padding.
  - Added functionality for a single block for ResNeXt, with a bottleneck.
  - Added ResNeXt support.

- `resnet\_run\_loop.py`:

  - Changes for enabling Horovod, e.g. data sharding in multi-node, usage of Horovod's TF DistributedOptimizer, etc.
  - Added options to enable the use of tf.data's 'experimental_slack' and 'experimental prefetch_to_device' options during the input processing.
  - Added support for specific size thread pool for tf.data operations.
  - TensorFlow v2 support: Changed dataset.apply(tf.contrib.data.map_and_batch(..)) to dataset.map(..., num_parallel_calls, ...) followed by dataset.batch().
  - Changed calls to tf.<api> to tf.compat.v1.<api> for backward compatibility when running training scripts on TensorFlow v2.
  - Other TF v2 replacements for tf.contrib usages.
  - Redefined learning_rate_with_decay to use warmup_epochs and use_cosine_lr.
  - Added functionality for label smoothing.
  - Commented out writing images to summary, for performance reasons.
  - Added check for non-tf.bfloat16 input images having the same data type as the dtype that training is run with.
  - Added functionality to define the cross-entropy function depending on whether label smoothing is enabled.
  - Added support for loss scaling of gradients.
  - Added flag for experimental_preloading that invokes the HabanaEstimator, besides other optimizations such as tf.data.experimental.prefetch_to_device.
  - Added 'TF_DISABLE_SCOPED_ALLOCATOR' environment variable flag to disable Scoped Allocator Optimization (enabled by default) for Horovod runs.
  - Added a flag to configure the save_checkpoint_steps.
  - If the flag "use_train_and_evaluate" is set, or in multi-worker training scenarios, there is a one-shot call to tf.estimator.train_and_evaluate.
  - resnet_main() returns a dictionary with keys 'eval_results' and 'train_hooks'.
  - Added flags in 'define_resnet_flags' for flags_core.define_base, flags_core.define_performance, flags_core.define_distribution, flags_core.define_experimental, and many others (please refer to this function for all the flags that are available).
  - Changed order of ops creating summaries to log them in TensorBoard with proper name. Added saving HParams to TensorBoard and exposed a flag for specifying frequency of summary updates.
  - Changed a name of directory, in which workers are saving logs and checkpoints, from "rank_N" to "worker_N".
  - Added support for TensorFlow profiling.


#### Hyperparameters

* Momentum (0.875)
* Learning rate (LR) = 0.256 for 256 batch size, for other batch sizes we linearly scale the learning rate.
* Cosine learning rate schedule
* Linear warmup of the learning rate during the first 8 epochs
* Weight decay: 6.103515625e-05
* Label Smoothing: 0.1

Weight decay on batch norm trainable parameters (gamma/bias) is not applied.
The training is done for:
  * 90 Epochs -> 90 epochs is a standard for ResNet family networks.
  * 250 Epochs -> best possible accuracy.
For 250 epoch training, [MixUp regularization](https://arxiv.org/pdf/1710.09412.pdf) is used.

#### Data Augmentation
This model uses the following data augmentation for training:
* Normalization
* Random resized crop to 224x224
    * Scale from 8% to 100%
    * Aspect ratio from 3/4 to 4/3
* Random horizontal flip

## Known issues
Final training accuracy is significantly lower than validation accuracy when the training is being run for 90 epochs with just one evaluation at the end.
This is the case for the exemplary commands given out in this README, when `-te 90` (or `--train_epochs 90`) and `-ebe 90` (or `--epochs_between_evals 90`) are being passed to the training script.
Note that the validation accuracy is still state of the art.
