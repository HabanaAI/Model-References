# ResNeXt for TensorFlow

This repository provides a script and recipe to train the ResNet v1.5 models to achieve state-of-the-art accuracy, and is tested and maintained by Habana. Please visit [this page](https://developer.habana.ai/resources/habana-training-models/#performance) for performance information.

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents

* [Model-References](../../../../README.md)
* [Model Overview](#model-overview)
* [Setup](#setup)
* [Training](#training)
* [Profile](#profile)
* [Changelog](#changelog)
* [Known Issues](#known-issues)


## Model Overview
ResNeXt is a modified version of the original ResNet v1 model. This implementation defines ResNeXt101, which features 101 layers.

### Training Script Modifications

Originally, scripts were taken from [Tensorflow Github](https://github.com/tensorflow/models.git), tag v1.13.0

Files used:

-   imagenet\_main.py
-   imagenet\_preprocessing.py
-   resnet\_model.py
-   resnet\_run\_loop.py

All of above files were converted to TF2 by using [tf\_upgrade\_v2](https://www.tensorflow.org/guide/upgrade?hl=en) tool. Additionally, some other changes were committed for specific files.

The following are the changes specific to Gaudi that were made to the original scripts:

#### imagenet\_main.py

1. Added Habana HPU support
2. Added Horovod support for multinode
3. Added mini_imagenet support
4. Changed the signature of input_fn with new parameters added, and num_parallel_batches removed
5. Changed parallel dataset deserialization to use dataset.interleave instead of dataset.apply
6. Added Resnext support
7. Added parameter to ImagenetModel::__init__ for selecting between resnet and resnext
8. Redefined learning_rate_fn to use warmup_epochs and use_cosine_lr from params
9. Added flags to specify the weight_decay and momentum
10. Added flag to enable horovod support
11. Added calls to tf.compat.v1.disable_eager_execution() and tf.compat.v1.enable_resource_variables() in main code section
12. Added flag to specify maximum number of cpus to be used

#### imagenet\_preprocessing.py

1. Changed the image decode, crop and flip function to take a seed to propagate to tf.image.sample_distorted_bounding_box
2. Changed the use of tf.expand_dims to tf.broadcast_to for better performance

#### resnet\_model.py

1. Added tf.bfloat16 to CASTABLE_TYPES
2. Changed calls to tf.<api> to tf.compat.v1.<api> for backward compatibility when running training scripts on TensorFlow v2
3. Deleted the call to pad the input along the spatial dimensions independently of input size for stride greater than 1
4. Added functionality for strided 2-D convolution with groups and explicit padding
5. Added functionality for a single block for ResNext, with a bottleneck
6. Added Resnext support

#### resnet\_run\_loop.py

1. Changes for enabling Horovod, e.g. data sharding in multi-node, usage of Horovod's TF DistributedOptimizer, etc.
2. Added options to enable the use of tf.data's 'experimental_slack' and 'experimental.prefetch_to_device' options during the input processing
3. Added support for specific size thread pool for tf.data operations
4. TensorFlow v2 support: Changed dataset.apply(tf.contrib.data.map_and_batch(..)) to dataset.map(..., num_parallel_calls, ...) followed by dataset.batch()
5. Changed calls to tf.<api> to tf.compat.v1.<api> for backward compatibility when running training scripts on TensorFlow v2
6. Other TF v2 replacements for tf.contrib usages
7. Redefined learning_rate_with_decay to use warmup_epochs and use_cosine_lr
8. Added functionality for label smoothing
9. Commented out writing images to summary, for performance reasons
10. Added check for non-tf.bfloat16 input images having the same data type as the dtype that training is run with
11. Added functionality to define the cross-entropy function depending on whether label smoothing is enabled
12. Added support for loss scaling of gradients
13. Added flag for experimental_preloading that invokes the HabanaEstimator, besides other optimizations such as tf.data.experimental.prefetch_to_device
14. Added 'TF_DISABLE_SCOPED_ALLOCATOR' environment variable flag to disable Scoped Allocator Optimization (enabled by default) for Horovod runs
15. Added a flag to configure the save_checkpoint_steps
16. If the flag "use_train_and_evaluate" is set, or in multi-worker training scenarios, there is a one-shot call to tf.estimator.train_and_evaluate
17. resnet_main() returns a dictionary with keys 'eval_results' and 'train_hooks'
18. Added flags in 'define_resnet_flags' for flags_core.define_base, flags_core.define_performance, flags_core.define_distribution, flags_core.define_experimental, and many others (please refer to this function for all the flags that are available)
19. Changed order of ops creating summaries to log them in TensorBoard with proper name. Added saving HParams to TensorBoard and exposed a flag for specifying frequency of summary updates.
20. Changed a name of directory, in which workers are saving logs and checkpoints, from "rank_N" to "worker_N".
21. Added support for TF profiling


#### Hyperparameters
**ResNeXt:**

* Momentum (0.875).
* Learning rate (LR) = 0.256 for 256 batch size, for other batch sizes we linearly scale the learning rate.
* Cosine learning rate schedule.
* Linear warmup of the learning rate during the first 8 epochs.
* Weight decay: 6.103515625e-05
* Label Smoothing: 0.1.

We do not apply Weight decay on batch norm trainable parameters (gamma/bias).
We train for:
  * 90 Epochs -> 90 epochs is a standard for ResNet family networks.
  * 250 Epochs -> best possible accuracy.
For 250 epoch training we also use [MixUp regularization](https://arxiv.org/pdf/1710.09412.pdf).

#### Data Augmentation
This model uses the following data augmentation:

* For training:
    * Normalization.
    * Random resized crop to 224x224.
        * Scale from 8% to 100%.
        * Aspect ratio from 3/4 to 4/3.
    * Random horizontal flip.

## Setup
Please follow the instructions given in the following link for setting up the
environment including the `$PYTHON` and `$MPI_ROOT` environment variables: [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html).
This guide will walk you through the process of setting up your system to run
the model on Gaudi.

### Training Data

The script operates on ImageNet 1k, a widely popular image classification dataset from the ILSVRC challenge.
In order to obtain the dataset, follow these steps:
1. Sign up with http://image-net.org/download-images and acquire the rights to download original images
2. Follow the link to the 2012 ILSVRC
 and download `ILSVRC2012_img_val.tar` and `ILSVRC2012_img_train.tar`.
3. Use the commands below - they will prepare the dataset under `/data/tensorflow/imagenet/tf_records`. This is the default data_dir for the training script.

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
```

### Clone Habana Model-References

In the docker container, clone this repository and switch to the branch that
matches your SynapseAI version. (Run the
[`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.)
```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```
Note: If the repository is not in the PYTHONPATH, make sure you update it.
```bash
export PYTHONPATH=/path/to/Model-References:$PYTHONPATH
```

### Install Model Requirements

In the docker container, go to the ResNeXt directory
```bash
cd /root/Model-References/TensorFlow/computer_vision/Resnets/ResNeXt
```
Install required packages using pip
```bash
$PYTHON -m pip install -r requirements.txt
```

### Setup jemalloc

There are instances where the performace improvement is noticed if jemalloc is setup before running the below examples. Generally it is recommended to setup jemalloc before training ResNeXt model.
To setup jemalloc, export the LD_PRELOAD with the path to libjemalloc.so
Only one of the libjemalloc.so.1 or libjemalloc.so.2 will be present.
To locate the files, search inside directories
* /usr/lib/x86_64-linux-gnu/
* /usr/lib64/

Once any of the above version is detected, export is using command export LD_PRELOAD=/path/to/libjemalloc.so

Example:
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
```

## Training

> ### ⚠ Performance issue related to data_loader_image_type
> There is known performance issue where disabling bf16 data loading for ResNeXt is a workaround that improves 1-HPU performance.
It turns out we get the best performance when `dtype == bf16` and `data_loader_image_type == fp32`.
Note that, examples below **do not** take it into account and use `-dlit bf16`.

### Examples
**Run training on 1 HPU**

- ResNeXt101, bf16, batch size 128, 90 epochs
    ```bash
    $PYTHON imagenet_main.py -dt bf16 -dlit bf16 -bs 128 -te 90 -ebe 90 --data_dir /data/tensorflow/imagenet/tf_records/
    ```

**Run training on 8 HPU- Horovod**
*<br>mpirun map-by PE attribute value may vary on your setup and should be calculated as:<br>
socket:PE = floor((number of physical cores) / (number of gaudi devices per each node))*

- ResNeXt101, bf16, batch size 128, 90 epochs
    ```bash
    mpirun --allow-run-as-root --tag-output --merge-stderr-to-stdout --output-filename /root/tmp/resnext_log --bind-to core --map-by socket:PE=7 -np 8 $PYTHON imagenet_main.py --use_horovod -dt bf16 -dlit bf16 -bs 128 -te 90 -ebe 90 --data_dir /data/tensorflow/imagenet/tf_records/
    ```

**Run training on 64 HPU, multiple boxes - Horovod**

Multi-server training works by setting these environment variables:
- NOTE: MODIFY IP ADDRESS BELOW TO MATCH YOUR SYSTEM.
- **`-H`**: set this to a comma-separated list of host IP addresses
- **`--mca btl_tcp_if_include`**: Provide network interface associated with IP address. More details: [Open MPI documentation](https://www.open-mpi.org/faq/?category=tcp#tcp-selection). If you get mpirun `btl_tcp_if_include` errors, try un-setting this environment variable and let the training script automatically detect the network interface associated with the host IP address.
- **`HCCL_SOCKET_IFNAME`**: HCCL_SOCKET_IFNAME defines the prefix of the network interface name that is used for HCCL sideband TCP communication. If not set, the first network interface with a name that does not start with lo or docker will be used.

*<br>mpirun map-by PE attribute value may vary on your setup and should be calculated as:<br>
socket:PE = floor((number of physical cores) / (number of gaudi devices per each node))*

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
   $PYTHON imagenet_main.py \
     --use_horovod -dt bf16 \
     -dlit bf16 \
     -bs 128 \
     -te 90 \
     -ebe 90 \
     --weight_decay 7.93E-05 \
     --data_dir /data/tensorflow/imagenet/tf_records/
```

Note: On a large scale, saving checkpoints has a significant impact on scalability. To disable checkpoints and improve preformance during training, use --disable_checkpoints

**In order to see all possible arguments to imagenet_main.py, run**
```bash
$PYTHON imagenet_main.py --helpfull
```

## Profile

**Run training on 1 HPU with profiler**

```bash
$PYTHON imagenet_main.py -dt bf16 -dlit bf16 -bs 128 -te 90 -ebe 90 --data_dir /data/tensorflow/imagenet/tf_records/ --hooks ProfilerHook --profile_steps 5,8 --max_train_steps 10
```
The above example will produce profile trace for 4 steps (5,6,7,8)

## Supported Configuration

| Device | SynapseAI Version | TensorFlow Version(s)  |
|:------:|:-----------------:|:-----:|
| Gaudi  | 1.4.0             | 2.8.0 |
| Gaudi  | 1.4.0             | 2.7.1 |

## Changelog
### 1.2.0
- removed support for LARS optimizer
- distribution_utils moved from main dir `/TensorFlow/utils/` to ResNeXt script dir
### 1.3.0
- move files from TensorFlow/common/ and TensorFlow/utils/ to model dir, align imports
- setup BF16 conversion pass using a config from habana-tensorflow package instead of recipe json file
- remove usage of HBN_TF_REGISTER_DATASETOPS as prerequisite for experimantal preloading
- update imagenet and align to new naming convention (img_train->train, img_val->validation)
- add flag to specify maximum number of cpus to be used
- postpone image transfer to device in order to leave DMA free for more critical data
- add support for TF profiling
- update requirements.txt
### 1.4.0
- import horovod-fork package directly instead of using Model-References' TensorFlow.common.horovod_helpers; wrapped horovod import with a try-catch block so that the user is not required to install this library when the model is being run on a single card
- references to custom demo script were replaced by community entry points in README.
- add profile steps range support to ResNeXt model.
- removed setup_jemalloc() from demo_resnext
- reduce frequency of logging
- switched from depracated flag TF_ENABLE_BF16_CONVERSION to TF_BF16_CONVERSION

## Known issues
Final training accuracy is significantly lower than validation accuracy when the training is being run for 90 epochs with just one evaluation at the end.
This is the case for the examplary commands given out in this README, when `-te 90` (or `--train_epochs 90`) and `-ebe 90` (or `--epochs_between_evals 90`) are being passed to the training script.
Note, that the validation accuracy is still state of the art.
