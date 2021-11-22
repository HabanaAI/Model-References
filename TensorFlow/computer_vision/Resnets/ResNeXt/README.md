# ResNeXt for TensorFlow

This repository provides a script and recipe to train the ResNet v1.5 models to achieve state-of-the-art accuracy, and is tested and maintained by Habana. Please visit [this page](https://developer.habana.ai/resources/habana-training-models/#performance) for performance information.

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

## Table Of Contents
  * [Model-References](../../../README.md)
  * [Model Overview](#model-overview)
    * [Training Script Modifications](#training-script-modifications)
  * [Setup](#setup)
  * [Training](#training)
  * [Preview Python Script](#Preview-Python-Script)

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

### The following are the changes specific to Gaudi that were made to the original scripts:

### imagenet\_main.py

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

### imagenet\_preprocessing.py

1. Changed the image decode, crop and flip function to take a seed to propagate to tf.image.sample_distorted_bounding_box
2. Changed the use of tf.expand_dims to tf.broadcast_to for better performance

### resnet\_model.py

1. Added tf.bfloat16 to CASTABLE_TYPES
2. Changed calls to tf.<api> to tf.compat.v1.<api> for backward compatibility when running training scripts on TensorFlow v2
3. Deleted the call to pad the input along the spatial dimensions independently of input size for stride greater than 1
4. Added functionality for strided 2-D convolution with groups and explicit padding
5. Added functionality for a single block for ResNext, with a bottleneck
6. Added Resnext support

### resnet\_run\_loop.py

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


### Hyperparameters
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

### Data Augmentation
This model uses the following data augmentation:

* For training:
    * Normalization.
    * Random resized crop to 224x224.
        * Scale from 8% to 100%.
        * Aspect ratio from 3/4 to 4/3.
    * Random horizontal flip.

## Setup
Please follow the instructions given in the following link for setting up the environment including the `$PYTHON` environment variable: [Gaudi Setup and Installation Guide](https://github.com/HabanaAI/Setup_and_Install). Please answer the questions in the guide according to your preferences. This guide will walk you through the process of setting up your system to run the model on Gaudi.

Set the `MPI_ROOT` environment variable to the directory where OpenMPI is installed.

For example, in Habana containers, use

```bash
export MPI_ROOT=/usr/lib/habanalabs/openmpi/
```

### Training Data

The script operates on ImageNet 1k, a widely popular image classification dataset from the ILSVRC challenge.
In order to obtain the dataset, follow these steps:
1. Sign up with http://image-net.org/download-images and acquire the rights to download original images
2. Follow the link to the 2012 ILSVRC
 and download `ILSVRC2012_img_val.tar` and `ILSVRC2012_img_train.tar`.
3. Use the commands below - they will prepare the dataset under `/data/tensorflow_datasets/imagenet/tf_records`. This is the default data_dir for the training script.

```
export IMAGENET_HOME=/data/tensorflow_datasets/imagenet/tf_records
mkdir -p $IMAGENET_HOME/img_val
mkdir -p $IMAGENET_HOME/img_train
tar xf ILSVRC2012_img_val.tar -C $IMAGENET_HOME/img_val
tar xf ILSVRC2012_img_train.tar -C $IMAGENET_HOME/img_train
cd $IMAGENET_HOME/img_train
for f in *.tar; do
  d=`basename $f .tar`
  mkdir $d
  tar xf $f -C $d
done
cd $IMAGENET_HOME
rm $IMAGENET_HOME/img_train/*.tar # optional
wget -O synset_labels.txt https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt
cd Model-References/TensorFlow/computer_vision/Resnets
$PYTHON preprocess_imagenet.py \
  --raw_data_dir=$IMAGENET_HOME \
  --local_scratch_dir=$IMAGENET_HOME/tf_records \
  --nogcs_upload
```

## Training

In the docker container, clone this repository and switch to the branch that
matches your SynapseAI version. (Run the
[`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.)
```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```
Go to ResNets directory:

```bash
cd Model-References/TensorFlow/computer_vision/Resnets/ResNeXt
```

Note: If the repository is not in the PYTHONPATH, make sure you update it.
```bash
export PYTHONPATH=/path/to/Model-References:$PYTHONPATH
```

### The `demo_resnext.py` script

The script for training and evaluating ResNeXt supports a variety of paramaters.
```
demo_resnext.py is a distributed launcher for imagenet_main.py.

usage: $PYTHON demo_resnext.py [arguments]

optional arguments:


        -dt <data_type>,   --dtype <data_type>                  Data type, possible values: fp32, bf16. Defaults to fp32
        -dlit <data_type>, --data_loader_image_type <data_type> Data loader images output. Should normally be set to the same data_type as the '--dtype' param
        -bs <batch_size>,  --batch_size <batch_size>            Batch size, defaults to 256
        -rs <size>,        --resnet_size <size>                 The size of the ResNet model to use. Defaults to 101.
        -te <epochs>,      --train_epochs <epochs>              Number of training epochs, defaults to 1
        -dd <data_dir>,    --data_dir <data_dir>                Data dir, defaults to `/data/tensorflow_datasets/imagenet/tf_records/`.
                                                                Needs to be specified if the above does not exists.
        -md <model_dir>,   --model_dir <model_dir>              Model dir, defaults to /tmp/resnet
                           --clean                              If set, model_dir will be removed if it exists. Unset by default.
                                                                Important: --clean may return errors in distributed environments. If that happens, try again
        -mts <steps>,      --max_train_steps <steps>            Max train steps
                           --log_steps <steps>                  How often display step status, defaults to 100
        -ebe <epochs>      --epochs_between_evals <epochs>      Number of training epochs between evaluations, defaults to 1.
                                                                To achieve fastest 'time to train', set to the same number as '--train_epochs' to only run one evaluation after the training.
                           --experimental_preloading            Enables support for 'data.experimental.prefetch_to_device' TensorFlow operator.
                                                                Enabled by default - pass --experimental_preloading=False to disable.
                           --num_workers_per_hls <num_workers>  Number of Horovod workers per node. Defaults to 1.
                                                                In case num_workers_per_hls>1, it runs 'resnet_ctl_imagenet_main.py [ARGS] --use_horovod' via mpirun with generated HCL config.
                           --kubernetes_run                     Setup kubernetes run for multi server training

examples:

        $PYTHON demo_resnext.py -bs 128 -dt bf16 -te 90 -ebe 90
        $PYTHON demo_resnext.py -bs 128 -dt bf16 -dlit bf16 -te 90 -ebe 90 --num_workers_per_hls 8

In order to see all possible arguments to imagenet_main.py, run "$PYTHON imagenet_main.py --helpfull"
```

## Examples
> ### ⚠ Performance issue related to data_loader_image_type
> There is known performance issue where disabling bf16 data loading for ResNeXt is a workaround that improves 1-HPU performance.
It turns out we get the best performance when `dtype == bf16` and `data_loader_image_type == fp32`.
Note that, examples below **do not** take it into account and use `-dlit bf16`.

**Run training on 1 HPU**

- ResNeXt101, bf16, batch size 128, 90 epochs, experimental preloading
    ```bash
    $PYTHON demo_resnext.py -dt bf16 -dlit bf16 -bs 128 -te 90 -ebe 90
    ```

**Run training on 8 HPU - Horovod**

- ResNeXt101, bf16, batch size 128, 90 epochs, experimental preloading
    ```bash
    $PYTHON demo_resnext.py -dt bf16 -dlit bf16 -bs 128 -te 90 -ebe 90 --num_workers_per_hls 8
    ```

**Run training on 8 HPU with mpirun and demo_resnext.py - Horovod**

If running multi-card training, set up the `HCL_CONFIG_PATH` environment variable to point to a valid HCL config JSON file for the server type being used. For documentation on creating an HCL config JSON file, please refer to [HCL JSON Config File Format](https://docs.habana.ai/en/latest/API_Reference_Guides/HCL_API_Reference.html#hcl-json-config-file-format).
- ResNeXt101, bf16, batch size 128, 90 epochs, experimental preloading
*<br>mpirun map-by PE attribute value may vary on your setup and should be calculated as:<br>
socket:PE = floor((number of physical cores) / (number of gaudi devices per each node))*
    ```bash
    mpirun --allow-run-as-root --tag-output --merge-stderr-to-stdout --output-filename /root/tmp/demo_resnext_log --bind-to core --map-by socket:PE=7 -np 8 -x HCL_CONFIG_PATH=hcl_config_8.json $PYTHON demo_resnext.py -dt bf16 -dlit bf16 -bs 128 -te 90 -ebe 90 --num_workers_per_hls 8 --kubernetes_run=True
    ```

**Run training on 8 HPU with the modified community script - Horovod**

If running multi-card training, set up the `HCL_CONFIG_PATH` environment variable to point to a valid HCL config JSON file for the server type being used. For documentation on creating an HCL config JSON file, please refer to [HCL JSON Config File Format](https://docs.habana.ai/en/latest/API_Reference_Guides/HCL_API_Reference.html#hcl-json-config-file-format).
- ResNeXt101, bf16, batch size 128, 90 epochs, no experimental preloading
    ```bash
    mpirun --allow-run-as-root --tag-output --merge-stderr-to-stdout --output-filename /root/tmp/demo_resnext_log --bind-to core --map-by socket:PE=7 -np 8 $PYTHON Model-References/TensorFlow/computer_vision/Resnets/ResNeXt/imagenet_main.py --use_horovod -dt bf16 -dlit bf16 -bs 128 -te 90 -ebe 90
    ```
