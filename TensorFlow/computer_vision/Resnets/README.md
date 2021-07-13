# ResNeXt for TensorFlow

This repository provides a script and recipe to train the ResNet v1.5 models to achieve state-of-the-art accuracy, and is tested and maintained by Habana. Please visit [this page](../../../README.md#tensorflow-model-performance) for performance information.

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

## Table Of Contents
  * [Model-References](../../../README.md)
  * [Model Overview](#model-overview)
    * [Training Script Modifications](#training-script-modifications)
  * [Setup](#setup)
  * [Training](#training)
  * [Preview Python Script](#Preview-Python-Script)

## Model Overview
ResNeXt is a modified version of the original ResNet v1 model. It supports 50, 101, and 152 layers.

### Training Script Modifications

Originally, scripts were taken from [Tensorflow
Github](https://github.com/tensorflow/models.git), tag v1.13.0

Files used:

-   imagenet\_main.py
-   imagenet\_preprocessing.py
-   resnet\_model.py
-   resnet\_run\_loop.py

All of the above files were converted to TF2 by using the
[tf\_upgrade\_v2](https://www.tensorflow.org/guide/upgrade?hl=en) tool.
Additionally, some other changes were committed for specific files. For a detailed list of changes,
please see [changes.md](changes.md).

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
* For inference:
    * Normalization.
    * Scale to 256x256.
    * Center crop to 224x224.

## Setup

Please follow the instructions given in the following link for setting up the environment: [Gaudi Setup and Installation Guide](https://github.com/HabanaAI/Setup_and_Install). Please answer the questions in the guide according to your preferences. This guide will walk you through the process of setting up your system to run the model on Gaudi.

### Training Data

The script operates on ImageNet 1k, a widely popular image classification dataset from the ILSVRC challenge.
In order to obtain the dataset, follow these steps:
1. Sign up with http://image-net.org/download-images and acquire the rights to download original images
2. Follow the link to the 2012 ILSVRC
 and download `ILSVRC2012_img_val.tar` and `ILSVRC2012_img_train.tar`.
3. Ensure python3 and the following Python packages are installed: TensorFlow 2.4.1 or Tensorflow 2.5.0 and `absl-py`.
4. Use the commands below - they will prepare the dataset under `/data/tensorflow_datasets/imagenet/tf_records`. This is the default data_dir for the training script.

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
python3 preprocess_imagenet.py \
  --raw_data_dir=$IMAGENET_HOME \
  --local_scratch_dir=$IMAGENET_HOME/tf_records \
  --nogcs_upload
```

## Training

Clone the repository and go to ResNets directory:

```bash
git clone https://github.com/HabanaAI/Model-References.git
cd Model-References/TensorFlow/computer_vision/Resnets
```

Note: If the repository is not in the PYTHONPATH, make sure you update it.
```bash
export PYTHONPATH=/path/to/Model-References:$PYTHONPATH
```

### The `demo_resnext.py` script

The script for training and evaluating the ResNet model has a variety of paramaters.
```
demo_resnext.py is a distributed launcher for imagenet_main.py.

usage: python demo_resnext.py [arguments]

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
                           --experimental_preloading            Enables support for 'data.experimental.prefetch_to_device' TensorFlow operator. If set:
                                                                - loads extension dynpatch_prf_remote_call.so
                                                                (demo_resnext.py handles it. On the other hand, if you use imagenet_main.py,
                                                                you should set the LD_PRELOAD environment variable to dynpatch_prf_remote_call.so)
                                                                - sets environment variable HBN_TF_REGISTER_DATASETOPS to 1
                                                                - this feature is experimental and works only with single node
                           --num_workers_per_hls <num_workers>  Number of Horovod workers per node. Defaults to 1.
                                                                In case num_workers_per_hls>1, it runs 'resnet_ctl_imagenet_main.py [ARGS] --use_horovod' via mpirun with generated HCL config.
                           --hls_type <hls_type>                HLS type: either HLS1 (8-cards) or HLS1-H (4-cards). Defaults to HLS1.
                           --kubernetes_run                     Setup kubernetes run for multi HLS training

examples:

        python demo_resnext.py -bs 64 -rs 50 --clean
        python demo_resnext.py -bs 128 -dt bf16 --experimental_preloading -te 90
        python demo_resnext.py -bs 128 -dt bf16 -dlit bf16 --experimental_preloading -te 90 --num_workers_per_hls 8

In order to see all possible arguments to imagenet_main.py, run "python imagenet_main.py --helpfull"
```

## Examples

**Run training on 1 HPU**

- ResNeXt101, bf16, batch size 128, 90 epochs, experimental preloading
    ```bash
    python3 demo_resnext.py -dt bf16 -dlit bf16 -bs 128 -te 90 --experimental_preloading
    ```

- ResNeXt50, fp32, batch size 64, 90 epochs
    ```bash
    python3 demo_resnext.py -rs 50 -bs 64 -te 90
    ```

**Run training on 8 HPU - Horovod**

- ResNeXt101, bf16, batch size 128, 90 epochs, experimental preloading
    ```bash
    python3 demo_resnext.py -dt bf16 -dlit bf16 -bs 128 -te 90 --experimental_preloading --num_workers_per_hls 8
    ```

- ResNeXt50, fp32, batch size 64, 90 epochs
    ```bash
    python3 demo_resnext.py -rs 50 -bs 64 -te 90 --num_workers_per_hls 8
    ```

**Run training on 8 HPU with mpirun and demo_resnext.py - Horovod**

If running multi-card training, set up the `HCL_CONFIG_PATH` environment variable to point to a valid HCL config JSON file for the HLS type being used. For documentation on creating an HCL config JSON file, please refer to [HCL JSON Config File Format](https://docs.habana.ai/en/latest/API_Reference_Guides/HCL_API_Reference.html#hcl-json-config-file-format).
- ResNeXt101, bf16, batch size 128, 90 epochs, experimental preloading
    ```bash
    mpirun --allow-run-as-root --tag-output --merge-stderr-to-stdout --output-filename /root/tmp/demo_resnext_log --bind-to core --map-by socket:PE=7 -np 8 -x HCL_CONFIG_PATH=hcl_config_8.json python3 demo_resnext.py -dt bf16 -dlit bf16 -bs 128 -te 90 --experimental_preloading --num_workers_per_hls 8 --kubernetes_run=True
    ```

**Run training on 8 HPU with the modified community script - Horovod**

If running multi-card training, set up the `HCL_CONFIG_PATH` environment variable to point to a valid HCL config JSON file for the HLS type being used. For documentation on creating an HCL config JSON file, please refer to [HCL JSON Config File Format](https://docs.habana.ai/en/latest/API_Reference_Guides/HCL_API_Reference.html#hcl-json-config-file-format).
- ResNeXt101, bf16, batch size 128, 90 epochs, no experimental preloading
    ```bash
    mpirun --allow-run-as-root --tag-output --merge-stderr-to-stdout --output-filename /root/tmp/demo_resnext_log --bind-to core --map-by socket:PE=7 -np 8 -x HCL_CONFIG_PATH=hcl_config_8.json python3 Model-References/TensorFlow/computer_vision/Resnets/imagenet_main.py --use_horovod -dt bf16 -dlit bf16 -bs 128 -te 90
    ```

- ResNeXt50, fp32, batch size 64, 90 epochs
    ```bash
    mpirun --allow-run-as-root --tag-output --merge-stderr-to-stdout --output-filename /root/tmp/demo_resnext_log --bind-to core --map-by socket:PE=7 -np 8 -x HCL_CONFIG_PATH=hcl_config_8.json python3 Model-References/TensorFlow/computer_vision/Resnets/imagenet_main.py --use_horovod -rs 50 -bs 64 -te 90
    ```