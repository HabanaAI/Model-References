# ResNet50 Keras for TensorFlow

This directory provides a script and recipe to train a ResNet Keras model to achieve state-of-the-art accuracy, and is tested and maintained by Habana. To obtain model performance data, refer to the [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

For more information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents

* [Model-References](../../../../README.md)
* [Model Overview](#model-overview)
* [Setup](#setup)
* [Media Loading Acceleration](#media-loading-acceleration)
* [Training and Examples](#training-and-examples)
* [Pre-trained Model](#pre-trained-model)
* [Profile](#profile)
* [Supported Configuration](#supported-configuration)
* [Changelog](#changelog)

## Model Overview
The ResNet Keras model is a modified version of the original model located in [TensorFlow model garden](https://github.com/tensorflow/models/tree/master/official/legacy/image_classification/resnet). It uses a custom training loop, supports 50 layers and can work with both SGD and LARS optimizers.

## Setup
Please follow the instructions provided in the [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the
environment including the `$PYTHON` environment variable.
The guide will walk you through the process of setting up your system to run the model on Gaudi.

### Training Data

The ResNet50 Keras script operates on ImageNet 1k, a widely popular image classification dataset from the ILSVRC challenge.
In order to obtain the dataset, follow these steps:
1. Sign up with http://image-net.org/download-images and acquire the rights to download original images.
2. Follow the link to the 2012 ILSVRC to download `ILSVRC2012_img_val.tar` and `ILSVRC2012_img_train.tar`.
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

**Note:** If the repository is not in the PYTHONPATH, make sure to update by running the below.
```bash
export PYTHONPATH=/path/to/Model-References:$PYTHONPATH
```

### Install Model Requirements

1. In the docker container, go to the `resnet_keras` directory:
```bash
cd /root/Model-References/TensorFlow/computer_vision/Resnets/resnet_keras
```
2. Install the required packages using pip:
```bash
$PYTHON -m pip install -r requirements.txt
```

### Set up jemalloc

There are instances where the performance improvement is noticed if jemalloc is set up before running the below examples. Generally, it is recommended to set up jemalloc before training ResNet model.
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
Gaudi2 offers a dedicated hardware engine for Media Loading operations such as JPEG decoding and data augmentation. This can be leveraged in ResNet models to decrease CPU usage and exceed performance limitation when data processing on CPU is a bottleneck.
Currently, the supported file format is JPEG only.

**Note:** TFRecord will be supported in a future release.

ResNet automatically uses hardware Media Loading Acceleration unless:
1. Training is done on first-gen Gaudi processors. First-gen Gaudi does not have a dedicated hardware for Media Loading Acceleration.
2. `hpu_media_loader` Python package is not installed.
3. `FORCE_HABANA_IMAGENET_LOADER_FALLBACK` environment variable is set to 1.
4. A location of the ImageNet dataset containing JPEGs (--jpeg_data_dir parameter) is not provided.

In the above cases, media processing will be done on CPU.

## Training and Examples

This directory contains modified community scripts. The modifications include changing the default values of parameters, setting environment variables, changing include paths, etc. The changes are described in the license headers of the respective files.

**To see all possible arguments for `resnet_ctl_imagenet_main.py`, run**
```bash
$PYTHON resnet_ctl_imagenet_main.py --helpfull
```

### Using LARS Optimizer
Using LARS optimizer usually requires changing the default values of some hyperparameters and should be manually set for `resnet_ctl_imagenet_main.py`. The recommended parameters together with their default values are presented below:

| Parameter          | Value      |
| ------------------ | ---------- |
| optimizer          | LARS       |
| base_learning_rate | 2.5 or 9.5*|
| warmup_epochs      | 3          |
| lr_schedule        | polynomial |
| label_smoothing    | 0.1        |
| weight_decay       | 0.0001     |
| single_l2_loss_op  | True       |

*2.5 is the default value for single card (1 HPU) trainings, otherwise, the default is 9.5. These values have been determined experimentally.

The following command shows an example of how to set LARS optimizer:
```bash
$PYTHON resnet_ctl_imagenet_main.py --optimizer LARS --base_learning_rate 9.5 --warmup_epochs 3 --lr_schedule polynomial --label_smoothing 0.1 --weight_decay 0.0001 --single_l2_loss_op
```

### Single Card and Multi-Card Training Examples
**Note:** When using the `resnet_ctl_imagenet_main.py` for training, some changes to the default values of parameters, setting environment variables, changing include paths, etc are required. These changes are described in the license headers of the file.

**Run training on 1 HPU:**

- 1 HPU, batch 256, 90 epochs, BF16 precision, SGD:
  ```bash
  $PYTHON resnet_ctl_imagenet_main.py -dt bf16 -dlit bf16 -te 90 -ebe 90 -bs 256 --data_dir /data/tensorflow/imagenet/tf_records --enable_tensorboard
  ```
- 1 HPU, batch 256, 40 epochs, BF16 precision, LARS:
  ```bash
  $PYTHON resnet_ctl_imagenet_main.py -bs 256 -te 40 -ebe 40 -dt bf16 --data_dir /data/tensorflow/imagenet/tf_records \
  --optimizer LARS --base_learning_rate 9.5 --warmup_epochs 3 --lr_schedule polynomial --label_smoothing 0.1 --weight_decay 0.0001 \
  --single_l2_loss_op --enable_tensorboard
  ```
- 1 HPU, batch 256, 90 epochs, BF16 precision, SGD, **Gaudi2 with media acceleration**:
  ```bash
  $PYTHON resnet_ctl_imagenet_main.py -dt bf16 -dlit bf16 -te 90 -ebe 90 -bs 256 --jpeg_data_dir /data/tensorflow/imagenet --enable_tensorboard
  ```
- 1 HPU, batch 256, 40 epochs, BF16 precision, LARS, **Gaudi2 with media acceleration**:
  ```bash
  $PYTHON resnet_ctl_imagenet_main.py -bs 256 -te 40 -ebe 40 -dt bf16 --jpeg_data_dir /data/tensorflow/imagenet \
  --optimizer LARS --base_learning_rate 9.5 --warmup_epochs 3 --lr_schedule polynomial --label_smoothing 0.1 --weight_decay 0.0001 \
  --single_l2_loss_op --enable_tensorboard
  ```

**Run training on 8 HPUs - Horovod:**

**NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration).

- 8 HPUs on 1 server, batch 256, 40 epochs, BF16 precision, LARS:
  ```bash 
  mpirun --allow-run-as-root --bind-to core -np 8 --map-by socket:PE=6 --merge-stderr-to-stdout \
    $PYTHON resnet_ctl_imagenet_main.py \
      --dtype bf16 \
      --data_loader_image_type bf16 \
      --use_horovod \
      -te 40 \
      -ebe 40 \
      -bs 256 \
      --optimizer LARS \
      --base_learning_rate 9.5 \
      --warmup_epochs 3 \
      --lr_schedule polynomial \
      --label_smoothing 0.1 \
      --weight_decay 0.0001 \
      --single_l2_loss_op \
      --data_dir /data/tensorflow/imagenet/tf_records \
      --enable_tensorboard
  ```

- 8 HPUs on 1 server, batch 256, 40 epochs, BF16 precision, LARS, **Gaudi2 with media acceleration**:

    ```bash 
    mpirun --allow-run-as-root --bind-to core -np 8 --map-by socket:PE=6 --merge-stderr-to-stdout \
      $PYTHON resnet_ctl_imagenet_main.py \
        --dtype bf16 \
        --data_loader_image_type bf16 \
        --use_horovod \
        -te 40 \
        -ebe 40 \
        -bs 256 \
        --optimizer LARS \
        --base_learning_rate 9.5 \
        --warmup_epochs 3 \
        --lr_schedule polynomial \
        --label_smoothing 0.1 \
        --weight_decay 0.0001 \
        --single_l2_loss_op \
        --jpeg_data_dir /data/tensorflow/imagenet \
        --enable_tensorboard
    ```
**Run training on 8 HPUs - tf.distribute:**

**NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration).

- 8 HPUs on 1 server, batch 256, 40 epochs, BF16 precision, LARS:

    ```bash 
    mpirun --allow-run-as-root --bind-to core -np 8 --map-by socket:PE=6 --merge-stderr-to-stdout \
      $PYTHON resnet_ctl_imagenet_main.py \
        --dtype bf16 \
        --data_loader_image_type bf16 \
        --distribution_strategy=hpu \
        -te 40 \
        -ebe 40 \
        -bs 2048 \
        --optimizer LARS \
        --base_learning_rate 9.5 \
        --use_tf_while_loop=False \
        --warmup_epochs 3 \
        --lr_schedule polynomial \
        --label_smoothing 0.1 \
        --weight_decay 0.0001 \
        --single_l2_loss_op \
        --data_dir /data/tensorflow/imagenet/tf_records \
        --enable_tensorboard
    ```

- 8 HPUs on 1 server, batch 256, 40 epochs, BF16 precision, LARS, **Gaudi2 with media acceleration**:

    ```bash 
    mpirun --allow-run-as-root --bind-to core -np 8 --map-by socket:PE=6 --merge-stderr-to-stdout \
      $PYTHON resnet_ctl_imagenet_main.py \
        --dtype bf16 \
        --data_loader_image_type bf16 \
        --distribution_strategy=hpu \
        -te 40 \
        -ebe 40 \
        -bs 2048 \
        --optimizer LARS \
        --base_learning_rate 9.5 \
        --use_tf_while_loop=False \
        --warmup_epochs 3 \
        --lr_schedule polynomial \
        --label_smoothing 0.1 \
        --weight_decay 0.0001 \
        --single_l2_loss_op \
        --jpeg_data_dir /data/tensorflow/imagenet \
        --enable_tensorboard
    ```

   Note:
   - Currently, `experimental_preloading` (enabled by default) is required to run `distribution_strategy=hpu`.
   - Unlike Horovod, global batch size must be specified for `tf.distribute`. In this case 256 * 8 workers = 2048.

### Multi-Server Training Setup

The following directions are generalized for use of multi-server setup:

1. Follow the [Setup](#setup) steps outlined above on all servers.
2. Configure ssh between servers. The following is required inside all servers' docker containers:
```
mkdir ~/.ssh
cd ~/.ssh
ssh-keygen -t rsa -b 4096
```
3. Copy id_rsa.pub contents from every server to every authorized_keys (all public keys need to be in all hosts' authorized_keys):
```
cat id_rsa.pub > authorized_keys
vi authorized_keys
```
3. Copy the contents from inside to other system.
Paste both hosts public keys in both host’s “authorized_keys” file:

On each system:
Add all hosts (including itself) to known_hosts:
```
ssh-keyscan -p 3022 -H 192.10.100.174 >> ~/.ssh/known_hosts
```
By default, the Habana docker uses port 3022 for ssh, and this is the default port configured in the training scripts. Sometimes, mpirun can fail to establish the remote connection when there is more than one Habana docker session running on the main server in which the Python training script is run. If this happens, you can set up a different ssh port as follows:

In each docker container:
```
vi /etc/ssh/sshd_config
vi /etc/ssh/ssh_config
```
Add a different port number:
Port 4022

Finally, restart sshd service on all hosts:
```
service ssh stop
service ssh start
```

**NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration).

### Multi-Server Training Examples


**Run training on 16 HPUs - Horovod:**

  - 16 HPUs on 2 servers, batch 256, 40 epochs, BF16 precision, LARS:
    - `-H`: Set this to a comma-separated list of host IP addresses. Make sure to modify IP addresses below to match your system.
    - `--mca btl_tcp_if_include`: Provide network interface associated with IP address. More details can be found in the [Open MPI documentation](https://www.open-mpi.org/faq/?category=tcp#tcp-selection). If you get mpirun `btl_tcp_if_include` errors, try un-setting this environment variable and let the training script automatically detect the network interface associated with the host IP address.
    - `HCCL_SOCKET_IFNAME`: Defines the prefix of the network interface name that is used for HCCL sideband TCP communication. If not set, the first network interface with a name that does not start with lo or docker will be used.
    - `$MPI_ROOT` environment variable is set automatically during Setup. See [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) for details.

    ```bash 
    mpirun \
     --allow-run-as-root --mca plm_rsh_args -p3022 \
     --bind-to core \
     --map-by socket:PE=6 -np 16 \
     --mca btl_tcp_if_include <interface_name> \
     --tag-output --merge-stderr-to-stdout --prefix $MPI_ROOT \
     -H 192.10.100.174:8,10.10.100.101:8 \
     -x GC_KERNEL_PATH -x HABANA_LOGS \
     -x PYTHONPATH -x HCCL_SOCKET_IFNAME=<interface_name> \
        $PYTHON resnet_ctl_imagenet_main.py \
          -dt bf16 \
          -dlit bf16 \
          -bs 256 \
          -te 40 \
          -ebe 40 \
          --use_horovod \
          --data_dir /data/tensorflow/imagenet/tf_records \
          --optimizer LARS \
          --base_learning_rate 9.5 \
          --warmup_epochs 3 \
          --lr_schedule polynomial \
          --label_smoothing 0.1 \
          --weight_decay 0.0001 \
          --single_l2_loss_op \
          --enable_tensorboard
    ```

**Note:** To run multi-server training over host NICs (required for AWS users), one of the following variants must take place:*

- In Horovod, the resnet_ctl_imagenet_main.py `--horovod_hierarchical_allreduce` option must be set.
- In HCCL using Libfabric, follow the steps detailed in [Scale-Out via Host-NIC over OFI](https://docs.habana.ai/en/latest/API_Reference_Guides/HCCL_APIs/Scale_Out_via_Host_NIC.html#scale-out-host-nic-ofi).

**Note:** Make sure to add any additional environment variables to the above mpirun command (for example, `-x <ENV_VARIABLE>`).

**Run training on 16 HPUs - tf.distribute:**

- 16 HPUs on 2 servers, batch 256, 40 epochs, BF16 precision, LARS:
    - `-H`: Set this to a comma-separated list of host IP addresses. Make sure to modify IP addresses below to match your system.
    - `--mca btl_tcp_if_include`: Provide network interface associated with IP address. More details can be found in the [Open MPI documentation](https://www.open-mpi.org/faq/?category=tcp#tcp-selection). If you get mpirun `btl_tcp_if_include` errors, try un-setting this environment variable and let the training script automatically detect the network interface associated with the host IP address.
    - `HCCL_SOCKET_IFNAME`: Defines the prefix of the network interface name that is used for HCCL sideband TCP communication. If not set, the first network interface with a name that does not start with lo or docker will be used.

   **Note:**
   - To run multi-server training over host NICs (required for AWS users), set the environment variable `HCCL_OVER_OFI=1`. Make sure to add the variable to the mpirun command `-x HCCL_OVER_OFI=1`.
   - `$MPI_ROOT` environment variable is set automatically during Setup. See [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) for details.

    ```bash
    # This environment variable is needed for multi-node training with tf.distribute.
    # Set this to be a comma-separated string of host IP addresses, e.g.:
    export MULTI_HLS_IPS=192.10.100.174,10.10.100.101 

    mpirun \
     --allow-run-as-root --mca plm_rsh_args -p3022 \
     --bind-to core \
     --map-by socket:PE=6 -np 16 \
     --mca btl_tcp_if_include <interface_name> \
     --tag-output --merge-stderr-to-stdout --prefix $MPI_ROOT \
     -H 192.10.100.174:8,10.10.100.101:8 \
     -x GC_KERNEL_PATH -x HABANA_LOGS \
     -x PYTHONPATH -x MULTI_HLS_IPS \
     -x HCCL_SOCKET_IFNAME=<interface_name> \
       $PYTHON resnet_ctl_imagenet_main.py \
        -dt bf16 \
        -dlit bf16 \
        -te 40 \
        -ebe 40 \
        -bs 4096 \
        --distribution_strategy hpu \
        --data_dir /data/tensorflow/imagenet/tf_records \
        --optimizer LARS \
        --use_tf_while_loop=False \
        --base_learning_rate 9.5 \
        --warmup_epochs 3 \
        --lr_schedule polynomial \
        --label_smoothing 0.1 \
        --weight_decay 0.0001 \
        --single_l2_loss_op \
        --enable_tensorboard
    ```

  Note:
  - Currently, `experimental_preloading` (enabled by default) is required to run `distribution_strategy=hpu`.
  - Unlike Horovod, global batch size must be specified for tf.distribute. In this case 256 * 16 workers = 4096.

**Top performance examples:**

**NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration).

- ResNet50 training on 8 HPUs 1 server - Horovod:

    ```bash 
    mpirun \
     --allow-run-as-root --bind-to core \
     -np 8 --map-by socket:PE=6 --merge-stderr-to-stdout \
       $PYTHON resnet_ctl_imagenet_main.py \
         --dtype bf16 \
         --data_loader_image_type bf16 \
         --use_horovod \
         -te 40 \
         -ebe 40 \
         --steps_per_loop 1000 \
         -bs 256 \
         --optimizer LARS \
         --base_learning_rate 9.5 \
         --warmup_epochs 3 \
         --lr_schedule polynomial \
         --label_smoothing 0.1 \
         --weight_decay 0.0001 \
         --single_l2_loss_op \
         --data_dir /data/tensorflow/imagenet/tf_records \
         --enable_tensorboard
    ```


- ResNet50 training on 32 HPUs 4 servers - Horovod:
    - `-H`: Set this to a comma-separated list of host IP addresses. Make sure to modify IP addresses below to match your system.
    - `--mca btl_tcp_if_include`: Provide network interface associated with IP address. More details can be found in the [Open MPI documentation](https://www.open-mpi.org/faq/?category=tcp#tcp-selection). If you get mpirun `btl_tcp_if_include` errors, try un-setting this environment variable and let the training script automatically detect the network interface associated with the host IP address.
    - `HCCL_SOCKET_IFNAME`: Defines the prefix of the network interface name that is used for HCCL sideband TCP communication. If not set, the first network interface with a name that does not start with lo or docker will be used.

    **NOTE:**
    - mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration).
    - `$MPI_ROOT` environment variable is set automatically during Setup. See [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) for details.

    ```bash 
    mpirun \
     --allow-run-as-root --mca plm_rsh_args -p3022 \
     --bind-to core \
     --map-by socket:PE=6 -np 32 \
     --mca btl_tcp_if_include <interface_name> \
     --tag-output \
     --merge-stderr-to-stdout --prefix $MPI_ROOT \
     -H 192.10.100.174:8,10.10.100.101:8,10.10.100.102:8,10.10.100.203:8 \
     -x GC_KERNEL_PATH -x HABANA_LOGS \
     -x PYTHONPATH -x HCCL_SOCKET_IFNAME=<interface_name> \
       $PYTHON resnet_ctl_imagenet_main.py \
        -dt bf16 \
        -dlit bf16 \
        -te 40 \
        -ebe 40 \
        --steps_per_loop 1000 \
        -bs 256 \
        --use_horovod \
        --data_dir /data/tensorflow/imagenet/tf_records \
        --optimizer LARS \
        --base_learning_rate 9.5 \
        --warmup_epochs 3 \
        --lr_schedule polynomial \
        --label_smoothing 0.1 \
        --weight_decay 0.0001 \
        --single_l2_loss_op \
        --enable_tensorboard
     ```

## Pre-trained Model
TensorFlow ResNet50 is trained on Habana Gaudi cards and the checkpoint and saved model files are created. You can use them for fine-tuning or transfer learning tasks with your own datasets. To download the checkpoint or saved model files, please refer to [Habana Catalog](https://developer.habana.ai/catalog/resnet-for-tensorflow/) to obtain the URL.

For more information on transfer learning with ResNet50, please check the [ResNet50 Transfer Learning Demo](transfer_learning_demo/README.md) file available in the `transfer_learning_demo` folder.


## Profile

To run training on 1 HPU with profiler:

```bash
$PYTHON resnet_ctl_imagenet_main.py -bs 128 --optimizer LARS --base_learning_rate 9.5 --warmup_epochs 3 --lr_schedule polynomial --label_smoothing 0.1 --weight_decay 0.0001 --single_l2_loss_op \
--data_dir /data/tensorflow/imagenet/tf_records --train_steps 100 --steps_per_loop 3 --profile_steps 51,53
```

**Note:**

- `steps_per_loop` default value is 200.
- `train_steps`, `profile_step`, and `steps_per_loop` flags should be coordinated.
For example:

```bash
$PYTHON resnet_ctl_imagenet_main.py -bs 128 --optimizer LARS --base_learning_rate 9.5 --warmup_epochs 3 --lr_schedule polynomial --label_smoothing 0.1 --weight_decay 0.0001 --single_l2_loss_op \
--data_dir /data/tensorflow/imagenet/tf_records --train_steps 100 --steps_per_loop 3 --profile_steps 51,54
```
**The above command will not produce any profile files.**

## Supported Configuration

| Device | SynapseAI Version | TensorFlow Version(s)  |
|:------:|:-----------------:|:-----:|
| Gaudi  | 1.7.1             | 2.10.1 |
| Gaudi  | 1.7.1             | 2.8.4 |
| Gaudi2 | 1.7.1             | 2.10.1 |
| Gaudi2 | 1.7.1             | 2.8.4 |

## Changelog

### 1.8.0
- Added Media API implementation for image processing on Gaudi2

### 1.7.0
- Added TimeToTrain callback for dumping evaluation timestamps

### 1.4.1
- Added support for image processing acceleration on Gaudi2 (JPEG format only).

### 1.4.0
- References to custom demo script were replaced by community entry points in README.
- Added Habana Media Loader to allow future performance improvements.
- Fixed Eager mode enabling in ResNet Keras.
- Removed setup_jemalloc from demo_resnet_keras.py.

### 1.3.0
- Import horovod-fork package directly instead of using Model-References' TensorFlow.common.horovod_helpers; wrapped horovod import with a try-catch block so that the user is not required to install this library when the model is being run on a single card
- Moved files from TensorFlow/common/ and TensorFlow/utils/ to model dir, align imports
- Set up BF16 conversion pass using a config from habana-tensorflow package instead of recipe json file.
- Removed usage of HBN_TF_REGISTER_DATASETOPS as prerequisite for experimantal preloading.
- Updated requirements.txt to align with TF 2.8.0.
- Updated imagenet and align to new naming convention (img_train->train, img_val->validation).

### 1.2.0
- distribution_utils moved from main dir `/TensorFlow/utils/` to `resnet_keras` script dir
