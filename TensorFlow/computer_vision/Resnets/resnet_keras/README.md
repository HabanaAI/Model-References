# ResNet50 Keras model

This repository provides a script and recipe to train the ResNet keras model to achieve state-of-the-art accuracy, and is tested and maintained by Habana. Please visit [this page](https://developer.habana.ai/resources/habana-training-models/#performance) for performance information.

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents

* [Model-References](../../../../README.md)
* [Model Overview](#model-overview)
* [Setup](#setup)
* [Media Loading Acceleration](#media-loading-acceleration)
* [Training](#training)
* [Pre-trained Checkpoint Files](#pre-trained-checkpoint-files)
* [Profile](#profile)
* [Supported Configuration](#supported-configuration)
* [Changelog](#changelog)

## Model Overview
ResNet keras model is a modified version of the original [TensorFlow model garden](https://github.com/tensorflow/models/tree/master/official/legacy/image_classification/resnet) model. It uses a custom training loop, supports 50 layers and can work both with SGD and LARS optimizers.

## Setup
Please follow the instructions given in the following link for setting up the
environment including the `$PYTHON` and `$MPI_ROOT` environment variables: [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html).
This guide will walk you through the process of setting up your system to run
the model on Gaudi.

## Media Loading Acceleration
Gaudi2 offers dedicated hardware engine for Media Loading operations, such as JPEG decoding and data augmentation. This can be leveraged in ResNet model to decrease CPU usage and exceed performance limitation when data processing on CPU is a bottleneck.
Currently the only supported file format is JPEG (TFRecord support is in plan).

ResNet automatically uses hardware Media Loading Acceleration unless:
1. Training is done on First-gen Gaudi Processors (they don't have dedicated hardware)
2. User doesn't have hpu_media_loader Python package installed
3. User has set FORCE_HABANA_IMAGENET_LOADER_FALLBACK environment variable
4. User hasn't provided location of ImageNet dataset containing JPEGs (--jpeg_data_dir parameter)

In the above cases media processing will be done on CPU.

### Training data

The ResNet50 Keras script operates on ImageNet 1k, a widely popular image classification dataset from the ILSVRC challenge.
In order to obtain the dataset, follow these steps:
1. Sign up with http://image-net.org/download-images and acquire the rights to download original images
2. Follow the link to the 2012 ILSVRC and download `ILSVRC2012_img_val.tar` and `ILSVRC2012_img_train.tar`.
3. Use the commands below - they will prepare the dataset under `/data/tensorflow/imagenet/tf_records`.
 This is the default data_dir for the training script.
 In `/data/tensorflow/imagenet/train` and `/data/tensorflow/imagenet/val` directories original JPEG files will stay
 and can be used for Media Loading Acceleration on Gaudi2.
 See examples with --data_dir and --jpeg_data_dir parameters for details.


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

In the docker container, go to the resnet_keras directory:
```bash
cd /root/Model-References/TensorFlow/computer_vision/Resnets/resnet_keras
```
Install required packages using pip:
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

This directory contains modified community scripts. The modifications include changing the default values of parameters, setting environment variables, changing include paths, etc. The changes are described in the license headers of the respective files.

**In order to see all possible arguments to resnet_ctl_imagenet_main.py, run**
```bash
$PYTHON resnet_ctl_imagenet_main.py --helpfull
```

### Using LARS optimizer
Using LARS optimizer usually requires changing the default values of some hyperparameters and should be manually set for resnet_ctl_imagenet_main.py. The recommended parameters together with their default values are presented below:

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

- The following commands shows the example to set LARS optimizer
  ```bash
  $PYTHON resnet_ctl_imagenet_main.py --optimizer LARS --base_learning_rate 9.5 --warmup_epochs 3 --lr_schedule polynomial --label_smoothing 0.1 --weight_decay 0.0001 --single_l2_loss_op
  ```

### Training Examples
Note: When using the resnet_ctl_imagenet_main.py for training, there are changes to the default values of parameters, setting environment variables, changing include paths, etc and these changes are described in the license headers of the file.

**Run training on 1 HPU**

- ResNet50, 1 HPU, batch 256, 90 epochs, bf16 precision, SGD
    ```bash
    $PYTHON resnet_ctl_imagenet_main.py -dt bf16 -dlit bf16 --train_epochs 90 -bs 256 --data_dir /data/tensorflow/imagenet/tf_records
    ```
- ResNet50, 1 HPU, batch 256, 40 epochs, bf16 precision, LARS
    ```bash
    $PYTHON resnet_ctl_imagenet_main.py -bs 256 --train_epochs 40 -dt bf16 --data_dir /data/tensorflow/imagenet/tf_records \
    --optimizer LARS --base_learning_rate 9.5 --warmup_epochs 3 --lr_schedule polynomial --label_smoothing 0.1 --weight_decay 0.0001 \
    --single_l2_loss_op
    ```
- ResNet50, 1 HPU, batch 256, 90 epochs, bf16 precision, SGD, **Gaudi2 with media acceleration**
    ```bash
    $PYTHON resnet_ctl_imagenet_main.py -dt bf16 -dlit bf16 --train_epochs 90 -bs 256 --jpeg_data_dir /data/tensorflow/imagenet
    ```
- ResNet50, 1 HPU, batch 256, 40 epochs, bf16 precision, LARS, **Gaudi2 with media acceleration**
    ```bash
    $PYTHON resnet_ctl_imagenet_main.py -bs 256 --train_epochs 40 -dt bf16 --jpeg_data_dir /data/tensorflow/imagenet \
    --optimizer LARS --base_learning_rate 9.5 --warmup_epochs 3 --lr_schedule polynomial --label_smoothing 0.1 --weight_decay 0.0001 \
    --single_l2_loss_op
    ```

**Run training on 8 HPU - Horovod**

**NOTE:** mpirun map-by PE attribute value may vary on your setup. Please refer to the instructions on [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration) for calculation.

- ResNet50, 8 HPU on 1 server, batch 256, 40 epochs, bf16 precision, LARS

    ```bash
    mpirun --allow-run-as-root --bind-to core -np 8 --map-by socket:PE=7 --merge-stderr-to-stdout \
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
        --data_dir /data/tensorflow/imagenet/tf_records
    ```

- ResNet50, 8 HPU on 1 server, batch 256, 40 epochs, bf16 precision, LARS, Gaudi2 with media acceleration

    ```bash
    mpirun --allow-run-as-root --bind-to core -np 8 --map-by socket:PE=7 --merge-stderr-to-stdout \
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
        --jpeg_data_dir /data/tensorflow/imagenet
    ```
**Run training on 8 HPU - tf.distribute**

**NOTE:** mpirun map-by PE attribute value may vary on your setup. Please refer to the instructions on [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration) for calculation.

- ResNet50, 8 HPU on 1 server, batch 256, 40 epochs, bf16 precision, LARS

    ```bash
    mpirun --allow-run-as-root --bind-to core -np 8 --map-by socket:PE=7 --merge-stderr-to-stdout \
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
        --data_dir /data/tensorflow/imagenet/tf_records
    ```

- ResNet50, 8 HPU on 1 server, batch 256, 40 epochs, bf16 precision, LARS, Gaudi2 with media acceleration

    ```bash
    mpirun --allow-run-as-root --bind-to core -np 8 --map-by socket:PE=7 --merge-stderr-to-stdout \
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
        --jpeg_data_dir /data/tensorflow/imagenet
    ```

   Note:
   - Currently experimental_preloading (enabled by default) is required to run hpu distribution_strategy.
   - Unlike Horovod, global batch size must be specified for tf.distribute. E.g. in this case 256 * 8 workers = 2048

### ***Multi Server Training***

The following directions are generalized for use of Multi-Chassis:

1. **Follow [Setup](#setup) above on all servers**
2. **Configure ssh between servers (Inside all dockers)**:
Do the following in all servers' docker containers:
```
mkdir ~/.ssh
cd ~/.ssh
ssh-keygen -t rsa -b 4096
```
Copy id_rsa.pub contents from every server to every authorized_keys (all public keys need to be in all hosts' authorized_keys):
```
cat id_rsa.pub > authorized_keys
vi authorized_keys
```
Copy the contents from inside to other system
Paste both hosts public keys in both host’s “authorized_keys” file

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

**NOTE:** mpirun map-by PE attribute value may vary on your setup. Please refer to the instructions on [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration) for calculation.

3. **Run training on 16 HPU - multi-server - Horovod**

  - ResNet50, 16 HPU on 2 boxes, batch 256, 40 epochs, bf16 precision, LARS
    - NOTE: MODIFY IP ADDRESS BELOW TO MATCH YOUR SYSTEM.
    - **`-H`**: set this to a comma-separated list of host IP addresses
    - **`--mca btl_tcp_if_include`**: Provide network interface associated with IP address. More details: [Open MPI documentation](https://www.open-mpi.org/faq/?category=tcp#tcp-selection). If you get mpirun `btl_tcp_if_include` errors, try un-setting this environment variable and let the training script automatically detect the network interface associated with the host IP address.
    - **`HCCL_SOCKET_IFNAME`**: HCCL_SOCKET_IFNAME defines the prefix of the network interface name that is used for HCCL sideband TCP communication. If not set, the first network interface with a name that does not start with lo or docker will be used.

    ```bash
    mpirun \
     --allow-run-as-root --mca plm_rsh_args -p3022 \
     --bind-to core \
     --map-by socket:PE=7 -np 16 \
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
          --single_l2_loss_op
    ```

   *Note that to run multi-server training over host NICs (required for AWS users), one of the following variant must take place:*

- In Horovod (*HorovodHierarchicalAllreduce*): The resnet_ctl_imagenet_main.py `--horovod_hierarchical_allreduce` option must be set.
- In HCCL using raw TCP sockets: environment variable `HCCL_OVER_TCP=1` must be set.
- In HCCL using Libfabric: follow [Scale-Out via Host-NIC over OFI](https://docs.habana.ai/en/latest/API_Reference_Guides/HCCL_APIs/Scale_Out_via_Host_NIC.html#scale-out-host-nic-ofi).

  *Remember to add any additional env variables to above mpirun command e.g. (`-x HCCL_OVER_TCP`)*

**Run training on 16 HPU - multi-server - tf.distribute**

- ResNet50, 16 HPU on 2 boxes, batch 256, 40 epochs, bf16 precision, LARS
    - NOTE: MODIFY IP ADDRESS BELOW TO MATCH YOUR SYSTEM.
    - **`-H`**: set this to a comma-separated list of host IP addresses
    - **`--mca btl_tcp_if_include`**: Provide network interface associated with IP address. More details: [Open MPI documentation](https://www.open-mpi.org/faq/?category=tcp#tcp-selection). If you get mpirun `btl_tcp_if_include` errors, try un-setting this environment variable and let the training script automatically detect the network interface associated with the host IP address.
    - **`HCCL_SOCKET_IFNAME`**: HCCL_SOCKET_IFNAME defines the prefix of the network interface name that is used for HCCL sideband TCP communication. If not set, the first network interface with a name that does not start with lo or docker will be used.

    ```bash
    # This environment variable is needed for multi-node training with tf.distribute.
    # Set this to be a comma-separated string of host IP addresses, e.g.:
    export MULTI_HLS_IPS=192.10.100.174,10.10.100.101

    mpirun \
     --allow-run-as-root --mca plm_rsh_args -p3022 \
     --bind-to core \
     --map-by socket:PE=7 -np 16 \
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
        --single_l2_loss_op
    ```

   *Note that to run multi-server training over host NICs (required for AWS users):*

- Environment variable `HCCL_OVER_TCP=1` must be set. Remember to add the variable to mpirun command `-x HCCL_OVER_TCP=1`

  Note:
  - Currently experimental_preloading (enabled by default) is required to run hpu distribution_strategy.
  - Unlike Horovod, global batch size must be specified for tf.distribute. E.g. in this case 256 * 16 workers = 4096

**Top performance examples**

**NOTE:** mpirun map-by PE attribute value may vary on your setup. Please refer to the instructions on [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration) for calculation.

- Resnet 50 training on 8 HPU 1 box - Horovod

    ```bash
    mpirun \
     --allow-run-as-root --bind-to core \
     -np 8 --map-by socket:PE=7 --merge-stderr-to-stdout \
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
         --data_dir /data/tensorflow/imagenet/tf_records
    ```


- Resnet 50 training on 32 HPU 4 boxes - Horovod
    - NOTE: MODIFY IP ADDRESS BELOW TO MATCH YOUR SYSTEM.
    - **`-H`**: set this to a comma-separated list of host IP addresses
    - **`--mca btl_tcp_if_include`**: Provide network interface associated with IP address. More details: [Open MPI documentation](https://www.open-mpi.org/faq/?category=tcp#tcp-selection). If you get mpirun `btl_tcp_if_include` errors, try un-setting this environment variable and let the training script automatically detect the network interface associated with the host IP address.
    - **`HCCL_SOCKET_IFNAME`**: HCCL_SOCKET_IFNAME defines the prefix of the network interface name that is used for HCCL sideband TCP communication. If not set, the first network interface with a name that does not start with lo or docker will be used.

    **NOTE:** mpirun map-by PE attribute value may vary on your setup. Please refer to the instructions on [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration) for calculation.

    ```bash
    mpirun \
     --allow-run-as-root --mca plm_rsh_args -p3022 \
     --bind-to core \
     --map-by socket:PE=7 -np 32 \
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
        --single_l2_loss_op
     ```

## Pre-trained Checkpoint Files
TensorFlow ResNet50 is trained on Habana Gaudi cards and the checkpoint files are created. The users can use the checkpoints for fine-tuning or transfer learning tasks with their own datasets. To download the pre-trained checkpoint files, please refer to [ResNet50 Catalog](https://developer.habana.ai/catalog/resnet-for-tensorflow/), select the checkpoint to obtain its URL, and run the following commands:

```bash
cd Model-References/TensorFlow/computer_vision/Resnets/resnet_keras
mkdir pretrained_checkpoint
wget </url/of/pretrained_checkpoint.tar.gz>
tar -xvf <pretrained_checkpoint.tar.gz> -C pretrained_checkpoint && rm <pretrained_checkpoint.tar.gz>
```

If you want to get more information about transfer learning with ResNet50, please check the [transfer learning README](transfer_learning_demo/README.md) file, available in transfer_learning_demo folder.

## Profile

**Run training on 1 HPU with profiler**

```bash
$PYTHON resnet_ctl_imagenet_main.py -bs 128 --optimizer LARS --base_learning_rate 9.5 --warmup_epochs 3 --lr_schedule polynomial --label_smoothing 0.1 --weight_decay 0.0001 --single_l2_loss_op \
--data_dir /data/tensorflow/imagenet/tf_records --train_steps 100 --steps_per_loop 3 --profile_steps 51,53
```

*Note:*

- steps_per_loop default value is 200.
- train_steps, profile_step, and steps_per_loop flags should be coordinated.
For example:

```bash
$PYTHON resnet_ctl_imagenet_main.py -bs 128 --optimizer LARS --base_learning_rate 9.5 --warmup_epochs 3 --lr_schedule polynomial --label_smoothing 0.1 --weight_decay 0.0001 --single_l2_loss_op \
--data_dir /data/tensorflow/imagenet/tf_records --train_steps 100 --steps_per_loop 3 --profile_steps 51,54
```
**Will not produce any profile files.**

## Supported Configuration

| Device | SynapseAI Version | TensorFlow Version(s)  |
|:------:|:-----------------:|:-----:|
| Gaudi  | 1.5.0             | 2.9.1 |
| Gaudi  | 1.5.0             | 2.8.2 |
| Gaudi2 | 1.5.0             | 2.9.1 |
| Gaudi2 | 1.5.0             | 2.8.2 |

## Changelog
### 1.4.1
- Added support for image processing acceleration on Gaudi2 (JPEG format only)

### 1.4.0
- References to custom demo script were replaced by community entry points in README.
- Added Habana Media Loader to allow future performance improvement
- Fixed Eager mode enabling in Resnet Keras
- removed setup_jemalloc from demo_resnet_keras.py

### 1.3.0
- Import horovod-fork package directly instead of using Model-References' TensorFlow.common.horovod_helpers; wrapped horovod import with a try-catch block so that the user is not required to install this library when the model is being run on a single card
- move files from TensorFlow/common/ and TensorFlow/utils/ to model dir, align imports
- setup BF16 conversion pass using a config from habana-tensorflow package instead of recipe json file
- remove usage of HBN_TF_REGISTER_DATASETOPS as prerequisite for experimantal preloading
- update requirements.txt to align with TF 2.8.0
- update imagenet and align to new naming convention (img_train->train, img_val->validation)

### 1.2.0
- distribution_utils moved from main dir `/TensorFlow/utils/` to `resnet_keras` script dir
