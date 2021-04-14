# ResNet50 Keras model

This repository provides a script and recipe to train the ResNet keras model to achieve state-of-the-art accuracy, and is tested and maintained by Habana.

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents

* [Model Overview](#model-overview)
* [Setup](#setup)
* [Training](#training)

## Model Overview
ResNet keras model is a modified version of the original [TensorFlow model garden](https://github.com/tensorflow/models/tree/master/official/vision/image_classification/resnet) model. It uses custom training loop. It supports layers 50. It supports both SGD and LARS optimizers.

## Setup

### Install Drivers
Follow steps in the [Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) to install the driver.

<br />

### Install container runtime
<details>
<summary>Ubuntu distributions</summary>

### Setup package fetching
1. Download and install the public key:
```
curl -X GET https://vault.habana.ai/artifactory/api/gpg/key/public | sudo apt-key add -
```
2. Create an apt source file /etc/apt/sources.list.d/artifactory.list.
3. Add the following content to the file:
```
deb https://vault.habana.ai/artifactory/debian focal main
```
4. Update Debian cache:
```
sudo dpkg --configure -a
sudo apt-get update
```
### Install habana-container-runtime:
Install the `habana-container-runtime` package:
```
sudo apt install -y habanalabs-container-runtime=0.14.0-420
```
### Docker Engine setup

To register the `habana` runtime, use the method below that is best suited to your environment.
You might need to merge the new argument with your existing configuration.

#### Daemon configuration file
```bash
sudo tee /etc/docker/daemon.json <<EOF
{
    "runtimes": {
        "habana": {
            "path": "/usr/bin/habana-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF
sudo systemctl restart docker
```

You can optionally reconfigure the default runtime by adding the following to `/etc/docker/daemon.json`:
```
"default-runtime": "habana"
```
</details>

<details>
<summary>CentOS distributions</summary>

### Setup package fetching
1. Create /etc/yum.repos.d/Habana-Vault.repo.
2. Add the following content to the file:
```
[vault]

name=Habana Vault

baseurl=https://vault.habana.ai/artifactory/centos7

enabled=1

gpgcheck=0

gpgkey=https://vault.habana.ai/artifactory/centos7/repodata/repomod.xml.key

repo_gpgcheck=0
```
3. Update YUM cache by running the following command:
```
sudo yum makecache
```
4. Verify correct binding by running the following command:
```
yum search habana
```
This will search for and list all packages with the word Habana.

### Install habana-container-runtime:
Install the `habana-container-runtime` package:
```
sudo yum install habanalabs-container-runtime-0.14.0-420* -y
```
### Docker Engine setup

To register the `habana` runtime, use the method below that is best suited to your environment.
You might need to merge the new argument with your existing configuration.

#### Daemon configuration file
```bash
sudo tee /etc/docker/daemon.json <<EOF
{
    "runtimes": {
        "habana": {
            "path": "/usr/bin/habana-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF
sudo systemctl restart docker
```

You can optionally reconfigure the default runtime by adding the following to `/etc/docker/daemon.json`:
```
"default-runtime": "habana"
```
</details>

<details>
<summary>Amazon linux distributions</summary>

### Setup package fetching
1. Create /etc/yum.repos.d/Habana-Vault.repo.
2. Add the following content to the file:
```
[vault]

name=Habana Vault

baseurl=https://vault.habana.ai/artifactory/AmazonLinux2

enabled=1

gpgcheck=0

gpgkey=https://vault.habana.ai/artifactory/AmazonLinux2/repodata/repomod.xml.key

repo_gpgcheck=0
```
3. Update YUM cache by running the following command:
```
sudo yum makecache
```
4. Verify correct binding by running the following command:
```
yum search habana
```
This will search for and list all packages with the word Habana.

### Install habana-container-runtime:
Install the `habana-container-runtime` package:
```
sudo yum install habanalabs-container-runtime-0.14.0-420* -y
```
### Docker Engine setup

To register the `habana` runtime, use the method below that is best suited to your environment.
You might need to merge the new argument with your existing configuration.

#### Daemon configuration file
```bash
sudo tee /etc/docker/daemon.json <<EOF
{
    "runtimes": {
        "habana": {
            "path": "/usr/bin/habana-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF
sudo systemctl restart docker
```

You can optionally reconfigure the default runtime by adding the following to `/etc/docker/daemon.json`:
```
"default-runtime": "habana"
```
</details>
<br />

### Training data

The ResNet50 Keras script operates on ImageNet 1k, a widely popular image
classification dataset from the ILSVRC challenge.

1. Sign up with http://image-net.org/download-images and acquire the rights to download original images
2. Follow the link to the 2012 ILSVRC
 and download `ILSVRC2012_img_val.tar` and `ILSVRC2012_img_train.tar` to this directory.
3. Ensure Python3 and the following Python packages are installed: TensorFlow 2.2 and `absl-py`.

```
mkdir /path/to/imagenet_data
export IMAGENET_HOME=/path/to/imagenet_data
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

The above commands will create `tf_record` files in `/path/to/imagenet_data/tf_records`

## Training

### Get the Habana Tensorflow docker image:

| TensorFlow version  | Ubuntu version        |                              Command Line                                                                                |
| ------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| TensorFlow 2.2      | Ubuntu 18.04          | `docker pull vault.habana.ai/gaudi-docker/0.14.0/ubuntu18.04/habanalabs/tensorflow-installer-tf-cpu-2.2.2:0.14.0-420`    |
| TensorFlow 2.4      | Ubuntu 18.04          | `docker pull vault.habana.ai/gaudi-docker/0.14.0/ubuntu18.04/habanalabs/tensorflow-installer-tf-cpu-2.4.1:0.14.0-420`    |
| TensorFlow 2.2      | Ubuntu 20.04          | `docker pull vault.habana.ai/gaudi-docker/0.14.0/ubuntu20.04/habanalabs/tensorflow-installer-tf-cpu-2.2.2:0.14.0-420`    |
| TensorFlow 2.4      | Ubuntu 20.04          | `docker pull vault.habana.ai/gaudi-docker/0.14.0/ubuntu20.04/habanalabs/tensorflow-installer-tf-cpu-2.4.1:0.14.0-420`    |

Note: For the sake of simplicity the Readme instructions follows the assumption that docker image is tf-cpu-2.2.2:0.14.0-420 and Ubuntu version is 18.04.

### 1. Download docker
```
docker pull vault.habana.ai/gaudi-docker/0.14.0/ubuntu18.04/habanalabs/tensorflow-installer-tf-cpu-2.2.2:0.14.0-420
```

### 2. Run docker

**NOTE:** This assumes the Imagenet dataset is under /opt/datasets/imagenet on the host. Modify accordingly.

```
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice  -v /sys/kernel/debug:/sys/kernel/debug -v /opt/datasets/imagenet:/root/tensorflow_datasets/imagenet --net=host vault.habana.ai/gaudi-docker/0.14.0/ubuntu18.04/habanalabs/tensorflow-installer-tf-cpu-2.2.2:0.14.0-420
```

OPTIONAL with mounted shared folder to transfer files out of docker:

```
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice  -v /sys/kernel/debug:/sys/kernel/debug -v ~/shared:/root/shared -v /opt/datasets/imagenet:/root/tensorflow_datasets/imagenet --net=host vault.habana.ai/gaudi-docker/0.14.0/ubuntu18.04/habanalabs/tensorflow-installer-tf-cpu-2.2.2:0.14.0-420
```

### 3. Clone the repository and go to resnet_keras directory:

```
git clone https://github.com/HabanaAI/Model-References.git
cd Model-References/TensorFlow/computer_vision/Resnets/resnet_keras
```

Note: If the repository is not in the PYTHONPATH, make sure you update it.
```
export PYTHONPATH=/path/to/Model-References:$PYTHONPATH
```

### Training Examples

The script can be run with python model runner.

You can run the following script: **Model-References/TensorFlow/habana_model_runner.py** which accepts two arguments:
- --model *model_name*
- --hb_config *path_to_yaml_config_file*

Example of config files can be found in **Model-References/TensorFlow/computer_vision/Resnets/resnet_keras/hb_configs/**.
Note: Config yaml files only for resnet_keras SGD. LARS should use bash script as mentioned in below sections.

You can use these scripts as such:
> cd Model-References/TensorFlow/computer_vision/Resnets/resnet_keras
  >
  > python3 ../../../../central/habana_model_runner.py --model resnet_keras --hb_config *path_to_yaml_config_file*
  >
  > **Example:**
  >
  > python3 ../../../../central/habana_model_runner.py --model resnet_keras --hb_config resnet_keras_default.yaml

**Single Card:**

- ResNet 50 , 1-node, BS=256, e=90, Precision=bf16, SGD, Ubuntu18
    ```python
    python3 ../../../../central/habana_model_runner.py --framework tensorflow --model resnet_keras --hb_config hb_configs/u18_single_node_resnet50_keras_bf16.yaml
    ```
- ResNet 50 , 1-node, BS=128, e=90, Precision=fp32, SGD, Ubuntu18
    ```python
    python3 ../../../../central/habana_model_runner.py --framework tensorflow --model resnet_keras --hb_config hb_configs/u18_single_node_resnet50_keras_fp32.yaml
    ```
Note:
- The parameters in yaml file can be altered or new yaml files can be created to train model on different values of the parameter for single card.
- A list of other yaml files with a description of parameter values passed via them for training in mentioned in Default configurations section below.

Note:
- Resnet 50 LARS should be run with demo bash script as described below.

**Running Single Card with demo_keras_resnet bash script**

Resnet50 with LARS optimizer should be executed using demo_keras_resnet.

This script uses Keras, a high-level neural networks library running on top of Tensorflow for training and evaluating the ResNet model:
```bash
usage: ./demo_keras_resnet [arguments]
mandatory arguments:
  -d <data_type>,    --dtype <data_type>          Data type, possible values: fp32, bf16
optional arguments:
  -b <batch_size>,   --batch-size <batch_size>    Batch size, default 128 for fp32, 256 for bf16"
  -e <epochs>,       --epochs <epochs>            Number of epochs, default to 1"
  -a <data_dir>,     --data-dir <data_dir>        Data dir, defaults to /software/data/tf/data/imagenet/tf_records/
  -m <model_dir>,    --model-dir <model_dir>      Model dir, defaults to $HOME/tmp/resnet/
  -o,                --use-horovod                Use horovod for training
  -s <steps>,        --steps <steps>              Max train steps
  -n,                --no-eval                    Don't do evaluation
  -v <steps>,        --display-steps <steps>      How often display step status
  -l <steps>,        --steps-per-loop             Number of steps per training loop. Will be capped at steps per epoch.
  -c,                --enable-checkpoint          Whether to enable a checkpoint callback and export the savedmodel.
  -r                 --recover                    If crashed restart training from last checkpoint. Requires -s to be set
  -k                 --no-experimental-preloading Disables support for 'data.experimental.prefetch_to_device' TensorFlow operator. If not set:
                                                  - loads extension dynpatch_prf_remote_call.so (via LD_PRELOAD)
                                                  - sets environment variable HBN_TF_REGISTER_DATASETOPS to 1
                                                  - this feature is experimental and works only with single node
example:
  ./demo_keras_resnet -d bf16
  ./demo_keras_resnet -d fp32
  ./demo_keras_resnet -d fp -rs 101
  ./demo_keras_resnet -d bf16 -s 1000
  ./demo_keras_resnet -d bf16 -s 1000 -l 50
  ./demo_keras_resnet -d bf16 -e 9"
  ./demo_keras_resnet -d fp32 -e 9 -b 128 -a home/user1/tensorflow_datasets/imagenet/tf_records/ -m /home/user1/tensorflow-training/demo/ck_81_080_450_bs128
```

To get better accuracy in singe node training it is needed to change base learninig rate value from 9.5 (dedicated for multinode run) to 2.5
It can be done by chaning resnet_keras.sh script.
```bash
    TRAIN_PARAMS="${TRAIN_PARAMS} --base_learning_rate=9.5"
```

**Example:**

- ResNet 50 , 1-node, BS=256, e=40, Precision=bf16, LARS
    ```bash
    /root/Model-References/TensorFlow/computer_vision/Resnets/demo_keras_resnet --data-dir <path_to_tf_records> -d bf16 --batch-size 256 --display-steps 100 --epochs 40 -c
    ```

**8 Card:**

Note: 
- The 8 card training for resnet_keras uses bash script currently.

#### The `demo_keras_resnet_hvd.sh` script

- ResNet 50 , 8-node, BS=256, e=90, Precision=bf16, SGD
    ```bash
    export IMAGENET_DIR=/path/to/tensorflow_datasets/imagenet/
    export BATCH_SIZE=256
    export USE_LARS_OPTIMIZER=0
    ./demo_keras_resnet_hvd.sh
    ```
- ResNet 50 , 8-node, BS=128, e=40, Precision=bf16, LARS

    ```bash
    export IMAGENET_DIR=/path/to/tensorflow_datasets/imagenet/
    export BATCH_SIZE=128
    export USE_LARS_OPTIMIZER=1
    ./demo_keras_resnet_hvd.sh
    ```

### Default configurations

The default training configurations can be found in the **Model-References/TensorFlow/computer_vision/Resnets/resnet_keras/hb_configs/**

### Single node

| Training configuration | File name |
| --------------------------------------------------------- | ---------------------------------------- |
| 1-node, BS=256, e=90, Precision=bf16, SGD, Ubuntu18       | u18_single_node_resnet50_keras_bf16.yaml |
| 1-node, BS=128, e=90, Precision=fp32, SGD, Ubuntu18       | u18_single_node_resnet50_keras_fp32.yaml |
| 1-node, BS=256, e=90, Precision=bf16, SGD, Ubuntu20       | u20_single_node_resnet50_keras_bf16.yaml |
| 1-node, BS=256, e=90, Precision=fp32, SGD, Ubuntu20       | u20_single_node_resnet50_keras_fp32.yaml |

### Configuration file parameters:

The training configuration file allows to define the values of all of the parameters supported by source training script.

Default configuration files define following parameters:

| Parameter                    |  Type  | Description                                                           |
| ---------------------------- | ------ | --------------------------------------------------------------------- |
| data_dir                     | string | Location of the input datasets                                        |
| model_dir                    | string | Lcation of the model checkpoint files                                 |
| batch_size                   |  int   | Training batch size                                                   |
| optimizer                    | string | Optimizer used in training {LARS, SGD}                                |
| lr_schedule                  | string | Learning rate schedule used for training {piecewise, polynomial}      |
| weight_decay                 | float  | Value of weight decay coefficient for L2 regularization               |
| label_smoothing              | float  | Value of label_smoothing applied to the loss                          |
| base_learning_rate           | float  | Base learning rate for batch size 256; scaled linearly for other batch sizes |
| warmup_epochs                |  int   | Override autoselected polynomial decay warmup epochs                  |
| steps_per_loop               |  int   | Steps per training loop                                               |
| train_epochs                 |  int   | Number of epochs used for training                                    |
| epochs_between_evals         |  int   | Number of training epochs to run between evaluations                  |
| experimental_preloading      |  bool  | Whether to enable exprimental data preloading                         |
| data_loader_image_type       | string | Type of images provided by data loader {bf16, fp32}                   |
| enable_checkpoint_and_export |  bool  | Whether to enable checkpoint callbacks and export the savedmodel      |
| enable_tensorboard           |  bool  | Whether to enable Tensorboard callbacks                               |
| use_horovod                  |  bool  | Wheter to use Horovod for distributed training                        |
| num_workers_per_hls          |  int   | Number of workers per HLS                                             |
| hls_type                     | string | Type of HLS {HLS1, HLS1H}                                             |

Full list of parameters can be obtained by running following command:

> cd Model-References/TensorFlow/computer_vision/Resnets/resnet_keras
  >
  > python3 resnet_ctl_imagenet_main.py --helpfull
  >
