# Densenet-121
## Table Of Contents
* [Model overview](#Model-overview)
* [Setup](#Setup)
* [Training the Model](#Training-the-model)

## Model overview
This code demonstrates training of the Densenet-121 model on ILSVRC2012 dataset (aka ImageNet).
The model architecture is described in this paper: https://arxiv.org/abs/1608.06993.
The implementation employs Keras.Application model (https://www.tensorflow.org/api_docs/python/tf/keras/applications/densenet).

### Changes to the original model:
1. A generic Keras-based training script was used from here: https://github.com/jkjung-avt/keras_imagenet.
As this script was originally developed in TF-1.12, it was migrated to TF-2.2 using the automatic migration tool
provided by TensorFlow (https://www.tensorflow.org/guide/upgrade).
2. Updates on the usage of tf.contrib module were made, as these are not available in TensorFlow 2.2.
An additional change is the addition of multi card support, using
[Mirrored strategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy). This change only applies to non-HPU hardware.
3. `set_learning_phase(True)` - Sets the learning phase to a fixed value.
4. Setting batch size before calling `model.fit`

## Setup

### Install Drivers
Follow steps in the [Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) to install the drivers.

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

### Data

The training script requires Imagenet data (ILSVRC2012) to be preprocessed into TFRecords.

**Note: Preprocessing requires use of TensorFlow version 1.12.2**

1.	Create a folder named `dataset_preparation`
	The `dataset_preparation` folder structure is described below:

	* `dataset_preparation`

        •	`keras_imagenet`  (a github repository for converting the files in train and validation folders  to TF records described in step #5)

        •	`tfrecords`  (The folder in which the training and validation TF record files will reside. This folder is the –dataset_dir folder used by the densnet training script)

        •	`train`

        •	`validation`

        •	`ILSVRC2012_img_train.tar`   (imagenet train file obtained by step #2)

        •	`ILSVRC2012_img_val.tar`	   (imagenet validation file obtained by step #2)

2.	Download imagenet train (`ILSVRC2012_img_train.tar`) and validation (`ILSVRC2012_img_val.tar`) files from http://image-net.org/download-images  to `dataset_preparation` directory.
3.	Imagent training files  creation of 1000 folder classes

    a.	`mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train`

    b.	`tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar`

    c.	`find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done`

    d.	`cd ..`

4.	Imagenet validation files creation in validation folder

    a.	`mkdir validation && mv ILSVRC2012_img_val.tar validation/ && cd validation`

    b.	`tar xvf ILSVRC2012_img_val.tar`

5.	Converting the validation and training images to TF records

    a.  In `dataset_preparation` folder `git clone  https://github.com/jkjung-avt/keras_imagenet.git`

    b.	`cd keras_imagenet/data`

    c.	`python3 preprocess_imagenet_validation_data.py ../../validation  imagenet_2012_validation_synset_labels.txt`

    d. `mkdir ../../tfrecords`

    d.	`python3 build_imagenet_data.py --output_directory ../../tfrecords --train_directory ../../train --validation_directory ../../validation`


After step #5 the `tfrecords` directory will contain 1024 training and 128 validation TFrecord files. The `tfrecords` directory is the value of the `--dataset_dir` command line argument for the densenet training script train.py.



### Model Setup

1. [Download](http://image-net.org/index) and [preprocess](https://gist.github.com/qfgaohao/51556faa527fba89a81d048dda37c504) the dataset.

2. Complete the Installation of the SynapseAI SW stack if needed.  Users can refer to the [Habana Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html)

2. Download docker

### Get the Habana Tensorflow docker image:

| TensorFlow version  | Ubuntu version        |                              Command Line                                                                                |
| ------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| TensorFlow 2.2      | Ubuntu 18.04          | `docker pull vault.habana.ai/gaudi-docker/0.14.0/ubuntu18.04/habanalabs/tensorflow-installer-tf-cpu-2.2.2:0.14.0-420`    |
| TensorFlow 2.2      | Ubuntu 20.04          | `docker pull vault.habana.ai/gaudi-docker/0.14.0/ubuntu20.04/habanalabs/tensorflow-installer-tf-cpu-2.2.2:0.14.0-420`    |

Note: For the sake of simplicity the Readme instructions follows the assumption that docker image is tf-cpu-2.2.2:0.14.0-420 and Ubuntu version is 20.04. Run docker ---
**NOTE:** This assumes Imagenet dataset is under /opt/datasets/imagenet on the host. Modify accordingly.
```
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice -v /sys/kernel/debug:/sys/kernel/debug -v /opt/datasets/imagenet:/root/tensorflow_datasets/imagenet --net=host vault.habana.ai/gaudi-docker/0.14.0/ubuntu20.04/habanalabs/tensorflow-installer-tf-cpu-2.2.2:0.14.0-420
```
OPTIONAL with mounted shared folder to transfer files out of docker:
```
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice -v /sys/kernel/debug:/sys/kernel/debug  -v ~/shared:/root/shared -v /opt/dataset/imagenet:/root/tensorflow_datasets/imagenet --net=host --ulimit memlock=-1:-1 vault.habana.ai/gaudi-docker/0.14.0/ubuntu20.04/habanalabs/tensorflow-installer-tf-cpu-2.2.2:0.14.0-420
```
5. The above command will open a shell inside the container. Clone the repository and go to densenet_keras directory:
$ git clone https://github.com/HabanaAI/Model-References.git /root/Model-References
$ cd Model-References/TensorFlow/computer_vision/densenet_keras
```

Note: If the repository is not in the PYTHONPATH, make sure you update it.

```
export PYTHONPATH=/path/to/Model-References:$PYTHONPATH
```

## Training the Model

1. `python3 train.py --dataset_dir <path TFRecords dataset>`
2. `python3 train.py --dataset_dir <path TFRecords dataset> --only_eval <path to saved model>`

Step 1 will save the trained model after each epoch under `saves`

The following params are default:
1. `--dropout_rate`: 0.0 (no dropout)
2. `--weight_decay`: 1e-4
3. `--optimizer`: sgd
4. `--batch_size`: 64
5. `--lr_sched`: steps
6. `--initial_lr`: 5e-2
7. `--final_lr`: 1e-5
8. `--bfloat16`: True
9. `--epochs`: 90
