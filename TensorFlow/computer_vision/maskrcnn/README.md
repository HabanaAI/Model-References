# Mask R-CNN for TensorFlow

## Table of Contents

- [Model overview](#model-overview)
    - [Model architecture](#model-architecture)
- [Setup](#setup)
    - [Install drivers](#install-drivers)
    - [Setting up COCO 2017 dataset](#setting-up-coco-2017-dataset)
- [Training the model](#training-the-model)
    - [Parameters](#parameters)
- [Examples](#examples)

## Model overview

Mask R-CNN is a convolution-based neural network for the task of object instance segmentation. The paper describing the model can be found [here](https://arxiv.org/abs/1703.06870). This repository provides a script to train Mask R-CNN for Tensorflow on Habana
Gaudi, and is an optimized version of the implementation in [NVIDIA's Mask R-CNN](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/MaskRCNN) for Gaudi.

Changes in the model:

- Support for Habana device was added
- Horovod support
- Custom PyramidRoiAlign implementation

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](developer.habana.ai).

### Model architecture

Mask R-CNN builds on top of Faster R-CNN adding an additional mask head for the task of image segmentation.

The architecture consists of the following:

- ResNet-50 backbone with Feature Pyramid Network (FPN)
- Region proposal network (RPN) head
- RoI Align
- Bounding and classification box head
- Mask head

## Setup

### Install drivers

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

### Docker setup
1. Stop running dockers
    ```bash
    docker stop $(docker ps -a -q)
    ```

2. Download docker

    Choose the Habana TensorFlow docker image for your TensorFlow version and Ubuntu version from the table below.

    | TensorFlow version | Ubuntu version | Habana TensorFlow docker image |
    |:------------------|:--------------|:-----------------------------------|
    | TensorFlow 2.2 | Ubuntu 18.04 | `vault.habana.ai/gaudi-docker/0.14.0/ubuntu18.04/habanalabs/tensorflow-installer-tf-cpu-2.2.2:0.14.0-420` |
    | TensorFlow 2.2 | Ubuntu 20.04 | `vault.habana.ai/gaudi-docker/0.14.0/ubuntu20.04/habanalabs/tensorflow-installer-tf-cpu-2.2.2:0.14.0-420` |
    | TensorFlow 2.4 | Ubuntu 18.04 | `vault.habana.ai/gaudi-docker/0.14.0/ubuntu18.04/habanalabs/tensorflow-installer-tf-cpu-2.4.1:0.14.0-420` |
    | TensorFlow 2.4 | Ubuntu 20.04 | `vault.habana.ai/gaudi-docker/0.14.0/ubuntu20.04/habanalabs/tensorflow-installer-tf-cpu-2.4.1:0.14.0-420` |

    ```bash
    docker pull <your_choice_of_Habana_TensorFlow_docker_image>
    ```

3. Run docker
    ```bash
    docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice -v /sys/kernel/debug:/sys/kernel/debug --net=host <your_choice_of_Habana_TensorFlow_docker_image>
    ```

4. Clone the repository
    ```bash
    git clone https://github.com/HabanaAI/Model-References
    cd Model-References/TensorFlow/computer_vision/maskrcnn/
    ```

### Setting up COCO 2017 dataset

This repository provides scripts to download and preprocess the [COCO 2017 dataset](http://cocodataset.org/#download). If you already have the data then you do not need to run the following script, proceed to [Download the pre-trained weights](#download-the-pre-trained-weights).

The following script will save TFRecords files to the data directory, `/data`.
```bash
cd dataset
bash download_and_preprocess_coco.sh /data
```

By default, the data directory is organized into the following structure:
```bash
<data_directory>
    raw-data/
        train2017/
        train2017.zip
        test2017/
        test2017.zip
        val2017/
        val2017.zip
        annotations_trainval2017.zip
        image_info_test2017.zip
    annotations/
        instances_train2017.json
        person_keypoints_train2017.json
        captions_train2017.json
        instances_val2017.json
        person_keypoints_val2017.json
        image_info_test2017.json
        image_info_test-dev2017.json
        captions_val2017.json
    train-*.tfrecord
    val-*.tfrecord
```

### Download the pre-trained weights
This repository also provides scripts to download the pre-trained weights of ResNet-50 backbone.
The script will make a new directory with the name `weights` in the current directory and download the pre-trained weights in it.

```bash
./download_and_process_pretrained_weights.sh
```

Ensure that the `weights` folder created has a `resnet` folder in it.
Inside the `resnet` folder there should be 3 folders for checkpoints and weights: `extracted_from_maskrcnn`, `resnet-nhwc-2018-02-07` and `resnet-nhwc-2018-10-14`.
Before moving to the next step, ensure `resnet-nhwc-2018-02-07` is not empty.

## Training the model

1. Add your Model-References repo path to PYTHONPATH.
    ```bash
    export PYTHONPATH=<MODEL_REFERENCES_ROOT_PATH>:$PYTHONPATH
    ```

2. Update the configuration file.

    The `hb_configs` folder has configuration files in YAML format. Users can choose one of the configuration files to run or create their own. Please make sure to update the `dataset` and the `checkpoint` fields in your configuration file with the folders you created in [Setting up COCO 2017 dataset](#setting-up-coco-2017-dataset) and [Download the pre-trained weights](#download-the-pre-trained-weights). For other parameters, find the explanations in [Parameters](#Parameters).

3. Run single-card training
    ```bash
    python3 ../../../central/habana_model_runner.py --framework tensorflow --model maskrcnn --hb_config ./hb_configs/<your_choice_of_yaml_file>
    ```

4. Run 8-cards training
    ```bash
    python3 demo_mask_rcnn.py --mode train --dataset /data -d bf16 --hvd_workers 8
    ```
    NOTE: The python training script, `habana_model_runner.py`, for 8-cards training will be available in the next release.

### Parameters

You can modify the training behavior through the various flags in the `demo_mask_rcnn.py` script and the `mask_rcnn_main.py`.
Flags in the `demo_mask_rcnn.py` script are as follows:

- `mode`: Run `train` or `train_and_eval` on MS COCO
- `dataset`: Dataset directory
- `checkpoint`: Path to model checkpoint
- `model_dir`: Model directory
- `total_steps`: The number of steps to use for training, should be adjusted according to the `train_batch_size` flag
- `dtype`: Data type, `fp32` or `bf16`.
`bf16` automatically converts the appropriate ops to the bfloat16 format. This approach is similar to Automatic Mixed Precision of TensorFlow, which can reduce memory requirements and speed up training.
- `hvd_workers`: Number of Horovod workers, disabled by default
- `train_batch_size`: Batch size for training
- `no_eval_after_training`: Disable single evaluation steps after training when `train` command is used
- `clean_model_dir`: Clean model directory before execution
- `use_fake_data`: Use fake input

## Examples

| Command | Notes |
| ------- | ----- |
|`python3 ../../../central/habana_model_runner.py --framework tensorflow --model maskrcnn --hb_config hb_configs/single_node_maskrcnn_bf16_full.yaml`| Single-card training in bf16 |
|`python3 ../../../central/habana_model_runner.py --framework tensorflow --model maskrcnn --hb_config hb_configs/single_node_maskrcnn_fp32.yaml`| Single-card training in fp32 |
|`python3 demo_mask_rcnn.py --mode train --dataset /data -d bf16 --hvd_workers 8`| 8-cards training in bf16 |
|`python3 demo_mask_rcnn.py --mode train --dataset /data -d fp32 --hvd_workers 8`| 8-cards training in fp32 |