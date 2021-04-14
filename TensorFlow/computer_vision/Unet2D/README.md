# UNet for 2D Medical Image Segmentation
This repository provides a script and recipe to train UNet Medical to achieve state of the art accuracy, and is tested and maintained by Habana Labs, Ltd. an Intel Company.

## Table of Contents

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

* [Model-References](../../../README.md)
* [Model overview](#model-overview)
* [Setup](#setup)
* [Training](#training)
* [Advanced](#Advanced)

## Model Overview

This repository provides script to train UNet Medical model for 2D Segmentation on Habana Gaudi (HPU). It is based on [NVIDIA UNet Medical Image Segmentation for TensorFlow 2.x](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/UNet_Medical) repository. Implementation provided in this repository covers UNet model as described in the original paper [UNet: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).

### Model architecture
UNet allows for seamless segmentation of 2D images, with high accuracy and performance, and can be adapted to solve many different segmentation problems.

The following figure shows the construction of the UNet model and its components. UNet is composed of a contractive and an expanding path, that aims at building a bottleneck in its centermost part through a combination of convolution and pooling operations. After this bottleneck, the image is reconstructed through a combination of convolutions and upsampling. Skip connections are added with the goal of helping the backward flow of gradients in order to improve the training.

![UNet](images/unet.png)
Figure 1. The architecture of a UNet model. Taken from the [UNet: Convolutional Networks for Biomedical Image Segmentation paper](https://arxiv.org/abs/1505.04597).

### Model changes
Major changes done to original model from [NVIDIA UNet Medical Image Segmentation for TensorFlow 2.x](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/UNet_Medical):
* GPU specific configurations have been removed;
* Some scripts were changed in order to run the model on Gaudi. It includes loading habana tensorflow modules and using HPU multinode helpers;
* Model is using bfloat16 precision instead of float16;
* tf.keras.activations.softmax was replaced with tf.nn.softmax due to performance issues described in https://github.com/tensorflow/tensorflow/pull/47572;
* Additional tensorboard and performance logging was added;
* GPU specific files (examples/*, Dockerfile etc.) and some unused code have been removed.

### Default configuration
- Execution mode: train and evaluate;
- Batch size: 8;
- Data type: bfloat16;
- Maximum number of steps: 6400;
- Weight decay: 0.0005;
- Learning rate: 0.0001;
- Number of Horovod workers (HPUs): 1;
- Data augmentation: False;
- Cross-validation: disabled;
- Using XLA: False;
- Logging losses and performance every N steps: 100.

The model was tested with tensorflow-cpu in version 2.2.0, 2.2.2 and 2.4.0.

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

### Get the Habana Tensorflow docker image:

| TensorFlow version  | Ubuntu version        |                              Command Line                                                                                |
| ------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| TensorFlow 2.2      | Ubuntu 18.04          | `docker pull vault.habana.ai/gaudi-docker/0.14.0/ubuntu18.04/habanalabs/tensorflow-installer-tf-cpu-2.2.2:0.14.0-420`    |
| TensorFlow 2.4      | Ubuntu 18.04          | `docker pull vault.habana.ai/gaudi-docker/0.14.0/ubuntu18.04/habanalabs/tensorflow-installer-tf-cpu-2.4.1:0.14.0-420`    |
| TensorFlow 2.2      | Ubuntu 20.04          | `docker pull vault.habana.ai/gaudi-docker/0.14.0/ubuntu20.04/habanalabs/tensorflow-installer-tf-cpu-2.2.2:0.14.0-420`    |
| TensorFlow 2.4      | Ubuntu 20.04          | `docker pull vault.habana.ai/gaudi-docker/0.14.0/ubuntu20.04/habanalabs/tensorflow-installer-tf-cpu-2.4.1:0.14.0-420`    |

Note: For the sake of simplicity the Readme instructions follows the assumption that docker image is tf-cpu-2.2.2:0.14.0-420 and Ubuntu version is 18.04.

```bash
  docker pull vault.habana.ai/gaudi-docker/0.14.0/ubuntu18.04/habanalabs/tensorflow-installer-tf-cpu-2.2.2:0.14.0-420
```

### Launch the Docker container:
```bash
  docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -v /tmp:/tmp -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --cap-add=SYS_PTRACE --net=host --user=root --workdir=/root -e TF_MODULES_RELEASE_BUILD=/usr/lib/habanalabs/ -v /sys/kernel/debug:/sys/kernel/debug vault.habana.ai/gaudi-docker/0.14.0/ubuntu18.04/habanalabs/tensorflow-installer-tf-cpu-2.2.2:0.14.0-420
```

### Clone Habana Model garden and go to UNet2D Medical repository:
```bash
  git clone https://github.com/HabanaAI/Model-References /root/Model-References
  cd /root/Model-References/TensorFlow/computer_vision/Unet2D
```

### Add Tensorflow packages from model_garden to python path:
```bash
  export PYTHONPATH=$PYTHONPATH:/root/Model-References/
```

### Download the [EM segmentation challenge dataset](http://brainiac2.mit.edu/isbi_challenge/home):
```python
  python3 download_dataset.py --data_dir /root/dataset
```
by default it will download the dataset to `./data` path

## Training
The model was tested both in single node and multinode configurations.

To run trainining using demo script without habana_model_runner.py:
```python
python3 unet2d_demo.py --data_dir <dataset_path> --model_dir <logs_path> --max_steps <steps_number> --batch_size <batch_size> --dtype <data_type> --exec_mode train_and_evaluate --fold 0 --augment --xla --hvd_workers <number_of_nodes>
```

For example:
- single card training with batch size 8 and bfloat16 precision:
    ```python
    python3 unet2d_demo.py --data_dir ./data --model_dir ./logs --max_steps 6400 --batch_size 8 --dtype bf16 --exec_mode train_and_evaluate --fold 0 --augment --xla --hvd_workers 1
    ```
- multinode 8 HPUs training with batch size 8 and bfloat16 precision:
    ```python
    python3 unet2d_demo.py --data_dir ./data --model_dir ./logs --max_steps 6400 --batch_size 8 --dtype bf16 --exec_mode train_and_evaluate --fold 0 --augment --xla --hvd_workers 8
    ```

Note:
- This is the preview of TensorFlow Unet2D Python scripts with yaml configuration of parameters.
- The training for single and 8 nodes should be performed using unet2d_demo.py as mentioned above in Training section.
- Below is a preview of how to use python scripts for the specific yaml files provided.

To run trainining using habana_model_runner:

```python
    python3 <path/of/habana_model_runner.py> --framework tensorflow --model unet2d --hb_config <hb_configs/config/file/name>
```

For example:
- single card training with batch size 8 and bfloat16 precision:
    ```python
    python3 /root/Model-References/central/habana_model_runner.py --framework tensorflow --model unet2d --hb_config hb_configs/single_node_bf16.yaml
    ```
- single card training with batch size 8 and fp32 precision:
    ```python
    python3 /root/Model-References/central/habana_model_runner.py --framework tensorflow --model unet2d --hb_config hb_configs/single_node_fp32.yaml
    ```
- multinode 8 HPUs training with batch size 8 and bfloat16 precision:
    ```python
    python3 /root/Model-References/central/habana_model_runner.py --framework tensorflow --model unet2d --hb_config hb_configs/1xhls_node_bf16.yaml
    ```
- multinode 8 HPUs training with batch size 8 and fp32 precision:
    ```python
    python3 /root/Model-References/central/habana_model_runner.py --framework tensorflow --model unet2d --hb_config hb_configs/1xhls_node_fp32.yaml
    ```

Note:
- The parameters in yaml file can be altered or new yaml files can be created to train model on different values of the parameter.

### Benchmark
To measure performance of training on HPU in terms of images per second:
```python
python3 unet2d_demo.py --data_dir <path/to/dataset> --log_dir <path/for/logs> --batch_size <batch size> --dtype <data type> --exec_mode train --benchmark --warmup_steps 200 --max_steps 1000 --augment --xla --hvd_workers <number of HPUs>
```
For example:
- single card benchmark with batch size 8 and bfloat16 precision:
    ```python
    python3 unet2d_demo.py --data_dir /root/dataset/ --log_dir ./logs --batch_size 8 --dtype bf16 --exec_mode train --benchmark --warmup_steps 200 --max_steps 1000  --augment --xla --hvd_workers 1
    ```
- multinode 8 HPUs benchmark with batch size 8 and bfloat16 precision:
    ```python
    python3 unet2d_demo.py --data_dir /root/dataset/ --log_dir ./logs --batch_size 8 --dtype bf16 --exec_mode train --benchmark --warmup_steps 200 --max_steps 1000  --augment --xla --hvd_workers 8
    ```

## Advanced

The following sections provide more details of scripts in the repository, available parameters, and command-line options

### Scripts definitions

* `unet2d_demo.py`: Serves as the entry point to the application. If it's going to be run in multinode, applies required command line prefix for OpenMPI.
* `download_dataset.py` - Script for downloading dataset.
* `requirements.txt`: Set of extra requirements for running UNet.
* `data_loading/data_loader.py`: Implements the data loading and augmentation.
* `model/layers.py`: Defines the different blocks that are used to assemble UNet.
* `model/unet.py`: Defines the model architecture using the blocks from the `layers.py` script.
* `runtime/arguments.py`: Implements the command-line arguments parsing.
* `runtime/losses.py`: Implements the losses used during training and evaluation.
* `runtime/run.py`: Implements the logic for training, evaluation, and inference.
* `runtime/parse_results.py`: Implements the intermediate results parsing.
* `runtime/setup.py`: Implements helper setup functions.
* `train_and_evaluate.sh`: Runs the topology training from scratch and evaluates the model for 5 cross-validation.

Other folders included in the root directory are:
* `images/`: Contains a model diagram.

### Parameters

The complete list of the available parameters for the `main.py` script contains:
* `--exec_mode`: Select the execution mode to run the model (default: `train_and_evaluate`). Modes available:
  * `train` - trains model from scratch.
  * `evaluate` - loads checkpoint (if available) and performs evaluation on validation subset (requires `--fold` other than `None`).
  * `train_and_evaluate` - trains model from scratch and performs validation at the end (requires `--fold` other than `None`).
  * `predict` - loads checkpoint (if available) and runs inference on the test set. Stores the results in `--model_dir` directory.
  * `train_and_predict` - trains model from scratch and performs inference.
* `--model_dir`: Set the output directory for information related to the model (default: `/logs`).
* `--data_dir`: Set the input directory containing the dataset (default: `None`).
* `--log_dir`: Set the output directory for logs (default: `logs`).
* `--batch_size`: Size of each minibatch per GPU (default: `8`).
* `--dtype`: Set precision to be used in model: fp32/bf16 (default: `bf16`).
* `--fold`: Selected fold for cross-validation (default: `None`).
* `--max_steps`: Maximum number of steps (batches) for training (default: `6400`).
* `--log_every`: Log data every n steps (default: `100`).
* `--evaluate_every`: Evaluate every n steps (default: `0` - evaluate once at the end).
* `--seed`: Set random seed for reproducibility (default: `0`).
* `--weight_decay`: Weight decay coefficient (default: `0.0005`).
* `--learning_rate`: Model’s learning rate (default: `0.0001`).
* `--augment`: Enable data augmentation (default: `False`).
* `--benchmark`: Enable performance benchmarking (default: `False`). If the flag is set, the script runs in a benchmark mode - each iteration is timed and the performance result (in images per second) is printed at the end. Works for both `train` and `predict` execution modes.
* `--warmup_steps`: Used during benchmarking - the number of steps to skip (default: `200`). First iterations are usually much slower since the graph is being constructed. Skipping the initial iterations is required for a fair performance assessment.
* `--xla`: Enable accelerated linear algebra optimization (default: `False`).
* `--hvd_workers`: Set number of HPUs to be used (default: `1`).
* `--tensorboard_logging`: Enable tensorboard logging (default: `False`).
* `--disable_hpu`: Disable execution on HPU (default: `False`).
* `--disable_ckpt_saving`: Disables saving checkpoints (default: `False`).

### Command-line options

To see the full list of available options and their descriptions, use the `-h` or `--help` command-line option, for example:
```bash
python3 unet2d_demo.py --help
```

The following example output is printed when running the model:
```python3 unet2d_demo.py --help
python3 unet2d_demo.py -h
usage: unet2d_demo.py [-h]
                      [--exec_mode {train,train_and_predict,predict,evaluate,train_and_evaluate}]
                      [--model_dir MODEL_DIR] --data_dir DATA_DIR
                      [--log_dir LOG_DIR] [--batch_size BATCH_SIZE]
                      [--dtype {fp32,bf16}] [--fold FOLD]
                      [--max_steps MAX_STEPS] [--log_every LOG_EVERY]
                      [--evaluate_every EVALUATE_EVERY]
                      [--warmup_steps WARMUP_STEPS]
                      [--weight_decay WEIGHT_DECAY]
                      [--learning_rate LEARNING_RATE] [--seed SEED]
                      [--hvd_workers HVD_WORKERS] [--dump_config DUMP_CONFIG]
                      [--augment] [--no-augment] [--benchmark]
                      [--no-benchmark] [--use_xla] [--resume_training]
                      [--disable_hpu] [--synth_data] [--disable_ckpt_saving]
                      [--use_horovod] [--tensorboard_logging]

UNet-medical

optional arguments:
  -h, --help            show this help message and exit
  --exec_mode {train,train_and_predict,predict,evaluate,train_and_evaluate}
                        Execution mode of running the model
  --model_dir MODEL_DIR
                        Output directory for information related to the model
  --data_dir DATA_DIR   Input directory containing the dataset for training
                        the model
  --log_dir LOG_DIR     Output directory for training logs
  --batch_size BATCH_SIZE
                        Size of each minibatch per GPU
  --dtype {fp32,bf16}, -d {fp32,bf16}
                        Data type: fp32 or bf16
  --fold FOLD           Chosen fold for cross-validation. Use None to disable
                        cross-validation
  --max_steps MAX_STEPS
                        Maximum number of steps (batches) used for training
  --log_every LOG_EVERY
                        Log data every n steps
  --evaluate_every EVALUATE_EVERY
                        Evaluate every n steps
  --warmup_steps WARMUP_STEPS
                        Number of warmup steps
  --weight_decay WEIGHT_DECAY
                        Weight decay coefficient
  --learning_rate LEARNING_RATE
                        Learning rate coefficient for AdamOptimizer
  --seed SEED           Random seed
  --hvd_workers HVD_WORKERS
                        Number of Horovod workers, default 1 - Horovod
                        disabled
  --dump_config DUMP_CONFIG
                        Directory for dumping debug traces
  --augment             Perform data augmentation during training
  --no-augment
  --benchmark           Collect performance metrics during training
  --no-benchmark
  --use_xla, --xla      Train using XLA
  --resume_training     Resume training from a checkpoint
  --disable_hpu         Disables execution on HPU
  --synth_data          Use deterministic and synthetic data
  --disable_ckpt_saving
                        Disables saving checkpoints
  --use_horovod         Enable horovod usage
  --tensorboard_logging
                        Enable tensorboard logging
```
