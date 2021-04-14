# ResNet For TensorFlow

This repository provides a script and recipe to train the ResNet v1.5 models to achieve state-of-the-art accuracy, and is tested and maintained by Habana.

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

## Table Of Contents
  * [Model-References](../../../README.md)
  * [Model Overview](#model-overview)
    * [Training Script Modifications](#training-script-modifications)
  * [Setup](#setup)
  * [Training](#training)
  * [Preview Python Script](#Preview-Python-Script)

## Model Overview
Both ResNet v1.5 and ResNeXt model is a modified version of the original ResNet v1 model. It supports layers 50, 101, and 152.

Keras version of the model is described in details in separate [readme](resnet_keras/README.md).

### Training Script Modifications

Originally, scripts were taken from [Tensorflow
Github](https://github.com/tensorflow/models.git), tag v1.13.0

Files used:

-   imagenet\_main.py
-   imagenet\_preprocessing.py
-   resnet\_model.py
-   resnet\_run\_loop.py

All of above files were converted to TF2 by using
[tf\_upgrade\_v2](https://www.tensorflow.org/guide/upgrade?hl=en) tool.
Additionally, some other changes were committed for specific files.

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
11. Added MLPerf support
12. Added calls to tf.compat.v1.disable_eager_execution() and tf.compat.v1.enable_resource_variables() in main code section

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
2. Added functionality to enable the use of LARS optimizer
3. Added MLPerf support, including in how the training schedule is defined, the usage of mlperf_logging, etc.
4. Added options to enable the use of tf.data's 'experimental_slack' and 'experimental.prefetch_to_device' options during the input processing
5. Added support for specific size thread pool for tf.data operations
6. TensorFlow v2 support: Changed dataset.apply(tf.contrib.data.map_and_batch(..)) to dataset.map(..., num_parallel_calls, ...) followed by dataset.batch()
7. Changed calls to tf.<api> to tf.compat.v1.<api> for backward compatibility when running training scripts on TensorFlow v2
8. Other TF v2 replacements for tf.contrib usages
9. Redefined learning_rate_with_decay to use warmup_epochs and use_cosine_lr
10. Adapt the warmup_steps based on MLPerf flag
11. Defined a new learning rate schedule for LARS optimizer, that handles linear scaling rule, gradual warmup, and LR decay
12. Added functionality for label smoothing
13. Commented out writing images to summary, for performance reasons
14. Added check for non-tf.bfloat16 input images having the same data type as the dtype that training is run with
15. Added functionality to define the cross-entropy function depending on whether label smoothing is enabled
16. Added support for loss scaling of gradients
17. Added flag for experimental_preloading that invokes the HabanaEstimator, besides other optimizations such as tf.data.experimental.prefetch_to_device
18. Added 'TF_DISABLE_SCOPED_ALLOCATOR' environment variable flag to disable Scoped Allocator Optimization (enabled by default) for Horovod runs
19. Added a flag to configure the save_checkpoint_steps
20. If the flag "use_train_and_evaluate" is set, or in multi-worker training scenarios, there is a one-shot call to tf.estimator.train_and_evaluate
21. resnet_main() returns a dictionary with keys 'eval_results' and 'train_hooks'
22. Added flags in 'define_resnet_flags' for flags_core.define_base, flags_core.define_performance, flags_core.define_distribution, flags_core.define_experimental, and many others (please refer to this function for all the flags that are available)
23. Changed order of ops creating summaries to log them in TensorBoard with proper name. Added saving HParams to TensorBoard and exposed a flag for specifying frequency of summary updates.
24. Changed a name of directory, in which workers are saving logs and checkpoints, from "rank_N" to "worker_N".


### Hyperparameters
**ResNet SGD:**

* Momentum (0.9),
* Learning rate (LR) = 0.128 (single node) or 0.1 (multinode) for 256 batch size. For other batch sizes we linearly scale the learning rate.
* Piecewise learning rate schedule.
* Linear warmup of the learning rate during the first 3 epochs.
* Weight decay: 0.0001.
* Label Smoothing: 0.1.

**ResNet LARS:**

* Momentum (0.9),
* Learning rate (LR) = 2.5 (single node) or 9.5 (1 HLS, i.e. 8 gaudis, global batch 2048 (8*256)) (1).
* Polynomial learning rate schedule.
* Linear warmup of the learning rate during the first 3 epochs (1).
* Weight decay: 0.0001.
* Label Smoothing: 0.1.

(1) These numbers apply for batch size lower than 8192. There are other configurations for higher global batch sizes. Note, however, that they haven't been tested yet:

* (8192 < batch size < 16384): LR = 10, warmup epochs = 5,
* (16384 < batch size < 32768): LR = 25, warmup epochs = 5,
* (bachsize > 32768): LR = 32, warmup epochs = 14.

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

### Training Data

The ResNet50 v1.5 script operates on ImageNet 1k, a widely popular image
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

```bash
docker pull vault.habana.ai/gaudi-docker/0.14.0/ubuntu18.04/habanalabs/tensorflow-installer-tf-cpu-2.2.2:0.14.0-420
```

### 2. Run docker

**NOTE:** This assumes the Imagenet dataset is under /opt/datasets/imagenet on the host. Modify accordingly.

```bash
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice  -v /sys/kernel/debug:/sys/kernel/debug -v /opt/datasets/imagenet:/root/tensorflow_datasets/imagenet --net=host vault.habana.ai/gaudi-docker/0.14.0/ubuntu18.04/habanalabs/tensorflow-installer-tf-cpu-2.2.2:0.14.0-420
```

OPTIONAL with mounted shared folder to transfer files out of docker:

```bash
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice  -v /sys/kernel/debug:/sys/kernel/debug -v ~/shared:/root/shared -v /opt/datasets/imagenet:/root/tensorflow_datasets/imagenet --net=host vault.habana.ai/gaudi-docker/0.14.0/ubuntu18.04/habanalabs/tensorflow-installer-tf-cpu-2.2.2:0.14.0-420
```

### 3. Clone the repository and go to resnet directory:

```bash
git clone https://github.com/HabanaAI/Model-References.git
cd Model-References/TensorFlow/computer_vision/Resnets
```

Note: If the repository is not in the PYTHONPATH, make sure you update it.
```bash
export PYTHONPATH=/path/to/Model-References:$PYTHONPATH
```

### Single-Card Training

#### The `demo_resnet` script

The script for training and evaluating the ResNet model has a variety of paramaters.
```
usage: ./demo_resnet [arguments]

mandatory arguments:
  -d <data_type>,    --dtype <data_type>             Data type, possible values: fp32, bf16

optional arguments:
  -rs <resnet_size>, --resnet-size <resnet_size>     ResNet size, default 50 (or 101, when --resnext flag is given),
                                                     possible values: 50, 101, 152
  -b <batch_size>,   --batch-size <batch_size>       Batch size, default 128 for fp32, 256 for bf16
  -e <epochs>,       --epochs <epochs>               Number of epochs, default to 1
  -a <data_dir>,     --data-dir <data_dir>           Data dir, defaults to /software/data/tf/data/imagenet/tf_records/
  -m <model_dir>,    --model-dir <model_dir>         Model dir, defaults to /home/user1/tmp/resnet50/
  -o,                --use-horovod                   Use horovod for training
  -s <steps>,        --steps <steps>                 Max train steps
  -l <steps>,        --eval-steps <steps>            Max evaluation steps
  -n,                --no-eval                       Don't do evaluation
  -v <steps>,        --display-steps <steps>         How often display step status
  -c <steps>,        --checkpoint-steps <steps>      How often save checkpoint
  -r                 --recover                       If crashed restart training from last checkpoint. Requires -s to be set
  -k                 --no-experimental-preloading    Disables support for 'data.experimental.prefetch_to_device' TensorFlow operator. If not set:
                                                     - loads extension dynpatch_prf_remote_call.so (via LD_PRELOAD)
                                                     - sets environment variable HBN_TF_REGISTER_DATASETOPS to 1
                                                     - this feature is experimental and works only with single node
                     --use-train-and-evaluate        If set, uses tf.estimator.train_and_evaluate for the training and evaluation
                     --epochs-between-evals <epochs> Number of training epochs between evaluations, default 1
                     --enable-lars-optimizer         If set uses LARSOptimizer instead of default one
                     --stop_threshold <accuracy>     Threshold accuracy which should trigger the end of training.
                     --resnext                       Run resnext
```

These are example run commands:
```
examples:
  ./demo_resnet -d bf16
  ./demo_resnet -d fp32
  ./demo_resnet -d fp32 -rs 101
  ./demo_resnet -d bf16 -s 1000
  ./demo_resnet -d bf16 -s 1000 -l 50
  ./demo_resnet -d bf16 -e 9
  ./demo_resnet -d fp32 -e 9 -b 128 -a /root/tensorflow_datasets/imagenet/tf_records/ -m /root/tensorflow-training/demo/ck_81_080_450_bs128
```

### Multi-Card Training

#### The `demo_resnet_hvd.sh` script

The script requires additional parameters.

```bash
export IMAGENET_DIR=/path/to/tensorflow_datasets/imagenet/
export RESNET_SIZE=<resnet-size>
./demo_resnet_hvd.sh
```

### Example Commands for single and multi cards

To benchmark the training performance on a specific batch size, run:

**For Single Gaudi**

ResNet 50 (FP32)

```
./demo_resnet -rs 50 -d fp32 -b 128 -a /path/to/tensorflow_datasets/imagenet/tf_records/ -m /root/tensorflow-training/demo/ck_81_080_450_bs128
```

ResNet 50 (BF16)

```
./demo_resnet -rs 50 -d bf16 -b 256 -a /path/to/tensorflow_datasets/imagenet/tf_records/ -m /root/tensorflow-training/demo/resnet50
```
ResNet 101 (BF16)

```
./demo_resnet -rs 101 -d bf16 -b 256 -a /path/to/tensorflow_datasets/imagenet/tf_records/ -m /root/tensorflow-training/demo/resnet101
```

ResNeXt 101 (BF16)

```
./demo_resnet -rs 101 --resnext -d bf16 -b 256 -a /path/to/tensorflow_datasets/imagenet/tf_records/ -m /root/tensorflow-training/demo/resnext101
```

**For multiple Gaudi cards**

ResNet 50 (BF16)
```
export IMAGENET_DIR=/path/to/tensorflow_datasets/imagenet/
export RESNET_SIZE=50
./demo_resnet_hvd.sh
```
ResNet 101 (BF16)
```
export IMAGENET_DIR=/path/to/tensorflow_datasets/imagenet/
export RESNET_SIZE=101
./demo_resnet_hvd.sh
```
ResNeXt 101 (BF16)
```
export IMAGENET_DIR=/path/to/tensorflow_datasets/imagenet/
export RESNET_SIZE=101
./demo_resnet_hvd.sh --resnext
```

### Multi-Chassis Training

The following directions are generalized for use of Multi-Chassis:

1. Follow [Setup](#setup) above on all chassis
2. Configure ssh between chassis (Inside all dockers):
Do the following on all chassis dockers:
```
mkdir ~/.ssh
cd ~/.ssh
ssh-keygen -t rsa -b 4096
```
Copy id_rsa.pub contents from every chassis to every authorized_keys (all public keys need to be in all hosts' authorized_keys):
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
By default, the Habana docker uses port 3022 for ssh, and this is the default port configured in the training scripts. Sometimes, mpirun can fail to establish the remote connection when there is more than one Habana docker session running on the main HLS in which the Python training script is run. If this happens, you can set up a different ssh port as follows:

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

3.	Run Rn50 16 cards:
NOTE: MODIFY IP ADDRESS BELOW TO MATCH YOUR SYSTEM.
```
MPI_TPC_INCLUDE=192.10.100.1/24 MULTI_HLS_IPS=192.10.100.174,10.10.100.101 IMAGENET_DIR=/root/tensorflow_datasets/imagenet/tf_records/ RESNET_SIZE=50 TRAIN_EPOCHS=90 ./demo_resnet_hvd.sh
```

## Preview Python Script

Note:
- This is the preview of TensorFlow ResNet Python scripts with yaml configuration of parameters.
- The training for single and 8 nodes should be performed using bash scripts as mentioned in above Training section.
- Below is a preview of how to use python scripts for the specific yaml files provided. Currenly python scripts should not be used with self modified yaml files.

You can run the following script: **/path/to/Model-References/central/habana_model_runner.py** which accepts two arguments:
- --model *model_name*
- --hb_config *path_to_yaml_config_file*

Example of config files can be found in the **/path/to/Model-References/TensorFlow/computer_vision/Resnets/hb_configs/** (for resnet_estimator model).

You can use these scripts as such:
> cd Model-References/TensorFlow/computer_vision/Resnets
  >
  > python3 ../../../central/habana_model_runner.py --model resnet_estimator --hb_config *path_to_yaml_config_file*

**Examples:**

Note: All the yaml configs assume the below path for data_dir and model_dir which can be modified as required.
data_dir= `/software/data/tf/data/imagenet/tfrecords`
model_dir= `/tmp/resnet_estimator`

**Single Card:**

- ResNet 50 , bf16, batch size= 256, enable_lars= flase, max_train_steps= 200
    ```python
    python3 ../../../central/habana_model_runner.py --framework tensorflow --model resnet_estimator --hb_config hb_configs/u18_single_node_resnet_estimator_bf16.yaml
    ```

- ResNet 50 , fp32, batch size= 128, enable_lars= flase, max_train_steps= 200
    ```python
    python3 ../../../central/habana_model_runner.py --framework tensorflow --model resnet_estimator --hb_config hb_configs/u18_single_node_resnet_estimator_fp32.yaml
    ```

- ResNeXt 101 , fp16, batch size= 256, enable_lars= flase, max_train_steps= 200
    ```python
    python3 ../../../central/habana_model_runner.py --framework tensorflow --model resnet_estimator --hb_config hb_configs/u18_single_node_resnext_estimator_bf16.yaml
    ```

- ResNeXt 101 , fp32, batch size= 256, enable_lars= flase, max_train_steps= 200
    ```python
    python3 ../../../central/habana_model_runner.py --framework tensorflow --model resnet_estimator --hb_config hb_configs/u18_single_node_resnext_estimator_fp32.yaml
    ```

**8 card:**

- ResNet 50 , bf16, batch size= 256, enable_lars= flase, max_train_steps= 200, num_workers_per_hls= 8, hls_type= "HLS1"
    ```python
    python3 ../../../central/habana_model_runner.py --framework tensorflow --model resnet_estimator --hb_config hb_configs/u18_1hls_resnet_estimator_bf16.yaml
    ```

- ResNet 50 , fp32, batch size= 128, enable_lars= flase, max_train_steps= 200, num_workers_per_hls= 8, hls_type= "HLS1"
    ```python
    python3 ../../../central/habana_model_runner.py --framework tensorflow --model resnet_estimator --hb_config hb_configs/u18_1hls_resnet_estimator_fp32.yaml
    ```

- ResNeXt 101 , fp16, batch size= 256, enable_lars= flase, max_train_steps= 200, num_workers_per_hls= 8, hls_type= "HLS1"
    ```python
    python3 ../../../central/habana_model_runner.py --framework tensorflow --model resnet_estimator --hb_config hb_configs/u18_1hls_resnext_estimator_bf16.yaml
    ```

- ResNeXt 101 , fp32, batch size= 256, enable_lars= flase, max_train_steps= 200, num_workers_per_hls= 8, hls_type= "HLS1"
    ```python
    python3 ../../../central/habana_model_runner.py --framework tensorflow --model resnet_estimator --hb_config hb_configs/u18_1hls_resnext_estimator_fp32.yaml
    ```

Note:
- The parameters in yaml file should not be modified currently.
