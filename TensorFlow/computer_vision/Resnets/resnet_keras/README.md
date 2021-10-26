# ResNet50 Keras model

This repository provides a script and recipe to train the ResNet keras model to achieve state-of-the-art accuracy, and is tested and maintained by Habana. Please visit [this page](https://developer.habana.ai/resources/habana-training-models/#performance) for performance information.

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents

* [Model Overview](#model-overview)
* [Setup](#setup)
* [Training](#training)

## Model Overview
ResNet keras model is a modified version of the original [TensorFlow model garden](https://github.com/tensorflow/models/tree/master/official/vision/image_classification/resnet) model. It uses a custom training loop, supports 50 layers and can work both with SGD and LARS optimizers.

## Setup
Please follow the instructions given in the following link for setting up the
environment including the `$PYTHON` environment variable: [Gaudi Setup and
Installation Guide](https://github.com/HabanaAI/Setup_and_Install). Please
answer the questions in the guide according to your preferences. This guide will
walk you through the process of setting up your system to run the model on
Gaudi.

### Training data

The ResNet50 Keras script operates on ImageNet 1k, a widely popular image classification dataset from the ILSVRC challenge.
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

```
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```
Go to resnet_keras directory:

```
cd Model-References/TensorFlow/computer_vision/Resnets/resnet_keras
```

Note: If the repository is not in the PYTHONPATH, make sure you update it.
```
export PYTHONPATH=/path/to/Model-References:$PYTHONPATH
```

### Multi-Chassis Training

The following directions are generalized for use of Multi-Chassis:

1. Follow [Setup](#setup) above on all servers
2. Follow [Training](#training) 1. - 3. above on all servers
3. Configure ssh between servers (Inside all dockers):
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

4.	Run Rn50 16 cards:
- NOTE: MODIFY IP ADDRESS BELOW TO MATCH YOUR SYSTEM.
- MPI_TCP_INCLUDE - comma-separated list of interfaces or subnets. This variable will set the mpirun parameter: `--mca btl_tcp_if_include`. This parameter tells mpi which TCP interfaces to use for communication between hosts. You can specify interface names or subnets in the include list in CIDR notation e.g. MPI_TCP_INCLUDE=eno1. More details: [Open MPI documentation](https://www.open-mpi.org/faq/?category=tcp#tcp-selection)
```
MPI_TCP_INCLUDE=interface_name MULTI_HLS_IPS=192.10.100.174,10.10.100.101 $PYTHON demo_resnet_keras.py -dt bf16 -dlit bf16 --use_horovod --num_workers_per_hls 8 -te 40 -ebe 40 --optimizer LARS --data_dir /data/tensorflow_datasets/imagenet/tf_records
```

### Two ways of running the model
Note: both scripts described below can be found under the following location: Model-References/TensorFlow/computer_vision/Resnets/resnet_keras

#### The `demo_resnet_keras` script
We've prepared a custom python wrapper called demo_resnet_keras.py which facilitates running the ResNet model. It wraps around the resnet_ctl_imagenet_main.py script described in the next section. Most importantly, the wrapper makes it easier for the user to run multicard trainings and also automates some hyperparameter setting when using LARS optimizer (see section "Using LARS optimizer").

```
demo_resnet_keras.py is a distributed launcher for resnet_ctl_imagenet_main.py

usage: $PYTHON demo_resnet_keras.py [arguments]

optional arguments:

-dt <data_type>,   --dtype <data_type>                     Data type, possible values: fp32, bf16. Defaults to fp32
-bs <batch_size>,  --batch_size <batch_size>               Batch size, defaults to 256
-te <epochs>,      --train_epochs <epochs>                 Number of training epochs, defaults to 1
-dd <data_dir>,    --data_dir <data_dir>                   Data dir, defaults to `/data/tensorflow_datasets/imagenet/tf_records/`.
                                                           Needs to be specified if the above does not exists.
-md <model_dir>,   --model_dir <model_dir>                 Model dir, defaults to /tmp/resnet
                   --clean                                 If set, model_dir will be removed if it exists. Unset by default
                   --train_steps <steps>                   The maximum number of steps per epoch. Ignored if larger than the number of steps needed to process the training set.
                   --log_steps <steps>                     How often display step status, defaults to 100
                   --steps_per_loop <steps>                Number of steps per training loop. Will be capped at steps per epoch, defaults to 50.
                   --enable_checkpoint_and_export          Enables checkpoint callbacks and exports the saved model.
                   --enable_tensorboard                    Enables Tensorboard callbacks.
-ebe <epochs>      --epochs_between_evals <epochs>         Number of training epochs between evaluations, defaults to 1.
                                                           To achieve fastest 'time to train', set to the same number as '--train_epochs' to only run one evaluation after the training.
                   --experimental_preloading               Enables support for 'data.experimental.prefetch_to_device' TensorFlow operator.
                                                           Enabled by default - pass --experimental_preloading=False to disable.
                   --optimizer <optimizer_type>            Name of optimizer preset, possible values: SGD, LARS. Defaults to SGD.
                   --num_workers_per_hls <num_workers>     Number of workers per node. Defaults to 1.
                                                           In case num_workers_per_hls > 1, it runs 'resnet_ctl_imagenet_main.py [ARGS]' via mpirun with generated HCL config.
                                                           Must be used together with --use_horovod either --distribution_strategy
                   --use_horovod                           Enable horovod for multicard scenarios
                   --distribution_strategy <strategy>      The Distribution Strategy to use for training. Defaults to off
                   --kubernetes_run                        Setup kubernetes run for multi server training
                   --use_keras_mixed_precision             Use native keras mixed precision policy instead of Habana bf16 conversion pass
                   --keras_applications_model <model_type> Use one of the natively supported resnet models from keras.applications:
                                                           ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2
                                                           Default: None

examples:

$PYTHON demo_resnet_keras.py
$PYTHON demo_resnet_keras.py -dt bf16 -dlit bf16 -bs 256 -te 90 -ebe 90
$PYTHON demo_resnet_keras.py --dtype bf16 -dlit bf16 --use_horovod --num_workers_per_hls 8 -te 40 -ebe 40 --optimizer LARS -bs 256
$PYTHON demo_resnet_keras.py --dtype bf16 -dlit bf16 --distribution_strategy hpu --num_workers_per_hls 8 -bs 256
In order to see all possible arguments to resnet_ctl_imagenet_main.py, run "$PYTHON resnet_ctl_imagenet_main.py --helpfull
```
#### The modified community script
Apart from the custom wrapper, users can also use the modified community script, resnet_ctl_imagenet_main.py, directly. The modifications include changing the default values of parameters, setting environment variables, changing include paths, etc. The changes are described in the license headers of the respective files.
### Using LARS optimizer
Using LARS optimizer usually requires changing the default values of some hyperparameters. This is done automatically in demo_resnet_keras.py, but needs to be passed manually when using resnet_ctl_imagenet_main.py. The recommended parameters together with their default values are presented below:

| Parameter          |   Value    |
| ------------------ | ---------- |
| optimizer          | LARS       |
| base_learning_rate | 2.5 or 9.5*|
| warmup_epochs      | 3          |
| lr_schedule        | polynomial |
| label_smoothing    | 0.1        |
| weight_decay       | 0.0001     |
| single_l2_loss_op  | True       |

*2.5 is the default value for single card (1 Gaudi) trainings, otherwise, the default is 9.5. These values have been determined experimentally.

The following commands would be equivalent:

1. Using custom wrapper
    ```bash
    $PYTHON demo_resnet_keras.py --optimizer LARS
    ```

2. Using modified community script
    ```bash
    $PYTHON resnet_ctl_imagenet_main.py --optimizer LARS --base_learning_rate 9.5 --warmup_epochs 3 --lr_schedule polynomial --label_smoothing 0.1 --weight_decay 0.0001 --single_l2_loss_op
    ```

### Training Examples
**Run training on 1 HPU**

- ResNet50, 1 Gaudi, batch 256, 90 epochs, bf16 precision, SGD, experimental preloading
    ```bash
    $PYTHON demo_resnet_keras.py -dt bf16 -dlit bf16 --train_epochs 90 -bs 256
    ```
- ResNet50, 1 Gaudi, batch 128, 1 epoch, fp32 precision, LARS
    ```bash
    $PYTHON demo_resnet_keras.py -bs 128 --optimizer LARS
    ```

**Run training on 8 HPU - Horovod**

- ResNet50, 8 Gaudis on 1 server, batch 256, 40 epochs, bf16 precision, LARS, experimental preloading
    ```bash
    $PYTHON demo_resnet_keras.py --dtype bf16 --data_loader_image_type bf16 --use_horovod --num_workers_per_hls 8 -te 40 -ebe 40 -bs 256 --optimizer LARS
    ```

**Run training on 8 HPU with mpirun and demo_resnet_keras.py - Horovod**

- You can also run demo_resnet_keras.py directly with mpirun. The following command will be roughly equivalent to the one presented above:
*<br>mpirun map-by PE attribute value may vary on your setup and should be calculated as:<br>
socket:PE = floor((number of physical cores) / (number of gaudi devices per each node))*
    ```bash
    mpirun --allow-run-as-root --bind-to core -np 8 --map-by socket:PE=7 --merge-stderr-to-stdout $PYTHON demo_resnet_keras.py --dtype bf16 --data_loader_image_type bf16 --use_horovod --num_workers_per_hls 8 --kubernetes_run=True -te 40 -ebe 40 -bs 256 --optimizer LARS
    ```

**Run training on 8 HPU with the modified community script - Horovod**

- You can also run the community script (resnet_ctl_imagenet_main.py) directly with mpirun. The following command will be roughly equivalent to the one presented above*:
    ```bash
    mpirun --allow-run-as-root --bind-to core -np 8 --map-by socket:PE=7 --merge-stderr-to-stdout $PYTHON resnet_ctl_imagenet_main.py --dtype bf16 --data_loader_image_type bf16 --use_horovod -te 40 -ebe 40 -bs 256 --optimizer LARS --base_learning_rate 9.5 --warmup_epochs 3 --lr_schedule polynomial --label_smoothing 0.1 --weight_decay 0.0001 --single_l2_loss_op
    ```

**Run training on 8 HPU - tf.distribute**

- ResNet50, 8 Gaudis on 1 server, batch 256, 40 epochs, bf16 precision, LARS
    ```bash
    $PYTHON demo_resnet_keras.py -dt bf16 -dlit bf16 --batch_size 2048 --distribution_strategy=hpu --num_workers_per_hls 8 -te 40 -ebe 40 --optimizer LARS
    ```
Note:
- Currently experimental_preloading (enabled by default) is required to run hpu distribution_strategy.
- Unlike Horovod, global batch size must be specified for tf.distribute. E.g. in this case 256 * 8 workers = 2048

**Run training on 16 HPU - multi-server - Horovod**

- ResNet50, 16 Gaudis on 2 boxes, batch 256, 40 epochs, bf16 precision, LARS, experimental preloading enabled
    ```bash
    # This environment variable is needed for multi-node training with Horovod.
    # Set this to be a comma-separated string of host IP addresses, e.g.:
    export MULTI_HLS_IPS=192.10.100.174,10.10.100.101

    # Set this to the network interface name or subnet that will be use by mpi to communicate e.g.:
    export MPI_TCP_INCLUDE=interface_name

    $PYTHON demo_resnet_keras.py -dt bf16 -dlit bf16 -bs 256 --use_horovod --num_workers_per_hls 8 -te 40 -ebe 40 --optimizer LARS
    ```
Note that to run multi-server over host NICs, use `--horovod_hierarchical_allreduce` flag.

**Run training on 16 HPU - multi-server with the modified community script - Horovod**

- You can also run the community script (resnet_ctl_imagenet_main.py) directly with mpirun
    ```bash
    mpirun --allow-run-as-root --mca plm_rsh_args -p3022 --bind-to core --map-by socket:PE=7 -np 16 --mca btl_tcp_if_include 192.10.100.174/24 --tag-output --merge-stderr-to-stdout --prefix /usr/local/openmpi/ -H 192.10.100.174:8,10.10.100.101:8 -x GC_KERNEL_PATH -x HABANA_LOGS -x PYTHONPATH -x HCL_CONFIG_PATH $PYTHON resnet_ctl_imagenet_main.py -dt bf16 -dlit bf16 -bs 256 --use_horovod
    ```
**Note that to run multi-server training over host NICs:**

- The resnet_ctl_imagenet_main.py `--horovod_hierarchical_allreduce` option must be set. This is required for AWS users.

**Top performance examples**

- Resnet 50 training on 8 HPU 1 box - Horovod
    ```bash
    $PYTHON demo_resnet_keras.py -dt bf16 -dlit bf16 -te 40 -ebe 40 --steps_per_loop 1000 --optimizer LARS --use_horovod --num_workers_per_hls 8 -bs 256
    ```

- Resnet 50 training on 32 HPU 4 boxes - Horovod
    ```bash
    export MULTI_HLS_IPS=IP1,IP2,IP3,IP4
    export MPI_TCP_INCLUDE: "interface_name"
    $PYTHON demo_resnet_keras.py -dt bf16 -dlit bf16 -te 40 -ebe 40 --steps_per_loop 1000 --optimizer LARS --use_horovod --num_workers_per_hls 8 -bs 256
    ```
