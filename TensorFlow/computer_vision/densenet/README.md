# DenseNet Keras model

This repository provides a script and recipe to train DenseNet121 to achieve state of the art accuracy, and is tested and maintained by Habana Labs, Ltd. an Intel Company.

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

## Table Of Contents
  * [Model Overview](#model-overview)
  * [Setup](#setup)
  * [Training](#training)
  * [Advanced](#advanced)

## Model overview

Paper: [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

### Model changes
Major changes done to original model from [https://github.com/flyyufelix/cnn_finetune](https://github.com/flyyufelix/cnn_finetunel):
* Some scripts were changed in order to run the model on Gaudi. It includes loading habana tensorflow modules and using multi Gaudi card helpers;
* Support for distributed training using HPUStrategy was addedd;
* Data pipeline was optimized;
* Color distortion and random image rotation was removed from training data augmentation;
* Learning rate scheduler with warmup was added and set as default;
* Kernel and bias regularization was added Conv2D and Dense layers;
* Additional tensorboard and performance logging options were added;
* Additional synthetic data and tensor dumping options were added.


## Setup

Please follow the instructions given in the following link for setting up the
environment including the `$PYTHON` environment variable: [Gaudi Setup and
Installation Guide](https://github.com/HabanaAI/Setup_and_Install). Please
answer the questions in the guide according to your preferences. This guide will
walk you through the process of setting up your system to run the model on
Gaudi.

### Training Data

The script operates on ImageNet 1k, a widely popular image classification dataset from the ILSVRC challenge.
In order to obtain the dataset, follow these steps:
1. Sign up with http://image-net.org/download-images and acquire the rights to download original images
2. Follow the link to the 2012 ILSVRC
 and download `ILSVRC2012_img_val.tar` and `ILSVRC2012_img_train.tar`.
3. Ensure python3 and the following Python packages are installed: Tensorflow 2.5.0 and `absl-py`.
4. Use the commands below - they will prepare the dataset under `/data/tensorflow_datasets/imagenet/tf_records`. This is the default data_dir for the training script.

```bash
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
Go to the Densenet directory:
```bash
cd Model-References/TensorFlow/computer_vision/densenet
```
### Training Examples
**Run training on 1 HPU**

- DenseNet121, 1 HPU, batch 256, 90 epochs, bf16 precision, SGD
    ```bash
    $PYTHON demo_densenet.py \
        --dataset_dir /data/tensorflow_datasets/imagenet/tf_records \
        --dtype bf16 \
        --model densenet121 \
        --optimizer    sgd \
        --batch_size   256 \
        --epochs       90 \
        --run_on_hpu
    ```
- DenseNet121, 1 HPU, batch 128, 1 epoch, fp32 precision, SGD
    ```bash
    $PYTHON demo_densenet.py \
        --dataset_dir /data/tensorflow_datasets/imagenet/tf_records \
        --dtype fp32 \
        --model densenet121 \
        --optimizer    sgd \
        --batch_size   128 \
        --epochs       1 \
        --run_on_hpu
    ```

**Run training on 8 HPU - tf.distribute**

- DenseNet121, 8 HPUs on 1 server, batch 2048 (256 per HPU), 90 epochs, bf16 precision, SGD
    ```bash
    $PYTHON demo_densenet.py \
        --dataset_dir /data/tensorflow_datasets/imagenet/tf_records \
        --dtype bf16 \
        --model densenet121 \
        --optimizer    sgd \
        --batch_size   2048 \
        --epochs       90 \
        --run_on_hpu \
        --use_hpu_strategy \
        --num_workers_per_hls 8
    ```

**Run training on 8 HPU with the modified community script - tf.distribute**

- You can also run demo_densenet.py directly with mpirun. The following command will be roughly equivalent to the one presented above:
*<br>mpirun map-by PE attribute value may vary on your setup and should be calculated as:<br>
socket:PE = floor((number of physical cores) / (number of gaudi devices per each node))*
    ```bash
    HCL_CONFIG_PATH=hcl_config.json mpirun --allow-run-as-root --bind-to core --map-by socket:PE=4 -np 8 \
    python3 train.py \
        --dataset_dir /data/tensorflow_datasets/imagenet/tf_records \
        --dtype bf16 \
        --model densenet121 \
        --optimizer sgd \
        --batch_size 2048 \
        --initial_lr 1e-1 \
        --epochs 90 \
        --run_on_hpu \
        --use_hpu_strategy \
        --num_workers_per_hls 8
    ```

Note that user needs to manually generate an HCL config JSON file as described in `Example 1 - Single server format` in [HCL JSON Config File Format](https://docs.habana.ai/en/latest/API_Reference_Guides/HCL_API_Reference.html#hcl-json-config-file-format).

## Advanced

The following sections provide greater details of the dataset, running training and inference, and the training results.

### Scripts and sample code

In the root directory, the most important files are:
* `demo_densenet.py`: Demo distributed launcher script, that gives a possibility to run the training in multi Gaudi card configuration without any additional setup requirements from user. If it is executed with `--num_workers_per_hls > 1` argument, it applies required command line prefix for OpenMPI.
* `train.py`: Serves as the entry point to the application. Encapsulates the training routine.
* `densenet.py`: Defines the model architecture.

The `utils/` folder contains the necessary utilities to train DenseNet model. Its main components are:
* `arguments.py`: Defines command line argumnet parser.
* `dataset.py`: Implements the functions defining the data pipeline.
* `image_processing.py`: Implements the data preprocessing and augmentation.

The `models/` folder contains utilities allowing for training
* `models.py`: Defines utilities necessary for training.
* `optimizers.py`: Defines optimizer utilities.

### Parameters

The list of the available parameters for the `demo_densenet.py` script contains:
```
Optional arguments:
  Name                          Type        Description
  --dataset_dir                 string      Path to dataset (default: /data/imagenet/tf_records)
  --model_dir                   string      Directory for storing saved model and logs (default:./)
  --dtype                       string      Training precision {fp32,bf16} (default: bf16)
  --dropout_rate                float       Dropout rate (default: 0.000000)
  --optimizer                   string      Optimizer used for training {sgd,adam,rmsprop}
  --batch_size                  int         Training batch size (default: 256)
  --initial_lr                  float       Initial lerning rate (default: 0.100000)
  --weight_decay                float       Weight decay (default: 0.000100)
  --epochs                      int         Total number of epochs for training (default: 90)
  --steps_per_epoch             int         Number of steps per epoch (default: None)
  --validation_steps            int         Number of validation steps, set to 0 to disable validation (default: None)
  --model                       string      Model size {densenet121,densenet161,densenet169}
  --train_subset                string      Name of training subset from dataset_dir
  --val_subset                  string      Name of validation subset from dataset_dir
  --resume_from_checkpoint_path string      Path to checkpoint from which to resume training (default: None)
  --resume_from_epoch           int         Epoch from which to resume training (used in conjunction with resume_from_checkpoint_path argument) (default: 0)
  --evaluate_checkpoint_path    string      Checkpoint path for evaluating the model on --val_subset (default: None)
  --seed                        int         Random seed (default: None)
  --warmup_epochs               int         Number of epochs with learning rate warmup (default: 5)
  --save_summary_steps          int         Steps between saving summaries to TensorBoard; when None, logging to TensorBoard is disabled. (enabling this option might affect the performance) (default: None)
  --num_workers_per_hls         int         number of workers per server (default: 1)
  --hls_type                    string      Type of server on which the training is conducted (default: HLS1)

Optional switches:
  Name                          Description
  --run_on_hpu                  Whether to use HPU for training (default: False)
  --use_hpu_strategy            Enables HPU strategy for distributed training
  --kubernetes_run              Whether it's kubernetes run (default: False)
  -h, --help                    Show this help message and exit
```
