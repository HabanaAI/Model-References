# UNet2D for TensorFlow 2

This directory provides a script and recipe to train a UNet2D Medical model to achieve state of the art accuracy, and is tested and maintained by Habana.
For further information on performance, refer to [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

For further information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents

* [Model-References](../../../README.md)
* [Model overview](#model-overview)
* [Setup](#setup)
* [Training and Examples](#training-and-examples)
* [Advanced](#Advanced)
* [Supported Configuration](#supported-configuration)
* [Changelog](#changelog)

## Model Overview

This directory describes how to train UNet Medical model for 2D Segmentation on Habana Gaudi (HPU). The UNet Medical model is a modified version of the original model located in [NVIDIA UNet Medical Image Segmentation for TensorFlow 2.x](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/UNet_Medical). The implementation provided covers UNet model as described in the original [UNet: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) paper.

### Model Architecture

The UNet model allows seamless segmentation of 2D images with high accuracy and performance, and can be adapted to solve many different segmentation issues.

The following figure shows the construction of the UNet model and its components. A UNet is composed of a contractive and an expanding path, that aims at building a bottleneck in its centermost part through a combination of convolution and pooling operations. After this bottleneck, the image is reconstructed through a combination of convolutions and upsampling. Skip connections are added with the goal of helping the backward flow of gradients in order to improve the training.

![UNet](images/unet.png)
Figure 1. The architecture of a UNet model from [UNet: Convolutional Networks for Biomedical Image Segmentation paper](https://arxiv.org/abs/1505.04597).

### Model Changes

The below lists the major changes applied to the model:

* Removed GPU specific configurations.
* Changed some scripts to run the model on Gaudi. This includes loading Habana TensorFlow modules and using  multiple Gaudi cards helpers.
* Added support for using bfloat16 precision instead of float16.
* Replaced tf.keras.activations.softmax with tf.nn.softmax due to performance issues described in https://github.com/tensorflow/tensorflow/pull/47572;
* Added further TensorBoard and performance logging options.
* Removed GPU specific files (examples/*, Dockerfile etc.) and some unused codes.
* Enabled the tf.data.experimental.prefetch_to_device for HPU device to improve performance.

### Default Configuration

- Execution mode: train and evaluate
- Batch size: 8
- Data type: bfloat16
- Maximum number of steps: 6400
- Weight decay: 0.0005
- Learning rate: 0.0001
- Number of Horovod workers (HPUs): 1
- Data augmentation: True
- Cross-validation: disabled
- Using XLA: False
- Logging losses and performance every N steps: 100

## Setup

Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) to set up the environment including the `$PYTHON` environment variable.  To achieve the best performance, please follow the methods outlined in the [Optimizing Training Platform guide](https://docs.habana.ai/en/latest/TensorFlow/Model_Optimization_TensorFlow/Optimization_Training_Platform.html).  
The guides will walk you through the process of setting up your system to run the model on Gaudi.  


### Clone Habana Model-References

In the docker container, clone this repository and switch to the branch that matches your SynapseAI version. You can run the [`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the SynapseAI version.

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References /root/Model-References
```

**Note:** If Model-References repository path is not in the PYTHONPATH, make sure you update it:
```bash
export PYTHONPATH=$PYTHONPATH:/root/to/Model-References
```

### Install Model Requirements

1. In the docker container, go to the UNet2D directory:

```bash
cd /root/Model-References/TensorFlow/computer_vision/Unet2D
```
2. Install the required packages using pip:

```bash
$PYTHON -m pip install -r requirements.txt
```

### Download the Dataset

Download the [EM segmentation challenge dataset](http://brainiac2.mit.edu/isbi_challenge/home)*:

```bash
  $PYTHON download_dataset.py
```

By default, it will download the dataset to `./data` path. Use `--data_dir <path>` to change it.

**NOTE:** If the original location is unavailable, the dataset is also mirrored on [Kaggle](https://www.kaggle.com/soumikrakshit/isbi-challenge-dataset). Registration is required.

## Training and Examples

### Single Card and Multi-Card Training Examples

**Run training on 1 HPU:**

```bash
$PYTHON unet2d.py --data_dir <path/to/dataset> --batch_size <batch_size> \
--dtype <precision> --model_dir <path/to/model_dir> --fold <fold>
```

- 1 HPU training with batch size 8, bfloat16 precision and fold 0:

    ```bash
    $PYTHON unet2d.py --data_dir /data/tensorflow/unet2d --batch_size 8 --dtype bf16 --model_dir /tmp/unet2d_1_hpu --fold 0 --tensorboard_logging
    ```

- 1 HPU training with batch size 8, float32 precision and fold 0:

    ```bash
    $PYTHON unet2d.py --data_dir /data/tensorflow/unet2d --batch_size 8 --dtype fp32 --model_dir /tmp/unet2d_1_hpu --fold 0 --tensorboard_logging
    ```

**Run training on 8 HPUs:**

Running the script via mpirun requires`--use_horovod` argument, and the mpirun prefix with several parameters.

**NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration).

```bash
mpirun --allow-run-as-root --bind-to core --map-by socket:PE=6 -np 8 \
 $PYTHON unet2d.py --data_dir <path/to/dataset> --batch_size <batch_size> \
 --dtype <precision> --model_dir <path/to/model_dir> --fold <fold> --use_horovod
```
- 8 HPUs training with batch size 8, bfloat16 precision and fold 0:

    ```bash
    mpirun --allow-run-as-root --tag-output --merge-stderr-to-stdout --bind-to core --map-by socket:PE=6 -np 8 \
    $PYTHON unet2d.py --data_dir /data/tensorflow/unet2d/ --batch_size 8 \
    --dtype bf16 --model_dir /tmp/unet2d_8_hpus --fold 0 --tensorboard_logging --log_all_workers --use_horovod
    ```
- 8 HPUs training with batch size 8, float32 precision and fold 0:

    ```bash
    mpirun --allow-run-as-root --tag-output --merge-stderr-to-stdout --bind-to core --map-by socket:PE=6 -np 8 \
    $PYTHON unet2d.py --data_dir /data/tensorflow/unet2d/ --batch_size 8 \
    --dtype fp32 --model_dir /tmp/unet2d_8_hpus --fold 0 --tensorboard_logging --log_all_workers --use_horovod
    ```

**Run 5-fold Cross-Validation and compute average dice score:**

All the commands described above train and evaluate the model on the dataset with fold 0. To perform 5-fold-cross-validation on the dataset and compute average dice score across 5 folds, you can execute training the script 5 times and calculate the average dice score manually or run bash script `train_and_evaluate.sh`:

```bash
bash train_and_evaluate.sh <path/to/dataset> <path/for/results> <batch_size> <precision> <number_of_HPUs>
```

- 1 HPU 5-fold-cross-validation training with batch size 8 and bfloat16 precision:

    ```bash
    bash train_and_evaluate.sh /data/tensorflow/unet2d/ /tmp/unet2d_1_hpu 8 bf16 1
    ```

- 1 HPU 5-fold-cross-validation training with batch size 8 and float32 precision:

    ```bash
    bash train_and_evaluate.sh /data/tensorflow/unet2d/ /tmp/unet2d_1_hpu 8 fp32 1
    ```

- 8 HPUs 5-fold-cross-validation training with batch size 8 and bfloat16 precision:

    ```bash
    bash train_and_evaluate.sh /data/tensorflow/unet2d/ /tmp/unet2d_8_hpus 8 bf16 8
    ```

- 8 HPUs 5-fold-cross-validation training with batch size 8 and float32 precision:

    ```bash
    bash train_and_evaluate.sh /data/tensorflow/unet2d/ /tmp/unet2d_8_hpus 8 fp32 8
    ```

## Advanced

The following sections provide further details on the scripts in the directory, available parameters and command-line options.

### Scripts Definitions

* `unet2d.py`: The training script of the UNet2D model, entry point to the application.
* `download_dataset.py`: Script for downloading dataset.
* `data_loading/data_loader.py`: Implements the data loading and augmentation.
* `model/layers.py`: Defines the different blocks that are used to assemble UNet.
* `model/unet.py`: Defines the model architecture using the blocks from the `layers.py` script.
* `runtime/arguments.py`: Implements the command-line arguments parsing.
* `runtime/losses.py`: Implements the losses used during training and evaluation.
* `runtime/run.py`: Implements the logic for training, evaluation, and inference.
* `runtime/parse_results.py`: Implements the intermediate results parsing.
* `runtime/setup.py`: Implements helper setup functions.
* `train_and_evaluate.sh`: Runs the topology training and evaluates the model for 5 cross-validation.

Other folders included in the root directory are:
* `images/`: Contains a model diagram.

### Parameters

The complete list of the available parameters for the `unet2d.py` script contains:
* `--exec_mode`: Select the execution mode to run the model (default: `train_and_evaluate`). Modes available:
  * `train` - trains model from scratch.
  * `evaluate` - loads checkpoint from `--model_dir` (if available) and performs evaluation on validation subset (requires `--fold` other than `None`).
  * `train_and_evaluate` - trains model from scratch and performs validation at the end (requires `--fold` other than `None`).
  * `predict` - loads checkpoint from `--model_dir` (if available) and runs inference on the test set. Stores the results in `--model_dir` directory.
  * `train_and_predict` - trains model from scratch and performs inference.
* `--model_dir`: Set the output directory for information related to the model (default: `/tmp/unet2d`).
* `--data_dir`: Set the input directory containing the dataset (default: `None`).
* `--log_dir`: Set the output directory for logs (default: `/tmp/unet2d`).
* `--batch_size`: Size of each minibatch per HPU (default: `8`).
* `--dtype`: Set precision to be used in model: fp32/bf16 (default: `bf16`).
* `--fold`: Selected fold for cross-validation (default: `None`).
* `--max_steps`: Maximum number of steps (batches) for training (default: `6400`).
* `--log_every`: Log data every n steps (default: `100`).
* `--evaluate_every`: Evaluate every n steps (default: `0` - evaluate once at the end).
* `--warmup_steps`: Used during benchmarking - the number of steps to skip (default: `200`). First iterations are usually much slower since the graph is being constructed. Skipping the initial iterations is required for a fair performance assessment.
* `--weight_decay`: Weight decay coefficient (default: `0.0005`).
* `--learning_rate`: Model’s learning rate (default: `0.0001`).
* `--seed`: Set random seed for reproducibility (default: `123`).
* `--dump_config`: Directory for dumping debug traces (default: `None`).
* `--augment`: Enable data augmentation (default: `True`).
* `--benchmark`: Enable performance benchmarking (default: `False`). If the flag is set, the script runs in a benchmark mode - each iteration is timed and the performance result (in images per second) is printed at the end. Works for both `train` and `predict` execution modes.
* `--xla`: Enable accelerated linear algebra optimization (default: `False`).
* `--resume_training`: Resume training from a checkpoint (default: `False`).
* `--no_hpu`: Disable execution on HPU, train on CPU  (default: `False`).
* `--synth_data`: Use deterministic and synthetic data (default: `False`).
* `--disable_ckpt_saving`: Disables saving checkpoints (default: `False`).
* `--use_horovod`: Enable horovod usage (default: `False`).
* `--tensorboard_logging`: Enable tensorboard logging (default: `False`).
* `--log_all_workers`: Enable logging data for every horovod worker in a separate directory named `worker_N` (default: False).
* `--bf16_config_path`: Path to custom mixed precision config to use given in JSON format.
* `--tf_verbosity`: If set changes logging level from Tensorflow:
    * `0` - all messages are logged (default behavior);
    * `1` - INFO messages are not printed;
    * `2` - INFO and WARNING messages are not printed;
    * `3` - INFO, WARNING, and ERROR messages are not printed.

### Command-line Options

To see the full list of the available options and their descriptions, use the `-h` or `--help` command-line option, for example:

```bash
$PYTHON unet2d.py --help
```

## Supported Configuration

| Validated on | SynapseAI Version | TensorFlow Version(s) | Mode |
|:------:|:-----------------:|:-----:|:----------:|
| Gaudi   | 1.11.0             | 2.12.1         | Training |
| Gaudi2  | 1.11.0             | 2.12.1         | Training |

## Changelog
### 1.12.0
- Removed limited number of nodes inserted into a single HPU graph.

### 1.11.0

- Limited number of nodes inserted into a single HPU graph to improve model performance.

### 1.10.0

- Changed default seed value for higher accuracy.

### 1.7.0

- Added TimeToTrain callback for dumping evaluation timestamps

### 1.6.0

- Model enabled on Gaudi2 with the same config as first-gen Gaudi.
- Added num_parallel_calls for data loader to improve performance on Gaudi2.

### 1.4.0

- Enabled tf.data.experimental.prefetch_to_device for HPU device for better performance.
- Changed `python` or `python3` to `$PYTHON` to execute correct version based on environment setup.
- Added support to import horovod-fork package directly instead of using Model-References' TensorFlow.common.horovod_helpers. Wrapped horovod import with a try-catch block so that installing the library is not required when the model is running on a single card.
- Replaced references to custom demo script by community entry points in the README and
`train_and_evaluate.sh`.

### 1.3.0

- Moved BF16 config json file from TensorFlow/common/ to model's directory.
- Updated the requirements.txt

### 1.2.0

- Removed the setting number of parallel calls in dataloader mapping to improve performance for different TensorFlow versions.
- Updated requirements.txt

