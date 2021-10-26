# 3D-UNet Medical Image Segmentation for TensorFlow 2.x

This repository provides a script and recipe to train 3D-UNet to achieve state of the art accuracy, and is tested and maintained by Habana Labs, Ltd. an Intel Company.

## Table of Contents

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

* [Model-References](../../../README.md)
* [Model overview](#model-overview)
* [Setup](#setup)
* [Training](#training)
* [Advanced](#advanced)

## Model overview

The U-Net model is a convolutional neural network for 3D image segmentation. This repository contains a 3D-UNet implementation introduced in [3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://arxiv.org/pdf/1606.06650), with modifications described in [No New-Net](https://arxiv.org/pdf/1809.10483). It is based on [3D-UNet Medical Image Segmentation for TensorFlow 1.x](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Segmentation/UNet_3D_Medical) repository.

### Model architecture

3D-UNet was first introduced by Olaf Ronneberger, Philip Fischer, and Thomas Brox in the paper: [3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://arxiv.org/pdf/1606.06650). In this repository we host a 3D-UNet version adapted by Fabian Isensee et al. to brain tumor segmentation. 3D-UNet allows for seamless segmentation of 3D volumes, with high accuracy and performance, and can be adapted to solve many different segmentation problems.

The following figure shows the construction of the 3D-UNet model and its different components. 3D-UNet is composed of a contractive and an expanding path, that aims at building a bottleneck in its centermost part through a combination of convolution and pooling operations. After this bottleneck, the image is reconstructed through a combination of convolutions and upsampling. Skip connections are added with the goal of helping the backward flow of gradients in order to improve the training.

![U-Net3D](images/unet3d.png)

3D-UNet consists of a contractive (left-side) and expanding (right-side) path. It repeatedly applies unpadded convolutions followed by max pooling for downsampling. Every step in the expanding path consists of an upsampling of the feature maps and a concatenation with the correspondingly cropped feature map from the contractive path.

### Model changes
Major changes done to original model from [3D-UNet Medical Image Segmentation for TensorFlow 1.x](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Segmentation/UNet_3D_Medical):
* Some scripts were changed in order to run the model on Gaudi. It includes loading habana tensorflow modules and using multi Gaudi card helpers;
* All scripts have been converted to Tensorflow 2.x version;
* Model is using bfloat16 precision instead of float16;
* Additional tensorboard and performance logging options were added;
* GPU specific files (examples/*, Dockerfile etc.) and some unused code have been removed.

### Default configuration
- Execution mode: train and evaluate;
- Batch size: 2;
- Data type: bfloat16;
- Maximum number of steps: 16000;
- Learning rate: 0.0002;
- Loss: dice+ce;
- Normalization block: instancenorm;
- Include background in predictions and labels: False;
- Number of Horovod workers (HPUs): 1;
- Data augmentation: True;
- Using XLA: True;
- Resume training from checkpoint: False;
- Logging losses and performance every N steps: 100;
- Tensorboard logging: False;
- Logging data from every worker: False;

# Setup
Please follow the instructions given in the following link for setting up the
environment including the `$PYTHON` environment variable: [Gaudi Setup and
Installation Guide](https://github.com/HabanaAI/Setup_and_Install). Please
answer the questions in the guide according to your preferences. This guide will
walk you through the process of setting up your system to run the model on
Gaudi.

### Clone Habana Model garden and go to UNet3D repository:
```bash
git clone https://github.com/HabanaAI/Model-References /root/Model-References
cd /root/Model-References/TensorFlow/computer_vision/UNet3D
```

### Add Tensorflow packages from model_garden to python path:
```bash
export PYTHONPATH=$PYTHONPATH:/root/Model-References/
```

### Download and pre-process the dataset:

Dataset can be obtained by registering on [Brain Tumor Segmentation 2019 dataset](https://www.med.upenn.edu/cbica/brats-2019/) website. The data should be downloaded and placed where `/dataset` in the container is mounted. The `dataset/preprocess_data.py` script will convert the raw data into tfrecord format used for training and evaluation.

The script can be launched as:
```bash
$PYTHON -m pip install -r requirements.txt
$PYTHON dataset/preprocess_data.py -i /dataset -o /dataset_preprocessed -v
```

## Training
The model was tested both in single Gaudi and 8x Gaudi cards configurations.

### Run training on single Gaudi card

```bash
$PYTHON demo_unet3d.py --data_dir <path/to/dataset> --dtype <precision> --model_dir <path/to/model_dir> --log_dir <path/to/log_dir>
```

For example:
- single Gaudi card training with batch size 2, bfloat16 precision and fold 0:
    ```bash
    $PYTHON demo_unet3d.py --data_dir /dataset_preprocessed --dtype bf16 --model_dir /tmp/unet3d_1_hpu --log_dir /tmp/unet3d_1_hpu
    ```

### Run training on single server (8 Gaudi cards)

```bash
$PYTHON demo_unet3d.py --data_dir <path/to/dataset> --dtype <precision> --model_dir <path/to/model_dir> --log_dir <path/to/log_dir> --num_workers_per_hls 8
```

For example:
- 8 Gaudi cards training with batch size 2, bfloat16 precision and fold 0:
    ```bash
    $PYTHON demo_unet3d.py --data_dir /dataset_preprocessed --dtype bf16 --model_dir /tmp/unet3d_8_hpus --log_dir /tmp/unet3d_8_hpus --num_workers_per_hls 8
    ```

### Run training on single server (8 Gaudi cards) via mpirun
Running the script via mpirun requires providing HCL config file and `--use_horovod` argument. For documentation on creating an HCL config JSON file, please refer to [HCL JSON Config File Format](https://docs.habana.ai/en/latest/API_Reference_Guides/HCL_API_Reference.html#hcl-json-config-file-format).

For example to run 8 Gaudi cards training via mpirun with batch size 2, bfloat16 precision and fold 0:
*<br>mpirun map-by PE attribute value may vary on your setup and should be calculated as:<br>
socket:PE = floor((number of physical cores) / (number of gaudi devices per each node))*
```bash
mpirun --allow-run-as-root --bind-to core --map-by socket:PE=7 --np 8 \
$PYTHON main.py --use_horovod --data_dir /dataset_preprocessed --dtype bf16 --model_dir /tmp/unet3d_8_hpus --log_dir /tmp/unet3d_8_hpus
```

### Run 5-fold Cross-Validation and compute average dice score
All the commands described above will train and evaluate the model on the dataset with fold 0. To perform 5-fold-cross-validation on the dataset and compute average dice score across 5 folds, the user can execute training script 5 times and calculate the average dice score manually or run bash script `5-cross-validation.sh`:
```bash
bash 5-cross-validation.sh <path/to/dataset> <path/for/results> <batch_size> <precision>
```
For example:
- 8 Gaudi cards 5-fold-cross-validation with batch size 2 and bfloat16 precision
    ```bash
    bash 5-cross-validation.sh /dataset_preprocessed /tmp/unet3d_8_hpus 2 bf16
    ```

### Benchmark
To measure performance of the training on HPU in terms of images per second user can take the data printed during regular training or run the script in a dedicated benchmark mode. Benchmark mode will calculate the average of throughput between warmup and max steps. For example to run benchmark mode:
```bash
$PYTHON demo_unet3d.py --data_dir <path/to/dataset> --dtype <precision> --model_dir <path/to/model_dir> --log_dir <path/to/log_dir> --num_workers_per_hls <number of Gaudi cards> --exec_mode train --benchmark --warmup_steps 40 --max_steps 80
```
For example:
- single Gaudi card benchmark with batch size 8 and bfloat16 precision:
    ```bash
    $PYTHON demo_unet3d.py --data_dir /dataset_preprocessed --dtype bf16 --model_dir /tmp/unet3d_1_hpu --log_dir /tmp/unet3d_1_hpu --num_workers_per_hls 1 --exec_mode train --benchmark --warmup_steps 40 --max_steps 80
    ```
- 8 Gaudi cards benchmark with batch size 8 and bfloat16 precision:
    ```bash
    $PYTHON demo_unet3d.py --data_dir /dataset_preprocessed --dtype bf16 --model_dir /tmp/unet3d_8_hpus --log_dir /tmp/unet3d_8_hpus --num_workers_per_hls 8 --exec_mode train --benchmark --warmup_steps 40 --max_steps 80
    ```
- 8 Gaudi cards benchmark via mpirun with batch size 8 and bfloat16 precision:
    ```bash
    mpirun --allow-run-as-root --bind-to core --map-by socket:PE=7 --np 8 \
    $PYTHON main.py --use_horovod --data_dir /dataset_preprocessed --dtype bf16 --model_dir /tmp/unet3d_8_hpus --log_dir /tmp/unet3d_8_hpus --num_workers_per_hls 8 --exec_mode train --benchmark --warmup_steps 40 --max_steps 80
    ```

## Advanced

The following sections provide greater details of the dataset, running training and inference, and the training results.

### Scripts and sample code

In the root directory, the most important files are:
* `demo_unet3d.py`: Demo distributed launcher script, that gives a possibility to run the training in multi Gaudi card configuration without any additional setup requirements from user. If it is executed with `--num_workers_per_hls > 1` argument, it applies required command line prefix for OpenMPI.
* `main.py`: Serves as the entry point to the application. Encapsulates the training routine.
* `requirements.txt`: Set of extra requirements for running U-Net.

The `dataset/` folder contains the necessary tools to train and perform inference using U-Net. Its main components are:
* `data_loader.py`: Implements the data loading and augmentation.
* `transforms.py`: Implements the data augmentation functions.
* `preprocess_data.py`: Implements the data conversion and pre-processing functionality.

The `runtime/` folder contains scripts with training and inference logic. Its contents are:
* `arguments.py`: Implements the command-line arguments parsing.
* `hooks.py`: Collects different metrics to be used for benchmarking and testing.
* `parse_results.py`: Defines a set of functions used for parsing the partial results.
* `setup.py`: Defines a set of functions to set the environment up.

 The `model/` folder contains information about the building blocks of 3D-UNet and the way they are assembled. Its contents are:
* `layers.py`: Defines the different blocks that are used to assemble 3D-UNet.
* `losses.py`: Defines the different losses used during training and evaluation.
* `model_fn.py`: Defines the computational graph to optimize.
* `unet3d.py`: Defines the model architecture using the blocks from the `layers.py` file.

Other folders included in the root directory are:
* `images/`: Contains the model diagram

### Parameters

The complete list of the available parameters for the `demo_unet3d.py` script contains:
* `--exec_mode`: Select the execution mode to run the model (default: `train_and_evaluate`). Modes available:
  * `train` - trains a model and stores checkpoints in the directory passed using `--model_dir`
  * `evaluate` - loads checkpoint (if available) and performs evaluation on validation subset (requires `--fold` other than `None`).
  * `train_and_evaluate` - trains model from scratch and performs validation at the end (requires `--fold` other than `None`).
  * `predict` - loads checkpoint (if available) and runs inference on the test set. Stores the results in the `--model_dir` directory.
  * `train_and_predict` - trains model from scratch and performs inference.
* `--model_dir`: Set the output directory for information related to the model.
* `--log_dir`: Set the output directory for logs (default: `/tmp/unet3d_logs`).
* `--data_dir`: Set the input directory containing the preprocessed dataset.
* `--batch_size`: Size of each minibatch per device (default: `2`).
* `--dtype`: Set precision to be used in model on HPU: fp32/bf16 (default: `bf16`).
* `--bf16_config_path`: Path to custom mixed precision config to be used (default: `../../common/bf16_config/unet.json`).
* `--fold`: Selected fold for cross-validation (default: `0`).
* `--num_folds`: Number of folds in k-cross-validation of dataset (default: `5`).
* `--max_steps`: Maximum number of steps (batches) for training (default: `16000`).
* `--seed`: Set random seed for reproducibility (default: `None`).
* `--log_every`: Log performance every n steps (default: `100`).
* `--learning_rate`: Model’s learning rate (default: `0.0002`).
* `--loss`: Loss function to be used during training (default: `dice+ce`).
* `--normalization`: Normalization block to be applied in the model (default: `instancenorm`).
* `--include_background`: Include background both in preditions and labels (default: `False`).
* `--no-augment`: Disable data augmentation (enabled by default).
* `--benchmark`: Enable performance benchmarking (disabled by default). If the flag is set, the script runs in a benchmark mode - each iteration is timed and the performance result (in images per second) is printed at the end. Works for both `train` and `predict` execution modes.
* `--warmup_steps`: Used during benchmarking - the number of steps to skip (default: `40`). First iterations are usually much slower since the graph is being constructed. Skipping the initial iterations is required for a fair performance assessment.
* `--resume_training`: Whether to resume training from a checkpoint, if there is one (disabled by default)
* `--no_xla`: Disable accelerated linear algebra optimization (enabled by default).
* `--use_amp`: Enable automatic mixed precision for GPU (disabled by default).
* `--no_hpu`: Disable execution on HPU, train on CPU/GPU (default: `False`).
* `--num_workers_per_hls`: Set number of Gaudi cards to be used for single server (default: `1`).
* `--kubernetes_run`: Indicates if the model is executed on kubernetes (default: `False`).
* `--use_horovod`: Enable horovod usage (default: `False`).
* `--tensorboard_logging`: Enable tensorboard logging (default: `False`).
* `--log_all_workers`: Enable logging data for every horovod worker in a separate directory named `worker_N` (default: `False`).

### Command line options

To see the full list of available options and their descriptions, use the `-h` or `--help` command-line option, for example:
```bash
$PYTHON demo_unet3d.py --help
```

### Dataset description

The 3D-UNet model was trained in the [Brain Tumor Segmentation 2019 dataset](https://www.med.upenn.edu/cbica/brats-2019/). Test images provided by the organization were used to produce the resulting masks for submission. Upon registration, the challenge's data is made available through the https//ipp.cbica.upenn.edu service.

The dataset consists of 335 240x240x155 `nifti` volumes. Each volume is represented by 4 modalities and a corresponding segmentation mask.
The modalities are:
* Native T1-weighted (T1),
* Post-contrast T1-weighted (T1Gd),
* Native T2-weighted (T2),
* T2 Fluid Attenuated Inversion Recovery (FLAIR).

Each voxel in a segmentation mask belongs to one of four classes:
* 0 corresponds to healthy tissue or background,
* 1 indicates the presence of the necrotic and non-enhancing tumor core (TC),
* 2 indicates the presence of the peritumoral edema (ED),
* 4 indicates the presence of the GD-enhancing tumor (ET).

The objective is to produce a set of masks that segment the data as accurately as possible. The results are expected to be submitted as a 12-bit `nifti` 3D image, with values corresponding to the underlying class.

### Dataset guidelines

The training and test datasets are given as 3D `nifti` volumes that can be read using the Nibabel library and NumPy.

Initially, all modalities are loaded, stacked and converted into 240x240x155x4 NumPy arrays using Nibabel. To decrease the size of the dataset, each volume is clipped to 85% of the maximal value, normalized to 255 for each modality separately, casted to 8-bit, grouped by 4 volumes, and saved as a `tfrecord` file. The process of converting from `nifti` to `tfrecord` can be found in the `preprocess_data.py` script.

The `tfrecord` files are fed to the model through `tf.data.TFRecordDataset()` to achieve high performance.

The foreground voxel intensities then z-score normalized, whereas labels are one-hot encoded for their later use in dice or pixel-wise cross-entropy loss, becoming 240x240x155x4 tensors.

If augmentation is enabled, the following set of augmentation techniques are applied:
* Random horizontal flipping
* Random 128x128x128x4 crop
* Random brightness shifting

In addition, random vertical flip and random gamma correction augmentations were implemented, but are not used. The process of loading, normalizing and augmenting the data contained in the dataset can be found in the `data_loader.py` script.

#### Multi-dataset

This implementation is tuned for the Brain Tumor Segmentation 2019 dataset. Using other datasets is possible, but might require changes to the code (data loader) and tuning some hyperparameters (e.g. learning rate, number of iterations).

In the current implementation, the data loader works with tfrecord files. It should work seamlessly with any dataset containing 3D data stored in tfrecord format, as long as features (with corresponding mean and standard deviation) and labels are stored as bytestream in the same file as `X`, `Y`, `mean`, and `stdev`.  See the data pre-processing script for details. If your data is stored in a different format, you will have to modify the parsing function in the `dataset/data_loader.py` file. For a walk-through, check the [TensorFlow tf.data API guide](https://www.tensorflow.org/guide/data_performance)

### Training process

The model trains for a total 16,000 (16,000 / number of devices) iterations for each fold, with the default 3D-UNet setup:
* Adam optimizer with learning rate of 0.0002.
* Training and evaluation batch size of 2.

The default configuration minimizes a function _L = 1 - DICE + cross entropy_ during training and reports achieved convergence as dice score per class, mean dice score, and dice score for whole tumor vs background. The training with a combination of dice and cross entropy has been proven to achieve better convergence than a training using only dice.

If the `--exec_mode train_and_evaluate` parameter was used, and if `--fold` parameter is set to an integer value of {0, 1, 2, 3, 4}, the evaluation of the validation set takes place after the training is completed. The results of the evaluation will be printed to the console.

### Inference process

Inference can be launched with the same script used for training by passing the `--exec_mode predict` flag:
```bash
$PYTHON main.py --exec_mode predict --data_dir <path/to/data/preprocessed> --model_dir <path/to/checkpoint> [other parameters]
```

The script will then:
* Load the checkpoint from the directory specified by the `<path/to/checkpoint>` directory,
* Run inference on the test dataset,
* Save the resulting masks in the `numpy` format in the `--model_dir` directory.
