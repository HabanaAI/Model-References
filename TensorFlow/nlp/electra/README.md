# ELECTRA for TensorFlow

This repository provides a script and recipe to pre-train and fine-tune the ELECTRA model for Tensorflow on Habana Gaudi device. For further information on performance, refer to [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

For more information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents

* [Model-References](../../../README.md)
* [Model Overview](#model-overview)
* [Setup](#setup)
* [Pre-training, Fine-tuning and Examples](#pre-training-fine-tuning-and-examples)
* [Advanced](#advanced)
* [Supported Configuration](#supported-configuration)
* [Changelog](#changelog)

## Model Overview

Electra (Efficiently Learning an Encoder that Classifies Token Replacements Accurately), is a novel pre-training method for language representations which outperforms existing techniques, given the same compute budget on a wide array of Natural Language Processing (NLP) tasks. This model is based on the [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators paper](https://openreview.net/forum?id=r1xMH1BtvB).

The scripts are Habana modified pre-training and fine-tuning scripts taken from [NVIDIA GitHub](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/LanguageModeling/ELECTRA) which in turn is an optimized version of the [Hugging Face implementation](https://huggingface.co/transformers/model_doc/electra.html). The script has been embedded with Habana device support.

This repository contains the scripts to interactively launch data download and training in a Docker container for pre-training on your own dataset (Wikipedia and BookCorpus shown as an example), and fine-tuning on SQuAD dataset.

### Model Changes

The model implementation is based on ELECTRA BASE configuration.

## Setup

Please follow the instructions provided in the [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the
environment including the `$PYTHON` environment variable.
The guide will walk you through the process of setting up your system to run the model on Gaudi.

### Clone Habana Model-References

In the docker container, clone this repository and switch to the branch that matches your SynapseAI version. You can run the [`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the SynapseAI version.

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References /root/Model-References
```

**Note:** If Model-References repository path is not in the PYTHONPATH, make sure you update it:
```bash
export PYTHONPATH=$PYTHONPATH:/root/Model-References
```

### Install Model Requirements

1. Go to the ELECTRA directory:

```bash
cd /root/Model-References/TensorFlow/nlp/electra
```

2. Install the required packages using pip:
```bash
$PYTHON -m pip install -r requirements.txt
```

### Download and Pre-process Datasets

Please note that the pre-trained model variant, ELECTRA Base, will be automatically downloaded the first time model training with `run_pretraining.py` is run in the docker container. These next steps will go over how to download the training datasets.

Please follow the dataset download instructions in BERT README under [download dataset](../bert/README.md#download-and-preprocess-the-datasets-for-pretraining-and-finetuning-for-non-k8s-and-k8s-configurations) section to download below mentioned datsets:

- [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) (fine-tuning for question answering)
- Wikipedia (pre-training)
- BookCorpus (pre-training)

#### Download Pre-training Datasets

For more information on ELECTRA pre-training, refer to [Pre-training Dataset section](../bert/README.md#pretraining-datasets-download-instructions).

**Note:** The unpacked dataset version is used for electra. Post creation of dataset, it will be located under `/data/tensorflow/bert/books_wiki_en_corpus/tfrecord/seq_len_128/books_wiki_en_corpus/training/` for `seq_len` 128 and `/data/tensorflow/bert/books_wiki_en_corpus/tfrecord/seq_len_512/books_wiki_en_corpus/training/` for `seq_len` 512.

#### Download Fine-tuning Datasets

The SQuAD v1.1 datasets are located in [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json) and [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json).

1. To download ELECTRA fine-tuning datasets file, run:
   `/path-to-datasets/squad/`

2. Use this location `/path-to-datasets/squad/` as dataset path at `squad_dir` variable in the `script/electra_squad.sh`.

3. Download `evaluate-v1.1.py` evaluation file and place it under the same path where the above SQuAD dataset files are placed i.e. `/path-to-datasets/squad/`. The placement of `evaluate-v1.1.py` file in the same folder is important because `--eval_script` parameter in `script/electra_squad.sh` searches for the evaluate python file in the same folder as provided to `squad_dir` parameter.
```bash
wget https://worksheets.codalab.org/rest/bundles/0xbcd57bee090b421c982906709c8c27e1/contents/blob/ -O /path-to-datasets/squad/evaluate-v1.1.py
```

Once the dataset is downloaded, the path to the dataset should be properly configured inside the script `script/electra.sh` in the `DATASET_P1` and `DATASET_P2` variable of the scripts for `seq_len` 128 and `seq_len` 512 respectively. For the SQuAD dataset, the `script/electra_squad.sh` should be modified to provide a proper dataset path at `squad_dir` variable and a checkpoint location at `init_checkpoint`.

### Create Results Directory

The `results` directory should be created inside the ELECTRA directory before running pre-training and finetuning to store the checkpoints.
To create the ELECTRA directory, run:

```bash
mkdir results
```

## Pre-training, Fine-tuning and Examples

### ELECTRA Pre-Training

- Located in: `Model-References/TensorFlow/nlp/electra`
- The training script is **`run_pretraining.py`**
- The main wrapper bash script for initiating training is **`scripts/electra.sh`**
- Suited for datasets:
  - `bookcorpus`
  - `wikipedia`
- Uses optimizer: **LAMB** ("Layer-wise Adaptive Moments optimizer for Batch training").
- Uses a learning rate of 6e-3
- Has BF16 precision enabled
- Runs for 20 steps, where the first 5 are warm-up steps
- Saves a checkpoint every 2 iterations (keeps only the latest 5 checkpoints) and at the end of training. All checkpoints, and training logs are saved to the `results/models/<model_name>` directory.
- Consists of 2 phases:
  - Phase 1 - Pretraining on samples of sequence length 128 and at most 15% masked predictions per sequence.
  - Phase 2 - Pretraining on samples of sequence length 512 and at most 15% masked predictions per sequence.
- Heavy-weight: the training takes several hours or days.

**Run pre-training on 1 HPU:**
```bash
bash scripts/electra.sh hpu
```

**NOTE:** The default hyperparameters are set in the `electra.sh` script which can be altered.

### ELECTRA Fine-Tuning

- Located in: `Model-References/TensorFlow/nlp/electra`
- The finetuning script is **`run_tf_squad.py`**
- The main wrapper bash script for initiating finetuning is **`scripts/electra_squad.sh`**
- Suited for dataset:
  - `squad`: Stanford Question Answering Dataset (**SQuAD**) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.
- Uses optimizer: **AdamW** ("ADAM with Weight Decay Regularization").
- Uses a learning rate of 4e-4
- Has BF16 precision enabled
- Saves a checkpoint at the end of epoch to the `checkpoints/` directory.
- Creates a log file containing all the output.
- Based on model weights trained with pretraining.
- Light-weight: the training takes few minutes.

**Run fine-tuning on 1 HPU:**

The above pre-trained ELECTRA model representations can be fine-tuned with one additional output layer only for a state-of-the-art question answering system. Running the following script extracts and saves the discriminator and generator from the pre-trained checkpoint and fine-tunes the discriminator on SQuAD:

```bash
checkpoints=./results/models/base/checkpoints bash scripts/finetune_ckpts_on_squad.sh
```

Run fine-tuning and evaluation with the SQuAD dataset on Google's pre-trained checkpoints by running the following:

```bash
bash scripts/electra_squad.sh hpu
```

Note:
* Running training first is required to generate the required checkpoints for `eval` mode only during validation/evaluation.
* The `<pretrained/checkpoint/dir/path/>` should be provided to the parameter **init_checkpoint** in **/scripts/electra_squad.sh** script.

## Advanced

### Scripts

Provided below the descriptions of the key scripts and folders.

-   `data/` - Contains scripts for downloading and preparing individual datasets, and will contain downloaded and processed datasets.
-   `scripts/` - Contains shell scripts to launch pre-training and fine-tuning.
-   `results/` - Folder where all training and inference results get stored by default.
-   `electra_squad.sh`  - Interface for launching question answering fine-tuning with `run_tf_squad.py`.
-   `electra.sh`  - Interface for launching ELECTRA pre-training with `run_pretraining.py`.
-   `finetune_ckpts_on_squad.sh` - Interface for extracting and saving discriminator and generator from the pretrained checkpoint and run SQuAD fine-tuning on discriminator.
-   `build_pretraining_dataset.py` - Creates `tfrecord` files from shared text files in the final step of dataset creation.
-   `postprocess_pretrained_ckpt.py` - Converts pretrained checkpoint to discriminator checkpoint and generator checkpoint which can be fed into `run_tf_squad_new.py`.
-   `modeling.py` - Implements the ELECTRA pre-training and fine-tuning model architectures with TensorFlow2.
-   `optimization.py` - Implements the Adam optimizer, LAMB and the learning rate schedule with TensorFlow2.
-   `configuration.py` - Implements parent class for model config.
-   `tokenization.py` - Implements the ELECTRA tokenizer.
-   `run_pretraining.py` - Implements ELECTRA pre-training.
-   `pretrain_utils.py` - Utilities required for pre-training such as dynamic masking etc.
-   `run_tf_squad.py` - Implements fine-tuning training and evaluation for question answering on the [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) dataset.

### Parameters

#### Pre-training Parameters

ELECTRA is designed to pre-train deep bidirectional networks for language representations. The following scripts replicate pre-training on Wikipedia + BookCorpus from [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators paper](https://openreview.net/forum?id=r1xMH1BtvB). These scripts are general and can be used for pre-training language representations on any corpus of choice.

In the parameters expected by `scripts/run_pretraining.sh`, `p1` stands for phase 1 whereas `p2` stands for phase 2 training. They are as follows:

-   `<training_batch_size_p1>` is batch size used for training. Default is `2`.
-   `<learning_rate_p1>` is the base learning rate for training. Default is `6e-3`.
-   `<precision>` is the type of math in your model, can be either `fp32` or `hmp`. Default is `fp32`.
-   `<warmup_steps_p1>` is the percentage of training steps used for warm-up at the start of training. Default is `5`.
-   `<train_steps_p1>` is the total number of training steps. Default is `20`.
-   `<save_checkpoint_steps>` controls how often checkpoints are saved. Default is `20`.
-   `<resume_training>` if set to `true`, training should resume from the latest model in `/results/models/<model_name>/checkpoints/`. Default is `false`.
-   `<accumulate_gradient>` a flag indicating whether a larger batch should be simulated with gradient accumulation. Default is `true`.
-   `<gradient_accumulation_steps_p1>` an integer indicating the number of steps to accumulate gradients over. Default is `4`.
-   `<seed>` Default is `100`.
-   `<training_batch_size_p2>` is batch size used for training in phase 2. Default is `2`.
-   `<learning_rate_p2>` is the base learning rate for training phase 2. Default is `4e-3`.
-   `<warmup_steps_p2>` is the percentage of training steps used for warm-up at the start of training. Default is `5`.
-   `<training_steps_p2>` is the total number of training steps for phase 2, to be continued in addition to phase 1. Default is `20`.
-   `<gradient_accumulation_steps_p2>` an integer indicating the number of steps to accumulate gradients over in phase 2. Default is `4`.
-   `<init_checkpoint>` A checkpoint to start the pre-training routine on (Usually a ELECTRA pretrained checkpoint). Default is `None`.

To see the full list of the available options and their descriptions in the main script `run_pretraining.py`, use the `-h` or `--help` command line option, for example:
```python
$PYTHON run_pretraining.py --help
```

#### Fine-tuning Parameters

Default arguments are listed below in the order `scripts/electra_squad.sh` expects:

-   `<electra_model>` - The default is `"google/electra-base-discriminator"`.
-   `<epochs>` - The default is `2`.
-   `<batch_size>` - The default is `16`.
-   `<learning_rate>` - The default is `4e-4`.
-   `<precision>` (either `hmp` or `fp32`) - The default is `fp32`.
-   `<seed>` - The default is `RANDOM`.
-   `<SQUAD_VERSION>` - The default is `1.1`.
-   `<squad_dir>` -  The location where squad dataset is located `<path/to/dir/of/squad>`.
-   `<OUT_DIR>` for result - The default is `results/`.
-   `<init_checkpoint>` - Parameter for initial checkpoint. The `<pretrained/checkpoint/dir/path/>` should be provided.
-   `<mode>` (`train`, `eval`, `train_eval`, `prediction`) - The default is `train_eval`.

The script saves the checkpoint at the end of each epoch to the `checkpoints/` folder.

To see the full list of the available options and their descriptions in the main script `run_tf_squad.py`, use the `-h` or `--help` command line option, for example:
```python
$PYTHON run_tf_squad.py --help
```

## Supported Configuration

| Device | SynapseAI Version | TensorFlow Version(s)  |
|:------:|:-----------------:|:-----:|
| Gaudi  | 1.6.1             | 2.9.1 |
| Gaudi  | 1.6.1             | 2.8.2 |

## Changelog

### 1.4.0

* Added support to import horovod-fork package directly instead of using Model-References' TensorFlow.common.horovod_helpers; wrapped horovod import with a try-catch block so that the user is not required to install this library when the model is being run on a single card.
* Switched from deprecated flag `TF_ENABLE_BF16_CONVERSION` to `TF_BF16_CONVERSION`.

### 1.3.0

* Added support for Habana devices:
  - Loading Habana specific library.
  - Certain environment variables are defined for habana device.
* Removed support for other devices.
* Changed `python` or `python3` to `$PYTHON` to execute correct version based on environment setup.