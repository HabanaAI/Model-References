# BERT

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

**Table of Contents**
* [Model Overview](#model-overview)
* [Setup](#setup)
* [BERT Pre-Training](#bert-pre-training)
* [BERT Fine-Tuning](#bert-fine-tuning)
* [TensorFlow BERT bash training scripts](#tensorflow-bert-bash-training-scripts)
* [TensorFlow BERT Python training scripts with YAML configuration of parameters](#tensorflow-bert-python-training-scripts-with-yaml-configuration-of-parameters)
* [Docker setup and dataset generation](#docker-setup-and-dataset-generation)
* [BERT Pretraining using bash script](#bert-pretraining-using-bash-script)
* [BERT Finetuning using bash script](#bert-finetuning-using-bash-script)
* [Running TensorFlow BERT Pretraining and Finetuning with the new Python command-line interface](#running-tensorflow-bert-pretraining-and-finetuning-with-the-new-python-command-line-interface)
* [Multi-HLS Training Support in BERT Python scripts](#multi-hls-training-support-in-bert-python-scripts)
* [Examples](#examples)
* [Known Issues](#known-issues)

## Model Overview

Bidirectional Encoder Representations from Transformers (BERT) is a technique for natural language processing (NLP) pre-training developed by Google.
BERT was created and published in 2018 by Jacob Devlin and his colleagues from Google.
Google is leveraging BERT to better understand user searches.

The original English-language BERT model comes with two pre-trained general types: (1) the BERTBASE model, a 12-layer, 768-hidden, 12-heads, 110M parameter neural network architecture, and (2) the BERTLARGE model, a 24-layer, 1024-hidden, 16-heads, 340M parameter neural network architecture; both of which were trained on the BooksCorpus with 800M words, and a version of the English Wikipedia with 2,500M words.

The scripts are a mix of Habana modified pre-training scripts taken from [NVIDIA GitHub](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT) and Habana modified fine-tuning scripts taken from [Google GitHub](https://github.com/google-research/bert). We converted the training scripts to TensorFlow 2, added Habana device support and modified Horovod usage to Horovod function wrappers and global `hvd` object. For the details on changes, go to [CHANGES.md](./CHANGES.md).

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

## BERT Pre-Training
- Located in: `Model-References/TensorFlow/nlp/bert`
- Suited for datasets:
  - `bookswiki`
  - `overfit`
- Uses optimizer: **LAMB** ("Layer-wise Adaptive Moments optimizer for Batch training").
- Consists of 2 phases:
  - Phase 1 - **Masked Language Model** - where given a sentence, a randomly chosen word is guessed.
  - Phase 2 - **Next Sentence Prediction** - where the model guesses whether sentence B comes after sentence A
- The resulting (trained) model weights are language-specific (here: English) and must be further "fitted" to do a specific task (with finetuning).
- Heavy-weight: the training takes several hours or days.

## BERT Fine-Tuning
- Located in: `Model-References/TensorFlow/nlp/bert`
- Suited for tasks:
  - `mrpc`: Microsoft Research Paraphrase Corpus (**MRPC**) is a paraphrase identification dataset, where systems aim to identify if two sentences are paraphrases of each other.
  - `squad`: Stanford Question Answering Dataset (**SQuAD**) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.
- Uses optimizer: **AdamW** ("ADAM with Weight Decay Regularization").
- Based on model weights trained with pretraining.
- Light-weight: the training takes a minute or so.

## TensorFlow BERT bash training scripts

**For the BERT Pretraining, it is recommended that bash scripts be used for all multi card training. Python or bash may be used for single card execution.**
**For the BERT Finetuning, it is recommended that bash scripts be used for all BERT Fine Tuning.**

The following is a description of the **demo_bert** bash script's parameters:

```bash
./demo_bert --help
usage: demo_bert subcommand [global and subcommand's local arguments]

subcommands:
  help
  finetuning
  pretraining

global mandatory arguments:
  -d <data_type>,  --dtype <data_type>       Data type, possible values: fp32, bf16
  -m <model>,      --model <model>           Model variant, possible values: tiny, mini, small, medium, base, large

global optional arguments:
                --bert-config-dir            Path to directory containing bert config files needed for choosen training type.
                                             If not specified the zip file will be downloaded
  -o <dir>,     --output-dir <dir>           Output directory (estimators model_dir)
  -v [num_workers_per_hls] --use_horovod [num_w]
                                             Use Horovod for training. num_workers_per_hls parameter is optional and defaults to 8
  -hls <hls_type>,  --hls_type <hls_type>    HLS Type, possible values: HLS1, HLS1-H. Default: HLS1

finetuning mandatory arguments:
  -t <test_set>,   --test-set <test_set>     Benchmark dataset, possible values: mrpc, squad

finetuning optional arguments:
  -e <val>,     --epochs <val>               Number of epochs. If not set defaults to 3.0 for mrpc, 2.0 for squad and 40.0 for bookswiki
  -b <val>,     --batch_size <val>           Batch size
  -s <val>,     --max_seq_length <val>       Number of tokens in each sequence

pretraining optional arguments:
  -t <test_set>,      --test-set <test_set>               Benchmark dataset, possible values: bookswiki [default], overfit.
  -fpo,               --fast-perf-only                    Run smaller global batch size for perf measurement.
  For below options, 2nd value is valid only for -t bookswiki:
  -i <val1> [val2], --iters <val1> [val2]                 Number of steps per worker for each phase of pretraining.
                                                          Default: 7038 782 for bookswiki and 200 for overfit
  -w <val1> [val2], --warmup <val1> [val2]                Number of warmup steps for each phase of pretraining.
                                                          Default: 2000 200 for bookswiki and 10 for overfit
  -b <val1> [val2], --batch_size <val1> [val2]            Batch size for each phase of pretraining.
                                                          Default: 64 8 for bookswiki if -d bf16 else 32 8, and 32 for overfit
  -s <val1> [val2], --max_seq_length <val1> [val2]        Number of tokens in each sequence for each phase of pretraining.
                                                          Default: 128 512 for bookswiki and 128 for overfit
                    --save_ckpt_steps <val>               How often to save the model checkpoint. Default: 100

example:
  ./demo_bert finetuning -d bf16 -m base -t mrpc -e 0.5
  ./demo_bert pretraining -d fp32 -m large -i 100 30
```

## TensorFlow BERT Python training scripts with YAML configuration of parameters

### Introduction

Habana Model-References now uses Python scripts for running single-card, 8-cards and multi-HLS (16-cards, 32-cards, etc.) training of BERT models. BERT and several of the other models we release are launched using **Model-References/central/habana_model_runner.py** and a YAML configuration file that contains the model's hyperparameters, training related parameters and environment variables. The unified Python script-based training invocation command-line is as follows:

```bash
export PYTHONPATH=/path/to/Model-References:$PYTHONPATH

cd /path/to/Model-References/<TensorFlow|PyTorch>/<computer_vision or nlp model directory>

python3 /path/to/Model-References/central/habana_model_runner.py --framework <tensorflow|pytorch> --model <model_name> --hb_config hb_configs/<config.yaml>
```

For example, to run BERT Finetuning in bf16 and the MRPC dataset:

```bash
export PYTHONPATH=/path/to/Model-References:$PYTHONPATH

cd /path/to/Model-References/TensorFlow/nlp/bert

python3 /path/to/Model-References/central/habana_model_runner.py --framework tensorflow --model bert --hb_config hb_configs/bert_base_default.yaml >& ~/hlogs/bert_logs/bert_base_finetuning_mrpc_bf16.txt
```

In 0.14.0, all BERT Pretraining and Finetuning in 1-card, 8-cards, and multi-HLS (16-cards, 32-cards) scaleout scenarios is run using Python scripts and the above methodology. The **Model-References/TensorFlow/nlp/bert** directory has a new **demo_bert.py** and additional Python scripts for running BERT training.

In the following sub-sections, we will describe the structure of the YAML configuration files. Subsequent sections will provide directions to setup the docker environment and run BERT training.

### Structure of YAML configuration files

The format of the YAML file is:

```
model: <model_name>
env_variables:
  <env_var>: <value>
  <env_var>: <value>
parameters:
  <parameter>: <value>
  <parameter>: <value>
  store_true:
    <parameter_name_whose_value_will_be_set_to_true>
    <parameter_name_whose_value_will_be_set_to_true>
```

The “model” and “parameters” sections are required; the “env_variables” section is optional and so is the “store_true” sub-section within the “parameters” section. All variables listed under "store_true" will be passed with a default value of "True" to the Python scripts. The rest of the environment variables and parameters will be passed with their specified values to the Python scripts.

### BERT YAML configuration files

We release the following YAML files for BERT in Model-References/TensorFlow/nlp/bert/hb_configs for running different BERT Base and BERT Large configurations:
- bert_base_default.yaml
- bert_large_default.yaml
- u20.04_sn_bert_base_mrpc_ft_bf16.yaml
- u20.04_sn_bert_base_mrpc_ft_fp32.yaml
- u20.04_sn_bert_base_squad_ft_bf16.yaml
- u20.04_sn_bert_base_squad_ft_fp32.yaml
- u20.04_sn_bert_large_mrpc_ft_bf16.yaml
- u20.04_sn_bert_large_mrpc_ft_fp32.yaml
- u20.04_sn_bert_large_squad_ft_bf16.yaml
- u20.04_sn_bert_large_squad_ft_fp32.yaml
- u20.04_1xhls_bert_large_bookswiki_pt_bf16_hcl.yaml
- u20.04_1xhls_bert_large_bookswiki_pt_bf16_hccl.yaml
- u20.04_1xhls_bert_large_squad_ft_bf16_hcl.yaml
- u20.04_1xhls_bert_large_squad_ft_bf16_hccl.yaml

Each of these files specifies:

- model: "bert" (the model name)
- env_variables: The environment variables to set for the training run
- parameters: The parameters to use:
  - model_variant: The BERT model variant name: base / large / tiny / mini / small / medium
  - command: The training command name: finetuning / pretraining
  - test_set: The dataset to use for the training command: mrpc / squad (finetuning), bookswiki / overfit (pretraining)
  - data_type: The data_type to use for computation: bf16 (mixed precision bfloat16 + float32) or fp32 (float32)
  - use_horovod: False (single-card training) or True (Horovod-based multi-card training)
  - num_workers_per_hls: Number of Gaudi cards to use per HLS, typically 8
  - hls_type: HLS1 or HLS1-H
  - dataset_parameters: The hyperparameters for the test_set to use during training (epochs / batch_size / max_seq_length / learning_rate), including data_type-specific settings, parameters for the dataset location and output_dir to store intermediate check points, log files, etc. for the training run

The user can edit these yaml files and create different versions that represent frequently run configurations.

## Docker setup and dataset generation

These are the instructions for getting the Habana docker image, setting up the container environment, etc.

### Docker setup

- Get the Habana TensorFlow docker image:

  | TensorFlow version | Ubuntu version | Habana docker pull command                                              |
  |:-------------------|:---------------|:------------------------------------------------------------------------|
  | TensorFlow 2.2     | Ubuntu 18.04 | `docker pull vault.habana.ai/gaudi-docker/0.14.0/ubuntu18.04/habanalabs/tensorflow-installer-tf-cpu-2.2.2:0.14.0-420` |
  | TensorFlow 2.2     | Ubuntu 20.04 | `docker pull vault.habana.ai/gaudi-docker/0.14.0/ubuntu20.04/habanalabs/tensorflow-installer-tf-cpu-2.2.2:0.14.0-420` |
  | TensorFlow 2.4     | Ubuntu 18.04 | `docker pull vault.habana.ai/gaudi-docker/0.14.0/ubuntu18.04/habanalabs/tensorflow-installer-tf-cpu-2.4.1:0.14.0-420` |
  | TensorFlow 2.4     | Ubuntu 20.04 | `docker pull vault.habana.ai/gaudi-docker/0.14.0/ubuntu20.04/habanalabs/tensorflow-installer-tf-cpu-2.4.1:0.14.0-420` |


- Run the docker container:

  We will assume there is a directory "$HOME/hlogs" on the host system which we will map as a container volume "<CONTAINER'S $HOME>/hlogs". The BERT Python training examples given below re-direct stdout/stderr to a file in the container's "~/hlogs" directory. We will also assume that there is a directory "$HOME/tmp" on the host system, that contains sufficient disk space to hold the training output directories. We will map this directory as a container volume "<CONTAINER'S $HOME>/tmp".

  Please substitute "<CONTAINER'S $HOME>" with the path that running "echo $HOME" in the container returns, e.g. "/home/user1" or "/root".

  This docker run command-line also assumes that the datasets that will be generated in the next sub-section titled [Download and preprocess the datasets for Pretraining and Finetuning](#download-and-preprocess-the-datasets-for-pretraining-and-finetuning) will be manually copied to a directory "/software/data/tf/data" on the host and mapped back to the container for subsequent training runs. This is because generating BERT datasets is a time-consuming process and we would like to generate the datasets once and reuse them for subsequent training runs in new docker sessions. Users can modify "/software/data/tf/data" to a path of their choice. The BERT YAML configs that are released in the Model-References/TensorFlow/nlp/bert/hb_configs directory contain "dataset_path" parameters that refer to "/software/data/tf/data", and will need to be modified accordingly.

  The docker run command-line also maps a directory "/software/data/bert_checkpoints" from the host to the container as a placeholder for the initial checkpoint data that will be required to run BERT Pretraining with the "overfit" dataset. Again, this directory name is customizable.

  For TensorFlow 2.2:
  ```bash
  docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice -v /sys/kernel/debug:/sys/kernel/debug --net=host -v $HOME/hlogs:<CONTAINER'S $HOME>/hlogs -v $HOME/tmp:<CONTAINER'S $HOME>/tmp -v /software/data/tf/data:/software/data/tf/data -v /software/data/bert_checkpoints:/software/data/bert_checkpoints vault.habana.ai/gaudi-docker/0.14.0/ubuntu18.04/habanalabs/tensorflow-installer-tf-cpu-2.2.2:0.14.0-420
  ```
  For TensorFlow 2.4:
  ```bash
  docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice -v /sys/kernel/debug:/sys/kernel/debug --net=host -v $HOME/hlogs:<CONTAINER'S $HOME>/hlogs -v $HOME/tmp:<CONTAINER'S $HOME>/tmp -v /software/data/tf/data:/software/data/tf/data -v /software/data/bert_checkpoints:/software/data/bert_checkpoints vault.habana.ai/gaudi-docker/0.14.0/ubuntu18.04/habanalabs/tensorflow-installer-tf-cpu-2.4.1:0.14.0-420
  ```

- In the docker container:
  ```bash
  git clone https://github.com/HabanaAI/Model-References

  cd Model-References/TensorFlow/nlp/bert/
  ```

### Download and preprocess the datasets for Pretraining and Finetuning

Please note that the pre-trained model variant, e.g. BERT Base or BERT Large, will be automatically downloaded the first time the model is run in the docker container. These next steps will go over how to download the training datasets.

#### Pretraining datasets download instructions

In `Model-References/TensorFlow/nlp/bert/data_preprocessing` folder, we provide scripts to download, extract and preprocess [Wikipedia](https://dumps.wikimedia.org/) and [BookCorpus](http://yknzhu.wixsite.com/mbweb) datasets.
To run the scripts, set Python 3.6 as default Python in the container, go to `data_preprocessing` folder and install required Python packages:

```bash
ln -s /usr/bin/python3.6 /usr/bin/python

cd Model-References/TensorFlow/nlp/bert/data_preprocessing

pip install boto3 ipdb html2text nltk progressbar filelock tokenizers==0.7.0
```
The pretraining dataset is 170GB+ and takes 15+ hours to download. The BookCorpus server gets overloaded most of the time and also contains broken links resulting in HTTP 403 and 503 errors. Hence, it is recommended to skip downloading BookCorpus with the script by running:

```bash
bash create_datasets_from_start.sh
```
Users are welcome to download BookCorpus from other sources to match our accuracy, or repeatedly try our script until the required number of files are downloaded by running the following:

```bash
bash create_datasets_from_start.sh wiki_books
```

#### Finetuning datasets download instructions

The dataset for MRPC will be automatically downloaded the first time the model is run in the docker container.
The SQuAD dataset needs to be manually downloaded to `Model-References/TensorFlow/nlp/bert/data` for the bash scripts.
For the Python scripts, the SQuAD dataset can be downloaded to a location that is customizable via the "dataset_path" YAML config parameter for "squad".
The [SQuAD website](https://rajpurkar.github.io/SQuAD-explorer/) does not seem to link to the v1.1 datasets any longer,
but the necessary files can be found here:
- [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
- [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)

## BERT Pretraining using bash script
Note:
- The 8-cards training should be performed using bash scripts as mentioned in Examples section below.

In the docker container:
```
  cd Model-References/TensorFlow/nlp/bert/
```

Set the pre-training dataset location as follows if you run BERT pre-training:
```bash
export INPUT_FILES_PREFIX=/path/to/pre-train/dataset
```

To see the full list of available options and the descriptions, run:
```bash
./demo_bert -h  # for help
```

### Examples

The following table provides command lines to run the training under various configurations.

| Model        | Task & Dataset            | # Gaudi cards, Batch Size, Seq Length, Precision     | Command Line           |
|:----------------|-----------------|:------------------------:|:--------------------------------------------------|
| BERT Large | Pre-Training on BookCorpus and Wikipedia | 1-cards, BS=64, Seq=128, fp32 | `./demo_bert pretraining -d fp32 -m large -t bookswiki -b 64 8`   |
| BERT Large | Pre-Training on BookCorpus and Wikipedia | 8-cards, BS=64, Seq=128, fp32 | `./demo_bert pretraining -v -d fp32 -m large -t bookswiki -b 64 8`   |

## BERT Finetuning using bash script
Note:
- The 8-cards training should be performed using bash scripts as mentioned in Examples section below.

In the docker container:
```
  cd Model-References/TensorFlow/nlp/bert/
```

If you are running Finetuning with SQuAD, the dataset needs to be manually downloaded to `Model-References/TensorFlow/nlp/bert/data` for the bash scripts, as described in [Finetuning datasets download instructions](#finetuning-datasets-download-instructions).

To see the full list of available options and the descriptions, run:
```bash
./demo_bert -h  # for help
```

### Examples

The following table provides command lines to run the training under various configurations.

| Model        | Task & Dataset            | # Gaudi cards, Batch Size, Seq Length, Precision     | Command Line           |
|:----------------|-----------------|:------------------------:|:--------------------------------------------------|
| BERT Base   | Fine-Tuning for MRPC                     | 1-card, BS=64, Seq=128, bf16                        | `./demo_bert -d bf16 finetuning -m base -t mrpc -b 64 -s 128`   |
| BERT Base   | Fine-Tuning for SQuAD                     | 1-card, BS=64, Seq=128, bf16                        | `./demo_bert -d bf16 finetuning -m base -t squad -b 64 -s 128`   |
| BERT Base  | Fine-Tuning for MRPC                     | 8-cards, BS=64, Seq=128, bf16                        | `./demo_bert -v -d bf16 finetuning -m base -t mrpc -b 64 -s 128`   |
| BERT Base  | Fine-Tuning for SQuAD                     | 8-cards, BS=64, Seq=128, bf16                        | `./demo_bert -v -d bf16 finetuning -m base -t squad -b 64 -s 128`   |
| BERT Large   | Fine-Tuning for MRPC                     | 1-card, BS=64, Seq=128, bf16                        | `./demo_bert -d bf16 finetuning -m large -t mrpc -b 64 -s 128`   |
| BERT Large   | Fine-Tuning for SQuAD                     | 1-card, BS=64, Seq=128, bf16                        | `./demo_bert -d bf16 finetuning -m large -t squad -b 64 -s 128`   |
| BERT Large  | Fine-Tuning for MRPC                     | 8-cards, BS=64, Seq=128, bf16                        | `./demo_bert -v -d bf16 finetuning -m large -t mrpc -b 64 -s 128`   |
| BERT Large  | Fine-Tuning for SQuAD                     | 8-cards, BS=64, Seq=128, bf16                        | `./demo_bert -v -d bf16 finetuning -m large -t squad -b 64 -s 128`   |

Keep an eye on the opening log messages, as they show variables, which can be overridden by the environment. For instance, if you want to enable HCCL, just:
```bash
HABANA_NCCL_COMM_API=true ./demo_bert ...
```

## Running TensorFlow BERT Pretraining and Finetuning with the new Python command-line interface

Now that we have completed the [Setup](#setup) and [Docker setup and dataset generation](#docker-setup-and-dataset-generation) steps, we are ready to run BERT training. Training is invoked via the **Model-References/central/habana_model_runner.py** Python script that requires three parameters:

- **--framework** ***tensorflow***
- **--model** ***model_name***
- **--hb_config** ***path_to_yaml_config_file***

Furthermore, to run 8-cards and multi-node training with Horovod support, these flags need to be set in the yaml config's "parameters" section:

```bash
use_horovod: True
num_workers_per_hls: 8
hls_type: HLS1
```

In the docker container, go to the bert directory:
```bash
cd Model-References/TensorFlow/nlp/bert/
```
If `Model-References` repository path is not in the PYTHONPATH, make sure you update it:
```bash
export PYTHONPATH=/path/to/Model-References:$PYTHONPATH
```

Make sure to also update the **dataset_path** parameter in the yaml config file that you are using, for the test-set that you are working with, to refer to the location where the generated dataset from the download and pre-processing step resides. The dataset_path must be a valid path that is available in the container.
For example: If you are running BERT Pretraining with the BooksWiki dataset, this is the dataset_path that should be set correctly in the yaml config file:
```bash
parameters:
    bookswiki:
      dataset_path: "/software/data/tf/data/bert/books_wiki_en_corpus/tfrecord/"
```

Run BERT training:

```bash
python3 /path/to/Model-References/central/habana_model_runner.py --framework tensorflow --model bert --hb_config hb_configs/<config.yaml>

Example:

python3 /path/to/Model-References/central/habana_model_runner.py --framework tensorflow --model bert --hb_config hb_configs/bert_base_default.yaml >& ~/hlogs/bert_logs/bert_base_finetuning_mrpc_bf16.txt
```

In 0.14.0, the Python scripts with YAML configuration support:

- single-card, 8-cards and multi-HLS Pretraining of BERT Base and BERT Large with Books & Wiki and Overfit datasets, and
- single-card, 8-cards and multi-HLS Finetuning of BERT Base and BERT Large with MRPC and SQuAD datasets

## Multi-HLS Training Support in BERT Python scripts

Multi-HLS support in the BERT Python scripts has been enabled using mpirun and Horovod, and has been tested with 2HLS1 (16 Gaudi cards) and 4HLS1 (32 Gaudi cards) configurations.

### YAML config and docker ssh port setup

Multi-HLS training works by setting the **MULTI_HLS_IPS** environment variable to a comma-separated list of host IP addresses in the yaml config file. Additionally, Model-References/TensorFlow/nlp/bert/hb_configs/bert_base_default.yaml provides default settings for the host's network interface name and the rsh port used by the sshd service in the docker container. This example shows how to configure a "bert_base_mrpc_32_cards.yaml" file for 4HLS1 training:

```bash
# This yaml config environment variable is needed for multi-node training with Horovod.
# Set this to be a comma-separated string of host IP addresses, e.g.:
MULTI_HLS_IPS: "192.10.100.174,10.10.100.101,10.10.102.181,10.10.104.192"

# Set this to the network interface name for the ping-able IP address of the host on
# which the training script is run. This appears in the output of ip addr.
MPI_TPC_INCLUDE: "eno1"

# This is the port number used for rsh from the docker container, as configured
# in /etc/ssh/sshd_config
DOCKER_SSHD_PORT: 3022
```
By default, the Habana docker uses port 3022 for ssh, and this is the default port configured in the training scripts. Sometimes, mpirun can fail to establish the remote connection when there is more than one Habana docker session running on the main HLS in which the Python training script is run. If this happens, you can set up a different ssh port as follows:

Follow [Setup](#setup) and [Docker setup and dataset generation](#docker-setup-and-dataset-generation) steps above on all HLS machines. In each HLS host's docker container:
```bash
vi /etc/ssh/sshd_config
```
Uncomment "#Port 22" and replace the port number with a different port number, example "Port 4022". Next, restart the sshd service:
```bash
service ssh stop
service ssh start
```

Change the yaml config environment variable value to reflect this change into the Python scripts:
```bash
DOCKER_SSHD_PORT: 4022
```

Remember to set these flags in the yaml config's "parameters" section:

```bash
use_horovod: True
num_workers_per_hls: 8
hls_type: HLS1
```

### Multi-HLS BERT Training

Please follow these steps for multi-HLS training over connected HLS systems:

1. Follow [Setup](#setup) and [Docker setup and dataset generation](#docker-setup-and-dataset-generation) steps above on all HLS machines
2. Configure password-less ssh between all nodes:

   Do the following in all the nodes' docker sessions:
   ```bash
   mkdir ~/.ssh
   cd ~/.ssh
   ssh-keygen -t rsa -b 4096
   ```
   Copy id_rsa.pub contents from every node's docker to every other node's docker's ~/.ssh/authorized_keys (all public keys need to be in all hosts' authorized_keys):
   ```bash
   cat id_rsa.pub > authorized_keys
   vi authorized_keys
   ```
   Copy the contents from inside to other systems.
   Paste all hosts' public keys in all hosts' “authorized_keys” file.

   On each system:
   Add all hosts (including itself) to known_hosts. If you configured a different docker sshd port, say "Port 4022", in [YAML config and docker ssh port setup](#yaml-config-and-docker-ssh-port-setup), replace "-p 3022" with "-p 4022":
   ```bash
   ssh-keyscan -p 3022 -H 192.10.100.174 >> ~/.ssh/known_hosts
   ssh-keyscan -p 3022 -H 10.10.100.101 >> ~/.ssh/known_hosts
   ssh-keyscan -p 3022 -H 10.10.102.181 >> ~/.ssh/known_hosts
   ssh-keyscan -p 3022 -H 10.10.104.192 >> ~/.ssh/known_hosts
   ```

3. Run BERT training with 32 cards using bash script:

   In the main HLS node's docker container:

   ```bash
   cd Model-References/TensorFlow/nlp/bert/

   export PYTHONPATH=/path/to/Model-References:$PYTHONPATH
   # Set this to the network interface name for the ping-able IP address of the host on
   # which the training script is run. This appears in the output of ip addr.
   export MPI_TPC_INCLUDE="eno1"

   # This yaml config environment variable is needed for multi-node training with Horovod.
   # Set this to be a comma-separated string of host IP addresses, e.g.:
   export MULTI_HLS_IPS="192.10.100.174,10.10.100.101,10.10.102.181,10.10.104.192"

   ./demo_bert pretraining -v -d bf16 -m large -b 64 8

   ```

4. Run BERT training with 32 cards using Python scripts:

   In the main HLS node's docker container:

   ```bash
   cd Model-References/TensorFlow/nlp/bert/

   export PYTHONPATH=/path/to/Model-References:$PYTHONPATH

   python3 /path/to/Model-References/central/habana_model_runner.py --framework tensorflow --model bert --hb_config hb_configs/<config.yaml>

   Example:

   python3 /path/to/Model-References/central/habana_model_runner.py --framework tensorflow --model bert --hb_config hb_configs/bert_base_mrpc_32_cards.yaml >& ~/hlogs/bert_base_finetuning_bf16_mrpc_32_cards.txt
   ```

## Examples

The following table provides command lines to run BERT training under various configurations:

| Model variant | Command | Dataset | Precision | Hardware configuration | Hyperparameters | Training run Command-line                                           |
|:----------------|:-------------|:-------------:|:-------------:|:-------------:|:--------------------------|:--------------------------------------------------|
| BERT Base | Finetuning | MRPC | bf16 | 1-card | epochs=3, BS=64, Seq=128 | `python3 /path/to/Model-References/central/habana_model_runner.py --framework tensorflow --model bert --hb_config hb_configs/bert_base_default.yaml` |
| BERT Large | Finetuning | MRPC | bf16 | 1-card | epochs=3, BS=64, Seq=128 | `python3 /path/to/Model-References/central/habana_model_runner.py --framework tensorflow --model bert --hb_config hb_configs/bert_large_default.yaml` |
| BERT Base | Finetuning | MRPC | bf16 | 1-card | epochs=3, BS=64, Seq=128 | `python3 /path/to/Model-References/central/habana_model_runner.py --framework tensorflow --model bert --hb_config hb_configs/u20.04_sn_bert_base_mrpc_ft_bf16.yaml` |
| BERT Base | Finetuning | MRPC  | fp32 | 1-card  | epochs=3, BS=32, Seq=128 | `python3 /path/to/Model-References/central/habana_model_runner.py --framework tensorflow --model bert --hb_config hb_configs/u20.04_sn_bert_base_mrpc_ft_fp32.yaml` |
| BERT Base | Finetuning | SQuAD | bf16 | 1-card | epochs=2, BS=32, Seq=384 | `python3 /path/to/Model-References/central/habana_model_runner.py --framework tensorflow --model bert --hb_config hb_configs/u20.04_sn_bert_base_squad_ft_bf16.yaml` |
| BERT Base | Finetuning | SQuAD | fp32 | 1-card | epochs=2, BS=32, Seq=384 | `python3 /path/to/Model-References/central/habana_model_runner.py --framework tensorflow --model bert --hb_config hb_configs/u20.04_sn_bert_base_squad_ft_fp32.yaml` |
| BERT Large | Finetuning | MRPC | bf16 | 1-card | epochs=3, BS=64, Seq=128 | `python3 /path/to/Model-References/central/habana_model_runner.py --framework tensorflow --model bert --hb_config hb_configs/u20.04_sn_bert_large_mrpc_ft_bf16.yaml` |
| BERT Large | Finetuning | MRPC | fp32 | 1-card | epochs=3, BS=32, Seq=128 | `python3 /path/to/Model-References/central/habana_model_runner.py --framework tensorflow --model bert --hb_config hb_configs/u20.04_sn_bert_large_mrpc_ft_fp32.yaml` |
| BERT Large | Finetuning | SQuAD | bf16 | 1-card | epochs=2, BS=24, Seq=384 | `python3 /path/to/Model-References/central/habana_model_runner.py --framework tensorflow --model bert --hb_config hb_configs/u20.04_sn_bert_large_squad_ft_bf16.yaml` |
| BERT Large | Finetuning | SQuAD | fp32 | 1-card | epochs=2, BS=10, Seq=384 | `python3 /path/to/Model-References/central/habana_model_runner.py --framework tensorflow --model bert --hb_config hb_configs/u20.04_sn_bert_large_squad_ft_fp32.yaml` |
| BERT Large | Pretraining | BooksWiki | bf16 | 8-cards, HCCL | epochs=1, BS=[64, 8], Seq=[128, 512] | `python3 /path/to/Model-References/central/habana_model_runner.py --framework tensorflow --model bert --hb_config hb_configs/u20.04_1xhls_bert_large_bookswiki_pt_bf16_hccl.yaml` |
| BERT Large | Pretraining | BooksWiki | bf16 | 8-cards | epochs=1, BS=[64, 8], Seq=[128, 512] | `python3 /path/to/Model-References/central/habana_model_runner.py --framework tensorflow --model bert --hb_config hb_configs/u20.04_1xhls_bert_large_bookswiki_pt_bf16_hcl.yaml` |
| BERT Large | Finetuning | SQuAD | bf16 | 8-cards, HCCL | epochs=2, BS=24, Seq=384 | `python3 /path/to/Model-References/central/habana_model_runner.py --framework tensorflow --model bert --hb_config hb_configs/u20.04_1xhls_bert_large_squad_ft_bf16_hccl.yaml` |
| BERT Large | Finetuning | SQuAD | bf16 | 8-cards | epochs=2, BS=24, Seq=384 | `python3 /path/to/Model-References/central/habana_model_runner.py --framework tensorflow --model bert --hb_config hb_configs/u20.04_1xhls_bert_large_squad_ft_bf16_hcl.yaml` |

Habana environment variables for logging, profiling, etc. can be prepended to the training run command-line. For instance, if you want to enable profiling with Synapse traces with SynapseLoggerHook:
```bash
HABANA_SYNAPSE_LOGGER=range python3 /path/to/Model-References/central/habana_model_runner.py ...
```


## Known Issues

Running BERT Base and BERT Large Finetuning with SQuAD in 16-cards configuration, BS=24, Seq=384 raises a DataLossError "Data loss: corrupted record". Also, running BERT Base and BERT Large Pretraining in fp32 precision with BooksWiki and 16-cards configuration gives errors about "nodes in a cycle". These issues are being investigated.
