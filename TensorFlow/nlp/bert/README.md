# BERT

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents
   * [Model-References/README.md](https://github.com/HabanaAI/Model-References/blob/master/README.md)
   * [Model Overview](#model-overview)
   * [Setup](#setup)
   * [BERT Pre-Training](#bert-pre-training)
   * [BERT Fine-Tuning](#bert-fine-tuning)
   * [Docker setup and dataset generation](#docker-setup-and-dataset-generation)
      * [Run the docker container and clone the Model-References repository for non-K8s configurations only](#run-the-docker-container-and-clone-the-model-references-repository-for-non-k8s-configurations-only)
      * [Download and preprocess the datasets for Pretraining and Finetuning for non-K8s and K8s configurations](#download-and-preprocess-the-datasets-for-pretraining-and-finetuning-for-non-k8s-and-k8s-configurations)
         * [Pretraining datasets download instructions](#pretraining-datasets-download-instructions)
         * [Finetuning datasets download instructions](#finetuning-datasets-download-instructions)
   * [Training TensorFlow BERT using Python training scripts](#training-tensorflow-bert-using-python-training-scripts)
   * [Training BERT in non-Kubernetes environments using demo_bert.py](#training-bert-in-non-kubernetes-environments-using-demo_bertpy)
      * [Single-card training](#single-card-training)
      * [Multi-card/single-HLS Horovod-based distributed training](#multi-cardsingle-hls-horovod-based-distributed-training)
      * [Multi-HLS Horovod-based scale-out distributed training](#multi-hls-horovod-based-scale-out-distributed-training)
         * [Docker ssh port setup for Multi-HLS training](#docker-ssh-port-setup-for-multi-hls-training)
         * [Setup password-less ssh between all connected HLS systems used in the scale-out training](#setup-password-less-ssh-between-all-connected-hls-systems-used-in-the-scale-out-training)
         * [Run BERT training with 32 cards using demo_bert.py](#run-bert-training-with-32-cards-using-demo_bertpy)
         * [Run BERT training with 16 cards using demo_bert.py](#run-bert-training-with-16-cards-using-demo_bertpy)
   * [Training BERT in Kubernetes environments using demo_bert.py](#training-bert-in-kubernetes-environments-using-demo_bertpy)
      * [Single-card training on K8s](#single-card-training-on-k8s)
      * [Multi-card Horovod-based distributed training on K8s](#multi-card-horovod-based-distributed-training-on-k8s)
   * [Multi-card training using mpirun with run_pretraining.py, run_classifier.py and run_squad.py](#multi-card-training-using-mpirun-with-run_pretrainingpy-run_classifierpy-and-run_squadpy)
   * [Known Issues](#known-issues)

## Model Overview

Bidirectional Encoder Representations from Transformers (BERT) is a technique for natural language processing (NLP) pre-training developed by Google.
BERT was created and published in 2018 by Jacob Devlin and his colleagues from Google.
Google is leveraging BERT to better understand user searches.

The original English-language BERT model comes with two pre-trained general types: (1) the BERTBASE model, a 12-layer, 768-hidden, 12-heads, 110M parameter neural network architecture, and (2) the BERTLARGE model, a 24-layer, 1024-hidden, 16-heads, 340M parameter neural network architecture; both of which were trained on the BooksCorpus with 800M words, and a version of the English Wikipedia with 2,500M words.

The scripts are a mix of Habana modified pre-training scripts taken from [NVIDIA GitHub](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT) and Habana modified fine-tuning scripts taken from [Google GitHub](https://github.com/google-research/bert). We converted the training scripts to TensorFlow 2, added Habana device support and modified Horovod usage to Horovod function wrappers and global `hvd` object. For the details on changes, go to [CHANGES.md](./CHANGES.md).

Please visit [this page](../../../README.md#tensorflow-model-performance) for performance information.

## Setup

Please follow the instructions given in the following link for setting up the environment: [Gaudi Setup and Installation Guide](https://github.com/HabanaAI/Setup_and_Install). Please answer the questions in the guide according to your preferences. This guide will walk you through the process of setting up your system to run the model on Gaudi.

## BERT Pre-Training
- Located in: `Model-References/TensorFlow/nlp/bert`
- The main training script is **`demo_bert.py`**
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
- The main training script is **`demo_bert.py`**
- Suited for tasks:
  - `mrpc`: Microsoft Research Paraphrase Corpus (**MRPC**) is a paraphrase identification dataset, where systems aim to identify if two sentences are paraphrases of each other.
  - `squad`: Stanford Question Answering Dataset (**SQuAD**) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.
- Uses optimizer: **AdamW** ("ADAM with Weight Decay Regularization").
- Based on model weights trained with pretraining.
- Light-weight: the training takes a minute or so.


## Docker setup and dataset generation

In this section, we will first provide instructions to launch a Habana TensorFlow docker container and clone the [Model-References repository](https://github.com/HabanaAI/Model-References/). This is mainly applicable to non-Kubernetes configurations.

Next, we will provide instructions to download and preprocess the datasets, and copy them to locations that are on volumes mapped to the host (for persistence across container runs, since generating BERT datasets is a time-consuming process). This step is applicable to both non-K8s and K8s configurations.

### Run the docker container and clone the Model-References repository for non-K8s configurations only

We will assume there is a directory `$HOME/hlogs` on the host system which we will map as a container volume `<CONTAINER'S $HOME>/hlogs`. The BERT Python training examples given below re-direct stdout/stderr to a file in the container's `~/hlogs` directory. We will also assume that there is a directory `$HOME/tmp` on the host system, that contains sufficient disk space to hold the training output directories. We will map this directory as a container volume `<CONTAINER'S $HOME>/tmp`.

Please substitute `<CONTAINER'S $HOME>` with the path that running `echo $HOME` in the container returns, e.g. `/home/user1` or `/root`.

The following docker run command-line also assumes that the datasets that will be generated in the next sub-section titled [Download and preprocess the datasets for Pretraining and Finetuning for non-K8s and K8s configurations](#download-and-preprocess-the-datasets-for-pretraining-and-finetuning-for-non-k8s-and-k8s-configurations) will be manually copied to a directory `/software/data/tf/data` on the host and mapped back to the container for subsequent training runs. This is because generating BERT datasets is a time-consuming process and we would like to generate the datasets once and reuse them for subsequent training runs in new docker sessions. Users can modify `/software/data/tf/data` to a path of their choice.

The following docker run command-line also maps a directory `/software/data/bert_checkpoints` from the host to the container as a placeholder for the initial checkpoint data that will be required to run BERT Pretraining with the `overfit` dataset. Again, this directory name is customizable.

For TensorFlow 2.4:
```bash
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host -v $HOME/hlogs:<CONTAINER'S $HOME>/hlogs -v $HOME/tmp:<CONTAINER'S $HOME>/tmp -v /software/data/tf/data:/software/data/tf/data -v /software/data/bert_checkpoints:/software/data/bert_checkpoints vault.habana.ai/gaudi-docker/0.15.0/ubuntu20.04/habanalabs/tensorflow-installer-tf-cpu-2.4.1:0.15.0-547
```

In the docker container:
```bash
git clone https://github.com/HabanaAI/Model-References

export PYTHONPATH=/path/to/Model-References:$PYTHONPATH

cd /path/to/Model-References/TensorFlow/nlp/bert/
```

### Download and preprocess the datasets for Pretraining and Finetuning for non-K8s and K8s configurations

Please note that the pre-trained model variant, e.g. BERT Base or BERT Large, will be automatically downloaded the first time model training with `demo_bert.py` is run in the docker container. These next steps will go over how to download the training datasets.

#### Pretraining datasets download instructions

In `Model-References/TensorFlow/nlp/bert/data_preprocessing` folder, we provide scripts to download, extract and preprocess [Wikipedia](https://dumps.wikimedia.org/) and [BookCorpus](http://yknzhu.wixsite.com/mbweb) datasets.
To run the scripts, set Python 3.8 as default Python in the container, go to `data_preprocessing` folder and install required Python packages:

```bash
ln -s /usr/bin/python3.8 /usr/bin/python

cd /path/to/Model-References/TensorFlow/nlp/bert/data_preprocessing

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

When BERT Finetuning with MRPC is run using `demo_bert.py`, the script will look for the MRPC dataset in the directory specified as the `--dataset_path` option. If it doesn't exist, the dataset will be automatically downloaded to this directory. If the `--dataset_path` option is not specified, the MRPC dataset will be downloaded to `Model-References/TensorFlow/nlp/bert/dataset/MRPC`. If needed, the MRPC dataset can be moved to a shared directory, and this location can be provided as the `--dataset_path` option to `demo_bert.py` during subsequent training runs. The examples that follow use `/software/data/tf/data/bert/MRPC` as this shared folder specified to `--dataset_path`.

The SQuAD dataset needs to be manually downloaded to a location of your choice, preferably a shared directory. This location should be provided as the `--dataset_path` option to `demo_bert.py` when running BERT Finetuning with SQuAD. The examples that follow use `/software/data/tf/data/bert/SQuAD` as this shared folder specified to `--dataset_path`.
The [SQuAD website](https://rajpurkar.github.io/SQuAD-explorer/) does not seem to link to the v1.1 datasets any longer,
but the necessary files can be found here:
- [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
- [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)

## TensorFlow BERT training

## Training TensorFlow BERT using Python training scripts

The Habana Model-References repository now uses Python scripts for running single-card, 8-cards and multi-HLS (16-cards, 32-cards, etc.) pretraining and finetuning of TensorFlow BERT models. The main training script is **`demo_bert.py`**.

`demo_bert.py` provides a uniform command-line scale-out training user interface that runs on non-Kubernetes (non-K8s) and Kubernetes (K8s) platforms. It does all the necessary setup work, such as downloading the pre-trained model, preparing output directories, generating HCL config JSON files for SynapseAI during Horovod multi-card runs, checking for required files and directories at relevant stages during training, etc. before calling the `run_pretraining.py`, `run_classifier.py` and `run_squad.py` TensorFlow training scripts. `demo_bert.py` internally uses mpirun to invoke these TensorFlow scripts in Horovod-based training mode for multi-card runs in non-K8s setup. For usage in K8s clusters, the user is expected to launch `demo_bert.py` via `mpirun`, as subsequent sections will describe.

The command-line options of demo_bert.py are described in the script's help message:

In the docker container:

```bash
cd /path/to/Model-References/TensorFlow/nlp/bert/

python3 demo_bert.py --help

usage: demo_bert.py [-h] -c <command> -d <data_type> -m <model_variant>
                    [--bert-config-dir <bert_config_dir>] [--model {bert}]
                    [-o <dir>] [-v <num_horovod_workers_per_hls>]
                    [--hls_type <hls_type>] [-t <test_set>] [-e <val>]
                    [-s <val>] [-i <val1[,val2]>] [-w <val1[,val2]>]
                    [-fpo <for_perf_measurements>] [--save_ckpt_steps <val1>]
                    [--no_steps_accumulation <for_bookswiki_pretraining>]
                    [-b <batch_size>] [--learning_rate <learning_rate>]
                    [--dataset_path <dataset_path>]
                    [--init_checkpoint_path <overfit_init_checkpoint_path>]
                    [--checkpoint_folder <checkpoint_folder>]
                    [--bf16_config_path </path/to/custom/bf16/config>]
                    [--kubernetes_run <kubernetes_run>]
                    [--run_phase2 <run_phase2>]
                    [--enable_scoped_allocator <enable_scoped_allocator>]
                    [--horovod_hierarchical_allreduce <horovod_hierarchical_allreduce>]

optional arguments:
  -h, --help            show this help message and exit
  -c <command>, --command <command>
                        Command, possible values: pretraining, finetuning
  -d <data_type>, --data_type <data_type>
                        Data type, possible values: fp32, bf16
  -m <model_variant>, --model_variant <model_variant>
                        Model variant, possible values: tiny, mini, small,
                        medium, base, large
  --bert-config-dir <bert_config_dir>
                        Path to directory containing bert config files needed
                        for chosen training type. If not specified the zip
                        file will be downloaded.
  --model {bert}        The model name is bert
  -o <dir>, --output_dir <dir>
                        Output directory. Default is /tmp/bert.
  -v <num_horovod_workers_per_hls>, --use_horovod <num_horovod_workers_per_hls>
                        Use Horovod for multi-card distributed training with
                        specified num_horovod_workers_per_hls.
  --hls_type <hls_type>
                        HLS Type, possible values: HLS1, HLS1-H
  -t <test_set>, --test_set <test_set>
                        Benchmark dataset, possible finetuning values: mrpc,
                        squad; possible pretraining values: bookswiki,
                        overfit. Default: bookswiki.
  -e <val>, --epochs <val>
                        Number of epochs. If not set, defaults to 3.0 for
                        mrpc, 2.0 for squad and 40.0 for bookswiki.
  -s <val>, --max_seq_length <val>
                        Number of tokens in each sequence. If not set,
                        defaults to 128 for mrpc; 384 for squad; 128,512 for
                        bookswiki; 128 for overfit.
  -i <val1[,val2]>, --iters <val1[,val2]>
                        Number of steps for each phase of pretraining.
                        Default: 7038,782 for bookswiki and 200 for overfit.
  -w <val1[,val2]>, --warmup <val1[,val2]>
                        Number of warmup steps for each phase of pretraining.
                        Default: 2000,200 for bookswiki and 10 for overfit.
  -fpo <for_perf_measurements>, --fast_perf_only <for_perf_measurements>
                        Defaults to 0. Set to 1 to run smaller global batch
                        size for perf measurement.
  --save_ckpt_steps <val1>
                        How often to save the model checkpoint. Default: 100.
  --no_steps_accumulation <for_bookswiki_pretraining>
                        Defaults to 0. Set to 1 for no steps accumulation
                        during BooksWiki pretraining.
  -b <batch_size>, --batch_size <batch_size>
                        Batch size. Defaults for bf16/fp32: 64/32 for mrpc;
                        24/10 for squad; 64,8/32,8 for bookswiki; 32 for
                        overfit.
  --learning_rate <learning_rate>
                        Learning rate. Default: 2e-5.
  --dataset_path <dataset_path>
                        Path to training dataset. Default: ./dataset/
  --init_checkpoint_path <overfit_init_checkpoint_path>
                        Init checkpoint path for use in overfit pretraining
  --checkpoint_folder <checkpoint_folder>
                        Init checkpoint folder for use in finetuning
  --bf16_config_path </path/to/custom/bf16/config>
                        Path to custom mixed precision config to use, given in
                        JSON format. Defaults to /root/model_garden/TensorFlow
                        /common/bf16_config/bert.json. Applicable only if
                        --data_type = bf16.
  --kubernetes_run <kubernetes_run>
                        Set to True for running training on a Kubernetes
                        cluster. Default: False.
  --run_phase2 <run_phase2>
                        Set to True for running Phase 2 of multi-card
                        pretraining on a Kubernetes cluster, after Phase 1 has
                        successfully completed. Default: False.
  --enable_scoped_allocator <enable_scoped_allocator>
                        Set to True to enable scoped allocator optimization.
                        Default: False.
  --horovod_hierarchical_allreduce <horovod_hierarchical_allreduce>
                        Enables hierarchical allreduce in Horovod. Default:
                        False. Set this option to True to run multi-HLS scale-
                        out training over host NICs. This will cause the
                        environment variable `HOROVOD_HIERARCHICAL_ALLREDUCE`
                        to be set to `1`.
```

For more details on the mixed precision training recipe customization via the `--bf16_config_path` option, please refer to the [TensorFlow Mixed Precision Training on Gaudi](https://docs.habana.ai/en/latest/Tensorflow_User_Guide/Tensorflow_User_Guide.html#tensorflow-mixed-precision-training-on-gaudi) documentation.

For more details on Horovod-based scaling of Gaudi on TensorFlow and using Host NICs vs. Gaudi NICs for multi-HLS scale-out training, please refer to the [Distributed Training with TensorFlow](https://docs.habana.ai/en/latest/Tensorflow_Scaling_Guide/TensorFlow_Gaudi_Scaling_Guide.html) documentation.

Next, we will describe how to run BERT training in non-K8s and K8s environments.

## Training BERT in non-Kubernetes environments using demo_bert.py

### Single-card training

```bash
cd /path/to/Model-References/TensorFlow/nlp/bert/

python3 demo_bert.py \
   --command <command> \
   --model_variant <model_variant> \
   --data_type <data_type> \
   --test_set <dataset_name> \
   --dataset_path </path/to/dataset> \
   --output_dir </path/to/outputdir>
```

#### Examples

- Single Gaudi card pretraining of BERT Large in bfloat16 precision using BooksWiki dataset:
  ```bash
  cd /path/to/Model-References/TensorFlow/nlp/bert/

  python3 demo_bert.py \
     --command pretraining \
     --model_variant large \
     --data_type bf16 \
     --test_set bookswiki \
     --dataset_path /software/data/tf/data/bert/books_wiki_en_corpus/tfrecord/ \
  2>&1 | tee ~/hlogs/bert_large_pretraining_bf16_bookswiki_1_card.txt
  ```
- Single Gaudi card finetuning of BERT Large in bfloat16 precision using MRPC dataset:
  ```bash
  cd /path/to/Model-References/TensorFlow/nlp/bert/

  python3 demo_bert.py \
     --command finetuning \
     --model_variant large \
     --data_type bf16 \
     --test_set mrpc \
     --dataset_path /software/data/tf/data/bert/MRPC \
  2>&1 | tee ~/hlogs/bert_large_finetuning_bf16_mrpc_1_card.txt
  ```
- Single Gaudi card finetuning of BERT Large in bfloat16 precision using SQuAD dataset:
  ```bash
  cd /path/to/Model-References/TensorFlow/nlp/bert/

  python3 demo_bert.py \
     --command finetuning \
     --model_variant large \
     --data_type bf16 \
     --test_set squad \
     --dataset_path /software/data/tf/data/bert/SQuAD \
  2>&1 | tee ~/hlogs/bert_large_finetuning_bf16_squad_1_card.txt
  ```

### Multi-card/single-HLS Horovod-based distributed training

Multi-card training has been enabled in BERT Python scripts using mpirun and Horovod.

```bash
cd /path/to/Model-References/TensorFlow/nlp/bert/

python3 demo_bert.py \
        --command <command> \
        --model_variant <model_variant> \
        --data_type <data_type> \
        --test_set <dataset_name> \
        --dataset_path </path/to/dataset> \
        --output_dir </path/to/outputdir> \
        --use_horovod <num_horovod_workers_per_hls> \
        --hls_type <HLS1|HLS1-H>
```

#### Examples

- 8 Gaudi cards pretraining of BERT Large in bfloat16 precision using BooksWiki dataset on an HLS1 system:
  ```bash
  cd /path/to/Model-References/TensorFlow/nlp/bert/

  python3 demo_bert.py \
     --command pretraining \
     --model_variant large \
     --data_type bf16 \
     --test_set bookswiki \
     --dataset_path /software/data/tf/data/bert/books_wiki_en_corpus/tfrecord/ \
     --use_horovod 8 \
     --hls_type HLS1 \
  2>&1 | tee ~/hlogs/bert_large_pretraining_bf16_bookswiki_8_cards.txt
  ```
- 8 Gaudi cards finetuning of BERT Large in bfloat16 precision using MRPC dataset on an HLS1 system:
  ```bash
  cd /path/to/Model-References/TensorFlow/nlp/bert/

  python3 demo_bert.py \
     --command finetuning \
     --model_variant large \
     --data_type bf16 \
     --test_set mrpc \
     --dataset_path /software/data/tf/data/bert/MRPC \
     --use_horovod 8 \
     --hls_type HLS1 \
  2>&1 | tee ~/hlogs/bert_large_finetuning_bf16_mrpc_8_cards.txt
  ```
- 8 Gaudi cards finetuning of BERT Large in bfloat16 precision using SQuAD dataset on an HLS1 system:
  ```bash
  cd /path/to/Model-References/TensorFlow/nlp/bert/

  python3 demo_bert.py \
     --command finetuning \
     --model_variant large \
     --data_type bf16 \
     --test_set squad \
     --dataset_path /software/data/tf/data/bert/SQuAD \
     --use_horovod 8 \
     --hls_type HLS1 \
  2>&1 | tee ~/hlogs/bert_large_finetuning_bf16_squad_8_cards.txt
  ```

### Multi-HLS Horovod-based scale-out distributed training

Multi-HLS support in the BERT Python scripts has been enabled using mpirun and Horovod, and has been tested with 2xHLS1 (16 Gaudi cards) and 4xHLS1 (32 Gaudi cards) configurations.

#### Docker ssh port setup for Multi-HLS training

Multi-HLS training works by setting these environment variables:

- **`MULTI_HLS_IPS`**: set this to a comma-separated list of host IP addresses
- `MPI_TCP_INCLUDE`: comma-separated list of interfaces or subnets. This variable will set the mpirun parameter: `--mca btl_tcp_if_include`. This parameter tells mpi which TCP interfaces to use for communication between hosts. You can specify interface names or subnets in the include list in CIDR notation e.g. MPI_TCP_INCLUDE=eno1. More details: [Open MPI documentation](https://www.open-mpi.org/faq/?category=tcp#tcp-selection).
- `DOCKER_SSHD_PORT`: set this to the rsh port used by the sshd service in the docker container

This example shows how to setup for a 4xHLS1 training configuration. The IP addresses used are only examples:

```bash
# This environment variable is needed for multi-HLS training with Horovod.
# Set this to be a comma-separated string of host IP addresses, e.g.:
export MULTI_HLS_IPS="192.10.100.174,10.10.100.101,10.10.102.181,10.10.104.192"

# Set this to the network interface name for the ping-able IP address of the host on
# which the demo_bert.py script is run. This appears in the output of "ip addr".
export MPI_TCP_INCLUDE="eno1"

# This is the port number used for rsh from the docker container, as configured
# in /etc/ssh/sshd_config
export DOCKER_SSHD_PORT=3022
```
By default, the Habana docker uses `port 3022` for ssh, and this is the default port configured in the training scripts. Sometimes, mpirun can fail to establish the remote connection when there is more than one Habana docker session running on the main HLS in which the Python training script is run. If this happens, you can set up a different ssh port as follows:

Follow [Setup](#setup) and [Docker setup and dataset generation](#docker-setup-and-dataset-generation) steps above on all HLS machines. In each HLS host's docker container:
```bash
vi /etc/ssh/sshd_config
```
Uncomment `#Port 22` and replace the port number with a different port number, example `Port 4022`. Next, restart the sshd service:
```bash
service ssh stop
service ssh start
```

Change the `DOCKER_SSHD_PORT` environment variable value to reflect this change into the Python scripts:
```bash
export DOCKER_SSHD_PORT=4022
```

#### Setup password-less ssh between all connected HLS systems used in the scale-out training

1. Follow [Setup](#setup) and [Docker setup and dataset generation](#docker-setup-and-dataset-generation) steps above on all HLS machines used in the scale-out training
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

3. On each system:
   Add all hosts (including itself) to known_hosts. If you configured a different docker sshd port, say `Port 4022`, in [Docker ssh port setup for Multi-HLS training](#docker-ssh-port-setup-for-multi-hls-training), replace `-p 3022` with `-p 4022`. The IP addresses used are only examples:
   ```bash
   ssh-keyscan -p 3022 -H 192.10.100.174 >> ~/.ssh/known_hosts
   ssh-keyscan -p 3022 -H 10.10.100.101 >> ~/.ssh/known_hosts
   ssh-keyscan -p 3022 -H 10.10.102.181 >> ~/.ssh/known_hosts
   ssh-keyscan -p 3022 -H 10.10.104.192 >> ~/.ssh/known_hosts
   ```

#### Run BERT training with 32 cards using demo_bert.py

**Note that to run multi-HLS training over host NICs, the demo_bert.py `--horovod_hierarchical_allreduce=True` option must be set.**

In the main HLS node's docker container:

```bash
export PYTHONPATH=/path/to/Model-References:$PYTHONPATH

cd /path/to/Model-References/TensorFlow/nlp/bert/

python3 demo_bert.py \
        --command <command> \
        --model_variant <model_variant> \
        --data_type <data_type> \
        --test_set <dataset_name> \
        --dataset_path </path/to/dataset> \
        --output_dir </path/to/outputdir> \
        --use_horovod <num_horovod_workers_per_hls> \
        --hls_type <HLS1|HLS1-H>
```

##### Examples

This runs 32-cards BERT Large pretraining in bfloat16 precision with the Bookswiki dataset:

```bash
cd /path/to/Model-References/TensorFlow/nlp/bert/

python3 demo_bert.py \
        --model_variant=large \
        --command=pretraining \
        --test_set=bookswiki \
        --data_type=bf16 \
        --epochs=2 \
        --batch_size=64,8 \
        --max_seq_length=128,512 \
        --iters=7,7 \
        --warmup=7,7 \
        --dataset_path=/software/data/tf/data/bert/books_wiki_en_corpus/tfrecord/ \
        --output_dir=$HOME/tmp/horovod_16_large_bookswiki_pretraining/ \
        --fast_perf_only=0 \
        --save_ckpt_steps=100 \
        --no_steps_accumulation=0 \
        --use_horovod=8 \
        --hls_type=HLS1 \
2>&1 | tee ~/hlogs/bert_large_pretraining_bf16_bookswiki_32_cards.txt
```

This runs 32-cards BERT Large finetuning in bfloat16 precision with the SQuAD dataset:

```bash
cd /path/to/Model-References/TensorFlow/nlp/bert/

python3 demo_bert.py \
        --model_variant=large \
        --command=finetuning \
        --test_set=squad \
        --data_type=bf16 \
        --epochs=2 \
        --batch_size=24 \
        --max_seq_length=384 \
        --learning_rate=3e-5 \
        --output_dir=$HOME/tmp/horovod_squad_large/ \
        --dataset_path=/software/data/tf/data/bert/SQuAD \
        --use_horovod=8 \
        --hls_type=HLS1 \
2>&1 | tee ~/hlogs/bert_large_finetuning_bf16_squad_32_cards.txt
```

#### Run BERT training with 16 cards using demo_bert.py

**Note that to run multi-HLS training over host NICs, the demo_bert.py `--horovod_hierarchical_allreduce=True` option must be set.**

##### Examples

In the main HLS node's docker container:

This runs 16 Gaudi cards BERT Large pretraining in bfloat16 precision using BooksWiki dataset:
```bash
cd /path/to/Model-References/TensorFlow/nlp/bert/
export MULTI_HLS_IPS="192.10.100.174,10.10.100.101"
python3 demo_bert.py -c pretraining -m large -d bf16 -t bookswiki --dataset_path /software/data/tf/data/bert/books_wiki_en_corpus/tfrecord/ -v 8
```
This runs 16 Gaudi cards BERT Large finetuning in bfloat16 precision using SQuAD dataset:
```bash
cd /path/to/Model-References/TensorFlow/nlp/bert/
export MULTI_HLS_IPS="192.10.100.174,10.10.100.101"
python3 demo_bert.py -c finetuning -m large -d bf16 -v 8 -t squad --output_dir /root/tmp/bert_large --dataset_path /software/data/tf/data/bert/SQuAD
```

Habana environment variables for logging, profiling, etc. can be prepended to the training run command-line. For instance, if you want to enable profiling with Synapse traces with SynapseLoggerHook:
```bash
HABANA_SYNAPSE_LOGGER=range python3 demo_bert.py ...
```

## Training BERT in Kubernetes environments using demo_bert.py

Set up the `PYTHONPATH` environment variable based on the file-system path to the Model-References repository. If running multi-card training, set up the `HCL_CONFIG_PATH` environment variable to point to a valid HCL config JSON file for the HLS type being used. For documentation on creating an HCL config JSON file, please refer to [HCL JSON Config File Format](https://docs.habana.ai/en/latest/API_Reference_Guides/HCL_API_Reference.html#hcl-json-config-file-format).

For example, if `/path/to/Model-References` is `/root/Model-References`:

```bash
export PYTHONPATH=/root/Model-References:/usr/lib/habanalabs
export HCL_CONFIG_PATH=/root/hcl/hcl_config_8.json
```

The `demo_bert.py` command-line options for K8s environments are similar to those described earlier for non-K8s configurations, except for these main differences:

- `demo_bert.py` is invoked with the **`--kubernetes_run=True`** command-line option.
- For multi-card Horovod-based distributed training, `mpirun` is used to invoke `demo_bert.py` in the K8s launcher shell. `demo_bert.py` will **not** invoke `mpirun` internally to call the `run_pretraining.py`, `run_classifier.py` and `run_squad.py` TensorFlow training scripts.
- Multi-card distributed training is called with `mpirun ... -np <num_workers_total> python3 demo_bert.py --use_horovod=1 --kubernetes_run=True ...`.
- Multi-card BERT pretraining Phase 1 and Phase 2 are run as two separate steps. The examples in the next sub-sections provide more details.

### Single-card training on K8s

```bash
cd /root/Model-References/TensorFlow/nlp/bert/

python3 demo_bert.py \
        --command <command> \
        --model_variant <model_variant> \
        --data_type <data_type> \
        --test_set <dataset_name> \
        --dataset_path </path/to/dataset> \
        --output_dir </path/to/outputdir> \
        --hls_type <HLS1|HLS1-H> \
        --kubernetes_run=True
```

#### Examples

- Single Gaudi card pretraining of BERT Large in bfloat16 precision using BooksWiki dataset on a K8s HLS1 system:
  ```bash
  cd /root/Model-References/TensorFlow/nlp/bert/

  python3 /root/Model-References/TensorFlow/nlp/bert/demo_bert.py \
     --model_variant=large \
     --command=pretraining \
     --test_set=bookswiki \
     --data_type=bf16 \
     --epochs=1 \
     --batch_size=64,8 \
     --max_seq_length=128,512 \
     --iters=7,7 \
     --warmup=7,7 \
     --dataset_path=/software/data/tf/data/bert/books_wiki_en_corpus/tfrecord/ \
     --output_dir=$HOME/tmp/bert_large_bookswiki/ \
     --fast_perf_only=0 \
     --no_steps_accumulation=0 \
     --hls_type=HLS1 \
     --kubernetes_run=True  \
  2>&1 | tee ~/hlogs/bert_large_pt_1_card.txt
  ```
- Single Gaudi card finetuning of BERT Large in bfloat16 precision using MRPC dataset on a K8s HLS1 system:
  ```bash
  cd /root/Model-References/TensorFlow/nlp/bert/

  python3 /root/Model-References/TensorFlow/nlp/bert/demo_bert.py \
     --model_variant=large \
     --command=finetuning \
     --test_set=mrpc \
     --data_type=bf16 \
     --epochs=3 \
     --batch_size=64 \
     --max_seq_length=128 \
     --learning_rate=2e-5 \
     --output_dir=$HOME/tmp/mrpc_output/ \
     --dataset_path=/software/data/tf/data/bert/MRPC \
     --hls_type=HLS1 \
     --kubernetes_run=True \
  2>&1 | tee ~/hlogs/bert_large_ft_mrpc_1_card.txt
  ```
- Single Gaudi card finetuning of BERT Large in bfloat16 precision using SQuAD dataset on a K8s HLS1 system:
  ```bash
  cd /root/Model-References/TensorFlow/nlp/bert/

  python3 /root/Model-References/TensorFlow/nlp/bert/demo_bert.py \
     --model_variant=large \
     --command=finetuning \
     --test_set=squad \
     --data_type=bf16 \
     --epochs=2 \
     --batch_size=24 \
     --max_seq_length=384 \
     --learning_rate=3e-5 \
     --output_dir=$HOME/tmp/squad_output/ \
     --dataset_path=/software/data/tf/data/bert/SQuAD \
     --hls_type=HLS1 \
     --kubernetes_run=True \
  2>&1 | tee ~/hlogs/bert_large_ft_squad_1_card.txt
  ```

### Multi-card Horovod-based distributed training on K8s

For multi-card Horovod-based distributed training, `mpirun` is used to invoke `demo_bert.py` in the K8s launcher shell. `demo_bert.py` will **not** invoke `mpirun` internally to call the `run_pretraining.py`, `run_classifier.py` and `run_squad.py` TensorFlow training scripts.

**Note: Invoke `demo_bert.py` with the `--kubernetes_run=True` command-line option.**

**Note: For mpirun execution, run `demo_bert.py` with `--use_horovod=1` for one worker per MPI process.**

**Note: To run multi-card/single-HLS training:**

- The user needs to manually generate an HCL config JSON file for the HLS type (`HLS1` or `HLS1-H`) as described in `Example 1 - Single HLS-1 format` in [HCL JSON Config File Format](https://docs.habana.ai/en/latest/API_Reference_Guides/HCL_API_Reference.html#hcl-json-config-file-format). The HCL_CONFIG_PATH environment variable should be set to point to this HCL config JSON file.

**Note: To run multi-HLS scale-out training over Host NICs:**

- The demo_bert.py `--horovod_hierarchical_allreduce=True` option must be set.
- The user needs to manually generate an HCL config JSON file as described in `Example 1 - Single HLS-1 format` in [HCL JSON Config File Format](https://docs.habana.ai/en/latest/API_Reference_Guides/HCL_API_Reference.html#hcl-json-config-file-format). This is because multi-HLS scale-out training over host NICs requires an HCL config JSON file that is the same as the one used for single HLS training. The HCL_CONFIG_PATH environment variable should be set to point to this HCL config JSON file.

**Note: To run multi-HLS scale-out training over Gaudi NICs:**

- The user needs to manually generate an HCL config JSON file as described in `Example 2 - Multiple HLS-1 format` in [HCL JSON Config File Format](https://docs.habana.ai/en/latest/API_Reference_Guides/HCL_API_Reference.html#hcl-json-config-file-format). The HCL_CONFIG_PATH environment variable should be set to point to this HCL config JSON file.

The general command-line for multi-card Horovod-based distributed training over mpirun in K8s setups is as follows:

```bash
mpirun --allow-run-as-root \
       --bind-to core \
       --map-by socket:PE=6 \
       -np <num_workers_total> \
       --tag-output \
       --merge-stderr-to-stdout \
       bash -c "cd /root/Model-References/TensorFlow/nlp/bert;\
       python3 /root/Model-References/TensorFlow/nlp/bert/demo_bert.py \
               --command=<command> \
               --model_variant=<model_variant> \
               --data_type=<data_type> \
               --test_set=<dataset_name> \
               --dataset_path=</path/to/dataset> \
               --output_dir=</path/to/outputdir> \
               --use_horovod=1 \
               --hls_type=<HLS1|HLS1-H> \
               --kubernetes_run=True"
```

#### Examples

- 8 Gaudi cards pretraining of BERT Large in bfloat16 precision using BooksWiki dataset on a K8s HLS1 system:

  **Important note about multi-card BERT pretraining on K8s:**

  Phase 1 and Phase 2 of multi-card Horovod-based TF BERT pretraining will be run as separate `demo_bert.py` invocations on K8s systems:
  - The first `demo_bert.py` invocation will run Phase 1 pretraining only.
  - The second `demo_bert.py` invocation will only run Phase 2 of the pretraining, and needs to be called with the exact same command-line options as the first run, and with an additional option **`--run_phase2=True`**. In particular, the **`--output_dir`** option for running Phase 2 is required and should be set to the same `--output_dir` option used for running Phase 1. demo_bert.py will terminate with an error if the `--output_dir` option is missing when running Phase 2. demo_bert.py checks for the existence of the last Phase 1 step's checkpoint in the `--output_dir` before it begins Phase 2.

  **Run multi-card pretraining with bookswiki, Phase 1:**

```bash
  mpirun --allow-run-as-root \
         --bind-to core \
         --map-by socket:PE=6 \
         -np 8 \
         --tag-output \
         --merge-stderr-to-stdout \
         bash -c "cd /root/Model-References/TensorFlow/nlp/bert;\
                  python3 /root/Model-References/TensorFlow/nlp/bert/demo_bert.py \
                   --model_variant=large \
                   --command=pretraining \
                   --test_set=bookswiki \
                   --data_type=bf16 \
                   --epochs=1 \
                   --batch_size=64,8 \
                   --max_seq_length=128,512 \
                   --iters=7,7 \
                   --warmup=7,7 \
                   --dataset_path=/software/data/tf/data/bert/books_wiki_en_corpus/tfrecord/ \
                   --output_dir=$HOME/tmp/bert_large_bookswiki_8cards/ \
                   --fast_perf_only=0 \
                   --no_steps_accumulation=0 \
                   --use_horovod=1 \
                   --hls_type=HLS1 \
                   --kubernetes_run=True" \
  2>&1 | tee ~/hlogs/bert_large_pt_8cards_phase1.txt
```

  **Run multi-card pretraining with bookswiki, Phase 2:**

```bash
  mpirun --allow-run-as-root \
         --bind-to core \
         --map-by socket:PE=6 \
         -np 8 \
         --tag-output \
         --merge-stderr-to-stdout \
         bash -c "cd /root/Model-References/TensorFlow/nlp/bert;\
                  python3 /root/Model-References/TensorFlow/nlp/bert/demo_bert.py \
                   --model_variant=large \
                   --command=pretraining \
                   --test_set=bookswiki \
                   --data_type=bf16 \
                   --epochs=1 \
                   --batch_size=64,8 \
                   --max_seq_length=128,512 \
                   --iters=7,7 \
                   --warmup=7,7 \
                   --dataset_path=/software/data/tf/data/bert/books_wiki_en_corpus/tfrecord/ \
                   --output_dir=$HOME/tmp/bert_large_bookswiki_8cards/ \
                   --fast_perf_only=0 \
                   --no_steps_accumulation=0 \
                   --use_horovod=1 \
                   --hls_type=HLS1 \
                   --kubernetes_run=True \
                   --run_phase2=True" \
  2>&1 | tee ~/hlogs/bert_large_pt_8cards_phase2.txt
```

- 8 Gaudi cards finetuning of BERT Large in bfloat16 precision using MRPC dataset on a K8s HLS1 system:
```bash
  mpirun --allow-run-as-root \
         --bind-to core \
         --map-by socket:PE=6 \
         -np 8 \
         --tag-output \
         --merge-stderr-to-stdout \
         bash -c "cd /root/Model-References/TensorFlow/nlp/bert;\
                  python3 /root/Model-References/TensorFlow/nlp/bert/demo_bert.py \
                   --model_variant=large \
                   --command=finetuning \
                   --test_set=mrpc \
                   --data_type=bf16 \
                   --epochs=3 \
                   --batch_size=64 \
                   --max_seq_length=128 \
                   --learning_rate=2e-5 \
                   --output_dir=$HOME/tmp/mrpc_output_8cards/ \
                   --dataset_path=/software/data/tf/data/bert/MRPC \
                   --use_horovod=1 \
                   --hls_type=HLS1 \
                   --kubernetes_run=True" \
  2>&1 | tee ~/hlogs/bert_large_ft_mrpc_8cards.txt
```

- 8 Gaudi cards finetuning of BERT Large in bfloat16 precision using SQuAD dataset on a K8s HLS1 system:
```bash
  mpirun --allow-run-as-root \
         --bind-to core \
         --map-by socket:PE=6 \
         -np 8 \
         --tag-output \
         --merge-stderr-to-stdout \
         bash -c "cd /root/Model-References/TensorFlow/nlp/bert;\
                  python3 /root/Model-References/TensorFlow/nlp/bert/demo_bert.py \
                   --model_variant=large \
                   --command=finetuning \
                   --test_set=squad \
                   --data_type=bf16 \
                   --epochs=2 \
                   --batch_size=24 \
                   --max_seq_length=384 \
                   --learning_rate=3e-5 \
                   --output_dir=$HOME/tmp/squad_output_8cards/ \
                   --dataset_path=/software/data/tf/data/bert/SQuAD \
                   --use_horovod=1 \
                   --hls_type=HLS1 \
                   --kubernetes_run=True" \
  2>&1 | tee ~/hlogs/bert_large_ft_squad_8cards.txt
```


## Multi-card training using mpirun with run_pretraining.py, run_classifier.py and run_squad.py

Summary of BERT scripts released in Model-References:

* `demo_bert.py`: Distributed launcher script, enables single-card, multi-card and multi-HLS training for both pretraining and finetuning tasks on non-K8s and K8s platforms. Calls `run_pretraining.py`, `run_classifier.py` and `run_squad.py`. Users are free to reuse Python modules from [Model-References/central](https://github.com/HabanaAI/Model-References/blob/master/central/README.md) and Python scripts from [Model-References/TensorFlow/nlp/bert/](https://github.com/HabanaAI/Model-References/tree/master/TensorFlow/nlp/bert), such as `demo_bert.py`, `bert_pretraining_bookswiki_main.py`, `bert_mrpc_main.py`, etc. to construct their training flows to run on Habana.
* `run_pretraining.py`: Script implementing pretraining.
* `run_classifier.py`: Script implementing finetuning with MRPC.
* `run_squad.py`: Script implementing finetuning with SQuAD.
* `create_datasets_from_start.sh`: Script for downloading, creating and preprocessing BooksWiki dataset.
* `download/download_pretrained_model.py`: Script for downloading pretrained models for BERT Base and BERT Large.

Users who prefer to run multi-card training by directly calling `run_pretraining.py`, `run_classifier.py` and `run_squad.py` can do so using the `mpirun` command and passing the `--horovod` flag to these scripts. In this scenario, correct setup of `HCL_CONFIG_PATH` environment variable is necessary. For documentation on creating an HCL config JSON file, please refer to [HCL JSON Config File Format](https://docs.habana.ai/en/latest/API_Reference_Guides/HCL_API_Reference.html#hcl-json-config-file-format).
Arguments for `mpirun` command in the subsequent examples are setup for best performance on a 56 core CPU host. To run it on a system with lower core count, change the `--map-by` argument value.

```bash
cd /path/to/Model-References/TensorFlow/nlp/bert/

mpirun --allow-run-as-root \
       --tag-output \
       --merge-stderr-to-stdout \
       --output-filename /root/tmp//demo_bert_log/ \
       --bind-to core \
       --map-by socket:PE=7 \
       -np 8 \
       python3 <bert_script> --horovod ...
```

### Pre-requisites

Please run the following additional setup steps before calling `run_pretraining.py`, `run_classifier.py` and `run_squad.py`.

- Set the PYTHONPATH:
  ```bash
  cd /path/to/Model-References/TensorFlow/nlp/bert/
  export PYTHONPATH=./:../../common:../../:../../../central/:$PYTHONPATH
  ```
- Download the pretrained model for the appropriate BERT model variant:

  For BERT Base:
```bash
  python3 download/download_pretrained_model.py \
          "https://storage.googleapis.com/bert_models/2020_02_20/" \
          "uncased_L-12_H-768_A-12" \
          False
```

  For BERT Large:
```bash
  python3 download/download_pretrained_model.py \
          "https://storage.googleapis.com/bert_models/2019_05_30/" \
          "wwm_uncased_L-24_H-1024_A-16" \
          True
```

### Examples

-  8 Gaudi cards pretraining Phase 1 of BERT Large in bfloat16 precision using BooksWiki dataset:
```bash
cd /path/to/Model-References/TensorFlow/nlp/bert/

mpirun --allow-run-as-root \
       --tag-output \
       --merge-stderr-to-stdout \
       --output-filename /root/tmp//demo_bert_log/ \
       --bind-to core \
       --map-by socket:PE=7 \
       -np 8 \
       python3 ./run_pretraining.py \
              --input_files_dir=seq_len_128/books_wiki_en_corpus/training \
              --eval_files_dir=seq_len_128/books_wiki_en_corpus/test \
              --output_dir=/root/tmp/pretraining/phase_1 \
              --bert_config_file=wwm_uncased_L-24_H-1024_A-16/bert_config.json \
              --do_train=True \
              --do_eval=False \
              --train_batch_size=32 \
              --eval_batch_size=8 \
              --max_seq_length=128 \
              --max_predictions_per_seq=20 \
              --num_train_steps=100 \
              --num_accumulation_steps=256 \
              --num_warmup_steps=2000 \
              --save_checkpoints_steps=100 \
              --learning_rate=7.500000e-04 \
              --horovod \
              --noamp \
              --nouse_xla \
              --allreduce_post_accumulation=True \
              --dllog_path=/root/tmp/pretraining/phase_1/bert_dllog.json
```
-  8 Gaudi cards pretraining Phase 2 of BERT Large in bfloat16 precision using BooksWiki dataset, and checkpoint from Phase 1:
```bash
cd /path/to/Model-References/TensorFlow/nlp/bert/

mpirun --allow-run-as-root \
       --tag-output \
       --merge-stderr-to-stdout \
       --output-filename /root/tmp//demo_bert_log/ \
       --bind-to core \
       --map-by socket:PE=7 \
       -np 8 \
       python3 ./run_pretraining.py \
                --input_files_dir=seq_len_512/books_wiki_en_corpus//training \
                --init_checkpoint=/root/tmp/pretraining/phase_1/model.ckpt-100 \
                --eval_files_dir=seq_len_512/books_wiki_en_corpus//test \
                --output_dir=/root/tmp/pretraining/phase_2 \
                --bert_config_file=wwm_uncased_L-24_H-1024_A-16/bert_config.json \
                --do_train=True \
                --do_eval=False \
                --train_batch_size=8 \
                --eval_batch_size=8 \
                --max_seq_length=512 \
                --max_predictions_per_seq=80 \
                --num_train_steps=60 \
                --num_accumulation_steps=512 \
                --num_warmup_steps=200 \
                --save_checkpoints_steps=100 \
                --learning_rate=5.000000e-04 \
                --horovod \
                --noamp \
                --nouse_xla \
                --allreduce_post_accumulation=True \
                --dllog_path=/root/tmp/pretraining/phase_2/bert_dllog.json
```
-  8 Gaudi cards finetuning of BERT Large in bfloat16 precision using MRPC dataset:
```bash
cd /path/to/Model-References/TensorFlow/nlp/bert/

mpirun --allow-run-as-root \
       --tag-output \
       --merge-stderr-to-stdout \
       --output-filename /root/tmp//demo_bert_log/ \
       --bind-to core \
       --map-by socket:PE=7 \
       -np 8 \
       python3 ./run_classifier.py \
                --task_name=MRPC \
                --do_train=true \
                --do_eval=true \
                --data_dir=/software/data/tf/data/bert/MRPC \
                --vocab_file=wwm_uncased_L-24_H-1024_A-16/vocab.txt \
                --bert_config_file=wwm_uncased_L-24_H-1024_A-16/bert_config.json \
                --init_checkpoint=wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt \
                --max_seq_length=128 \
                --train_batch_size=64 \
                --learning_rate=2e-5 \
                --num_train_epochs=0.5 \
                --output_dir=/root/tmp/mrpc_output/ \
                --use_horovod=true
```
-  8 Gaudi cards finetuning of BERT Large in bfloat16 precision using SQuAD dataset:
```bash
cd /path/to/Model-References/TensorFlow/nlp/bert/

mpirun --allow-run-as-root \
       --tag-output \
       --merge-stderr-to-stdout \
       --output-filename /root/tmp//demo_bert_log/ \
       --bind-to core \
       --map-by socket:PE=7 \
       -np 8 \
       python3 ./run_squad.py \
                --vocab_file=wwm_uncased_L-24_H-1024_A-16/vocab.txt \
                --bert_config_file=wwm_uncased_L-24_H-1024_A-16/bert_config.json \
                --init_checkpoint=wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt \
                --do_train=True \
                --train_file=./data/train-v1.1.json \
                --do_predict=True \
                --predict_file=./data/dev-v1.1.json \
                --do_eval=True \
                --train_batch_size=24 \
                --learning_rate=3e-5 \
                --num_train_epochs=0.5 \
                --max_seq_length=384 \
                --doc_stride=128 \
                --output_dir=/root/tmp/squad_large/ \
                --use_horovod=true
```
-  16 Gaudi cards pretraining Phase 1 of BERT Large in bfloat16 precision using BooksWiki dataset:
(The IP addresses in mpirun command are only examples.)

    **Note that to run multi-HLS training over host NICs:**

    - The `--horovod_hierarchical_allreduce=true` flag must be set.
    - The user needs to manually generate an HCL config JSON file as described in `Example 1 - Single HLS-1 format` in [HCL JSON Config File Format](https://docs.habana.ai/en/latest/API_Reference_Guides/HCL_API_Reference.html#hcl-json-config-file-format). This is because multi-HLS scale-out training over host NICs requires an HCL config JSON file that is the same as the one used for single HLS training. The HCL_CONFIG_PATH environment variable should be set to point to this HCL config JSON file.


```bash
cd /path/to/Model-References/TensorFlow/nlp/bert/

mpirun --allow-run-as-root \
       --mca plm_rsh_args -p3022 \
       --bind-to core \
       --map-by socket:PE=7 \
       -np 16 \
       --mca btl_tcp_if_include 192.10.100.174/24 \
       --tag-output \
       --merge-stderr-to-stdout \
       --prefix /usr/lib/habanalabs/openmpi/ \
       -H 192.10.100.174:8,10.10.100.101:8 \
       -x GC_KERNEL_PATH \
       -x HABANA_LOGS \
       -x PYTHONPATH \
       -x HCL_CONFIG_PATH \
       python3 ./run_pretraining.py \
                --input_files_dir=seq_len_128/books_wiki_en_corpus/training \
                --eval_files_dir=seq_len_128/books_wiki_en_corpus/test \
                --output_dir=/root/tmp/pretraining/phase_1 \
                --bert_config_file=wwm_uncased_L-24_H-1024_A-16/bert_config.json \
                --do_train=True \
                --do_eval=False \
                --train_batch_size=32 \
                --eval_batch_size=8 \
                --max_seq_length=128 \
                --max_predictions_per_seq=20 \
                --num_train_steps=100 \
                --num_accumulation_steps=256 \
                --num_warmup_steps=2000 \
                --save_checkpoints_steps=100 \
                --learning_rate=7.500000e-04 \
                --horovod \
                --noamp \
                --nouse_xla \
                --allreduce_post_accumulation=True \
                --dllog_path=/root/tmp/pretraining/phase_1/bert_dllog.json
```
-  16 Gaudi cards finetuning of BERT Large in bfloat16 precision using SQuAD dataset:
(The IP addresses in mpirun command are only examples.)

    **Note that to run multi-HLS training over host NICs:**

    - The `--horovod_hierarchical_allreduce=true` flag must be set.
    - The user needs to manually generate an HCL config JSON file as described in `Example 1 - Single HLS-1 format` in [HCL JSON Config File Format](https://docs.habana.ai/en/latest/API_Reference_Guides/HCL_API_Reference.html#hcl-json-config-file-format). This is because multi-HLS scale-out training over host NICs requires an HCL config JSON file that is the same as the one used for single HLS training. The HCL_CONFIG_PATH environment variable should be set to point to this HCL config JSON file.

```bash
cd /path/to/Model-References/TensorFlow/nlp/bert/

mpirun --allow-run-as-root \
       --mca plm_rsh_args -p3022 \
       --bind-to core \
       --map-by socket:PE=7 \
       -np 16 \
       --mca btl_tcp_if_include 10.211.162.97/24 \
       --tag-output \
       --merge-stderr-to-stdout \
       --prefix /usr/lib/habanalabs/openmpi/ \
       -H 10.211.162.97,10.211.160.140 \
       -x GC_KERNEL_PATH \
       -x HABANA_LOGS \
       -x PYTHONPATH \
       -x HCL_CONFIG_PATH \
       python3 ./run_squad.py \
                --vocab_file=wwm_uncased_L-24_H-1024_A-16/vocab.txt \
                --bert_config_file=wwm_uncased_L-24_H-1024_A-16/bert_config.json \
                --init_checkpoint=wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt \
                --do_train=True \
                --train_file=./data/train-v1.1.json \
                --do_predict=True \
                --predict_file=./data/dev-v1.1.json \
                --do_eval=True \
                --train_batch_size=24 \
                --learning_rate=3e-5 \
                --num_train_epochs=0.5 \
                --max_seq_length=384 \
                --doc_stride=128 \
                --output_dir=/root/tmp/squad_large/ \
                --use_horovod=true
```
## Known Issues

Running BERT Base and BERT Large Finetuning with SQuAD in 16-cards configuration, BS=24, Seq=384 raises a DataLossError "Data loss: corrupted record". Also, running BERT Base and BERT Large Pretraining in fp32 precision with BooksWiki and 16-cards configuration gives errors about "nodes in a cycle". These issues are being investigated.
