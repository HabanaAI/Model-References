# DeepSpeed BERT-1.5B
This folder contains scripts to pre-train BERT-1.5B model using DeepSpeed on Habana Gaudi<sup>TM</sup> device to achieve an accurate training of a large scale model.
For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

# Table of Contents
- [Model-References](../../../../README.md)
- [Model Overview](#model-overview)
  - [BERT-1.5B Pre-Training](#bert-15b-pre-training)
- [Setup](#setup)
  - [DeepSpeed Installation](#deepspeed-installation)
  - [Install requirements](#install-requirements)
  - [Pre-training dataset preparation](#pre-training-dataset-preparation)
  - [Reference Script](#reference-script)
- [Pre-Training the Model](#pre-training-the-model)
  - [Multicard Training (Single Node)](#multicard-training-single-node)
  - [Multinode Training](#multinode-training)
- [Advanced](#advanced)
  - [Helper Scripts](#helper-scripts)
  - [Training Script Modifications](#training-script-modifications)
- [Supported Configurations](#supported-configurations)
- [Changelog](#changelog)
  - [1.5.0](#150)
- [Known Issues](#known-issues)

# Model Overview
Bidirectional Encoder Representations from Transformers (BERT) is a technique for natural language processing (NLP) pre-training developed by Google.
The original English-language BERT model comes with two pre-trained general types: (1) the BERTBASE model, a 12-layer, 768-hidden, 12-heads, 110M parameter neural network architecture, and (2) the BERTLARGE model, a 24-layer, 1024-hidden, 16-heads, 340M parameter neural network architecture; both of which were trained on the BooksCorpus with 800M words, and a version of the English Wikipedia with 2,500M words.
The Pre-training modeling scripts are derived from a clone of https://github.com/NVIDIA/DeepLearningExamples.git

In this pre-training model script we will introduce a 48-layer, 1600-hidden, 25-heads, 1.5B parameters neural network architecture, which will be referred to as Bert-1.5B.

## BERT-1.5B Pre-Training
- BERT-1.5B pre-training with DeepSpeed library includes the following configurations:
  - Multi card data parallel with Zero1 and BF16.
  - Multi node data parallel with Zero1 and BF16.
- Suited for datasets:
  - `wiki`, `bookswiki`(combination of BooksCorpus and Wiki datasets)
- Uses optimizer:
  - **LAMB** ("Layer-wise Adaptive Moments optimizer for Batch training").
- Consists of 2 phases:
  - Task 1 - **Masked Language Model** - where when given a sentence, a randomly chosen word is guessed.
  - Task 2 - **Next Sentence Prediction** - where the model guesses whether sentence B comes after sentence A
- The resulting (trained) model weights are language-specific (here: english) and has to be further "fitted" to do a specific task (with finetuning).
- Heavy-weight: the training takes several hours or days.

# Setup
Please follow the instructions given in the following link for setting up the
environment including the `$PYTHON` environment variable: [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html).
This guide will walk you through the process of setting up your system to run
the model on Gaudi.

In the docker container, clone this repository and switch to the branch that
matches your SynapseAI version. (Run the
[`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.)
```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References /root/Model-References
```

Go to the deepspeed-bert directory:
```bash
cd /root/Model-References/PyTorch/nlp/pretraining/deepspeed-bert/
```

## DeepSpeed Installation
In order to run this model - DeepSpeed library is required.
Please follow the instruction in the [DeepSpeed](https://github.com/HabanaAI/deepspeed) repository on a release branch matching the SynapseAI version being used.

## Install requirements
Install the required Python packages in the container:
```bash
pip install -r ./requirements.txt
```

## Pre-training dataset preparation
In deepspeed-bert directory, `data` directory provides scripts
to download, extract and preprocess [Wikipedia](https://dumps.wikimedia.org/) and [BookCorpus](http://yknzhu.wixsite.com/mbweb) datasets.

Go to `data` directory and run the data preparation script.
```bash
cd ./data
```
It is recommended to download wiki data set alone using the following command.
```bash
bash create_datasets_from_start.sh
```
Wiki and BookCorpus data sets can be downloaded by runnining the script as follows.
```bash
bash create_datasets_from_start.sh wiki_books
```
**Note:** The pretraining dataset is huge and takes several hours to download. BookCorpus may have access and download constraints. The final accuracy may vary depending on the dataset and its size.
The script creates formatted dataset for the phase1 of pre-training.

## Reference Script
The base training and modeling scripts for pretraining are based on a clone of [BERT](../bert/) which is based on a clone of https://github.com/NVIDIA/DeepLearningExamples.

# Pre-Training the Model
Clone the Model-References repository and navigate to the deepspeed-bert directory.
Set up the dataset as mentioned in the section [Pre-training dataset preparation](#pre-training-dataset-preparation), install the [DeepSpeed](https://github.com/HabanaAI/deepspeed) library and the python packages as mentioned in the section [Install requirements](install-requirements).

## Multicard Training (Single Node)
Make sure the host machine has 512 GB of RAM installed.
Modify the docker run command to pass 8 Gaudi cards to the docker container.
This ensures the docker has access to all the 8 cards required for multi-card pre-training.

### Bert-1.5B With Zero1 and BF16 Mixed-Precision on 8xHPUs:
Note: Ensure the `DATA_DIR` variable in the [run_bert_1.5b_8x.sh](./scripts/run_bert_1.5b_8x.sh) script contains the correct path to the input dataset.

- Run pre-training for Phase1 8 Gaudi cards:
  ```bash
  bash ./scripts/run_bert_1.5b_8x.sh
  ```

## Multi-node Training
### Setup instructions
To run multi-node demo, make sure the host machine has 512 GB of RAM installed.
Also ensure you followed the [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html)
to install and set up docker, so that the docker has access to all the 8 cards
required for multi-node demo. Multi-node configuration for BERT PT training has been
verified with up to 4 servers, each with 8 Gaudi cards.

Before execution of the multi-node demo scripts, make sure all network interfaces are up. You can change the state of each network interface managed by the habanalabs driver using the following command:
```
sudo ip link set <interface_name> up
```
To identify if a specific network interface is managed by the habanalabs driver type, run:
```
sudo ethtool -i <interface_name>
```
#### Docker ssh port setup for multi-server training

By default, the Habana docker uses `port 22` for ssh. The default port configured in the demo script is `port 3022`. Run the following commands to configure the selected port number , `port 3022` in example below.

```bash
sed -i 's/#Port 22/Port 3022/g' /etc/ssh/sshd_config
sed -i 's/#Port 22/Port 3022/g' /etc/ssh/ssh_config
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
service ssh restart
```
#### Setup password-less ssh between all connected servers used in the scale-out training

1. Configure password-less ssh between all nodes:

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
   Note: Copy the contents from inside to other systems.
   Paste all hosts' public keys in all hosts' “authorized_keys” file.

2. On each system:

   Add all hosts (including itself) to known_hosts. The IP addresses used below are just for illustration:
   ```bash
   ssh-keyscan -p 3022 -H 10.10.100.101 >> ~/.ssh/known_hosts
   ssh-keyscan -p 3022 -H 10.10.100.102 >> ~/.ssh/known_hosts
   ssh-keyscan -p 3022 -H 10.10.100.103 >> ~/.ssh/known_hosts
   ssh-keyscan -p 3022 -H 10.10.100.104 >> ~/.ssh/known_hosts
   ```
   Install the [DeepSpeed](https://github.com/HabanaAI/deepspeed) library on all systems as mentioned in section [DeepSpeed Installation](#deepSpeed-installation).

   Install python packages required for BERT-1.5B Pre-training on all systems:
   ```bash
   pip install -r /root/Model-References/PyTorch/nlp/pretraining/deepspeed-bert/requirements.txt
   ```

### Bert-1.5B With Zero1 and BF16 Mixed-Precision on 32xHPUs:

- Please review [DeepSpeed documentation](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node) regarding multi-node training.
- A hostfile should be created according to DeepSpeed requirements. Here the [hostfile](./scripts/hostsfile) is present in scripts directory which should be edited with correct IP addresses of all hosts and respective cards.
- DeepSpeed allows to create a **~/.deepspeed_env** file to set environment vairables by DeepSpeed across the hosts. Please refer to [multi-node-environment-variables section](https://www.deepspeed.ai/getting-started/#multi-node-environment-variables).
- It is recommended to review [Habana HCCL documentation](https://docs.habana.ai/en/latest/API_Reference_Guides/HCCL_APIs/)
- If your setup requires HOST NICs communication please refer to [Scale out via Host NIC documentation](https://docs.habana.ai/en/latest/API_Reference_Guides/HCCL_APIs/Scale_Out_via_Host_NIC.html)
- For AWS DL1 users it is recommended to use the below ~/.deepspeed_env configuration:
  ```
  HCCL_OVER_TCP=0
  HCCL_OVER_OFI=1
  HCCL_SOCKET_IFNAME=eth0
  LD_LIBRARY_PATH=/root/hccl_ofi_wrapper:/opt/amazon/openmpi/lib:/opt/amazon/efa/lib
  ```
  Note: Ensure the `DATA_DIR` variable in the [run_bert_1.5b_32x.sh](./scripts/run_bert_1.5b_32x.sh) script contains the correct path to the input dataset.

- Run pre-training for Phase1 32 Gaudi cards:
  ```bash
  bash ./scripts/run_bert_1.5b_32x.sh
  ```

# Advanced
## Helper Scripts
Below are the helper scripts for configuration and training:

- Model_Config: [bert_1.5b_config.json](./scripts/bert_1.5b_config.json)<br>
- DeepSpeed Config: [deepspeed_config_bert_1.5b.json](./scripts/deepspeed_config_bert_1.5b.json)
- 8 card training helper script: [run_bert_1.5b_8x.sh](./scripts/run_bert_1.5b_8x.sh)
- 32 card training helper script: [run_bert_1.5b_32x.sh](./scripts/run_bert_1.5b_32x.sh)
- Hostfile: [hostfile](./scripts/hostsfile)

## Training Script Modifications
This section lists the training script modifications for the BERT-1.5B.

1. Functional Changes:
   1. Wrap model object with DeepSpeed wrapper
   2. Call DeepSpeed's model.backward(loss) instead of loss.backward()
   3. Call DeepSpeed's model.step() instead of optimizer.step()
   4. Remove all gradient_accumulation handling from the training loop, as it is being handled by deepspeed engine.
   5. Adding TensorLogger utility to enable accuracy debug
   6. Remove user configuration from user script that from now will be configured through DeepSpeed
   7. Enable CUDA functionlity in training script
   8. Remove torch.distributed intialization from the script (also moved to DeepSpeed)
   9. Save checkpoint flow should also go through DeepSpeed
   10. Adding new configuration flags to enable 1.5B training regime specification.

# Supported Configurations

| Device | SynapseAI Version | PyTorch Version |
|-----|-----|-----|
| Gaudi | 1.5.0 | 1.11.0 |

# Changelog
## 1.5.0
1. Created this training script based on a clone of [BERT](../bert/) version 1.4.0 .
2. Adjustment to DeepSpeed engine
3. Add recommended deepspeed configuration files
4. Add new model configuration file for Bert-1.5B

# Known Issues
1. Placing mark_step() arbitrarily may lead to undefined behaviour. Recommend to keep mark_step() as shown in provided scripts.
