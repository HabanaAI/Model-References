# DeepSpeed BERT-1.5B and BERT-5B for PyTorch
This folder contains scripts to pre-train BERT-1.5B and BERT-5B models using DeepSpeed on Habana Gaudi device to achieve an accurate training of a large scale model. To obtain model performance data, refer to the [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

For more information about training deep learning models using Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents
- [Model-References](../../../../README.md)
- [Model Overview](#model-overview)
- [Setup](#setup)
- [Pre-Training the Model](#pre-training-the-model)
- [Advanced](#advanced)
- [Supported Configurations](#supported-configurations)
- [Changelog](#changelog)
- [Known Issues](#known-issues)

## Model Overview
Bidirectional Encoder Representations from Transformers (BERT) is a technique for natural language processing (NLP) pre-training developed by Google.
The original English-language BERT model comes with two pre-trained general types: (1) the BERT BASE model, a 12-layer, 768-hidden, 12-heads, 110M parameter neural network architecture, and (2) the BERT LARGE model, a 24-layer, 1024-hidden, 16-heads, 340M parameter neural network architecture; both of which were trained on the BooksCorpus with 800M words, and a version of the English Wikipedia with 2,500M words.
The pre-training modeling scripts are derived from a clone of https://github.com/NVIDIA/DeepLearningExamples.git.

In this pre-training model script we will introduce the following models:
- **BERT-1.5B**: 48-layer, 1600-hidden, 25-heads, 1.5B parameters neural network architecture. 
- **BERT-5B**: a 63-layer, 2560-hidden, 40-heads, 5B parameters neural network architecture.

### BERT-1.5B and BERT-5B Pre-Training
BERT-1.5B and BERT-5B pre-training with DeepSpeed library includes the following configurations:
- Multi-card data parallel with Zero1 and BF16 (BERT-1.5B).
- Multi-server data parallel with Zero1 and BF16 (BERT-1.5B and BERT-5B).
- Suited for datasets: `wiki`, `bookswiki` (combination of BooksCorpus and Wiki datasets).
- BERT-1.5B uses optimizer: **LANS**
- BERT-5B uses optimizer: **LANS**
- Consists of two tasks:
  - Task 1 - **Masked Language Model** - where when given a sentence, a randomly chosen word is guessed.
  - Task 2 - **Next Sentence Prediction** - where the model guesses whether sentence B comes after sentence A
- The resulting (trained) model weights are language-specific (here: english) and has to be further "fitted" to do a specific task (with fine-tuning).
- Heavy-weight: The training takes several hours or days.



## Setup
Please follow the instructions provided in the [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the
environment including the `$PYTHON` environment variable.
The guide will walk you through the process of setting up your system to run the model on Gaudi.

### Clone Habana Model-References
In the docker container, clone this repository and switch to the branch that
matches your SynapseAI version. You can run the
[`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.
```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References /root/Model-References
```

Go to the deepspeed-bert directory:
```bash
cd /root/Model-References/PyTorch/nlp/pretraining/deepspeed-bert/
```
**Note:** If the repository is not in the PYTHONPATH, make sure to update by running the below.
```bash
export PYTHONPATH=/path/to/Model-References:$PYTHONPATH
```

### Install DeepSpeed Library
In order to run this model, installing DeepSpeed library is required.
Please follow the instruction provided in the [DeepSpeed](https://github.com/HabanaAI/deepspeed) repository on a release branch matching the SynapseAI version being used.

### Install Model Requirements
Install the required Python packages in the docker container:
```bash
pip install -r ./requirements.txt
```

### Pre-training Dataset Preparation
In `deepspeed-bert` directory, `data` directory provides scripts
to download, extract and pre-process [Wikipedia](https://dumps.wikimedia.org/) and [BookCorpus](http://yknzhu.wixsite.com/mbweb) datasets.

Go to `data` directory and run the data preparation script:
```bash
cd ./data
```
It is recommended to download wiki dataset alone using the following command:
```bash
bash create_datasets_from_start.sh
```
Wiki and BookCorpus datasets can be downloaded by running the script as follows:
```bash
bash create_datasets_from_start.sh wiki_books
```
**Note:** The pre-training dataset is huge and takes several hours to download. BookCorpus may have access and download constraints. The final accuracy may vary depending on the dataset and its size.
The script creates formatted dataset for the phase1 of pre-training.

### Reference Script
The base training and modeling scripts for pre-training are based on a clone of [BERT](../bert/) which is based on a clone of https://github.com/NVIDIA/DeepLearningExamples.

## Pre-Training the Model

### Multi-Card Training (Single Server)
Make sure the host machine has 512 GB of RAM installed.
Modify the docker run command to pass 8 Gaudi cards to the docker container.
This ensures the docker has access to all the 8 cards required for multi-card pre-training.

#### Bert-1.5B With Zero1 and BF16 Mixed-Precision on 8 HPUs
**Note:** Ensure the `DATA_DIR` variable in the [run_bert_1.5b_8x.sh](./scripts/run_bert_1.5b_8x.sh) script contains the correct path to the input dataset.

Run pre-training for Phase 1 on 8 HPUs:
  ```bash
  bash ./scripts/run_bert_1.5b_8x.sh
  ```

### Multi-Server Training
To run multi-server demo, make sure the host machine has 512 GB of RAM installed.
Also ensure you followed the [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html)
to install and set up docker, so that the docker has access to all the 8 cards
required for multi-server demo. Multi-server configuration for BERT PT training has been
verified with up to 4 servers, each with 8 Gaudi cards.

Before execution of the multi-server demo scripts, make sure all network interfaces are up. You can change the state of each network interface managed by the habanalabs driver using the following command:
```
sudo ip link set <interface_name> up
```
To identify if a specific network interface is managed by the habanalabs driver type, run:
```
sudo ethtool -i <interface_name>
```
#### Docker ssh Port Setup for Multi-Server Training

By default, the Habana docker uses `port 22` for ssh. The default port configured in the demo script is `port 3022`. Run the following commands to configure the selected port number , `port 3022` in example below.

```bash
sed -i 's/#Port 22/Port 3022/g' /etc/ssh/sshd_config
sed -i 's/#   Port 22/    Port 3022/g' /etc/ssh/ssh_config
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
service ssh restart
```
#### Set Up Password-less ssh
To set up password-less ssh between all connected servers used in scale-out training, follow the below steps:

1. Do the following in all the nodes' docker sessions:
   ```bash
   mkdir ~/.ssh
   cd ~/.ssh
   ssh-keygen -t rsa -b 4096
   ```
   a. Copy id_rsa.pub contents from every node's docker to every other node's docker's ~/.ssh/authorized_keys (all public keys need to be in all hosts' authorized_keys):
   ```bash
   cat id_rsa.pub > authorized_keys
   vi authorized_keys
   ```
   b. Copy the contents from inside to other systems.

   c. Paste all hosts' public keys in all hosts' “authorized_keys” file.

2. On each system, add all hosts (including itself) to known_hosts. The IP addresses used below are just for illustration:
   ```bash
   ssh-keyscan -p 3022 -H 10.10.100.101 >> ~/.ssh/known_hosts
   ssh-keyscan -p 3022 -H 10.10.100.102 >> ~/.ssh/known_hosts
   ssh-keyscan -p 3022 -H 10.10.100.103 >> ~/.ssh/known_hosts
   ssh-keyscan -p 3022 -H 10.10.100.104 >> ~/.ssh/known_hosts
   ```
3. Install the [DeepSpeed](https://github.com/HabanaAI/deepspeed) library on all systems as mentioned in [Install DeepSpeed Library](#install-deepspeed-library).

4. Install the python packages required for BERT-1.5B pre-training on all systems:
   ```bash
   pip install -r /root/Model-References/PyTorch/nlp/pretraining/deepspeed-bert/requirements.txt
   ```

#### Bert-1.5B With Zero1 and BF16 Mixed-Precision on 32 HPUs

- Please review [DeepSpeed documentation](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node) regarding multi-node training.
- A hostfile should be created according to DeepSpeed requirements. Here the [hostfile](./scripts/hostsfile) is present in scripts directory which should be edited with correct IP addresses of all hosts and respective cards.
- DeepSpeed allows to create a **~/.deepspeed_env** file to set environment variables by DeepSpeed across the hosts. Please refer to [multi-node-environment-variables section](https://www.deepspeed.ai/getting-started/#multi-node-environment-variables).
- It is recommended to review [Habana HCCL documentation](https://docs.habana.ai/en/latest/API_Reference_Guides/HCCL_APIs/)
- If your setup requires HOST NICs communication please refer to [Scale out via Host NIC documentation](https://docs.habana.ai/en/latest/API_Reference_Guides/HCCL_APIs/Scale_Out_via_Host_NIC.html)
- For AWS DL1 users it is recommended to use the below `~/.deepspeed_env` configuration:
  ```
  HCCL_OVER_OFI=1
  HCCL_SOCKET_IFNAME=eth0
  LD_LIBRARY_PATH=/root/hccl_ofi_wrapper:/opt/amazon/openmpi/lib:/opt/amazon/efa/lib
  ```
  **Note:** Ensure the `DATA_DIR` variable in the [run_bert_1.5b_32x.sh](./scripts/run_bert_1.5b_32x.sh) script contains the correct path to the input dataset.

Run pre-training for Phase 1 on 32 HPUs:
```bash
bash ./scripts/run_bert_1.5b_32x.sh
```

#### Bert-5B With Zero2 and BF16 Mixed-Precision on 128 HPUs

- Please follow the guidelines described in *Bert-1.5B With Zero1 and BF16 Mixed-Precision on 32 HPUs*

Run pre-training for Phase 1 on 128 HPUs:
```bash
bash ./scripts/run_bert_5b_128x.sh
```

## Advanced
### BERT-1.5B Helper Scripts
Below are the helper scripts for BERT-1.5B configuration and training:

- Model_Config: [bert_1.5b_config.json](./scripts/bert_1.5b_config.json)<br>
- DeepSpeed Config: [deepspeed_config_bert_1.5b.json](./scripts/deepspeed_config_bert_1.5b.json)
- 8 card training helper script: [run_bert_1.5b_8x.sh](./scripts/run_bert_1.5b_8x.sh)
- 32 card training helper script: [run_bert_1.5b_32x.sh](./scripts/run_bert_1.5b_32x.sh)
- Hostfile: [hostfile](./scripts/hostsfile)

### BERT-5B Helper Scripts
Below are the helper scripts for BERT-5B configuration and training:

- Model_Config: [bert_5b_config.json](./scripts/bert_5b_config.json)<br>
- DeepSpeed Config: [deepspeed_config_bert_5b.json](./scripts/deepspeed_config_bert_5b.json)
- 128 card training helper script: [run_bert_5b_128x.sh](./scripts/run_bert_5b_128x.sh)
- Hostfile: [hostfile](./scripts/hostsfile)

## Supported Configurations

| Validated on | SynapseAI Version | PyTorch Version | Mode |
|--------|-------------------|-----------------|----------------|
| Gaudi   | 1.8.0             | 1.13.1          | Training |
| Gaudi2  | 1.8.0             | 1.13.1          | Training |

## Changelog
### 1.8.0
1. Added support for profiling via --profile and --profile_steps script flag.
2. Removed non-required WAs.
3. Disabled Accumulation thread as a WA.
4. Reduced Max HCCL comms to 1 for memory purposes.
5. Re-used DeepSpeed engine data_parallel_group in the script.
6. Added support for BS=8 in Bert-5B over x128 cards.
7. Added support for BS=24 in Bert-5B with checkpoint-activation over x128 cards.
8. Removed weight sharing WA for decoder and embeddings.
9. Added mark_step before and after the loss for memory purposes for memory reasons.
10. Added BWD hook that will mark_step for the BertLMPredictionHead layer for memory reasons.
11. Modified Bert-5B example to run with checkpoint-activations and BS=24. 

### 1.7.0
1. Added support for distributed emulation mode.
2. Improved memory consumption for Bert-5B model via environment flags.
3. Fixed time_per_step metric to report time per model step instead of acc-step.

### 1.6.1
1. Moved Bert-5B model specific workarounds to run_pretraining.py.
2. Removed lazy_mode argument.

### 1.6.0
1. Changed default optimizer of 1.5b model to LANS.
2. Added script support for using deepspeed activation checkpointing.
3. Added option for LANS optimizer.
4. Improved checkpoint mechanism for better reproducibility.
5. Added new model and deepspeed configuration files and script example for Bert-5B.

### 1.5.0
1. Created this training script based on a clone of [BERT](../bert/) version 1.4.0 .
2. Made adjustment to DeepSpeed engine.
3. Added recommended deepspeed configuration files.
4. Added new model configuration file for Bert-1.5B.

### Training Script Modifications
This section lists the training script modifications for the BERT-1.5B and BERT-5B.
1. Wrapped model object with DeepSpeed wrapper.
2. Call DeepSpeed's model.backward(loss) instead of loss.backward().
3. Call DeepSpeed's model.step() instead of optimizer.step().
4. Removed all gradient_accumulation handling from the training loop, as it is being handled by deepspeed engine.
5. Added TensorLogger utility to enable accuracy debug.
6. Removed user configuration from user script that from now will be configured through DeepSpeed.
7. Enabled CUDA functionality in training script.
8. Removed torch.distributed initialization from the script (also moved to DeepSpeed).
9. Saved checkpoint flow should also go through DeepSpeed.
10. Added new configuration flags to enable 1.5B training regime specification.

## Known Issues
Placing mark_step() arbitrarily may lead to undefined behavior. Recommend to keep mark_step() as shown in provided scripts.
