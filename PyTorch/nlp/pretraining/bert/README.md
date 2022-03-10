# Table of Contents
- [BERT for PyTorch](#bert-for-pytorch)
  - [BERT Pre-Training](#bert-pre-training)
- [Setup](#setup)
  - [Pre-training dataset preparation](#pre-training-dataset-preparation)
  - [Model Overview](#model-overview)
- [Pre-Training](#pre-training)
  - [Reference Script](#reference-script)
  - [Training the Model](#training-the-model)
  - [Multicard Training](#multicard-training)
  - [Multinode Training](#multinode-training)
      - [Docker ssh port setup for multi-server training](#docker-ssh-port-setup-for-multi-server-training)
      - [Setup password-less ssh between all connected servers used in the scale-out training](#setup-password-less-ssh-between-all-connected-servers-used-in-the-scale-out-training)
- [Changelog](#changelog)
  - [1.3.0](#130)
  - [1.2.0](#120)
- [Known Issues](#known-issues)
- [Training Script Modifications](#training-script-modifications)
  - [BERT Large Pre-training](#bert-large-pre-training)

# BERT for PyTorch

This folder contains scripts to pre-train BERT model on Habana Gaudi<sup>TM</sup> device to achieve state-of-the-art accuracy. Please visit [this page](https://developer.habana.ai/resources/habana-training-models/#performance) for performance information.

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

The BERT demos included in this release are as follows:

## BERT Pre-Training
- BERT Large pre-training for BF16 mixed precision for Wikipedia BookCorpus and Wiki dataset in Lazy mode.
- Multi card (1 server = 8 cards) support for BERT Large Pretraining with BF16 Mixed precision in Lazy mode

The Demo script is a wrapper for respective python training scripts. Additional environment variables are used in training scripts in order to achieve optimal results for each workload.

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
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```

## Pre-training dataset preparation

`Model-References/PyTorch/nlp/bert/pretraining/data` provides scripts to download, extract and preprocess [Wikipedia](https://dumps.wikimedia.org/) and [BookCorpus](http://yknzhu.wixsite.com/mbweb) datasets.
Install the required Python packages in the container:
```
 pip install -r Model-References/PyTorch/nlp/bert/pretraining/requirements.txt
```
Then, go to `data` folder and run the data preparation script.
```
cd Model-References/PyTorch/nlp/bert/pretraining/data
```
So it is recommended to download wiki data set alone using the following command.
```
bash create_datasets_from_start.sh
```
Wiki and BookCorpus data sets can be downloaded by runnining the script as follows.
```
bash create_datasets_from_start.sh wiki_books
```
Note that the pretraining dataset is huge and takes several hours to download. BookCorpus may have access and download constraints. The final accuracy may vary depending on the dataset and its size.
The script creates formatted dataset for the phase1 and 2 of pre-training.


## Model Overview
Bidirectional Encoder Representations from Transformers (BERT) is a technique for natural language processing (NLP) pre-training developed by Google.
The original English-language BERT model comes with two pre-trained general types: (1) the BERTBASE model, a 12-layer, 768-hidden, 12-heads, 110M parameter neural network architecture, and (2) the BERTLARGE model, a 24-layer, 1024-hidden, 16-heads, 340M parameter neural network architecture; both of which were trained on the BooksCorpus with 800M words, and a version of the English Wikipedia with 2,500M words.
The Pretraining modeling scripts are derived from a clone of https://github.com/NVIDIA/DeepLearningExamples.git and the fine tuning is based on https://github.com/huggingface/transformers.git.


# Pre-Training
- Located in: `Model-References/PyTorch/nlp/bert/pretraining`
- Suited for datasets:
  - `wiki`, `bookswiki`(combination of BooksCorpus and Wiki datasets)
- Uses optimizer: **LAMB** ("Layer-wise Adaptive Moments optimizer for Batch training").
- Consists of 2 phases:
  - Task 1 - **Masked Language Model** - where given a sentence, a randomly chosen word is guessed.
  - Task 2 - **Next Sentence Prediction** - where the model guesses whether sentence B comes after sentence A
- The resulting (trained) model weights are language-specific (here: english) and has to be further "fitted" to do a specific task (with finetuning).
- Heavy-weight: the training takes several hours or days.

BERT training script supports pre-training of  dataset on BERT large for both FP32 and BF16 mixed precision data type using **Lazy mode**.

## Reference Script
The base training and modeling scripts for pretraining are based on a clone of
https://github.com/NVIDIA/DeepLearningExamples.

## Training the Model
Clone the Model-References git.
Set up the data set as mentioned in the section "Set up dataset".

```
cd Model-References/PyTorch/nlp/pretraining/bert
```

Install python packages required for BERT Pre-training model
```
pip install -r Model-References/PyTorch/nlp/pretraining/bert/requirements.txt
```

Run `$PYTHON demo_bert.py pretraining -h` for command-line options.

You can use Python launcher of `habana_model_runner.py` located in `Model-References/central` folder to
launch the training for the specified model.


i. lazy mode, bf16 mixed precision, BS64 for phase1 and BS8 for phase2:
```
$PYTHON demo_bert.py pretraining --model_name_or_path large --mode lazy --data_type bf16 --batch_size 64 8 --accumulate_gradients --allreduce_post_accumulation --allreduce_post_accumulation_fp16 --data_dir <dataset_path_phase1> <dataset_path_phase2>
```

ii. lazy mode, fp32 precision, BS32 for phase1 and BS4 for phase2:
```
$PYTHON demo_bert.py pretraining --model_name_or_path large --mode lazy --data_type fp32 --batch_size 32 4 --accumulate_gradients --allreduce_post_accumulation --allreduce_post_accumulation_fp16 --data_dir <dataset_path_phase1> <dataset_path_phase2>
```

## Multicard Training
Follow the relevant steps under "Training the Model".
To run multi-card demo, make sure the host machine has 512 GB of RAM installed. Modify the docker run command to pass 8 Gaudi cards to the docker container. This ensures the docker has access to all the 8 cards required for multi-card demo.

Use the following commands to run multicard training on 8 cards:


i. lazy mode, bf16 mixed precision, per chip batch size of 64 for phase1 and 8 for phase2:
```
$PYTHON demo_bert.py pretraining --model_name_or_path large --mode lazy --data_type bf16 --world_size 8 --batch_size 64 8 --accumulate_gradients --allreduce_post_accumulation --allreduce_post_accumulation_fp16 --dist --data_dir <dataset_path_phase1> <data_path_phase2>
```
```
export MASTER_ADDR="localhost"
export MASTER_PORT="12345"
mpirun -n 8 --bind-to core --map-by socket:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON pretraining/run_pretraining.py --do_train --bert_model=bert-large-uncased --hmp --hmp_bf16=./ops_bf16_bert_pt.txt --hmp_fp32=./ops_fp32_bert_pt.txt --use_lazy_mode --config_file=./pretraining/bert_config.json --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 --json-summary=/tmp/BERT_PRETRAINING/results/dllogger.json --output_dir=/tmp/BERT_PRETRAINING/results/checkpoints --seed=12439 --use_fused_lamb --input_dir=/root/software/lfs/data/pytorch/bert/pretraining/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/books_wiki_en_corpus --train_batch_size=8192 --max_seq_length=128 --max_predictions_per_seq=20 --warmup_proportion=0.2843 --num_steps_per_checkpoint=200 --learning_rate=0.006 --gradient_accumulation_steps=128

```
```
export MASTER_ADDR="localhost"
export MASTER_PORT="12345"
mpirun -n 8 --bind-to core --map-by socket:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON pretraining/run_pretraining.py --do_train --bert_model=bert-large-uncased --hmp --hmp_bf16=./ops_bf16_bert_pt.txt --hmp_fp32=./ops_fp32_bert_pt.txt --use_lazy_mode --config_file=/root/model_garden/PyTorch/nlp/bert/pretraining/bert_config.json --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 --json-summary=/tmp/BERT_PRETRAINING/results/dllogger.json --output_dir=/tmp/BERT_PRETRAINING/results/checkpoints --seed=12439 --use_fused_lamb --input_dir=/root/software/lfs/data/pytorch/bert/pretraining/hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/books_wiki_en_corpus --train_batch_size=4096 --max_seq_length=512 --max_predictions_per_seq=80 --warmup_proportion=0.128 --num_steps_per_checkpoint=200 --learning_rate=0.004 --gradient_accumulation_steps=512 --resume_from_checkpoint --phase2

```

ii. lazy mode, fp32 precision, per chip batch size of 32 for phase1 and 4 for phase2:
```
$PYTHON demo_bert.py pretraining --model_name_or_path large --mode lazy --data_type fp32 --world_size 8 --batch_size 32 4  --accumulate_gradients --allreduce_post_accumulation --allreduce_post_accumulation_fp16 --dist --data_dir <dataset_path_phase1> <data_path_phase2>
```

## Multinode Training
To run multi-node demo, make sure the host machine has 512 GB of RAM installed.
Also ensure you followed the [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html)
to install and set up docker, so that the docker has access to all the 8 cards
required for multi-node demo. Multinode configuration for BERT PT training upto
4 servers, each with 8 Gaudi cards, have been verified.

Before execution of the multi-node demo scripts, make sure all network interfaces are up. You can change the state of each network interface managed by the habanalabs driver using the following command:
```
sudo ip link set <interface_name> up
```
To identify if a specific network interface is managed by the habanalabs driver type, run:
```
sudo ethtool -i <interface_name>
```
#### Docker ssh port setup for multi-server training

Multi-server training works by setting these environment variables:

- **`MULTI_HLS_IPS`**: set this to a comma-separated list of host IP addresses. Network id is derived from first entry of this variable

This example shows how to setup for a 4 server training configuration. The IP addresses used are only examples:

```bash
export MULTI_HLS_IPS="10.10.100.101,10.10.100.102,10.10.100.103,10.10.100.104"

# When using demo script, export the following variable to the required port number (default 3022)
export DOCKER_SSHD_PORT=3022
```
By default, the Habana docker uses `port 22` for ssh. The default port configured in the demo script is `port 3022`. Run the following commands to configure the selected port number , `port 3022` in example below.

```bash
sed -i 's/#Port 22/Port 3022/g' /etc/ssh/sshd_config
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
   Copy the contents from inside to other systems.
   Paste all hosts' public keys in all hosts' “authorized_keys” file.

2. On each system:
   Add all hosts (including itself) to known_hosts. The IP addresses used below are just for illustration:
   ```bash
   ssh-keyscan -p 3022 -H 10.10.100.101 >> ~/.ssh/known_hosts
   ssh-keyscan -p 3022 -H 10.10.100.102 >> ~/.ssh/known_hosts
   ssh-keyscan -p 3022 -H 10.10.100.103 >> ~/.ssh/known_hosts
   ssh-keyscan -p 3022 -H 10.10.100.104 >> ~/.ssh/known_hosts
   ```
   Install python packages required for BERT Pre-training model
   ```
   pip install -r Model-References/PyTorch/nlp/bert/pretraining/requirements.txt
   ```

i. BERT Pre-training, lazy mode, bf16 mixed precision, BS64 for phase1 and BS8 for phase2:
```python
$PYTHON -u demo_bert.py pretraining --world_size 32 --model_name_or_path large --config_file /root/model_garden/PyTorch/nlp/bert/pretraining/bert_config.json --output_dir /root/nfs/BERT_PRETRAINING/results/checkpoints --seed 42 --allreduce_post_accumulation --allreduce_post_accumulation_fp16 --accumulate_gradients --device hpu --mode lazy --data_type bf16 --data_dir <dataset_path_phase1> <data_path_phase2>  --max_seq_length 128 512 --train_batch_size 65536 32768 --batch_size 64 8 --learning_rate 0.006 0.004 --warmup 0.2843 0.128 --max_steps 7038 1563 --steps_this_run 7038 1563 --init_checkpoint None None --phase 1 --process_per_node 8
```

# Changelog
## 1.3.0
1. Single worker thread changes are removed.
2. Loss computation brought it back to training script.
3. Removed setting the embedding padding index as 0 explicitly.
4. Removed the select op implementation using index select and squeeze and retained the default code.
5. Permute and view is replaced as flatten.
6. Change `python` or `python3` to `$PYTHON` to execute correct version based on environment setup.

## 1.2.0
1. Enabled HCCL flow for distributed training.
2. Removed changes related to data type conversions for input_ids, segment ids, position_ids and input_mask.
3. Removed changes related to position ids from training script.
4. Removed changes related to no pinned memory and skip last batch.


# Known Issues
1. Placing mark_step() arbitrarily may lead to undefined behaviour. Recommend to keep mark_step() as shown in provided scripts.


# Training Script Modifications
This section lists the training script modifications for the BERT models.

## BERT Large Pre-training
The following changes have been added to training (run_pretraining.py) & modeling (modeling.py) scripts.

1. Added support for Habana devices:

    a. Load Habana specific library.

    b. Support required for cpu to work.

    c. Required environment variables are defined for habana device.

    d. Saving checkpoint: Bring tensors to CPU and save.

    e. Added Habana BF16 Mixed precision support.

    f. Added python version of LAMB optimizer and will be used as default(from lamb.py).

    g. Support for distributed training on Habana device.

    h. Added changes to support Lazy mode with required mark_step().

    i. Added changes to calculate the performance per step and report through dllogger.

    j. Using conventional torch layernorm, linear and activation functions.

    k. Changes for dynamic loading of HCCL library.

2. To improve performance:

    a. Added support for Fused LAMB optimizer.

    b. Bucket size set to 230MB for better performance in distributed training.

    c. Added support to use distributed all_reduce instead of default Distributed Data Parallel.

    d. Added support for lowering print frequency of loss and associated this with log_freq.

