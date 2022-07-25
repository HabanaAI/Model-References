# Table of Contents
- [Model References](../../../../README.md)
- [BERT for PyTorch](#bert-for-pytorch)
  - [BERT Pre-Training](#bert-pre-training)
- [Setup](#setup)
  - [Pre-training dataset preparation](#pre-training-dataset-preparation)
  - [Packing the data](#packing-the-data)
  - [Model Overview](#model-overview)
- [Pre-Training](#pre-training)
  - [Reference Script](#reference-script)
  - [Training the Model](#training-the-model)
  - [Multicard Training](#multicard-training)
  - [Multinode Training](#multinode-training)
      - [Docker ssh port setup for multi-server training](#docker-ssh-port-setup-for-multi-server-training)
      - [Setup password-less ssh between all connected servers used in the scale-out training](#setup-password-less-ssh-between-all-connected-servers-used-in-the-scale-out-training)
- [Supported Configurations](#supported-configurations)
- [Changelog](#changelog)
  - [1.5.0](#150)
  - [1.4.0](#140)
  - [1.3.0](#130)
  - [1.2.0](#120)
- [Known Issues](#known-issues)
- [Training Script Modifications](#training-script-modifications)
  - [BERT Large Pre-training](#bert-large-pre-training)

# BERT for PyTorch

This folder contains scripts to pre-train BERT model on Habana Gaudi<sup>TM</sup> device to achieve state-of-the-art accuracy. Please visit [this page](https://developer.habana.ai/resources/habana-training-models/#performance) for performance information.

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

**Note**: BERT is enabled on both Gaudi and **Gaudi2**.

The BERT demos included in this release are as follows:

## BERT Pre-Training
- BERT Large pre-training for BF16 mixed precision for Wikipedia BookCorpus and Wiki dataset in Lazy mode.
- Multi card (1 server = 8 cards) support for BERT Large Pretraining with BF16 Mixed precision in Lazy mode

Additional environment variables are used in training scripts in order to achieve optimal results for each workload.

# Setup
Please follow the instructions given in the following link for setting up the
environment including the `$PYTHON` environment variable: [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html).
This guide will walk you through the process of setting up your system to run
the model on Gaudi.

In the docker container, clone this repository and switch to the branch that
matches your SynapseAI version. (Run the
[`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
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

## Packing the data
We support the option to use a [data packing
technique](https://github.com/HabanaAI/Gaudi-tutorials/blob/main/TensorFlow/DataPackingMLperfBERT/Data_Packing_Process_for_MLPERF_BERT.ipynb),
called Non-Negative Least Squares Histogram. Here, instead of padding with zero,
we pack several short sequences into one multi-sequence of size `max_seq_len`.
Thus, we remove most of the padding, which can lead to a speedup of up to 2x times;
in time-to-train (TTT). This packing technique can be applied on other datasets
with high variability in samples length.

Please note that for each NLP dataset with sequential data samples, the speedup
with data packing is determined by the ratio of `max_seq_len` to
`average_seq_len` in that particular dataset. The larger the ratio, the higher
the speedup.

To pack the dataset, in docker run
```bash
cd /root/Model-References/PyTorch/nlp/pretraining/bert

$PYTHON pack_pretraining_data_pytorch.py --input_dir <dataset_path_phase1> --output-dir <packed_dataset_path_phase1> --max_sequence_length 128 --max_predictions_per_sequence 20

$PYTHON pack_pretraining_data_pytorch.py --input_dir <dataset_path_phase2> --output-dir <packed_dataset_path_phase2> --max_sequence_length 512 --max_predictions_per_sequence 80
```
Note: This will generate json at the path <output-dir>/../<tail_dir>_metadata.json with meta data info like: "avg_seq_per_sample" etc. This json will be
used as an input to run_pretraining.py to extract "avg_seq_per_sample" in case of packed dataset mode.

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

Please create a log directory to store dllogger.json and specify its location for --json_summary attribute.


i. lazy mode, bf16 mixed precision, BS64 for phase1 and BS8 for phase2:
```bash
$PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased --hmp \
      --hmp_bf16=./ops_bf16_bert_pt.txt --hmp_fp32=./ops_fp32_bert_pt.txt --config_file=./bert_config.json \
      --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
      --json-summary=/tmp/log_directory/dllogger.json --output_dir=/tmp/results/checkpoints --use_fused_lamb \
      --input_dir=/data/pytorch/bert_pretraining/hdf5_lower_case_1_seq_len_128/books_wiki_en_corpus \
      --train_batch_size=8192 --max_seq_length=128 --max_predictions_per_seq=20 --max_steps=7038 \
      --warmup_proportion=0.2843 --num_steps_per_checkpoint=200 --learning_rate=0.006 --gradient_accumulation_steps=128 \
      --enable_packed_data_mode False
```

```bash
$PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased --hmp \
      --hmp_bf16=./ops_bf16_bert_pt.txt --hmp_fp32=./ops_fp32_bert_pt.txt --config_file=./bert_config.json \
      --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
      --json-summary=/tmp/log_directory/dllogger.json --output_dir=/tmp/results/checkpoints --use_fused_lamb \
      --input_dir=/data/pytorch/bert_pretraining/hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/books_wiki_en_corpus \
      --train_batch_size=4096 --max_seq_length=512 --max_predictions_per_seq=80 --max_steps=1563 \
      --warmup_proportion=0.128 --num_steps_per_checkpoint=200 --learning_rate=0.004\
      --gradient_accumulation_steps=512 --resume_from_checkpoint --phase1_end_step=7038 --phase2 \
      --enable_packed_data_mode False
```

ii. lazy mode, fp32 precision, BS32 for phase1 and BS4 for phase2:
```bash
$PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased --config_file=./bert_config.json \
      --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
      --json-summary=/tmp/log_directory/dllogger.json --output_dir=/tmp/results/checkpoints --use_fused_lamb \
      --input_dir=/data/pytorch/bert_pretraining/hdf5_lower_case_1_seq_len_128/books_wiki_en_corpus \
      --train_batch_size=512 --max_seq_length=128 --max_predictions_per_seq=20 --max_steps=7038 \
      --warmup_proportion=0.2843 --num_steps_per_checkpoint=200 --learning_rate=0.006 --gradient_accumulation_steps=32 \
      --enable_packed_data_mode False
```

```bash
$PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased --config_file=./bert_config.json \
      --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
      --json-summary=/tmp/log_directory/dllogger.json --output_dir=/tmp/results/checkpoints --use_fused_lamb \
      --input_dir=/data/pytorch/bert_pretraining/hdf5_lower_case_1_seq_len_512/books_wiki_en_corpus \
      --train_batch_size=128 --max_seq_length=512 --max_predictions_per_seq=80 --max_steps=1563 \
      --warmup_proportion=0.128 --num_steps_per_checkpoint=200 --learning_rate=0.004 \
      --gradient_accumulation_steps=64 --resume_from_checkpoint --phase1_end_step=7038 --phase2 \
      --enable_packed_data_mode False
```

iii. Using packed data: lazy mode, bf16 mixed precision, BS64 for phase1 and BS8 for phase2:
```bash
$PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased --hmp \
      --hmp_bf16=./ops_bf16_bert_pt.txt --hmp_fp32=./ops_fp32_bert_pt.txt --config_file=./bert_config.json \
      --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
      --json-summary=/tmp/log_directory/dllogger.json --output_dir=/tmp/results/checkpoints \
      --use_fused_lamb \
      --input_dir=/data/pytorch/bert_pretraining/packed_data/phase1/train_packed_new \
      --train_batch_size=8192 --max_seq_length=128 --max_predictions_per_seq=20 --max_steps=7038 \
      --warmup_proportion=0.2843 --num_steps_per_checkpoint=200 --learning_rate=0.006 --gradient_accumulation_steps=128
```

```bash
$PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased --hmp \
      --hmp_bf16=./ops_bf16_bert_pt.txt --hmp_fp32=./ops_fp32_bert_pt.txt --config_file=./bert_config.json \
      --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
      --json-summary=/tmp/log_directory/dllogger.json --output_dir=/tmp/results/checkpoints \
      --use_fused_lamb \
      --input_dir=/data/pytorch/bert_pretraining/packed_data/phase2/train_packed_new \
      --train_batch_size=4096 --max_seq_length=512 --max_predictions_per_seq=80 --max_steps=1563 \
      --warmup_proportion=0.128 --num_steps_per_checkpoint=200 --learning_rate=0.004 \
      --gradient_accumulation_steps=512 --resume_from_checkpoint --phase1_end_step=7038 --phase2
```

iv. Using packed data: lazy mode, bf16 mixed precision, BS64 for phase1 and BS16 for phase2 (on **Gaudi2**):
```bash
$PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased --hmp \
      --hmp_bf16=./ops_bf16_bert_pt.txt --hmp_fp32=./ops_fp32_bert_pt.txt --config_file=./bert_config.json \
      --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
      --json-summary=/tmp/log_directory/dllogger.json --output_dir=/tmp/results/checkpoints \
      --use_fused_lamb \
      --input_dir=/data/pytorch/bert_pretraining/packed_data/phase1/train_packed_new \
      --train_batch_size=8192 --max_seq_length=128 --max_predictions_per_seq=20 --max_steps=7038 \
      --warmup_proportion=0.2843 --num_steps_per_checkpoint=200 --learning_rate=0.006 --gradient_accumulation_steps=128
```

```bash
$PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased --hmp \
      --hmp_bf16=./ops_bf16_bert_pt.txt --hmp_fp32=./ops_fp32_bert_pt.txt --config_file=./bert_config.json \
      --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
      --json-summary=/tmp/log_directory/dllogger.json --output_dir=/tmp/results/checkpoints \
      --use_fused_lamb \
      --input_dir=/data/pytorch/bert_pretraining/packed_data/phase2/train_packed_new \
      --train_batch_size=8192 --max_seq_length=512 --max_predictions_per_seq=80 --max_steps=1563 \
      --warmup_proportion=0.128 --num_steps_per_checkpoint=200 --learning_rate=0.004 \
      --gradient_accumulation_steps=512 --resume_from_checkpoint --phase1_end_step=7038 --phase2
```
## Multicard Training
Follow the relevant steps under "Training the Model".
To run multi-card demo, make sure the host machine has 512 GB of RAM installed. Modify the docker run command to pass 8 Gaudi cards to the docker container. This ensures the docker has access to all the 8 cards required for multi-card demo.

Use the following commands to run multicard training on 8 cards:

**NOTE:** mpirun map-by PE attribute value may vary on your setup. Please refer to the instructions on [mpirun Configuration](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Scaling_Guide/DDP_Based_Scaling.html#mpirun-configuration) for calculation.

i. lazy mode, bf16 mixed precision, per chip batch size of 64 for phase1 and 8 for phase2:
```bash
export MASTER_ADDR="localhost"
export MASTER_PORT="12345"
mpirun -n 8 --bind-to core --map-by socket:PE=7 --rank-by core --report-bindings --allow-run-as-root \
    $PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased --hmp \
        --hmp_bf16=./ops_bf16_bert_pt.txt --hmp_fp32=./ops_fp32_bert_pt.txt --use_lazy_mode=True \
        --config_file=./bert_config.json --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
        --json-summary=/tmp/log_directory/dllogger.json --output_dir=/tmp/BERT_PRETRAINING/results/checkpoints --use_fused_lamb \
        --input_dir=/data/pytorch/bert_pretraining/hdf5_lower_case_1_seq_len_128/books_wiki_en_corpus \
        --train_batch_size=8192 --max_seq_length=128 --max_predictions_per_seq=20 --warmup_proportion=0.2843 \
        --max_steps=7038 --num_steps_per_checkpoint=200 --learning_rate=0.006 --gradient_accumulation_steps=128 \
        --enable_packed_data_mode False
```

```bash
export MASTER_ADDR="localhost"
export MASTER_PORT="12345"
mpirun -n 8 --bind-to core --map-by socket:PE=7 --rank-by core --report-bindings --allow-run-as-root \
    $PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased --hmp \
        --hmp_bf16=./ops_bf16_bert_pt.txt --hmp_fp32=./ops_fp32_bert_pt.txt --use_lazy_mode=True \
        --config_file=./bert_config.json --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
        --json-summary=/tmp/log_directory/dllogger.json --output_dir=/tmp/BERT_PRETRAINING/results/checkpoints --use_fused_lamb \
        --input_dir=/data/pytorch/bert_pretraining/hdf5_lower_case_1_seq_len_512/books_wiki_en_corpus \
        --train_batch_size=4096 --max_seq_length=512 --max_predictions_per_seq=80 --warmup_proportion=0.128 \
        --max_steps=5 --num_steps_per_checkpoint=200 --learning_rate=0.004 --gradient_accumulation_steps=512 --resume_from_checkpoint --phase1_end_step=7038 --phase2 \
        --enable_packed_data_mode False
```

ii. lazy mode, fp32 precision, per chip batch size of 32 for phase1 and 4 for phase2:
```bash
export MASTER_ADDR="localhost"
export MASTER_PORT="12345"
mpirun -n 8 --bind-to core --map-by socket:PE=7 --rank-by core --report-bindings --allow-run-as-root \
    $PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased --config_file=./bert_config.json \
        --use_habana --allreduce_post_accumulation  --allreduce_post_accumulation_fp16 \
        --json-summary=/tmp/log_directory/dllogger.json --output_dir=/tmp/results/checkpoints \
        --use_fused_lamb --input_dir=/data/pytorch/bert_pretraining/hdf5_lower_case_1_seq_len_128/books_wiki_en_corpus \
        --train_batch_size=8192 --max_seq_length=128 --max_predictions_per_seq=20 --max_steps=3 --warmup_proportion=0.2843 \
        --num_steps_per_checkpoint=200 --learning_rate=0.006 --gradient_accumulation_steps=256 \
        --enable_packed_data_mode False
```

```bash
export MASTER_ADDR="localhost"
export MASTER_PORT="12345"
mpirun -n 8 --bind-to core --map-by socket:PE=7 --rank-by core --report-bindings --allow-run-as-root
    $PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased --config_file=./bert_config.json \
      --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 --json-summary=/tmp/log_directory/dllogger.json \
      --output_dir=/tmp/results/checkpoints --use_fused_lamb \
      --input_dir=/data/pytorch/bert_pretraining/hdf5_lower_case_1_seq_len_512/books_wiki_en_corpus \
      --train_batch_size=4096 --max_seq_length=512 --max_predictions_per_seq=80 --max_steps=1563 --warmup_proportion=0.128 \
      --num_steps_per_checkpoint=200 --learning_rate=0.004 --gradient_accumulation_steps=512 \
      --resume_from_checkpoint --phase1_end_step=7038 --phase2 \
      --enable_packed_data_mode False
```

iii. Using packed data: lazy mode, bf16 mixed precision, per chip batch size of 64 for phase1 and 8 for phase2:
```bash
export MASTER_ADDR="localhost"
export MASTER_PORT="12345"
mpirun -n 8 --bind-to core --map-by socket:PE=7 --rank-by core --report-bindings --allow-run-as-root \
    $PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased --hmp --hmp_bf16=./ops_bf16_bert_pt.txt \
      --hmp_fp32=./ops_fp32_bert_pt.txt --config_file=./bert_config.json --use_habana \
      --allreduce_post_accumulation --allreduce_post_accumulation_fp16 --json-summary=/tmp/log_directory/dllogger.json \
      --output_dir=/tmp/results/checkpoints --use_fused_lamb \
      --input_dir=/data/pytorch/bert_pretraining/packed_data/phase1/train_packed_new \
      --train_batch_size=8192 --max_seq_length=128 --max_predictions_per_seq=20 --max_steps=7038 \
      --warmup_proportion=0.2843 --num_steps_per_checkpoint=200 --learning_rate=0.006 --gradient_accumulation_steps=128
```

```bash
export MASTER_ADDR="localhost"
export MASTER_PORT="12345"
mpirun -n 8 --bind-to core --map-by socket:PE=7 --rank-by core --report-bindings --allow-run-as-root \
    $PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased --hmp --hmp_bf16=./ops_bf16_bert_pt.txt \
      --hmp_fp32=./ops_fp32_bert_pt.txt --config_file=./bert_config.json --use_habana \
      --allreduce_post_accumulation --allreduce_post_accumulation_fp16 --json-summary=/tmp/log_directory/dllogger.json \
      --output_dir=/tmp/results/checkpoints --use_fused_lamb \
      --input_dir=/data/pytorch/bert_pretraining/packed_data/phase2/train_packed_new \
      --train_batch_size=4096 --max_seq_length=512 --max_predictions_per_seq=80 --max_steps=1563 \
      --warmup_proportion=0.128 --num_steps_per_checkpoint=200 --learning_rate=0.004 \
      --gradient_accumulation_steps=512 --resume_from_checkpoint --phase1_end_step=7038 --phase2
```

iv. Using packed data: lazy mode, bf16 mixed precision, per chip batch size of 64 for phase1 and 16 for phase2 (on **Gaudi2**):
```bash
export MASTER_ADDR="localhost"
export MASTER_PORT="12345"
mpirun -n 8 --bind-to core --map-by socket:PE=7 --rank-by core --report-bindings --allow-run-as-root \
    $PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased --hmp --hmp_bf16=./ops_bf16_bert_pt.txt \
      --hmp_fp32=./ops_fp32_bert_pt.txt --config_file=./bert_config.json --use_habana \
      --allreduce_post_accumulation --allreduce_post_accumulation_fp16 --json-summary=/tmp/log_directory/dllogger.json \
      --output_dir=/tmp/results/checkpoints --use_fused_lamb \
      --input_dir=/data/pytorch/bert_pretraining/packed_data/phase1/train_packed_new \
      --train_batch_size=8192 --max_seq_length=128 --max_predictions_per_seq=20 --max_steps=7038 \
      --warmup_proportion=0.2843 --num_steps_per_checkpoint=200 --learning_rate=0.006 --gradient_accumulation_steps=128
```

```bash
export MASTER_ADDR="localhost"
export MASTER_PORT="12345"
mpirun -n 8 --bind-to core --map-by socket:PE=7 --rank-by core --report-bindings --allow-run-as-root \
    $PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased --hmp --hmp_bf16=./ops_bf16_bert_pt.txt \
      --hmp_fp32=./ops_fp32_bert_pt.txt --config_file=./bert_config.json --use_habana \
      --allreduce_post_accumulation --allreduce_post_accumulation_fp16 --json-summary=/tmp/log_directory/dllogger.json \
      --output_dir=/tmp/results/checkpoints --use_fused_lamb \
      --input_dir=/data/pytorch/bert_pretraining/packed_data/phase2/train_packed_new \
      --train_batch_size=8192 --max_seq_length=512 --max_predictions_per_seq=80 --max_steps=1563 \
      --warmup_proportion=0.128 --num_steps_per_checkpoint=200 --learning_rate=0.004 \
      --gradient_accumulation_steps=512 --resume_from_checkpoint --phase1_end_step=7038 --phase2
```
## Multinode Training
To run multi-node demo, make sure the host machine has 512 GB of RAM installed.
Also ensure you followed the [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html)
to install and set up docker, so that the docker has access to all the 8 cards
required for multi-node demo. Multinode configuration for BERT PT training upto
4 servers, each with 8 Gaudi cards, have been verified.

Before execution of the multi-node scripts, make sure all network interfaces are up. You can change the state of each network interface managed by the habanalabs driver using the following command:
```
sudo ip link set <interface_name> up
```
To identify if a specific network interface is managed by the habanalabs driver type, run:
```
sudo ethtool -i <interface_name>
```
#### Docker ssh port setup for multi-server training

By default, the Habana docker uses `port 22` for ssh. The default port configured in the script is `port 3022`. Run the following commands to configure the selected port number , `port 3022` in example below.

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

**NOTE:** mpirun map-by PE attribute value may vary on your setup. Please refer to the instructions on [mpirun Configuration](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Scaling_Guide/DDP_Based_Scaling.html#mpirun-configuration) for calculation.

i. BERT Pre-training, lazy mode, bf16 mixed precision, BS64 for phase1 and BS8 for phase2:
```bash
export MASTER_ADDR="10.10.100.101"
export MASTER_PORT="12345"
mpirun --allow-run-as-root --mca plm_rsh_args -p3022 --bind-to core -n 32 --map-by ppr:4:socket:PE=7 \
--rank-by core --report-bindings --prefix --mca btl_tcp_if_include 10.10.100.101/16 \
$MPI_ROOT -H 10.10.100.101:16,10.10.100.102:16,10.10.100.103:16,10.10.100.104:16 \
        -x LD_LIBRARY_PATH -x HABANA_LOGS -x PYTHONPATH -x GC_KERNEL_PATH -x MASTER_ADDR -x MASTER_PORT -x https_proxy -x http_proxy \
$PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased --hmp \
        --hmp_bf16=./ops_bf16_bert_pt.txt --hmp_fp32=./ops_fp32_bert_pt.txt --config_file=./bert_config.json \
        --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
        --json-summary=/tmp/log_directory/dllogger.json --output_dir= /tmp/results/checkpoints \
        --use_fused_lamb --input_dir=/data/pytorch/bert_pretraining/hdf5_lower_case_1_seq_len_128/books_wiki_en_corpus \
        --train_batch_size=2048 --max_seq_length=128 --max_predictions_per_seq=20
        --max_steps=7038 --warmup_proportion=0.2843 \
        --num_steps_per_checkpoint=200 --learning_rate=0.006 --gradient_accumulation_steps=32 \
        --enable_packed_data_mode False
```

```bash
export MASTER_ADDR="10.10.100.101"
export MASTER_PORT="12345"
mpirun --allow-run-as-root --mca plm_rsh_args -p3022 --bind-to core -n 32 --map-by ppr:4:socket:PE=7 \
--rank-by core --report-bindings --prefix --mca btl_tcp_if_include 10.10.100.101/16 \
      $MPI_ROOT -H 10.10.100.101:16,10.10.100.102:16,10.10.100.103:16,10.10.100.104:16 -x LD_LIBRARY_PATH \
        -x HABANA_LOGS -x PYTHONPATH -x GC_KERNEL_PATH -x MASTER_ADDR -x MASTER_PORT \
      $PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased \
        --hmp --hmp_bf16=./ops_bf16_bert_pt.txt --hmp_fp32=./ops_fp32_bert_pt.txt \
        --config_file=./bert_config.json --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
        --json-summary=/tmp/log_directory/dllogger.json --output_dir= /tmp/results/checkpoints \
        --use_fused_lamb --input_dir=/data/pytorch/bert_pretraining/hdf5_lower_case_1_seq_len_512/books_wiki_en_corpus \
        --train_batch_size=1024 --max_seq_length=512 --max_predictions_per_seq=80 --max_steps=1563 \
        --warmup_proportion=0.128 --num_steps_per_checkpoint=200 --learning_rate=0.004 \
        --gradient_accumulation_steps=128 --resume_from_checkpoint --phase1_end_step=7038 --phase2 \
        --enable_packed_data_mode False
```

ii. Using packed data: lazy mode, bf16 mixed precision, per chip BS64 for phase1 and BS8 for phase2:
```bash
export MASTER_ADDR="10.10.100.101"
export MASTER_PORT="12345"
mpirun --allow-run-as-root --mca plm_rsh_args "-p 3022" --bind-to core -n 32 --map-by ppr:4:socket:PE=7 \
--rank-by core --report-bindings --prefix --mca btl_tcp_if_include 10.10.100.101/16
      $MPI_ROOT -H 10.10.100.101:16,10.10.100.102:16,10.10.100.103:16,10.10.100.104:16 -x LD_LIBRARY_PATH \
          -x HABANA_LOGS -x PYTHONPATH -x GC_KERNEL_PATH -x MASTER_ADDR \
          -x MASTER_PORT \
          $PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased --hmp \
          --hmp_bf16=./ops_bf16_bert_pt.txt --hmp_fp32 =./ops_fp32_bert_pt.txt --config_file=./bert_config.json \
          --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
          --json-summary=/tmp/log_directory/dllogger.json --output_dir=/tmp/results/checkpoints \
          --use_fused_lamb --input_dir=/data/pytorch/bert_pretraining/packed_data/
          phase1/train_packed_new \
          --train_batch_size=2048 --max_seq_length=128 --max_predictions_per_seq=20 --max_steps=7038 \
          --warmup_proportion=0.2843 --num_steps_per_checkpoint=200 --learning_rate=0.006 \
          --gradient_accumulation_steps=32
```

```bash
export MASTER_ADDR="10.10.100.101"
export MASTER_PORT="12345"
 mpirun --allow-run-as-root --mca plm_rsh_args "-p 3022" --bind-to core -n 32 --map-by ppr:4:socket:PE=7 \
 --rank-by core --report-bindings --prefix --mca btl_tcp_if_include 10.10.100.101/16 \
      $MPI_ROOT -H 10.10.100.101:16,10.10.100.102:16,10.10.100.103:16,10.10.100.104:16 -x LD_LIBRARY_PATH \
         -x HABANA_LOGS -x PYTHONPATH -x GC_KERNEL_PATH -x MASTER_ADDR \
         -x MASTER_PORT \
         $PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased --hmp \
         --hmp_bf16=./ops_bf16_bert_pt.txt --hmp_fp32 =./ops_fp32_bert_pt.txt --config_file=./bert_config.json \
         --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
         --json-summary=/tmp/log_directory/dllogger.json --output_dir=/tmp/results/checkpoints \
         --use_fused_lamb --input_dir=/data/pytorch/bert_pretraining/packed_data/
 phase2/train_packed_new \
         --train_batch_size=1024 --max_seq_length=512 --max_predictions_per_seq=80 --max_steps=1563 --warmup_proportion=0.128 \ --num_steps_per_checkpoint=200 --learning_rate=0.004 --gradient_accumulation_steps=128 \
         --resume_from_checkpoint --phase1_end_step=7038 --phase2
```


# Supported Configurations

| Device | SynapseAI Version | PyTorch Version |
|-----|-----|-----|
| Gaudi | 1.5.0 | 1.11.0 |
| **Gaudi2** | 1.5.0 | 1.11.0 |

# Changelog

## 1.5.0
1. Packed dataset mode is set as default execution mode
2. Deprecated the flags `enable_packed_data_mode` and `avg_seq_per_pack` and added support for automatic detection of those parameters based on dataset metadata file.
3. Changes related to Saving and Loading checkpoint were removed.
4. Removed changes related to padding index and flatten.
5. Fixed throughput calculation for packed dataset.
6. Demo scripts were removed and references to custom demo script were replaced by community entry points in README
7. Reduced the number of distributed barrier calls to once per gradient accumulation steps
8. Simplified the distributed Initialization.
9. Added support for training on **Gaudi2** supporting up to 8 cards

## 1.4.0
1. Lazy mode is set as default execution mode,for eager mode set `use-lazy-mode` as False
2. Pretraining with packed dataset is supported

s
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

    d. Added Habana BF16 Mixed precision support.

    e. Added python version of LAMB optimizer and will be used as default(from lamb.py).

    f. Support for distributed training on Habana device.

    g. Added changes to support Lazy mode with required mark_step().

    h. Added changes to calculate the performance per step and report through dllogger.

    i. Using conventional torch layernorm, linear and activation functions.

    j. Changes for dynamic loading of HCCL library.

2. To improve performance:

    a. Added support for Fused LAMB optimizer.

    b. Bucket size set to 230MB for better performance in distributed training.

    c. Added support to use distributed all_reduce instead of default Distributed Data Parallel.

    d. Added support for lowering print frequency of loss and associated this with log_freq.
