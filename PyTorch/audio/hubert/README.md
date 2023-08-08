# HuBERT Pre-training Examples
This folder contains scripts to pre-train HuBERT model on Habana Gaudi device. To obtain model performance data, refer to the [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

For more information about training deep learning models using Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents
* [Model-References](../../../README.md)
* [Model Overview](#model-overview)
* [Setup](#setup)
* [Pre-training and Examples](#pre-training-and-examples)
* [Supported Configurations](#supported-configurations)
* [Script Modifications](#script-modifications)

## Model Overview
This directory contains sample implementations of pre-training pipeline based on [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](https://arxiv.org/abs/2106.07447). This version is based on PyTorch Audio version [v2.0.1](https://github.com/pytorch/audio/tree/v2.0.1).

## Setup
Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) 
to set up the environment including the `$PYTHON` environment variable. 
The guide will walk you through the process of setting up your system to run the model on Gaudi.


### Clone Habana Model-References
In the docker container, clone this repository and switch to the branch that matches your SynapseAI version. 
You can run the [`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the SynapseAI version.

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```

**Note:** If the repository is not in the PYTHONPATH, make sure to update by running the below.

```
export PYTHONPATH=/path/to/Model-References:$PYTHONPATH
```

### Install Model Requirements
1. In the docker container, go to the model directory:
```bash
cd Model-References/PyTorch/audio/hubert
```

2. Install the required packages using pip.
```bash
pip install -r requirements.txt
```

### Dataset Preparation
Download LibriSpeech dataset.
```
python dataset_itr1.py --download_dir=/scratch1/hubert/itr1_data --librispeech_split 960 --cleanup
```
**NOTE:**
* The above command generates the required data under /scratch1/hubert/itr1_data/960_extract.
* Rename /scratch1/hubert/itr1_data/960_extract to /data/pytorch/wav2vec/data/LibriSpeech/train-960/, that is used as root-dir in pre-processing example commands. 

###  Set up Environment for Autocast
The environment variables `LOWER_LIST` and `FP32_LIST` are set by default to enable Autocast on Gaudi. These environment variables are set with default paths when executing the script. This can be modified when providing custom lists. 

The following are the default Autocast Ops files:

- LOWER_LIST: ./ops_bf16_hubert.txt
- FP32_LIST: ./ops_fp32_hubert.txt

For further details, refer to [PyTorch Mixed Precision](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Mixed_Precision/PT_Mixed_Precision.html).

##  Pre-training and Examples

The Base architecture of HuBERT model requires two iterations of pre-training.
###  Pre-processing for 1st Iteration

[`preprocess.py`](./preprocess.py) generates the file list of training and validation data, trains a KMeans clustering model with either MFCC feature or the transformer layer's output from the pre-trained HuBERT model, then predict the cluster ID for each utterance as the label for masked prediction training.

Using MFCC feature to train KMeans model, see the following example:
```
python preprocess.py --root-dir /data/pytorch/wav2vec/data/LibriSpeech/train-960/ --feat-type mfcc --exp-dir /data/pytorch/hubert/exp --num-cluster 100
```

**NOTE:**
* Please make sure the first line in /data/pytorch/hubert/exp/data/mfcc/tsv/librispeech_train.tsv and /data/pytorch/hubert/exp/data/mfcc/tsv/librispeech_valid.tsv
 points to the correct directory as follows /data/pytorch/wav2vec/data/LibriSpeech/train-960/.
* It is assumed that the above LibriSpeech dataset is available at path /data/pytorch/wav2vec/data/LibriSpeech/train-960/.

### Pre-training for 1st Iteration

[`train.py`](./train.py) trains a HuBERTPretrainModel using PyTorch Lightning.

The first iteration is trained for ~250k steps on 8 cards, each rank has at most 175 seconds of audio in a mini-batch.

*NOTE:* OMP_NUM_THREADS value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Scaling_Guide/DDP_Based_Scaling.html#mpirun-configuration).

- Run on Gaudi2 with BF16 and on 8 cards:
```
LOWER_LIST=ops_bf16_hubert.txt FP32_LIST=ops_fp32_hubert.txt PT_HPU_USE_UNSORTED_SCATTER_ADD=1 OMP_NUM_THREADS=10 PT_RECIPE_CACHE_PATH="/tmp/iter1_recipe_cache/" PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES=0 python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 --rdzv_id=JOB_ID --rdzv_backend=c10d train.py --dataset-path /data/pytorch/hubert/exp/data/mfcc/ --exp-dir /root/scratch1/exp_iter1_cache --feature-type mfcc --num-class 100 --max-updates 245594 --learning-rate 0.0005 --hpus 8 --num-nodes 1 --num-buckets=10 --align-buckets="bucket1" --accumulate-grad-batches 1 --all-static --use-conv2d --use-instancenorm --use-fused-clip --optimizer fusedadamw --warmup-updates 31436 --autocast --seconds-per-batch 350 --use-max-sub-softmax-opt --recompilation-optimization 2>&1 | tee /root/scratch1/log_8x_hpu_bf16_iter1.txt
```

###  Pre-processing for 2nd Iteration
After the first iteration of pre-training, the intermediate transformer layer's output of the pre-trained HuBERTPretrainModel can be applied to train a new KMeans clustering model. Then the KMeans clustering model can be used to generate new clustering labels for the second iteration of masked prediction training.

- Using 6th transformer layer's output the input feature for training KMeans model, see the following example: 
```
python preprocess.py --root-dir /data/pytorch/wav2vec/data/LibriSpeech/train-960/ --feat-type hubert --exp-dir /data/pytorch/hubert/exp --layer-index 6 --num-rank 40 --checkpoint-path exp_iter1/checkpoints_librispeech_hubert_pretrain_base/epoch=220-step=241553.ckpt --num-cluster 500 --percent 0.1 2>&1 | tee log_preprocess_2_1.txt
```
**NOTE:**
* Please make sure the first line in /data/pytorch/hubert/exp/data/hubert_6/tsv/librispeech_train.tsv and /data/pytorch/hubert/exp/data/hubert_6/tsv/librispeech_valid.tsv points to the correct directory as follows /data/pytorch/wav2vec/data/LibriSpeech/train-960/.
* It is assumed that the above LibriSpeech dataset is available at path /data/pytorch/wav2vec/data/LibriSpeech/train-960/.

###  Pre-training for 2nd Iteration
The second iteration is trained for ~400k steps.

*NOTE:* OMP_NUM_THREADS value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Scaling_Guide/DDP_Based_Scaling.html#mpirun-configuration).

- Run on Gaudi2 with BF16 and on 8 cards:
```
LOWER_LIST=ops_bf16_hubert.txt FP32_LIST=ops_fp32_hubert.txt PT_HPU_USE_UNSORTED_SCATTER_ADD=1 OMP_NUM_THREADS=10 PT_RECIPE_CACHE_PATH="/tmp/recipe_cache_iter2/" PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES=0 python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 --rdzv_id=JOB_ID --rdzv_backend=c10d train.py --dataset-path /data/pytorch/hubert/exp/data/hubert_6/ --exp-dir /root/scratch1/exp_iter2_cache --feature-type hubert --num-class 500 --max-updates 395120 --learning-rate 0.0005 --hpus 8 --num-nodes 1 --num-buckets=10 --align-buckets="bucket1" --accumulate-grad-batches 2 --all-static --use-conv2d --use-instancenorm --use-fused-clip --optimizer fusedadamw --warmup-updates 31610 --seconds-per-batch 175 --autocast --use-max-sub-softmax-opt --recompilation-optimization --split-logits 2>&1 | tee /root/scratch1/log_8x_hpu_2nditer.txt
```

## Supported Configurations

**Training**

**Hubert Pretrain1, Pretrain2**

| Validated on | SynapseAI Version | PyTorch Lightning Version | Torch Audio Version | PyTorch Version | Mode |
|-----|-----|-----|---------|---------|---------|
| Gaudi2  | 1.10.0 | 2.0.0 | 2.0.1 | 2.0.1 | Training |

## Script Modifications
- Aligned input shape with bucket boundary to reduce input dynamicity. Adjusted train/warmup step based on dataloader change introduced by this bucket change.
- Removed boolean index dynamic in maskgenerator, encoder, and _compute_logits.
- Used torch.where to implement static layer drop.
- Made loss static by combining masked/unmasked accuracy calculation together.
- Used FusedAdam, FusedClipNorm.
- Used instancenorm to replace groupnorm, conv2d to replace conv1d, and checkpoint option with 1d/2d for feature extractor.
- Added log interval and reduce log frequency.
- Added the mixed precision, and autocast support.
- Zero-ed out gradients by setting them to None.
- Added conv2d to 1d parameter change for preprocess 2.
- Added max-sub-softmax optimization.
- Added recipe cache across ranks.
- Added distributed eval and increased batch sizes.
- Updated to support torch 2.0 and torch audio v2.0.1