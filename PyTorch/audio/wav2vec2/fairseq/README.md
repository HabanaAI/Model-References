# Wav2vec 2.0

This folder contains scripts to train Wav2vec 2.0 model on Habana Gaudi device. For more information on training and inference of deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

For model performance data, refer to the [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

# Table of Contents
* [Model-References](../../../../README.md)
* [Model Overview](#model-overview)
* [Setup](#setup)
* [Training and Examples](#training_and_examples)
* [Supported Configurations](#supported-configurations)
* [Changelog](#changelog)
* [Known Issues](#known-issues)
  
## Model Overview
The base model used is from [Wav2vec 2.0](https://github.com/pytorch/fairseq). It is described in the paper [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations (Baevski et al., 2020)](https://arxiv.org/abs/2006.11477) by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed and Michael Auli.

According to the paper, learning powerful representations from speech audio alone followed by fine-tuning on transcribed speech can outperform the best semi-supervised methods while being conceptually simpler.

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
In the docker container, go to PyTorch Wav2vec 2.0 directory:
```bash
cd Model-References/PyTorch/audio/wav2vec2
```
Install fairseq and the required packages using pip:
```bash
cd fairseq
pip install --editable .
$PYTHON setup.py develop
pip install soundfile
```

### Setting up the Dataset
Follow the steps below to setup Wav2vec dataset:
1. Download the dataset from http://www.openslr.org/12.
2. Create the train-960 directory comprised of the untared train-clean-100, train-clean-360 train-other-500 (totaling 960 hours of speech).
3. Create the manifest file by running the following command:
```bash
$PYTHON wav2vec_manifest.py /path/to/dataset/train-960/ --dest /path/to/dataset/train-960/manifest --valid-percent 0.05
```
The `wav2vec_manifest.py` file can be fetched from `/path/to/wav2vec2/fairseq/examples/wav2vec`. 

The following is an example on the dataset layout:
```
100/
1001/
1006/
101/
1012/
...
manifest/
```

**Notes:**
* Please make sure the first line in `/path/to/dataset/train-960/manifest/train.tsv` and `/path/to/dataset/train-960/manifest/valid.tsv` 
  points to the correct directory as follows `/data/pytorch/wav2vec/data/LibriSpeech/train-960`. 
* It is assumed that the above Wav2vec dataset is available at path `/data/pytorch/wav2vec/data/LibriSpeech/train-960`

## Training and Examples

### Single Card Training 

**Run training on 1 HPU:**
- Run training on 1 HPU, Gradient accumulation=64, mixed precision (BF16):
```bash
PT_HPU_ENABLE_SYNAPSE_OUTPUT_PERMUTE=0 PT_HPU_ENABLE_WEIGHT_CPU_PERMUTE=0 fairseq-hydra-train task.data=/data/pytorch/wav2vec/data/LibriSpeech/train-960/manifest/ --config-dir examples/wav2vec/config/pretraining --config-name wav2vec2_base_librispeech_hpu
```

### Multi-Card Training

To run multi-card demo, the following is required: 
- The host machine has 512 GB of RAM installed. 
- Make sure to follow the [Gaudi Setup and Installation Guide](https://github.com/HabanaAI/Setup_and_Install) to install and set up the docker, 
  so that it has access to all 8 cards required for multi-card demo.
- Before executing the multi-card demo scripts, make sure all server network interfaces are up. You can change the state of each network interface managed by the habanalabs driver using the following command:
  ```
  sudo ip link set <interface_name> up
  ```
  To identify if a specific network interface is managed by the habanalabs driver type, run:
  ```
  sudo ethtool -i <interface_name>
  ```

**Run training on 8 HPUs:**

**Note:** The number of cards can be configured using `--world_size` option in the demo script as shown below. 

1. Modify the `wav2vec2_base_librispeech_hpu.yaml` under `/path/to/wav2vec2/fairseq/examples/wav2vec/config/pretraining/`. 

2. Set `distributed_world_size` to 8:
```
distributed_training:
  distributed_world_size: 8
```
3. Set `update_freq` to 8:
```
optimization:
  max_update: 400000
  lr: [0.0005]
  update_freq: [8]
```
4. Run the following command (Gaudi):
```bash
PT_HPU_ENABLE_SYNAPSE_OUTPUT_PERMUTE=0 PT_HPU_ENABLE_WEIGHT_CPU_PERMUTE=0 fairseq-hydra-train task.data=/data/pytorch/wav2vec/data/LibriSpeech/train-960/manifest/ --config-dir examples/wav2vec/config/pretraining --config-name wav2vec2_base_librispeech_hpu
```
5. Run the following command (Gaudi2):
```bash
PT_HPU_ENABLE_SYNAPSE_OUTPUT_PERMUTE=0 PT_HPU_ENABLE_WEIGHT_CPU_PERMUTE=0 PT_RECIPE_CACHE_PATH="./cache_dir/" fairseq-hydra-train common.log_interval=111 task.data=/data/pytorch/wav2vec/data/LibriSpeech/train-960/manifest/ --config-dir examples/wav2vec/config/pretraining --config-name wav2vec2_base_librispeech_hpu
```

## Supported Configurations

| Device | SynapseAI Version | PyTorch Version |
|--------|-------------------|-----------------|
| Gaudi  | 1.8.0             | 1.13.1          |
| Gaudi2 | 1.8.0             | 1.13.1          |

## Changelog
### v1.8.0
  - Marked copy to device(inputs) as async.
  - Added async allreduce for sample_size.
  - Removed host barrier in wav2vec.
  - Replaced isnonzero with where op to unblock the host.

### v1.5.0
* To improve performance:
  - Only fetch the log statistics to CPU when needed.
  - Replace broadcast+sum with equal algorithm to save memory in Quantizer module.
  - Create a customized version of cos_similarity via removing the broadcast operations.
  - Move negative indices generation to HPU.
  - Change the data type of randint to int16 to save the memory copyfrom host to device when generating negative indics.
  - Replace conv1d with equivalent conv2d.
  - Replace group norm with equivalent instance norm.

### Training Script Modifications
The following are the changes made to the training scripts:

* Added support for Habana devices: 
  - Defined certain environment variables Habana devices.
  - Added support to run training in lazy mode in addition to the eager mode.
  - mark_step() is performed to trigger execution.
  - Added support of bucketting, padding, and Precompute loss for HPU.
  - Added support to use HPU accelerator plugin, DDP plugin(for multi-HPU training) and mixed precision plugin.
  - Added support of `fairseq_hydra_train` for multi-node training.

## Known Issues
- Only the above configurations mentioned are supported and verified.
- Training on 1 HPU with FP32 data type has OOM issue.
