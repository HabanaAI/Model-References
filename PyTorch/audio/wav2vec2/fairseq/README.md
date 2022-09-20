# Wav2vec 2.0
For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

This folder contains scripts to train Wav2vec 2.0 model on Habana Gaudi device. Please visit [this page](https://developer.habana.ai/resources/habana-training-models/#performance) for performance information.
# Table of Contents
- [Model Overview](#model-overview)
- [Setup](#setup)
  - [Install the requirements](#install-the-requirements)
  - [Setting up the dataset](#setting-up-the-dataset)
- [Training the Model](#training-the-model)
- [Changes](#changes)
- [Known Issues](#known-issues)

## Model Overview
The base model used is from [wav2vec 2.0](https://github.com/pytorch/fairseq). It is described in the paper [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations (Baevski et al., 2020)](https://arxiv.org/abs/2006.11477) by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli.

This paper shows that learning powerful representations from speech audio alone followed by fine-tuning on transcribed speech can outperform the best semi-supervised methods while being conceptually simpler.

## Setup
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

Go to PyTorch Wav2vec 2.0 directory:
```bash
cd /path/to/wav2vec2
```
Note: If the repository is not in the PYTHONPATH, make sure you update it.
```bash
export PYTHONPATH=/path/to/Model-References:$PYTHONPATH
```

### Install the required packages
To install fairseq and the requirement package:
```bash
cd fairseq
pip install --editable .
python setup.py develop
pip install soundfile
```

### Setting up the dataset
Follow the steps below to setup Wav2vec dataset
1. Download the dataset from http://www.openslr.org/12
2. Create the train-960 directory comprised of the untared train-clean-100, train-clean-360 train-other-500 ( totaling 960 hours of speech)
3. Run the following command to create the manifest file:
```bash
python wav2vec_manifest.py /path/to/dataset/train-960/ --dest /path/to/dataset/train-960/manifest --valid-percent 0.05
```
The “wav2vec_manifest.py” file can get from /path/to/wav2vec2/fairseq/examples/wav2vec

An example layout of the dataset will look like below:
```
100/
1001/
1006/
101/
1012/
...
manifest/
```

Note:
1. Please make sure the first line in /path/to/dataset/train-960/manifest/train.tsv and /path/to/dataset/train-960/manifest/valid.tsv points to the correct directory. e.g. `/software/data/pytorch/wav2vec/data/LibriSpeech/train-960`
2. Going forward we assume the above Wav2vec dataset is available at path `/software/data/pytorch/wav2vec/data/LibriSpeech/train-960`

## Training the Model

### Single HPU Training

Train on 1 HPU, Gradient accumulation=64 , mixed precision (BF16) :
```bash
PT_HPU_ENABLE_SYNAPSE_OUTPUT_PERMUTE=0 PT_HPU_ENABLE_WEIGHT_CPU_PERMUTE=0 fairseq-hydra-train task.data=/software/data/pytorch/wav2vec/data/LibriSpeech/train-960/manifest/ --config-dir examples/wav2vec/config/pretraining --config-name wav2vec2_base_librispeech_hpu
```

### Multi-HPU Training

To run multi-HPU demo, make sure the host machine has 512 GB of RAM installed.
Also ensure you followed the [Gaudi Setup and
Installation Guide](https://github.com/HabanaAI/Setup_and_Install) to install and set up docker,
so that the docker has access to all the 8 cards required for multi-HPU demo.

Before execution of the multi-HPU demo scripts, make sure all server network interfaces are up. You can change the state of each network interface managed by the habanalabs driver using the following command:
```
sudo ip link set <interface_name> up
```
To identify if a specific network interface is managed by the habanalabs driver type, run:
```
sudo ethtool -i <interface_name>
```

Finally to train the wav2vec on multiple HPUs with below script.

#### Train on 8 HPUs, Gradient accumulation = 8, mixed precision (BF16)

Modify the wav2vec2_base_librispeech_hpu.yaml under /path/to/wav2vec2/fairseq/examples/wav2vec/config/pretraining/:

Set distributed_world_size to 8
```
distributed_training:
  distributed_world_size: 8
```
Set update_freq to 8
```
optimization:
  max_update: 400000
  lr: [0.0005]
  update_freq: [8]
```
Then run the following commmand:
```bash
PT_HPU_ENABLE_SYNAPSE_OUTPUT_PERMUTE=0 PT_HPU_ENABLE_WEIGHT_CPU_PERMUTE=0 fairseq-hydra-train task.data=/software/data/pytorch/wav2vec/data/LibriSpeech/train-960/manifest/ --config-dir examples/wav2vec/config/pretraining --config-name wav2vec2_base_librispeech_hpu
```

#### Train on 16 HPUs (2 HLS), Gradient accumulation = 4, mixed precision (BF16)

Assume server 1 works as master server with IP address of "10.3.124.151". Server 2 works as slave server and can communicate with server 1 successfully.

Modify the wav2vec2_base_librispeech_hpu.yaml under /path/to/wav2vec2/fairseq/examples/wav2vec/config/pretraining/:

Set distributed_world_size to 8, i.e. 8 HPUs on one node.
```
distributed_training:
  distributed_world_size: 8
```
Set update_freq to 4. Make sure distributed_world_size * update_freq * node_num = 64
```
optimization:
  max_update: 400000
  lr: [0.0005]
  update_freq: [4]
```
Then run below commmand on server 1:
```bash
PT_HPU_ENABLE_SYNAPSE_OUTPUT_PERMUTE=0 PT_HPU_ENABLE_WEIGHT_CPU_PERMUTE=0 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="10.3.124.151" --master_port=12345 $(which fairseq-hydra-train) task.data=/software/data/pytorch/wav2vec/data/LibriSpeech/train-960/manifest/ --config-dir examples/wav2vec/config/pretraining --config-name wav2vec2_base_librispeech_hpu
```
And run below commmand on server 2 at same time:
```bash
PT_HPU_ENABLE_SYNAPSE_OUTPUT_PERMUTE=0 PT_HPU_ENABLE_WEIGHT_CPU_PERMUTE=0 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr="10.3.124.151" --master_port=12345 $(which fairseq-hydra-train) task.data=/software/data/pytorch/wav2vec/data/LibriSpeech/train-960/manifest/ --config-dir examples/wav2vec/config/pretraining --config-name wav2vec2_base_librispeech_hpu
```
Note: The console log is only printed out on master server.

#### Train on 32 HPUs (4 HLS), Gradient accumulation = 2, mixed precision (BF16)

Assume server 1 works as master server with IP address of "10.3.124.151". Server 2/3/4 work as slave servers and can communicate with server 1 successfully.

Modify the wav2vec2_base_librispeech_hpu.yaml under /path/to/wav2vec2/fairseq/examples/wav2vec/config/pretraining/:

Set distributed_world_size to 8, i.e. 8 HPUs on one node.
```
distributed_training:
  distributed_world_size: 8
```
Set update_freq to 2. Make sure distributed_world_size * update_freq * node_num = 64
```
optimization:
  max_update: 400000
  lr: [0.0005]
  update_freq: [2]
```
Then run below commmand on server 1:
```bash
PT_HPU_ENABLE_SYNAPSE_OUTPUT_PERMUTE=0 PT_HPU_ENABLE_WEIGHT_CPU_PERMUTE=0 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=4 --node_rank=0 --master_addr="10.3.124.151" --master_port=12345 $(which fairseq-hydra-train) task.data=/software/data/pytorch/wav2vec/data/LibriSpeech/train-960/manifest/ --config-dir examples/wav2vec/config/pretraining --config-name wav2vec2_base_librispeech_hpu
```
And run below commmand on server 2 at same time:
```bash
PT_HPU_ENABLE_SYNAPSE_OUTPUT_PERMUTE=0 PT_HPU_ENABLE_WEIGHT_CPU_PERMUTE=0 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=4 --node_rank=1 --master_addr="10.3.124.151" --master_port=12345 $(which fairseq-hydra-train) task.data=/software/data/pytorch/wav2vec/data/LibriSpeech/train-960/manifest/ --config-dir examples/wav2vec/config/pretraining --config-name wav2vec2_base_librispeech_hpu
```
And run below commmand on server 3 at same time:
```bash
PT_HPU_ENABLE_SYNAPSE_OUTPUT_PERMUTE=0 PT_HPU_ENABLE_WEIGHT_CPU_PERMUTE=0 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=4 --node_rank=2 --master_addr="10.3.124.151" --master_port=12345 $(which fairseq-hydra-train) task.data=/software/data/pytorch/wav2vec/data/LibriSpeech/train-960/manifest/ --config-dir examples/wav2vec/config/pretraining --config-name wav2vec2_base_librispeech_hpu
```
And run below commmand on server 4 at same time:
```bash
PT_HPU_ENABLE_SYNAPSE_OUTPUT_PERMUTE=0 PT_HPU_ENABLE_WEIGHT_CPU_PERMUTE=0 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=4 --node_rank=3 --master_addr="10.3.124.151" --master_port=12345 $(which fairseq-hydra-train) task.data=/software/data/pytorch/wav2vec/data/LibriSpeech/train-960/manifest/ --config-dir examples/wav2vec/config/pretraining --config-name wav2vec2_base_librispeech_hpu
```
Note: The console log is only printed out on master server.

## Supported Configurations

| Device | SynapseAI Version | PyTorch Version |
|-----|-----|-----|
| Gaudi | 1.6.1 | 1.12.0 |

## Changelog
### v1.5.0
The following are the changes made to the training scripts:

1. Added support for Habana devices

   a. Certain environment variables are defined for habana device.

   b. Added support to run training in lazy mode in addition to the eager mode.

   c. mark_step() is performed to trigger execution.

   d. Added support of bucketting, padding, and Precompute loss for HPU

   e. Added support to use HPU accelerator plugin, DDP plugin(for multi-HPU training) & mixed precision plugin.

   f. Added support of fairseq_hydra_train for multi-node training.

2. To improve performance

   a. Only fetch the log statistics to CPU when needed.

   b. Replace broadcast+sum with equal algorithm to save memory in Quantizer module

   c. Create a customized version of cos_similarity via removing the broadcast operations

   d. Move negative indices generation to HPU

   e. Change the data type of randint to int16 to save the memory copyfrom host to device when generating negitve indics

   f. Replace conv1d with equivalent conv2d

   g. Replace group norm with equivalent instance norm.

## Known Issues
- Only above configurations mentioned are supported and verified.
- Training on single HPU with FP32 data type has OOM issue.
