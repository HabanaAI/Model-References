# Deep Learning Recommendation Model for Personalization and Recommendation Systems for PyTorch

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

The DLRM demo included in this release are as follows:
- DLRM training for FP32 and BF16 mixed precision for [Medium configuration](#medium-configuration) in Eager mode.
- DLRM training for FP32 and BF16 mixed precision for [Medium configuration](#medium-configuration) in Lazy mode.

**Table of Contents**
* [Model overview](#model-overview)
* [Setup](#setup)
* [Training the Model](#training-the-model)
* [Known issues](#known-issues)

## Model overview
This repository provides a script to train the Deep Learning Recommendation Model(DLRM) for PyTorch on Habana Gaudi<sup>TM</sup>, and is based on Facebook's [DLRM repository](https://github.com/facebookresearch/dlrm). The implementation in this repository has been modified to support Habana Gaudi<sup>TM</sup> and to make use of custom extension operators. For the details on what was changed in the model, please see the [Training Script Modifications section](#training-script-modifications)

One input data consists of continuous and categorical features. Categorical features and continuous features are converted into a dense representation of the same length by the embedding tables and the bottom MLP, respectively. Then the dot interaction is performed on all pairs of the embedding vectors and the processed continuous features. The output of the dot interaction is concatenated with the original processed continuous features and provided to the top MLP. Finally, this is fed into the sigmoid function, giving a probability that a user will click on an ad.

### Medium Configuration
Each gaudi contains a single embedding table with 3000000 entries of feature size 64.
- Model parameters
    - Interaction between categorical & continuous features set to 'cat'
    - Feature size for categorical features set to 64
    - Bottom MLP size set to "1024-1024-1024-64"
    - Top MLP size set to "4096-4096-4096-4096-4096-4096-4096-1"
    - Embedding table size set to "3000000" (for 1x)
    - Number of indices per lookup set to 38

- Hyperparameters
    - Mini batch size: 512
    - Learning rate: 1e-5
    - Optimizer: Adagrad

## Setup
Please follow the instructions given in the following link for setting up the
environment including the `$PYTHON` environment variable: [Gaudi Setup and
Installation Guide](https://github.com/HabanaAI/Setup_and_Install). Please
answer the questions in the guide according to your preferences. This guide will
walk you through the process of setting up your system to run the model on
Gaudi.

Set the `MPI_ROOT` environment variable to the directory where OpenMPI is installed.

For example, in Habana containers, use

```bash
export MPI_ROOT=/usr/local/openmpi/
```

### Dataset setup

#### Random dataset for medium configuration
Input data for the medium configuration is obtained from the random input generator provided in the scripts.

The following settings are used:
- num-indices-per-lookup: 38
- mini-batch-size: 512
- data-size: 1024000

## Training the model

In the docker container, clone this repository and switch to the branch that
matches your SynapseAI version. (Run the
[`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.)

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```
Go to the PyTorch dlrm directory.
```bash
cd Model-References/PyTorch/recommendation/dlrm
```
Install python packages required for DLRM model
```bash
pip install -r Model-References/PyTorch/recommendation/dlrm/requirements.txt
```
Set up the dataset as mentioned in the section [Dataset setup](#dataset-setup).

i. Run single-card training using Random data in lazy mode using Medium configuration in bf16 mixed precision
```
$PYTHON demo_dlrm.py  --world_size 1 --print-time --print-freq 500 --mode lazy --arch-interaction-op cat --arch-sparse-feature-size 64 --arch-embedding-size 3000000 --arch-mlp-bot 1024-1024-1024-64 --arch-mlp-top 4096-4096-4096-4096-4096-4096-4096-1 --mini-batch-size 512 --learning-rate 1e-05 --num-batches 10000 --num-indices-per-lookup 38 --optimizer adagrad --data_type bf16
```
```
$PYTHON dlrm_s_pytorch_hpu_custom.py --arch-interaction-op=cat --arch-sparse-feature-size=64 --arch-mlp-bot=1024-1024-1024-64 --arch-mlp-top=4096-4096-4096-4096-4096-4096-4096-1 --arch-embedding-size=3000000 --num-indices-per-lookup=38 --mini-batch-size=512 --learning-rate=1e-05 --num-batches=10000 --numpy-rand-seed=123 --print-precision=5 --data-size=8 --activation-function=relu --loss-function=bce --loss-threshold=0.0 --data-generation=random --data-trace-file=./input/dist_emb_j.log --data-set=kaggle --nepochs=1 --processed-data-file= --data-randomize=total --max-ind-range=-1 --data-sub-sample-rate=0.0 --print-freq=500 --optimizer=adagrad --raw-data-file= --run-lazy-mode --print-time --hmp --hmp-bf16 ./ops_bf16_dlrm.txt --hmp-fp32 ./ops_fp32_dlrm.txt
```

ii. Run single-card training using Random data in eager mode using Medium configuration in fp32
```
$PYTHON demo_dlrm.py  --world_size 1 --print-time --print-freq 500 --arch-interaction-op cat --arch-sparse-feature-size 64 --arch-embedding-size 3000000 --arch-mlp-bot 1024-1024-1024-64 --arch-mlp-top 4096-4096-4096-4096-4096-4096-4096-1 --mini-batch-size 512 --learning-rate 1e-05 --num-batches 10000 --num-indices-per-lookup 38 --optimizer adagrad
```

## Known issues
- Multinode configurations are not supported for DLRM.
- Placing mark_step() arbitrarily may lead to undefined behaviour. Recommend to keep mark_step() as shown in provided scripts.
- Only scripts & configurations mentioned in this README are supported and verified.


## Training Script Modifications
The following changes were added to support Habana devices:

1. Support for Habana device was added

   a. Added support to run DLRM in lazy mode in addition to the eager mode.
   mark_step() is performed to trigger execution of the graph.

   b. Changes for dynamic loading of HCL library.

   c. Certain environment variables are defined for habana device.

2. Added new file, dlrm_habana_kernels.py, that encapsulates some operators and utilities for Habana specific adaptations/optimizations

    a. Inside model, nn.EmbeddingBag was replaced by a custom operator for using Habana specific implementation of embedding bag kernel. Also, inputs to embedding bag are pre-processed on the host; this step has also been added as a custom operator which runs on the CPU.

    b. Embedding weights are updated using a custom operator (running on Habana device) which uses some outputs of the pre-processing step described above.

    c. Maximum size tensors are created wherever variable sized input and intermediate tensors can change from iteration to iteration. Actual tensors from the iteration are copied in-place into these tensors.
3. Added support for Adagrad optimizer in scripts.
4. Added modifications to discard partial batches.
5. Control of few Habana lazy mode optimization passes in bridge.
6. Added flag to allow re-running multiple iterations of training with previously
    loaded data to allow measurement of iteration time excluding dataloader times.
7. To improve performance:

    a. In case print-frequency is larger than 1, then accuracy and loss are pulled out of device at the iteration where printing is actually performed.

    b. Enabled pinned memory for data loader using flag pin_memory.

    c. When SGD optimizer is used, torch.optim.SGD is used only for updating learnable parameters of bottom and top MLP.

    d. When Adagrad optimizer is used, fusedAdagrad operator is used for updating bottom and top MLP weights.


8. Enhanced training scripts with multi-node support with following (Multinode is not functional in this release):

    a. Embedding table is split across cards.

    b. Training script modified to make use of all to all.

    c. Script argument distributed added to trigger distributed training.

    d. Split input batches based on number of cards (world_size) used.

    e. Use HCL backend.

    f. Added support for DDP.

    g. Printing of loss.

