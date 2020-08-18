# Deep Learning Recommendation Model for Personalization and Recommendation Systems for PyTorch

The DLRM demo included in this release are as follows:
- DLRM training for FP32 and BF16 mixed precision for Medium configuration in Eager mode.
- DLRM training for FP32 and BF16 mixed precision for Medium configuration in Lazy mode.

**Table of Contents**
* [Model overview](#model-overview)
* [Setup](#setup)
* [Training the Model](#training-the-model)
* [Training Results](#training-results)
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
### Docker Setup
This model is tested with the Habana PyTorch docker container 0.13.0-380 for Ubuntu 20.04 and some dependencies contained within it.

#### Requirements
- Docker version 19.03.12 or newer
- Sudo access to install required drivers/firmware

#### Install Drivers
Follow steps in the [Installation Guide](https://docs.habana.ai/projects/SynapeAI-Gaudi/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) to install the driver.

### Model Setup
1. Stop running dockers
```
docker stop $(docker ps -a -q)
```

2. Download docker
```
docker pull vault.habana.ai/gaudi-docker/0.13.0/ubuntu20.04/habanalabs/pytorch-installer:0.13.0-380
```

3. Run docker
```
docker run -td --device=/dev/hl_controlD0:/dev/hl_controlD0 --device=/dev/hl_controlD1:/dev/hl_controlD1 --device=/dev/hl_controlD2:/dev/hl_controlD2 --device=/dev/hl_controlD3:/dev/hl_controlD3 --device=/dev/hl_controlD4:/dev/hl_controlD4 --device=/dev/hl_controlD5:/dev/hl_controlD5 --device=/dev/hl_controlD6:/dev/hl_controlD6 --device=/dev/hl_controlD7:/dev/hl_controlD7 --device=/dev/hl0:/dev/hl0 --device=/dev/hl1:/dev/hl1 --device=/dev/hl2:/dev/hl2 --device=/dev/hl3:/dev/hl3 --device=/dev/hl4:/dev/hl4 --device=/dev/hl5:/dev/hl5 --device=/dev/hl6:/dev/hl6 --device=/dev/hl7:/dev/hl7 -e DISPLAY=$DISPLAY -e LOG_LEVEL_ALL=6 -v /sys/kernel/debug:/sys/kernel/debug -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v /tmp:/tmp --net=host --user user1 --workdir=/home/user1 vault.habana.ai/gaudi-docker/0.13.0/ubuntu20.04/habanalabs/pytorch-installer:0.13.0-380
```

4. Check name of your docker
```
docker ps
```
This command will print a list of docker containers. Get the container name.

5. Run bash in your docker
```
docker exec -ti <CONTAINER_NAME> bash
```

6. Clone the repository
```
git clone https://github.com/HabanaAI/Model-References
cd PyTorch/recommendation/dlrm/
```

### Dataset setup

#### Random dataset for medium configuration
Input data for the medium configuration is obtained from the random input generator provided in the scripts.

The following settings are used:
- num-indices-per-lookup: 38
- mini-batch-size: 512
- data-size: 1024000

## Training the model
Clone the Model-References repository
```
git clone https://github.com/HabanaAI/Model-References
cd Model-References/PyTorch/recommendation/dlrm
```
Set up the dataset as mentioned in the section [Dataset setup](#dataset-setup). Run `./demo_dlrm -h` for command-line options.

i. Run single-card training using Random data in eager mode using Medium configuration in bf16 mixed precision
```
./demo_dlrm --arch-interaction-op cat --arch-sparse-feature-size 64 \
    --arch-mlp-bot 1024-1024-1024-64 --arch-mlp-top 4096-4096-4096-4096-4096-4096-4096-1 \
    --arch-embedding-size 3000000 --num-indices-per-lookup 38 --mini-batch-size 512 \
    --learning-rate 1e-5 --num-batches 10000 -d bf16 --mode custom \
    --optimizer adagrad --print-time True --print-freq 50
```
ii. Run single-card training using Random data in eager mode using Medium configuration in fp32
```
./demo_dlrm --arch-interaction-op cat --arch-sparse-feature-size 64 \
    --arch-mlp-bot 1024-1024-1024-64 --arch-mlp-top 4096-4096-4096-4096-4096-4096-4096-1 \
    --arch-embedding-size 3000000 --num-indices-per-lookup 38 --mini-batch-size 512 \
    --learning-rate 1e-5 --num-batches 10000 -d fp32 --mode custom \
    --optimizer adagrad --print-time True  --print-freq 50
```
iii. Run single-card training using Random data in Lazy mode and Medium configuration in bf16 mixed precision
```
./demo_dlrm --arch-interaction-op cat --arch-sparse-feature-size 64 \
    --arch-mlp-bot 1024-1024-1024-64 --arch-mlp-top 4096-4096-4096-4096-4096-4096-4096-1 \
    --arch-embedding-size 3000000 --num-indices-per-lookup 38 --mini-batch-size 512 \
    --learning-rate 1e-5 --num-batches 10000  -d bf16 --mode lazy \
    --optimizer adagrad --print-time True --print-freq 50
```
iv. Run single-card training using Random data in Lazy mode and Medium configuration in fp32 
```
./demo_dlrm --arch-interaction-op cat --arch-sparse-feature-size 64 \
    --arch-mlp-bot 1024-1024-1024-64 --arch-mlp-top 4096-4096-4096-4096-4096-4096-4096-1 \
    --arch-embedding-size 3000000 --num-indices-per-lookup 38 --mini-batch-size 512 \
    --learning-rate 1e-5 --num-batches 10000 -d fp32 --mode lazy \
    --optimizer adagrad --print-time True  --print-freq 50
```

## Training Results
The following results were obtained for DLRM training in Lazy mode:

| Model       | Dataset   | # Gaudi cards, cfg, Precision|  Throughput (Queries/Sec)  |
|:------------|:-------------------|:------------------------:|:-------------------|
| DLRM  | Random  | 1-card, Medium, bf16   | 36957 qps |
| DLRM  | Random  | 1-card, Medium, fp32   | 20197 qps |

To report performance results for DLRM Lazy mode for Medium configuration, with bf16 mixed precision, the following command can be used: 
```
./demo_dlrm --arch-interaction-op cat --arch-sparse-feature-size 64 --arch-mlp-bot 1024-1024-1024-64 --arch-mlp-top 4096-4096-4096-4096-4096-4096-4096-1 --arch-embedding-size 3000000 --num-indices-per-lookup 38 --mini-batch-size 512 --learning-rate 1e-5 --num-batches 10000 -d bf16 --mode lazy --optimizer adagrad --print-time True --print-freq 500 --mlperf-logging --measure-perf
```
The above measurement does not include dataloading time. 

## Known issues
- Multinode configurations are not supported for DLRM.

## Training Script Modifications
The following changes were added to support Habana devices:

1. Support for Habana device was added.
2. Added new file, dlrm_habana_kernels.py, that encapsulates some operators
   and utilities for Habana specific adaptations/optimizations
    a. Inside model, nn.EmbeddingBag was replaced by a custom operator for
    using Habana specific implementation of embedding bag kernel. Also,
    inputs to embedding bag are pre-processed on the host; this step has
    also been added as a custom operator which runs on the CPU.
    b. Embedding weights are updated using a custom
    operator (running on Habana device) which uses some outputs of the
    pre-processing step described above. 
    c. Maximum size tensors are created wherever variable sized input and
    intermediate tensors can change from iteration to iteration. Actual
    tensors from the iteration are copied in-place into these tensors.
3. Added support to run DLRM in lazy mode in addition to the eager mode.
   mark_step() is performed to trigger execution of the graph.
4. Added support in scripts for Adagrad optimizer.
5. Aligned interface of the embedding bag op to the TPC kernel.
6. Added modifications to discard partial batches.
7. Enabled Pinned memory for data loader using flag pin_memory
8. torch.optim.SGD is used only for updating learnable parameters of
    bottom and top MLP. Fused adagrad operator is used
    for updating bottom and top MLP weights when adagrad operator is used.
9. Control of few Habana lazy mode optimization passes in bridge
10. Added flag to allow re-running multiple iterations of training with previously
    loaded data to allow measurement of iteration time excluding dataloader times
11. In case print-frequency is larger than 1, then accuracy and loss are pulled out
    of device at the iteration where printing is actually performed.
12. Enhanced training scripts with multi-node support with following (Multinode is not functional in this release):
    a. Embedding table is split across cards
    b. Training script modified to make use of all to all
    c. Script argument distributed added to trigger distributed training
    d. Split input batches based on number of cards (world_size) used
    e. Use HCL backend
    f. Added support for DDP
    g. Printing of loss

