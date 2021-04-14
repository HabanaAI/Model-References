# Deep Learning Recommendation Model for Personalization and Recommendation Systems for PyTorch

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

The DLRM demo included in this release are as follows:
- DLRM training for FP32 and BF16 mixed precision for Medium configuration in Eager mode.
- DLRM training for FP32 and BF16 mixed precision for Medium configuration in Lazy mode.

**Table of Contents**
* [Model overview](#model-overview)
* [Setup](#setup)
* [Training the Model](#training-the-model)
* [Known issues](#known-issues)

## Model overview
This repository provides a script to train the Deep Learning Recommendation Model(DLRM) for PyTorch on Habana Gaudi<sup>TM</sup>, and is based on Facebook's [DLRM repository](https://github.com/facebookresearch/dlrm). The implementation in this repository has been modified to support Habana Gaudi<sup>TM</sup> and to make use of custom extension operators. For the details on what was changed in the model, please see the [Training Script Modifications section](#training-script-modifications).

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
This model is tested with the Habana PyTorch docker container and some dependencies contained within it.

#### Requirements
- Docker version 19.03.12 or newer
- Sudo access to install required drivers/firmware

#### Install Drivers
Follow steps in the [Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) to install the driver.

<br />

### Install container runtime
<details>
<summary>Ubuntu distributions</summary>

#### Setup package fetching
1. Download and install the public key:  
```
curl -X GET https://vault.habana.ai/artifactory/api/gpg/key/public | sudo apt-key add -
```
2. Create an apt source file /etc/apt/sources.list.d/artifactory.list.
3. Add the following content to the file:
```
deb https://vault.habana.ai/artifactory/debian focal main
```
4. Update Debian cache:  
```
sudo dpkg --configure -a
sudo apt-get update
```  
#### Install habana-container-runtime:
Install the `habana-container-runtime` package:
```
sudo apt install -y habanalabs-container-runtime=0.14.0-420
```
#### Docker Engine setup

To register the `habana` runtime, use the method below that is best suited to your environment.
You might need to merge the new argument with your existing configuration.

##### Daemon configuration file
```bash
sudo tee /etc/docker/daemon.json <<EOF
{
    "runtimes": {
        "habana": {
            "path": "/usr/bin/habana-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF
sudo systemctl restart docker
```

You can optionally reconfigure the default runtime by adding the following to `/etc/docker/daemon.json`:
```
"default-runtime": "habana"
```
</details>

<details>
<summary>CentOS distributions</summary>

#### Setup package fetching
1. Create /etc/yum.repos.d/Habana-Vault.repo.
2. Add the following content to the file:
```
[vault]

name=Habana Vault

baseurl=https://vault.habana.ai/artifactory/centos7

enabled=1

gpgcheck=0

gpgkey=https://vault.habana.ai/artifactory/centos7/repodata/repomod.xml.key

repo_gpgcheck=0
```
3. Update YUM cache by running the following command:
```
sudo yum makecache
```
4. Verify correct binding by running the following command:
```
yum search habana
```
This will search for and list all packages with the word Habana.

#### Install habana-container-runtime:
Install the `habana-container-runtime` package:
```
sudo yum install habanalabs-container-runtime-0.14.0-420* -y
```
#### Docker Engine setup

To register the `habana` runtime, use the method below that is best suited to your environment.
You might need to merge the new argument with your existing configuration.

##### Daemon configuration file
```bash
sudo tee /etc/docker/daemon.json <<EOF
{
    "runtimes": {
        "habana": {
            "path": "/usr/bin/habana-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF
sudo systemctl restart docker
```

You can optionally reconfigure the default runtime by adding the following to `/etc/docker/daemon.json`:
```
"default-runtime": "habana"
```
</details>

<details>
<summary>Amazon linux distributions</summary>

#### Setup package fetching
1. Create /etc/yum.repos.d/Habana-Vault.repo.
2. Add the following content to the file:
```
[vault]

name=Habana Vault

baseurl=https://vault.habana.ai/artifactory/AmazonLinux2

enabled=1

gpgcheck=0

gpgkey=https://vault.habana.ai/artifactory/AmazonLinux2/repodata/repomod.xml.key

repo_gpgcheck=0
```
3. Update YUM cache by running the following command:
```
sudo yum makecache
```
4. Verify correct binding by running the following command:
```
yum search habana
```
This will search for and list all packages with the word Habana.

#### Install habana-container-runtime:
Install the `habana-container-runtime` package:
```
sudo yum install habanalabs-container-runtime-0.14.0-420* -y
```
#### Docker Engine setup

To register the `habana` runtime, use the method below that is best suited to your environment.
You might need to merge the new argument with your existing configuration.

##### Daemon configuration file
```bash
sudo tee /etc/docker/daemon.json <<EOF
{
    "runtimes": {
        "habana": {
            "path": "/usr/bin/habana-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF
sudo systemctl restart docker
```

You can optionally reconfigure the default runtime by adding the following to `/etc/docker/daemon.json`:
```
"default-runtime": "habana"
```
</details>
<br />

### Model Setup
1. Stop running dockers
    ```
    docker stop $(docker ps -a -q)
    ```

2. Download docker

    Choose the Habana PyTorch docker image for your Ubuntu version from the table below.

    | Ubuntu version | Habana PyTorch docker image |
    |:--------------|:-----------------------------------|
    | Ubuntu 18.04 | `vault.habana.ai/gaudi-docker/0.14.0/ubuntu18.04/habanalabs/pytorch-installer:0.14.0-420` |
    | Ubuntu 20.04 | `vault.habana.ai/gaudi-docker/0.14.0/ubuntu20.04/habanalabs/pytorch-installer:0.14.0-420` |

    Download the image using the following command.
    ```
    docker pull <your_choice_of_Habana_PyTorch_docker_image>
    ```

3. Run docker
    ```
    docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice -v /sys/kernel/debug:/sys/kernel/debug --net=host <your_choice_of_Habana_PyTorch_docker_image>
    ```

### Dataset setup

#### Random dataset for medium configuration
Input data for the medium configuration is obtained from the random input generator provided in the scripts.

The following settings are used:
- num-indices-per-lookup: 38
- mini-batch-size: 512
- data-size: 1024000

## Training the model
Clone the Model-References repository.
```
git clone https://github.com/HabanaAI/Model-References
cd Model-References/PyTorch/recommendation/dlrm
```
Add your Model-References repo path to PYTHONPATH.
```
export PYTHONPATH=<MODEL_REFERENCES_ROOT_PATH>:$PYTHONPATH
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

