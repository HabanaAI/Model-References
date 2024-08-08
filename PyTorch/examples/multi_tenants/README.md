# Multi-Tenants Example for PyTorch

The Multi-Tenants example provides a script to demonstrate how to run multiple training jobs in parallel for PyTorch. The training jobs mentioned below utilize running multiple workloads while using partial Intel® Gaudi® AI accelerators on the same node simultaneously. Using this feature requires some additional configuration to make the workload run properly with part of the Gaudi processors and also avoid interference between the workloads which run in parallel. For further information, refer to [Multiple Tenants for PyTorch guide](https://docs.habana.ai/en/latest/PyTorch/PT_Multiple_Tenants_on_HPU/index.html). 

For further information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents

* [Model-References](../../../README.md)
* [Setup](#setup)
* [Training and Examples](#training-the-model)

## Setup

Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) to set up the
environment including the `$PYTHON` environment variable. The guide will walk you through the process of setting up your system to run the model on Gaudi.

### Create Docker Container and Set up Python

Please follow the instructions provided in [Run Using Containers on Habana Base AMI](https://docs.habana.ai/en/latest/AWS_User_Guides/Habana_Deep_Learning_AMI.html#run-using-containers-on-habana-base-ami) to pull the docker image and launch the container. And make sure to setup Python inside the docker container following [Model References Requirements](https://docs.habana.ai/en/latest/AWS_User_Guides/Habana_Deep_Learning_AMI.html#model-references-requirements).

### Clone Intel Gaudi Model-References

In the docker container, clone this repository and switch to the branch that matches your Intel Gaudi software version. You can run the [`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the Intel Gaudi software version.

```bash
git clone -b [Intel Gaudi software version] https://github.com/HabanaAI/Model-References /root/Model-References
```

**Note:** If Model-References repository path is not in the PYTHONPATH, make sure you update it:
```bash
export PYTHONPATH=$PYTHONPATH:/root/Model-References
```

### Install Model Requirements 

By default, the script installs the required packages. 

### Download the Dataset 

The Multi-tenants example script operates on ImageNet 1k, a widely popular image classification dataset from the ILSVRC challenge.
To obtain the dataset, perform the following:
1. Sign up with [ImageNet](http://image-net.org/download-images) and acquire the rights to download original images.
2. Follow the link to the 2012 ILSVRC to download `ILSVRC2012_img_val.tar` and `ILSVRC2012_img_train.tar`.
3. Use the below commands to prepare the dataset under `/data/pytorch/imagenet/ILSVRC2012`.
 This is the default `data_dir` for the training script.

```
export IMAGENET_HOME=/data/pytorch/imagenet/ILSVRC2012
# extract train data
mkdir -p $IMAGENET_HOME/train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
# extract validation data
mkdir -p $IMAGENET_HOME/val && mv ILSVRC2012_img_val.tar val/ && cd val
tar -xvf ILSVRC2012_img_val.tar && rm -f ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```


## Training Multi-Tenants Example 

You can run multiple jobs in parallel using the script described in the following sections. 

**NOTE:** This example is based on ResNet50.  

### multi_tenants_resnet_pt.sh

#### Run 2 ResNet50 Jobs on Total 8 HPUs with torch.compile Enabled

Running the script without setting any arguments invokes 2 ResNet50 jobs in parallel, each using 4 Gaudis. 

```bash
bash multi_tenants_resnet_pt.sh
```

#### Run 2 ResNet50 Jobs on Total 4 HPUs with torch.compile Enabled

User can also provide two sets of module IDs as the script arguments. The following command invokes 2 jobs in parallel, each using 2 Gaudis. 

```bash
bash multi_tenants_resnet_pt.sh "0,1" "2,3"
```

#### `HABANA_VISIBLE_MODULES`

Using the command `hl-smi -Q index,module_id -f csv` will produce a .csv file which will show the corresponding to the AIP number mapped to module_id. This can be used to find which module IDs are available for parallel training. The `HABANA_VISIBLE_MODULES` environment variable and model python script arguments need to be explicitly specified as different values for both jobs.

`HABANA_VISIBLE_MODULES` is an environment variable for the list of module IDs, composed by a sequence of single digit integers. The same integer should not be used by multiple jobs running in parallel: 
For jobs with 4 Gaudis, it is recommended to set this to "0,1,2,3" or "4,5,6,7".
For jobs with 2 Gaudis, it is recommended to set this to "0,1", "2,3", "4,5", or "6,7".

## Changelog
### 1.16.2
 - Added torch.compile support to improve training performance.
 - Lazy mode support is deprecated for this example.
