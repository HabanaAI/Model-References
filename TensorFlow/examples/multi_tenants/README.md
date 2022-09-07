# Multi-Tenants Example for TensorFlow

The Multi-Tenants example provides a script to demonstrate how to run multiple training jobs in parallel. The training jobs mentioned below utilize running multiple workloads while using partial Gaudi processors on the same node simultaneously. Using this feature requires some additional configuration to make the workload run properly with part of the Gaudi processors and also avoid interference between the workloads which run in parallel. For further information, refer to [Multiple Tenants on HPU guide](https://docs.habana.ai/en/latest/Orchestration/Multiple_Tenants_on_HPU/index.html?highlight=tenants). 

For further information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents

* [Model-References](../../../README.md)
* [Setup](#setup)
* [Training and Examples](#training-the-model)

## Setup

Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) to set up the
environment including the `$PYTHON` environment variable. The guide will walk you through the process of setting up your system to run the model on Gaudi.

### Clone Habana Model-References

In the docker container, clone this repository and switch to the branch that matches your SynapseAI version. You can run the [`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the SynapseAI version.

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References /root/Model-References
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
3. Use the below commands to prepare the dataset under `/data/tensorflow/imagenet/tf_records`.
 This is the default `data_dir` for the training script.

```
export IMAGENET_HOME=/data/tensorflow/imagenet
mkdir -p $IMAGENET_HOME/validation
mkdir -p $IMAGENET_HOME/train
tar xf ILSVRC2012_img_val.tar -C $IMAGENET_HOME/validation
tar xf ILSVRC2012_img_train.tar -C $IMAGENET_HOME/train
cd $IMAGENET_HOME/train
for f in *.tar; do
  d=`basename $f .tar`
  mkdir $d
  tar xf $f -C $d
done
cd $IMAGENET_HOME
rm $IMAGENET_HOME/train/*.tar # optional
wget -O synset_labels.txt https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt
cd Model-References/TensorFlow/computer_vision/Resnets
$PYTHON preprocess_imagenet.py \
  --raw_data_dir=$IMAGENET_HOME \
  --local_scratch_dir=$IMAGENET_HOME/tf_records
mv $IMAGENET_HOME/validation $IMAGENET_HOME/val
cd $IMAGENET_HOME/val
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```


## Training Multi-Tenants Example 

You can run multiple jobs in parallel using the script described in the following sections. 

**NOTE:** This example is based on ResNet-50.  

### multi_tenants_resnet.sh

`multi_tenants_resnet.sh` script invokes 2 ResNet-50 jobs in parallel, and each uses 4 Gaudis. The following environment variables and python script arguments need to be explicitly specified as different values for both jobs.

#### HABANA_VISIBLE_MODULES

`HABANA_VISIBLE_MODULES` is an environment variable for the list of module IDs, composed by a sequence of single digit integers. The same integer should not be used by multiple jobs running in parallel: 
For jobs with 4 Gaudis, it is recommended to set this to "0,1,2,3" or "4,5,6,7".
For jobs with 2 Gaudis, it is recommended to set this to "0,1", "2,3", "4,5", or "6,7".

#### -md/--model_dir

`-md/--model_dir` is a Python script argument where the model directory needs to be specified to different paths for the different jobs. 