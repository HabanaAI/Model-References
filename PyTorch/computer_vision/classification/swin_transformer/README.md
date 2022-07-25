# Swin-Transformer
For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

# Table of Contents
- [Model-References](../../../../README.md)
- [Model Overview](#model-overview)
- [Setup](#setup)
  - [Set up dataset](#set-up-dataset)
- [Training the Model](#training-the-model)
- [Multi-HPU Training](#multihpu-training)
- [Supported Configurations](#supported-configurations)
- [Changelog](#changelog)
  - [1.5.0](#150)
- [Known Issues](#known-issues)
- [Training Script Modifications](#training-script-modifications)

# Model Overview
The base model used is from [GitHub: Swin-Transformer](https://github.com/microsoft/Swin-Transformer#introduction), commit tag: 6bbd83ca617db8480b2fb9b335c476ffaf5afb1a. Please refer to a below section for a summary of model changes, changes to training script and the original files.

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
Go to PyTorch swin_transformer directory:
```bash
cd Model-References/PyTorch/computer_vision/classification/swin_transformer
```
Please install required packages using pip
```bash
$PYTHON -m pip install -r requirements.txt
```
## Set up dataset

Imagenet 2012 dataset needs to be organized as per PyTorch requirement. PyTorch requirements are specified in the link below which contains scripts to organize Imagenet data.

https://github.com/soumith/imagenet-multiGPU.torch#data-processing.

The training images for imagenet are already in appropriate subfolders (like n07579787, n07880968).
You need to get the validation groundtruth and move the validation images into appropriate subfolders.
To do this, download ILSVRC2012_img_train.tar ILSVRC2012_img_val.tar and use the following commands:
```bash
# extract train data
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
# extract validation data
cd ../ && mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```

# Training the Model

The following commands assume that Imagenet dataset is available at /data/pytorch/imagenet/ILSVRC2012/ directory.

Please refer to the following command for available training parameters:
```bash
$PYTHON -u main.py --help
```
i. Example: Swin Transformer, Tiny mode, bf16 mixed precision, Batch Size 128
```bash
$PYTHON -u main.py --data-path /data/pytorch/imagenet/ILSVRC2012/ --batch-size 128 --mode lazy --cfg ./configs/swin_tiny_patch4_window7_224.yaml --hmp --hmp-bf16 ops_bf16_swin_transformer.txt --hmp-fp32 ops_fp32_swin_transformer.txt
```

## Multi-HPU Training
To run multi-HPU demo, make sure the host machine has 512 GB of RAM installed.
Also ensure you followed the [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html)
to install and set up docker, so that the docker has access to all the 8 cards
required for multi-HPU demo.

Before execution of the multi-HPU demo scripts, make sure all server network interfaces are up. You can change the state of each network interface managed by the habanalabs driver using the following command:
```
sudo ip link set <interface_name> up
```
To identify if a specific network interface is managed by the habanalabs driver type, run:
```
sudo ethtool -i <interface_name>
```

Finally to train the Swin Transformer on multiple HPUs with below script.

**NOTE:** mpirun map-by PE attribute value may vary on your setup. Please refer to the instructions on [mpirun Configuration](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Scaling_Guide/DDP_Based_Scaling.html#mpirun-configuration) for calculation.

i. Example: Swin Transformer, Tiny mode, lazy mode: bf16 mixed precision, Batch size 128, 8x on single server
```bash
mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON -u main.py --data-path /data/pytorch/imagenet/ILSVRC2012/ --batch-size 128 --mode lazy --cfg ./configs/swin_tiny_patch4_window7_224.yaml --hmp --hmp-bf16 ops_bf16_swin_transformer.txt --hmp-fp32 ops_fp32_swin_transformer.txt
```

# Supported Configurations

| Device | SynapseAI Version | PyTorch Version |
|-----|-----|-----|
| Gaudi | 1.5.0 | 1.11.0 |

# Changelog
### 1.5.0
 - Changed channels_last from True to False
 - Disable permute functions

# Known Issues
- Only channels-last = false is supported.
- Placing mark_step() arbitrarily may lead to undefined behaviour. Recommend to keep mark_step() as shown in provided scripts.
- Only scripts & configurations mentioned in this README are supported and verified.

# Training Script Modifications
The following are the changes added to the training script (train.py) and utilities(utils.py):

1. Added support for Habana devices

   a. Load Habana specific library.

   b. Certain environment variables are defined for habana device.

   c. Added support to run Swin-Transformer training in lazy mode in addition to the eager mode.
    mark_step() is performed to trigger execution of the graph.

   d. Added mixed precision support.

   e. Modified training script to support distributed training on HPU.

   f. Changed start_method for torch multiprocessing.


2. To improve performance

   a. Enhance with channels last data format (NHWC) support.

   b. Permute convolution weight tensors & and any other dependent tensors like 'momentum' for better performance.

   c. Checkpoint saving involves getting trainable params and other state variables to CPU and permuting the weight tensors.

   d. Optimized optimizer (FusedAdamW or FusedSGD) is used in place of original optimizer.

   e. Added additional parameter to DistributedDataParallel to increase the size of the bucket to combine all the all-reduce calls to a single call.

   f. Set pin_memory to false for data loader on HPU.

   g. The loss value is fetched to CPU only when needed (to be printed).

3. Others
   a. Change the device type in timm.mixup from cuda to cpu.


The model changes are listed below:
1. Change the 6D permute to equivalent 5D permute.
