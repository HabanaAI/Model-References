# Swin-Transformer for PyTorch

This directory provides a script and recipe to train the Swin-Transformer model to achieve state of the art accuracy, and is tested and maintained by Habana. For further information on performance, refer to [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance). 

For further information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources

## Table of Contents
- [Model-References](../../../../README.md)
- [Model Overview](#model-overview)
- [Setup](#setup)
- [Training Examples](#training-examples)
- [Advanced](#advanced)
- [Supported Configurations](#supported-configurations)
- [Changelog](#changelog)
- [Known Issues](#known-issues)

## Model Overview

This is a PyTorch implementation based on an earlier implementation from [Microsoft Swin-Transformer](https://github.com/microsoft/Swin-Transformer#introduction).

## Setup

Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) to set up the environment including the `$PYTHON` environment variable. The guide will walk you through the process of setting up your system to run the model on Gaudi.

### Clone Habana Model-References

In the docker container, clone this repository and switch to the branch that matches your SynapseAI version. You can run the [`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the SynapseAI version.

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References /root/Model-References
```

**Note:** If Model-References repository path is not in the PYTHONPATH, make sure you update it:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/Model-References
```

### Install Model Requirements

1. Go to PyTorch swin_transformer directory:
```bash
cd Model-References/PyTorch/computer_vision/classification/swin_transformer
```

2. Install the required packages using pip:
```bash
$PYTHON -m pip install -r requirements.txt
```
### Prepare the Dataset

ImageNet 2012 dataset needs to be organized as per PyTorch requirements. For the specific requirements, refer to [Data Processing section](https://github.com/soumith/imagenet-multiGPU.torch#data-processing). 

**NOTE:** It is assumed that the ImageNet dataset is downloaded and available at `/data/pytorch/imagenet/ILSVRC2012/` path. 

## Training Examples

### Single Card and Multi-Card Training Examples 

**Run training on 1 HPU:**

Run training on 1 HPU in tiny mode, BF16 mixed precision, Batch Size 128: 
```bash
$PYTHON -u main.py --data-path /data/pytorch/imagenet/ILSVRC2012/ --batch-size 128 --mode lazy --cfg ./configs/swin_tiny_patch4_window7_224.yaml --hmp --hmp-bf16 ops_bf16_swin_transformer.txt --hmp-fp32 ops_fp32_swin_transformer.txt
```

**Run training on 8 HPUs:**

To run multi-card demo, make sure to set the following prior to the training: 
- The host machine has 512 GB of RAM installed.
- The docker is installed and set up as per the [Gaudi Setup and Installation Guide](https://github.com/HabanaAI/Setup_and_Install), so that the docker has access to all 8 cards required for multi-card demo.
- All server network interfaces are up. You can change the state of each network interface managed by the habanalabs driver by running the following command:
   ```
   sudo ip link set <interface_name> up
   ```
**NOTE:** To identify if a specific network interface is managed by the habanalabs driver type, run:
   ```
   sudo ethtool -i <interface_name>
   ```

Run training on 8 HPUs in tiny mode, lazy mode: BF16 mixed precision, Batch size 128, 8x on a single server: 

**NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Scaling_Guide/DDP_Based_Scaling.html#mpirun-configuration).

```bash
mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON -u main.py --data-path /data/pytorch/imagenet/ILSVRC2012/ --batch-size 128 --mode lazy --cfg ./configs/swin_tiny_patch4_window7_224.yaml --hmp --hmp-bf16 ops_bf16_swin_transformer.txt --hmp-fp32 ops_fp32_swin_transformer.txt
```
## Advanced 

### Parameters 

To see the available training parameters, run the following command:
```bash
$PYTHON -u main.py --help
```

## Supported Configurations

| Device | SynapseAI Version | PyTorch Version |
|-----|-----|-----|
| Gaudi | 1.6.1 | 1.12.0 |

## Changelog

### 1.5.0
 - Changed `channels_last` from True to False.
 - Disabled permute functions.

### Training Script Modifications 

The following are the changes added to the training script `train.py` and utilities `utils.py`:

* Added support for Habana devices:
  - Load Habana specific library.
  - Certain environment variables are defined for habana device.
  - Added support to run Swin-Transformer training in lazy mode in addition to the eager mode. `mark_step()` is performed to trigger execution of the graph.
  - Added mixed precision support.
  - Modified training script to support distributed training on HPU.
  - Changed start_method for torch multiprocessing.

* Improved performance:
   - Enhance with channels last data format (NHWC) support.
   - Permute convolution weight tensors & and any other dependent tensors like 'momentum' for better performance.
   - Checkpoint saving involves getting trainable params and other state variables to CPU and permuting the weight tensors.
   - Optimized optimizer (FusedAdamW or FusedSGD) is used in place of original optimizer.
   - Added additional parameter to DistributedDataParallel to increase the size of the bucket to combine all the all-reduce calls to a single call.
   - Set pin_memory to false for data loader on HPU.
   - The loss value is fetched to CPU only when needed (to be printed).

* Other changes:  
  - Changed the device type in `timm.mixup` from CUDA to CPU.
  - Change the 6D permute to equivalent 5D permute.

## Known Issues
- Only channels-last = false is supported.
- Placing `mark_step()` arbitrarily may lead to undefined behavior. It is recommended to keep `mark_step()` as shown in the provided scripts.
- Only scripts & configurations mentioned in this README are supported and verified.

