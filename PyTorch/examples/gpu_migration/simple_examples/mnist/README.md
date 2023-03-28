# MNIST example Using GPU Migration

This README provides an example of how to simply migrate a model from GPU to HPU. For more details, refer to [Migrating PyTorch Models from GPU to HPU](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Model_Porting/Migrating_PyTorch_Models_to_HPU.html) documentation. 

This is a modified version of an MNIST example cloned from [PyTorch GitHub repository](https://github.com/pytorch/examples/tree/40289773aa4916fad0d50967917b3ae8aa534fd6mnist).

For further information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents

* [Model-References](../../../../../README.md)
* [Setup](#setup)
* [Run the Model](#run-the-model)
* [Supported Configurations](#supported-configurations)
* [Script Modifications](#script-modifications)
* [GPU Migration Logs](#gpu-migration-logs)

## Setup
Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the environment including the `$PYTHON` environment variable. This guide will walk you through the process of setting up your system to run the model on Gaudi.

### Clone Habana Model-References
In the docker container, clone this repository and switch to the branch that matches your SynapseAI version.
You can run the [`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the SynapseAI version.
```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```

### Install Model Requirements
1. In the docker container, go to the model directory:
```bash
cd Model-References/PyTorch/examples/simple_examples/mnist
```

2. Create a data directory
```bash
mkdir data
```
## Run the Model 
To run the model, execute the following command:
```bash
$PYTHON main.py
```

## Supported Configurations
| Device  | SynapseAI Version | PyTorch Version | Mode |
|---------|-------------------|-----------------|------|
| Gaudi   | 1.9.0             | 1.13.1          | Training |
| Gaudi2  | 1.9.0             | 1.13.1          | Training |

## Script Modifications 
The following lists the significant changes made to the original script. 

1. Import GPU Migration Package:
```python
import habana_frameworks.torch.gpu_migration
```

2. Import Habana Torch Library:
```python
import habana_frameworks.torch.core as htcore
```

3. Add `mark_step()`. In Lazy mode, `mark_step()` must be added in all training scripts right after `loss.backward()` and `optimizer.step()`.
```python
htcore.mark_step()
```
### Non-functional Script Modifications 
- Added a line to print whether the `use_cuda` parameter is set to true. 
- Updated the expected dataset location from ../data to data.

## GPU Migration Logs
You can review GPU Migration logs under [gpu_migration_logs/gpu_migration_66.log](gpu_migration_logs/gpu_migration_66.log).
For further information, refer to [GPU Migration Toolkit documentation](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Model_Porting/GPU_Migration_Toolkit/GPU_Migration_Toolkit.html#enabling-logging-feature).