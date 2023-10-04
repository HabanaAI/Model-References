
# Hello World in PyTorch 

This directory provides example training scripts to run Hello World on 1 HPU and 8 HPUs distributed training in lazy mode with FP32 data type and BF16 mixed data type.

For further information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents

* [Model Overview](#model-overview)
* [Setup](#setup)
* [Training Examples](#training-examples)
* [Changelog](#changelog)

## Example Overview

The PyTorch Hello World example, `mnist.py`, is based on the source code forked from GitHub repository
[pytorch/examples](https://github.com/pytorch/examples/tree/master/mnist).

## Setup

Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) to set up the environment including the `$PYTHON` environment variable. The guide will walk you through the process of setting up your system to run the model on Gaudi.

### Clone Habana Model-References

In the docker container, clone this repository and switch to the branch that matches your SynapseAI version. You can run the [`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the SynapseAI version.

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References /path/to/Model-References
```

**Note:** If Model-References repository path is not in the PYTHONPATH, make sure you update it:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/Model-References
```

## Training Examples

### Single Card and Multi-Card Training Examples 

#### Examples in Bash Scripts 

**Run training on 1 HPU:**

- 1 HPU in FP32 lazy mode:

```bash
PT_HPU_LAZY_MODE=1 $PYTHON mnist.py --batch-size=64 --epochs=1 --lr=1.0 --gamma=0.7 --hpu
```

- 1 HPU in BF16 lazy mode:

```bash
PT_HPU_LAZY_MODE=1 $PYTHON mnist.py --batch-size=64 --epochs=1 --lr=1.0 --gamma=0.7 --hpu --hmp --hmp-bf16=ops_bf16_mnist.txt --hmp-fp32=ops_fp32_mnist.txt
```

**Run training on 8 HPUs:**

**NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration).



- 8 HPUs, 1 server in FP32 lazy mode:

```bash
mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root -x PT_HPU_LAZY_MODE=1 $PYTHON mnist.py --batch-size=64 --epochs=1 --lr=1.0 --gamma=0.7 --hpu
```

- 8 HPU, 1 server in BF16 lazy mode:

```bash
mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root -x PT_HPU_LAZY_MODE=1 $PYTHON mnist.py --batch-size=64 --epochs=1 --lr=1.0 --gamma=0.7 --hpu --hmp --hmp-bf16=ops_bf16_mnist.txt --hmp-fp32=ops_fp32_mnist.txt
```

#### Examples in Python Script

The `example.py` presents a basic PyTorch code example. For more details, refer to [Getting Started with PyTorch and Gaudi](https://docs.habana.ai/en/latest/PyTorch/Getting_Started_with_PyTorch_and_Gaudi/Getting_Started_with_PyTorch.html).

**Run training on 1 HPU:**

On 1 HPU in lazy mode, run the following command:

```bash
$PYTHON example.py
```

## Changelog

### 1.12.0
 - Eager mode support is deprecated.

### 1.5.0
 - Changed channels_last from True to False.
 - Disabled permute functions.
