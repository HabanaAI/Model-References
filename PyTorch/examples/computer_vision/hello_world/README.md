# PyTorch MNIST Example

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents

* [Model Overview](#model-overview)
  * [Simple Example for Model Migration](#simple-example-for-model-migration)
* [Setup](#setup)
* [Training Examples](#training-examples)
  * [Training on 1 HPU Commands](#training-on-1-hpu-commands)
  * [Training on 8 HPU Commands](#training-on-8-hpu-commands)
* [Changelog](#changelog)
  * [1.5.0](#150)

## Model Overview

The PyTorch MNIST example, mnist.py, is based on the source code forked from GitHub repository
[pytorch/examples](https://github.com/pytorch/examples/tree/master/mnist).
The model has been enabled on 1 HPU and 8 HPUs of 1 HLS server, in eager mode and lazy mode with FP32 data type and BF16 mixed data type.

### A Simple Example on Model Migration

In addition to the mnist.py script, the simple example.py model shows the minimal modifications required to allow a model to run on Gaudi.
For further details, refer to [PyTorch Migration Guide]( https://docs.habana.ai/en/latest/PyTorch/Migration_Guide/Porting_Simple_PyTorch_Model_to_Gaudi.html).

On a single Gaudi and in lazy mode, run the following command:

```bash
$PYTHON example.py
```

## Setup

Follow the instructions given in the following link for setting up the
environment including the `$PYTHON` environment variable: [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html).
This guide will walk you through the process of setting up your system to run
the model on Gaudi.

### Clone Habana Model-References Repository

In the docker container, clone Model-References repository and switch to the branch that
matches your SynapseAI version. 

---
**NOTE:** 

To determine the SynapseAI version, run the
[`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility .)
---

To clone the Model-References repository, run the following command: 

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```

## Training Examples

This hello world example can run on 1 HPU and 8 HPUs in eager mode and lazy mode with FP32 data type and BF16 mixed data type.

### Training on 1 HPU Examples 

On a single HPU and in FP32 eager mode, run the following command:

```bash
$PYTHON mnist.py --batch-size=64 --epochs=1 --lr=1.0 --gamma=0.7 --hpu
```

On a single HPU and in BF16 eager mode, run the following command:

```bash
$PYTHON mnist.py --batch-size=64 --epochs=1 --lr=1.0 --gamma=0.7 --hpu --hmp --hmp-bf16=ops_bf16_mnist.txt --hmp-fp32=ops_fp32_mnist.txt
```

On a single HPU and in FP32 lazy mode, run the following command:

```bash
$PYTHON mnist.py --batch-size=64 --epochs=1 --lr=1.0 --gamma=0.7 --hpu --use_lazy_mode
```

On a single HPU and in BF16 lazy mode, run the following command:

```bash
$PYTHON mnist.py --batch-size=64 --epochs=1 --lr=1.0 --gamma=0.7 --hpu --hmp --hmp-bf16=ops_bf16_mnist.txt --hmp-fp32=ops_fp32_mnist.txt --use_lazy_mode
```

### Training on 8 HPU Examples

On 8 HPU, 1 HLS and in FP32 eager mode, run the following command:

```bash
mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON mnist.py --batch-size=64 --epochs=1 --lr=1.0 --gamma=0.7 --hpu
```

On 8 HPU, 1 HLS and in BF16 eager mode, run the following command:

```bash
mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON mnist.py --batch-size=64 --epochs=1 --lr=1.0 --gamma=0.7 --hpu --hmp --hmp-bf16=ops_bf16_mnist.txt --hmp-fp32=ops_fp32_mnist.txt
```

On 8 HPU, 1 HLS and in FP32 lazy mode, run the following command:

```bash
mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON mnist.py --batch-size=64 --epochs=1 --lr=1.0 --gamma=0.7 --hpu --use_lazy_mode
```

On 8 HPU, 1 HLS and in BF16 lazy mode, run the following command:

```bash
mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root $PYTHON mnist.py --batch-size=64 --epochs=1 --lr=1.0 --gamma=0.7 --hpu --hmp --hmp-bf16=ops_bf16_mnist.txt --hmp-fp32=ops_fp32_mnist.txt --use_lazy_mode
```
## Changelog
### 1.5.0
 - Changed channels_last from True to False
 - Disabled permute functions
