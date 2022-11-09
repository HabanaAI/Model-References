# CustomOp API Usage in PyTorch

This README provides an example of how to write custom PyTorch Ops using a TPC Kernel supported on an HPU device. For more details, refer to [PyTorch CustomOP API](https://docs.habana.ai/en/latest/PyTorch/PyTorch_CustomOp_API/page_index.html) documentation. 

For further information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/)

## Table of Contents

* [Model-References](../../../README.md)
* [Prerequisites](#prerequisites)
* [Content](#content)
* [Build and Run with Custom Kernels](#build-and-run-customDivOp-with-default-kernels)
* [Important to Know](#important-to-know)
  
## Prerequisites

- A TPC kernel on which the HpuKernel will run. To write a CustomOp, you must define the TPC kernel that HpuKernel will run on first. This document provides the required steps for using the existing default TPC kernel `topk` as well as the custom kernel `custom_op::custom_topk` to implement CustomOp. For further information on how to write TPC kernels, refer to the [Habana Custom Kernel GitHub page](https://github.com/HabanaAI/Habana_Custom_Kernel).

- **habana-torch-plugin** Python package must be installed. Make sure to install by following the instructions detailed in the [Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html).

## Content

- C++ file with `custom_op::custom_topk` definition and Kernel implementation on HPU:
    - `custom_topk` performs a topk on input parameters.
- `setup.py` file for building the solution:
    - To compile to Op, run ```python setup.py install```.
- Python test to run and validate custom_topk:
    - ```python hpu_custom_op_topk_test.py```

## Build and Run `custom_topk`

To build and run CustomOp with `custom_topk`, run the following: 

```python setup.py install```

## Build and Run `custom_topk` with topk TPC Kernel

To build and run `custom_topk` with topk Kernel, run the following: 

```pytest --custom_op_lib <custom_topk_lib.so> hpu_custom_op_topk_test.py```

## Important to Know

Implementing a single Op requires additional setup in order to use it in a trainable topology.

During the training, PyTorch injects Gradient Ops to all trainable Ops defined.
In this case, there is no autograd defined, therefore the Op cannot be used in a topology.
To make sure the Op can be used, **CustomTopkBackward** should be defined and registered in Python with [torch.autograd](https://pytorch.org/docs/stable/notes/extending.html) method.

