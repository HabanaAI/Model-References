# Example on CustomOp API Usage

This README provides an example of how to write a custom PyTorch Op with Kernel supported on HPU device.
For further information about training deep learning models on Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

## Prerequisites

- Defining TPC Kernels. To write CustomOp, the TPC kernel that HpuKernel will run should be defined.
  This document provides instructions for using the existing default TPC kernels **topk** and a custom kernel **custom_op::custom_topk** schema to implement CustomOp. For further information on how to write TPC kernels, refer to [Habana® Gaudi® TPC documentation](https://github.com/HabanaAI/Habana_Custom_Kernel).

- Python package installe: **habana-torch**. 

## Content

- C++ file with **custom_op::custom_topk** definition and Kernel implementation on HPU:
    - custom_topk performs a topk on input parameters.
- setup.py file for building the solution:
    - To compile to Op, run ```python setup.py install```.
- Python test to run and validate custom_topk:
    - ```python hpu_custom_op_topk_test.py```

## Commands to Build custom_topk
```python setup.py install```

## Build and Run custom_topk with topk TPC Kernel
```pytest --custom_op_lib <custom_topk_lib.so> hpu_custom_op_topk_test.py```

## Important to Know

Implementing a single Op is mostly not enough in order to use it in a *trainable* topology.

During the training, PyTorch injects Gradient Ops to all trainable Ops defined.
In this case, there is no autograd defined, therefore the Op cannot be used in a topology.
To make it work, **CustomTopkBackward** should be defined and registered in Python with [torch.autograd](https://pytorch.org/docs/stable/notes/extending.html).

