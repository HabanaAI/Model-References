# Example CustomOp API usage

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

This folder contains example of how to write custom Pytorch Op with Kernel supported on HPU device.

## Prerequisites

In order to write CustomOp, user needs to define TPC kernel, that the HpuKernel will run.
This document covers how to use existing default TPC kernels **topk** and a custom kernel **custom_op::custom_topk** schema to implement CustomOp.

For information how to write TPC kernels, please refer to [Habana® Gaudi® TPC documentation](https://github.com/HabanaAI/Habana_Custom_Kernel).

On top of TPC kernel, there are following requirements:
- **habana-torch** Python package installed

## Content
- C++ file with **custom_op::custom_topk** definition and Kernel implementation on HPU
    - custom_topk is performing a topk on input parameters
- setup.py file for building the solution
    - to compile to op run ```python setup.py install```
- Python test to run and validate custom_topk
    - ```python hpu_custom_op_topk_test.py```

## Commands to build custom_topk
```python setup.py install```

## Steps to build and run custom_topk with topk TPC kernel
```pytest --custom_op_lib <custom_topk_lib.so> hpu_custom_op_topk_test.py```

## Important to know

Implementing single Op is in most cases not enough in order to use it in a *trainable* topology.

During training, Pytorch is injecting Gradient Ops to all trainable Ops defined.

In this case, there is no autograd defined, so the Op cannot be used in a topology.
In order to make it work, user would need to define **CustomTopkBackward** first and register it in Python with [torch.autograd](https://pytorch.org/docs/stable/notes/extending.html).

