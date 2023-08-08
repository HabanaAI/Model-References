# CustomOp API Usage in PyTorch

This README provides an example of how to write custom PyTorch Ops using a TPC Kernel supported on an HPU device. For more details, refer to [PyTorch CustomOP API](https://docs.habana.ai/en/latest/PyTorch/PyTorch_CustomOp_API/page_index.html) documentation. 

For further information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents

* [Model-References](../../../../README.md)
* [Prerequisites](#prerequisites)
* [Content](#content)
* [Build and Run with Custom Kernels](#build-and-run-with-custom-kernels)
* [Important to Know](#important-to-know)
* [Applying CustomOps to a Real Training Model Example](#applying-customops-to-a-real-training-model-example)
* [Known Issues](#known-issues)


## Prerequisites

- A TPC kernel on which the HpuKernel will run. To write a CustomOp, you must define the TPC kernel that HpuKernel will run on first. This document provides the required steps for using the existing default TPC kernels `relu_fwd_f32`, `relu_bwd_f32` as we all as the custom kernel `custom_op::custom_relu` to implement CustomOp. For further information on how to write TPC kernels, refer to the [Habana Custom Kernel GitHub page](https://github.com/HabanaAI/Habana_Custom_Kernel).

- **habana-torch-plugin** Python package must be installed. Make sure to install by following the instructions detailed in the [Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html).

## Content

- C++ file with **custom_op::custom_relu**, **custom_op::custom_relu_backward** definition and Kernel implementation on HPU:
    - `custom_relu` performs a relu on input.
    - `custom_relu_backward` performs a threshold_backward on input.
- `setup.py` file for building the solution:
    - To compile to Op, run ```python setup.py build```.
- Python test to run and validate `custom_relu` and `custom_relu_backward`:
    - ```python hpu_custom_op_relu_test.py```

## Build and Run with Custom Kernels 

To build and run `custom_relu` and `custom_relu_backward`, run the following: 
```python setup.py build```

## Important to Know

This is an example of an Op implementing both forward and backward.
The forward and backward CustomOp is used for training the model by extending the [torch.autograd](https://pytorch.org/docs/stable/notes/extending.html) package.


## Applying CustomOps to a Real Training Model Example

This section provides an example for applying CustomOps to a real training model ResNet-50. 
Follow the below steps:

1. Apply the patch `custom_relu_op.patch` to replace the `torch.nn.ReLU` with `CustomReLU` in ResNet-50 model:
   - Go to the main directory in the repository.
   - Run `git apply --verbose PyTorch/computer_vision/classification/torchvision/custom_relu_op.patch`
2. Build the `custom_relu` and `custom_relu_backward` Ops with the default kernels `relu_fwd_f32` and `relu_bwd_f32` as described above. 
3. If the build steps are successful, follow the instructions in `<repo>/PyTorch/computer_vision/classification/torchvision/README.md` to try running ResNet-50 model using CustomOps `custom_relu` and `custom_relu_backward`.

## Known Issues

BF16 or HMP is not supported yet. To use CustomOp in topology, run FP32 variant only. 

