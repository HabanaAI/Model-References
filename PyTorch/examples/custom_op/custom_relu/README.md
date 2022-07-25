# Example on CustomOp API Usage

This README provides an example of how to write custom PyTorch Op with Kernel supported on HPU device.
For further information about training deep learning models on Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).


## Prerequisites

- Defining TPC Kernels. To write CustomOp, the TPC kernel that HpuKernel will run should be defined.
  This document provides instructions for using the existing default TPC kernels **relu_fwd_f32** , **relu_bwd_f32** and a custom kernel **custom_op::custom_relu** , **custom_op::custom_relu_backward** schema to implement CustomOp. For further information on how to write TPC kernels, refer to [Habana® Gaudi® TPC documentation](https://github.com/HabanaAI/Habana_Custom_Kernel).

- Python package installed: **habana-torch-plugin**. 

## Content
- C++ file with **custom_op::custom_relu** , **custom_op::custom_relu_backward** definition and Kernel implementation on HPU:
    - custom_relu performs a relu on input.
    - custom_relu_backward performs a threshold_backward on input.
- setup.py file for building the solution:
    - To compile to op run ```python setup.py build```.
- Python test to run and validate custom_relu and custom_relu_backward:
    - ```python hpu_custom_op_relu_test.py```

## Commands to build custom_relu and custom_relu_backward
```python setup.py build```

## Important to Know
This is an example of an Op implementing both forward and backward.
The forward and backward CustomOp is used for training the model by extending [torch.autograd](https://pytorch.org/docs/stable/notes/extending.html) package.

## Known Issues
BF16 or HMP is not supported yet. If custom op is used in topology, run FP32 variant only. 

## Examples on Applying CustomOps to a Real Training Model

This section provides an example on how to apply CustomOps to a real training model ResNet50. See the steps below: 

1. Apply the patch custom_relu_op.patch to replace the torch.nn.ReLU with CustomReLU in ResNet50 model:
   - Go to the main directory in the repository.
   - Run `git apply --verbose PyTorch/computer_vision/classification/torchvision/custom_relu_op.patch`
2. Build the custom_relu and custom_relu_backward ops with the existing kernels relu_fwd_f32 and relu_bwd_f32 as described above. 
3. If the build steps were successful, follow the README instructions in `<repo>/PyTorch/computer_vision/classification/torchvision/README.md` to try running ResNet50 model using CustomOps custom_relu and custom_relu_backward.

