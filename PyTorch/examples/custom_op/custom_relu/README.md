# Example CustomOp API usage

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

This folder contains example of how to write custom Pytorch Op with Kernel supported on HPU device.

## Prerequisites

In order to write CustomOp, user needs to define TPC kernel, that the HpuKernel will run.
This document covers how to use existing default TPC kernels **relu_fwd_f32** , **relu_bwd_f32** and a custom kernel **custom_op::custom_relu** , **custom_op::custom_relu_backward** schema to implement CustomOp.

For information how to write TPC kernels, please refer to [Habana® Gaudi® TPC documentation](https://github.com/HabanaAI/Habana_Custom_Kernel).

On top of TPC kernel, there are following requirements:
- **habana-torch-plugin** Python package installed

## Content
- C++ file with **custom_op::custom_relu** , **custom_op::custom_relu_backward** definition and Kernel implementation on HPU
    - custom_relu is performing a relu on input
    - custom_relu_backward is performing a threshold_backward on input
- setup.py file for building the solution
    - to compile to op run ```python setup.py build```
- Python test to run and validate custom_relu and custom_relu_backward
    - ```python hpu_custom_op_relu_test.py```

## Commands to build custom_relu and custom_relu_backward
```python setup.py build```

## Important to know
Here we show an example of an op implementing both forward and backward,
The forward and backward custom op is used for training the model, by extending [torch.autograd](https://pytorch.org/docs/stable/notes/extending.html) package.

## Known Issues
BF16 or HMP is not supported yet
If custom op is used in topology, need to run fp32 variant only

## Examples to apply custom ops to a real training model

In this section, we will show an example how to apply custom ops to a real training model renset50. The steps are as following:
1. Apply the patch custom_relu_op.patch to replace the torch.nn.ReLU with CustomReLU in resnet50 model
   - Go to main directory in the repository
   - `git apply --verbose PyTorch/computer_vision/classification/torchvision/custom_relu_op.patch`
2. Build the custom_relu and custom_relu_backward ops with existing kernels relu_fwd_f32 and relu_bwd_f32 as described above
3. If build steps were successful, users can follow the readme instruction on `<repo>/PyTorch/computer_vision/classification/torchvision/README.md` to try running the resnet50 model using custom ops custom_relu and custom_relu_backward.

