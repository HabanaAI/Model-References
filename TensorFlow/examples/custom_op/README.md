# Example CustomOp API usage

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

This folder contains example of how to write custom TensorFlow Op with Kernel supported on HPU device.
For complete documentation, please refer to **Habana® Gaudi® SynapseAI® documentation**, Section *API REFERENCE GUIDES*, Chapter 4 *TensorFlow CustomOp API Reference*.

## Prerequisites

In order to write CustomOp, user needs to define TPC kernel, that the HpuKernel will run.
This document covers how to use existing default TPC kernels **div_fwd_fp32 & div_fwd_bf16** and a custom kernel **customdiv_fwd_f32**
to implement CustomOp.

For information how to write TPC kernels, please refer to [Habana® Gaudi® TPC documentation](https://github.com/HabanaAI/Habana_Custom_Kernel).

On top of TPC kernel, there are following requirements:
- **habana-tensorflow** Python package installed
- PyTest (just for testing custom op library)

## Content

This example consists of:
- C++ file with CustomDivOp definition and Kernel implementation on HPU
    - CustomDivOp is performing a division on input parameters
    - As an addition, it's accepting also input attribute **example_attr**, just to demonstrate, how it can be retrieved in C++
- CMake file for building the solution
    - it looks for TensorFlow using Python3 and sets compiler/linker flags
    - it looks for Habana-TensorFlow using Python3 and sets include directory and compiler/linker flags
- Python test to run and validate CustomDivOp
    - Pytest is configured to accept command line argument:
        - **custom_op_lib** - a location of built library.

## Commands to build and run CustomDivOp with default kernel div_fwd_fp32 & div_fwd_bf16
```bash
mkdir build/ && cd build/
cmake .. && make
pytest -v --custom_op_lib lib/libhpu_custom_div_op.so ../test_custom_div_op.py
```
If all steps were successful, the environment is set up properly.

## Steps to build and run CustomDivOp with custom kernel customdiv_fwd_f32
1. Refer to https://github.com/HabanaAI/Habana_Custom_Kernel for custom kernel build and generate the custom kernel binary **libcustom_tpc_perf_lib.so**.
2. Remove build directory if it exists `rm -rf build`
3. Build solution
```bash
mkdir build/ && cd build/
cmake -DUSE_CUSTOM_KERNEL=1 .. && make
```
4. Run test
```bash
export GC_KERNEL_PATH=<path to custom kernel binary>/libcustom_tpc_perf_lib.so:$GC_KERNEL_PATH
# The custom kernel only implemented float32 flavor for now.
pytest -v --custom_op_lib lib/libhpu_custom_div_op.so -k float32 ../test_custom_div_op.py
```
5. If all steps were successful, the environment is set up properly.

## Important to know

Implementing single Op is in most cases not enough in order to use it in a *trainable* topology.

During training, TensorFlow is injecting Gradient Ops to all trainable Ops defined.

In this case, there is no Gradient defined, so the Op cannot be used in a topology.
In order to make it work, user would need to define **CustomDivGradOp** first and register it in Python with [tf.RegisterGradient](https://www.tensorflow.org/api_docs/python/tf/RegisterGradient) method.

## Examples to apply custom ops to a real training model

In this section, we will show an example how to apply custom ops to a real training model mobilenetv2. The steps are as following:
1. Apply the patch custom_relu6_op.patch to replace the tf.nn.relu6 with custom_relu6 in mobilenetv2 model
   - Go to main directory in the repository
   - `git apply --verbose TensorFlow/examples/custom_op/custom_relu6_op.patch`
2. Build the custom relu6 and relu6Grad ops with default kernel relu6_fwd_ and relu6_bwd_ (Note: users can also generate custom ops based on custom kernels following the above guidance in [Steps to build and run CustomDivOp with custom kernel customdiv_fwd_f32](https://github.com/HabanaAI/Habana_Custom_Kernel))
   - `cd TensorFlow/examples/custom_op/`
   - `mkdir build/ && cd build/`
   - `cmake -DUSE_CUSTOM_KERNEL=1 .. && make`
3. If build steps were successful, users can follow the readme instruction on `<repo>/staging/TensorFlow/computer_vision/mobilenetv2/research/slim/README.md` to try running the mobilenevt2 model using custom ops CustomRelu6Op and CustomRelu6GradOp.
