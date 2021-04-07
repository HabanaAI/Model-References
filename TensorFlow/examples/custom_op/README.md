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
- **habana_device.so** and all its dependencies
- TensorFlow 2.2.0 - Python package
- PyTest (just for testing custom op library)

## Content

This example consists of:
- C++ file with CustomDivOp definition and Kernel implementation on HPU
    - CustomDivOp is performing a division on input parameters
    - As an addition, it's accepting also input attribute **example_attr**, just to demonstrate, how it can be retrieved in C++
- CMake file for building the solution
    - it looks for TensorFlow, using Python3 and sets Compiler/Linker flags
    - it expects input variables **HPU_INCLUDE_DIR** and **HPU_HABANA_DEVICE_BINARY** that point to headers and **habana_device.so**
- Python test to run and validate CustomDivOp
    - Pytest is configured to accept command line argument:
        - **custom_op_lib** - a location of built library.

## Steps to build and run CustomDivOp with default kernel div\_fwd\_fp32 & div\_fwd\_bf16
1. `mkdir build/ && cd build/`
2. `cmake -DHPU_INCLUDE_DIR:PATH=<location of hpu headers> -DHPU_HABANA_DEVICE_BINARY:PATH=</path/to/habana_device.so> .. && make`
3. Make sure **LD_LIBRARY_PATH** includes path to the binaries. In 0.13.0_380 docker image, users need to set `export LD_LIBRARY_PATH=/usr/lib/habanalabs:$LD_LIBRARY_PATH`
4. `pytest -v --custom_op_lib lib/libhpu_custom_div_op.so ../test_custom_div_op.py`
5. If all steps were successful, the environment is set up properly.

## Steps to build and run CustomDivOp with custom kernel customdiv\_fwd\_f32
1. Refer to https://github.com/HabanaAI/Habana_Custom_Kernel for custom kernel build and generate the custom kernel binary **libcustom_tpc_perf_lib.so**.
2. Remove build directory if it exists `rm -rf build`
3. `mkdir build/ && cd build/`
4. `cmake -DUSE_CUSTOM_KERNEL=1 -DHPU_INCLUDE_DIR:PATH=<location of hpu headers> -DHPU_HABANA_DEVICE_BINARY:PATH=</path/to/habana_device.so> .. && make`
5. Make sure **LD_LIBRARY_PATH** includes path to the binaries. In 0.13.0_380 docker image, users need to set `export LD_LIBRARY_PATH=/usr/lib/habanalabs:$LD_LIBRARY_PATH`
6. `export GC_KERNEL_PATH=<path to custom kernel binary>/libcustom_tpc_perf_lib.so:$GC_KERNEL_PATH`
7. `pytest -v --custom_op_lib lib/libhpu_custom_div_op.so -k float32 ../test_custom_div_op.py` The custom kernel only implemented float32 flavor for now.
8. If all steps were successful, the environment is set up properly.

## Important to know

Implementing single Op is in most cases not enough in order to use it in a *trainable* topology.

During training, TensorFlow is injecting Gradient Ops to all trainable Ops defined.

In this case, there is no Gradient defined, so the Op cannot be used in a topology.
In order to make it work, user would need to define **CustomDivGradOp** first and register it in Python with [tf.RegisterGradient](https://www.tensorflow.org/api_docs/python/tf/RegisterGradient) method.
