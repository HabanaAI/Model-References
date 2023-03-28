# CustomOp API Usage Example

This README provides an example of how to write custom TensorFlow Ops using a TPC Kernel supported on an HPU device. For more details, refer to [TensorFlow CustomOP API](https://docs.habana.ai/en/latest/TensorFlow/TensorFlow_CustomOp_API/index.html) documentation. For further information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents

* [Model-References](../../../README.md)
* [Prerequisites](#prerequisites)
* [Content](#content)
* [Build and Run CustomDivOp with Default Kernels](#build-and-run-customdivop-with-default-kernels)
* [Build and Run CustomDivOp with Custom Kernel](#build-and-run-customdivop-with-custom-kernel)
* [Important to Know](#important-to-know)

## Prerequisites

- A TPC kernel on which the HpuKernel will run. To write a CustomOp, you must define the TPC kernel that HpuKernel will run on first. This document provides the required steps for using the existing default TPC kernels `div_fwd_fp32` and `div_fwd_bf16` as we all as the custom kernel `customdiv_fwd_f32` to implement CustomOp. For further information on how to write TPC kernels, refer to the [Habana Custom Kernel GitHub page](https://github.com/HabanaAI/Habana_Custom_Kernel).

- **habana-tensorflow** Python package must be installed. Make sure to install by following the instructions detailed in the [Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html).

- PyTest for testing the CustomOp library only.

## Content

- C++ file with CustomDivOp definition and Kernel implementation on HPU:
    - CustomDivOp performs a division on input parameters.
    - As an addition, it accepts input attribute `example_attr` only to demonstrate how it can be retrieved in C++. 
- CMake file for building the solution:
    - Looks for TensorFlow using Python3 and sets compiler/linker flags.
    - Looks for `habana-tensorflow` using Python3 and sets include directory as well as compiler/linker flags.
- Python test to run and validate CustomDivOp. PyTest is configured to accept a command line argument:
    - `custom_op_lib` - location of the built library.

## Build and Run CustomDivOp with Default Kernels

To build and run CustomDivOp with default kernels `div_fwd_fp32` and `div_fwd_bf16`, run the following:

```bash
mkdir build/ && cd build/
cmake .. && make
pytest -v --custom_op_lib lib/libhpu_custom_div_op.so ../test_custom_div_op.py
```

If all steps are successful, the environment is set up properly.

## Build and Run CustomDivOp with Custom Kernel

To build and run CustomDivOp with custom kernel `customdiv_fwd_f32`, perform the following steps:

1. Refer to the [Habana Custom Kernel GitHub page](https://github.com/HabanaAI/Habana_Custom_Kernel) to obtain the custom kernel build and generate the custom kernel binary `libcustom_tpc_perf_lib.so`.
2. Remove build directory `rm -rf build` if a build directory already exists.
3. Build the solution:
```bash
mkdir build/ && cd build/
cmake -DUSE_CUSTOM_KERNEL=1 .. && make
```
4. Run the PyTest:
```bash
export GC_KERNEL_PATH=<path to custom kernel binary>/libcustom_tpc_perf_lib.so:$GC_KERNEL_PATH
# The custom kernel only implemented float32 flavor for now.
pytest -v --custom_op_lib lib/libhpu_custom_div_op.so -k float32 ../test_custom_div_op.py
```
If all steps are successful, the environment is set up properly.

## Important to Know

Implementing a single Op requires additional setup in order to use it in a trainable topology.

During training, TensorFlow injects Gradient Ops to all trainable Ops defined. In this case, there is no Gradient defined, therefore the Op cannot be used in a topology.
To make sure the Op can be used, **CustomDivGradOp** should be defined and registered in Python with [tf.RegisterGradient](https://www.tensorflow.org/api_docs/python/tf/RegisterGradient) method.
