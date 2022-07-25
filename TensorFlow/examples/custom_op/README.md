# Example on CustomOp API Usage

This README provides an example of how to write custom TensorFlow Op with Kernel supported on HPU device.
For the complete documentation, refer to **Habana® Gaudi® SynapseAI® documentation**, Section *API REFERENCE GUIDES*, Chapter 4 *TensorFlow CustomOp API Reference*.
For further information about training deep learning models on Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).



## Prerequisites

- Defining TPC Kernels. To write CustomOp, the TPC kernel that HpuKernel will run should be defined.
  This document provides instructions for using the existing default TPC kernels **div_fwd_fp32 & div_fwd_bf16** and a custom kernel **customdiv_fwd_f32** to implement CustomOp. For further information on how to write TPC kernels, refer to [Habana® Gaudi® TPC documentation](https://github.com/HabanaAI/Habana_Custom_Kernel).

- Python package installed: **habana-tensorflow**. 

- PyTest for testing CustomOp library only. 

## Content

- C++ file with CustomDivOp definition and Kernel implementation on HPU:
    - CustomDivOp performs a division on input parameters.
    - As an addition, it accepts input attribute **example_attr** only to demonstrate how it can be retrieved in C++. 
- CMake file for building the solution:
    - It looks for TensorFlow using Python3 and sets compiler/linker flags.
    - It looks for Habana-TensorFlow using Python3 and sets include directory and compiler/linker flags.
- Python test to run and validate CustomDivOp
    - Pytest is configured to accept a command line argument:
        - **custom_op_lib** - a location of built library.

## Build and Run CustomDivOp with Default Kernels div_fwd_fp32 and div_fwd_bf16
```bash
mkdir build/ && cd build/
cmake .. && make
pytest -v --custom_op_lib lib/libhpu_custom_div_op.so ../test_custom_div_op.py
```
If all steps were successful, the environment is set up properly.

## Build and Run CustomDivOp with Custom Kernel customdiv_fwd_f32
1. Refer to https://github.com/HabanaAI/Habana_Custom_Kernel for custom kernel build and generate the custom kernel binary **libcustom_tpc_perf_lib.so**.
2. Remove build directory if it exists `rm -rf build`
3. Build solution:
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

## Important to Know

Implementing a single Op is mostly not enough in order to use it in a *trainable* topology.

During training, TensorFlow injects Gradient Ops to all trainable Ops defined. In this case, there is no Gradient defined, therefore the Op cannot be used in a topology.
To make it work, **CustomDivGradOp** should be defined and registered in Python with [tf.RegisterGradient](https://www.tensorflow.org/api_docs/python/tf/RegisterGradient) method.

## Examples of Applying CustomOps to a Real Training Model

This section gives an example for applying CustomOps to a real training model mobilenetv2. 
The steps are as follows:

1. Apply the patch custom_relu6_op.patch to replace the tf.nn.relu6 with custom_relu6 in mobilenetv2 model:
   - Go to main directory in the repository.
   - `git apply --verbose TensorFlow/examples/custom_op/custom_relu6_op.patch`.

2. Build the custom relu6 and relu6Grad Ops with the default kernel relu6_fwd_ and relu6_bwd_:  
   - `cd TensorFlow/examples/custom_op/`
   - `mkdir build/ && cd build/`
   - `cmake -DUSE_CUSTOM_KERNEL=1 .. && make`
CustomOps can be also generated based on custom kernels following the above guidance in [Steps to build and run CustomDivOp with custom kernel customdiv_fwd_f32](https://github.com/HabanaAI/Habana_Custom_Kernel).

3. If build steps were successful, follow the instruction in `Model-References/TensorFlow/computer_vision/mobilenetv2/research/slim/README.md` to try running the mobilenevt2 model using CustomOps CustomRelu6Op and CustomRelu6GradOp.
