/*******************************************************************************
 * Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */
#include <hpu_kernel.h>

#include <tensorflow/core/framework/common_shape_fns.h> // tensorflow::shape_inference::UnchangedShape
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/lib/bfloat16/bfloat16.h>
#include <tensorflow/core/lib/core/error_codes.pb.h>
#include <tensorflow/core/platform/default/logging.h>

#include <iostream>


REGISTER_OP("CustomRelu6Op")
    .Input("input1: T")
    .Output("output: T")
    .Attr("T: {float}")
    .SetShapeFn(tensorflow::shape_inference::UnchangedShape);

REGISTER_OP("CustomRelu6GradOp")
    .Input("input1: T")
    .Input("input2: T")
    .Output("output: T")
    .Attr("T: {float}")
    .SetShapeFn(tensorflow::shape_inference::UnchangedShape);

template <class DType> class CustomRelu6Functor : public habana::BaseFunctor {
public:
  void DefineNode(habana::HpuKernelContext *context) {
    habana::TensorInfo in0;
    OP_REQUIRES_OK(context, context->InputInfo(0, in0));

    if (in0.dtype != tensorflow::DataType::DT_BFLOAT16 && in0.dtype != tensorflow::DataType::DT_FLOAT) {
      context->SetStatus(
          tensorflow::Status(tensorflow::error::Code::UNAVAILABLE,
                             "This kernel is implemented only for fp32."));
    }

    OP_REQUIRES_OK(context, context->DefineOutputInfo(
                                0, habana::TensorInfo{in0.dtype, in0.shape}));

    // This is the core part of the kernel implementation.
    // TPC is defined with 'div_fwd_' name with appropriate suffix.
    // There are no custom parameters associated with this kernel.
    // TPC will be connected to input indices 0, 1. NOTE: Order of indices
    // matters, if it's {1, 0}, TPC would divide input2/input1 (see input names
    // in OP_REGISTER) TPC will be connected to output index 0.
    std::string tpc_suffix = "f32";
    OP_REQUIRES_OK(
        context,
        context->DefineNode(habana::NodeDesc{.guid = "relu6_fwd_" + tpc_suffix,
                                             .user_params = nullptr,
                                             .params_size = 0,
                                             .name = "custom_relu6"},
                            std::vector<int>{0}, std::vector<int>{0}));

  }
};

template <class DType> class CustomRelu6GradFunctor : public habana::BaseFunctor {
public:
  void DefineNode(habana::HpuKernelContext *context) {
    habana::TensorInfo in0, in1;
    OP_REQUIRES_OK(context, context->InputInfo(0, in0));

    if (in0.dtype != tensorflow::DataType::DT_BFLOAT16 && in0.dtype != tensorflow::DataType::DT_FLOAT) {
      context->SetStatus(
          tensorflow::Status(tensorflow::error::Code::UNAVAILABLE,
                             "This kernel is implemented only for fp32."));
    }

    OP_REQUIRES_OK(context, context->DefineOutputInfo(
                                0, habana::TensorInfo{in0.dtype, in0.shape}));

    // This is the core part of the kernel implementation.
    // TPC is defined with 'div_fwd_' name with appropriate suffix.
    // There are no custom parameters associated with this kernel.
    // TPC will be connected to input indices 0, 1. NOTE: Order of indices
    // matters, if it's {1, 0}, TPC would divide input2/input1 (see input names
    // in OP_REGISTER) TPC will be connected to output index 0.
    std::string tpc_suffix = "f32";
    OP_REQUIRES_OK(
        context,
        context->DefineNode(habana::NodeDesc{.guid = "relu6_bwd_" + tpc_suffix,
                                             .user_params = nullptr,
                                             .params_size = 0,
                                             .name = "custom_relu6grad"},
                            std::vector<int>{0, 1}, std::vector<int>{0}));

  }
};

REGISTER_KERNEL_BUILDER(
    ForHpuWithName("CustomRelu6Op").TypeConstraint<float>("T"),
    habana::HpuKernel<CustomRelu6Functor<float>>);
REGISTER_KERNEL_BUILDER(
    ForHpuWithName("CustomRelu6Op").TypeConstraint<tensorflow::bfloat16>("T"),
    habana::HpuKernel<CustomRelu6Functor<tensorflow::bfloat16>>);
REGISTER_KERNEL_BUILDER(
    ForHpuWithName("CustomRelu6GradOp").TypeConstraint<float>("T"),
    habana::HpuKernel<CustomRelu6GradFunctor<float>>);
REGISTER_KERNEL_BUILDER(
    ForHpuWithName("CustomRelu6GradOp").TypeConstraint<tensorflow::bfloat16>("T"),
    habana::HpuKernel<CustomRelu6GradFunctor<tensorflow::bfloat16>>);
