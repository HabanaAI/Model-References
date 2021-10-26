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
#ifdef INCLUDE_FROM_WHEEL
#include <habanalabs/hpu_kernel.h>
#else
#include <hpu_kernel.h>
#endif

#include <tensorflow/core/framework/common_shape_fns.h> // tensorflow::shape_inference::UnchangedShape
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/lib/bfloat16/bfloat16.h>
#include <tensorflow/core/lib/core/error_codes.pb.h>
#include <tensorflow/core/platform/default/logging.h>

REGISTER_OP("CustomDivOp")
    .Input("input1: T")
    .Input("input2: T")
    .Output("output: T")
    .Attr("T: {bfloat16, float}")
    .Attr("example_attr: int")
    .SetShapeFn(tensorflow::shape_inference::UnchangedShape);

template <class DType> class CustomDivFunctor : public habana::BaseFunctor {
public:
  void DefineNode(habana::HpuKernelContext *context) {
    habana::TensorInfo in0, in1;
    OP_REQUIRES_OK(context, context->InputInfo(0, in0));
    OP_REQUIRES_OK(context, context->InputInfo(1, in1));
    // Validate inputs
    OP_REQUIRES(context, in0.shape == in1.shape,
                tensorflow::errors::InvalidArgument(
                    "Shapes of both inputs do not match. Input1.shape: ",
                    in0.shape, ", Input2.shape: ", in1.shape));
    OP_REQUIRES(context, in0.dtype == in1.dtype,
                tensorflow::errors::InvalidArgument(
                    "Dtypes of both inputs do not match. Input1.dtype: ",
                    in0.dtype, ", Input2.dtype: ", in1.dtype));

    if (in0.dtype != tensorflow::DataType::DT_BFLOAT16 &&
        in0.dtype != tensorflow::DataType::DT_FLOAT) {
      context->SetStatus(
          tensorflow::Status(tensorflow::error::Code::UNAVAILABLE,
                             "This kernel is implemented only for bf16&fp32."));
    }

    // just an example of retrieving the Attribute of a kernel
    int example_attr;
    OP_REQUIRES_OK(context, context->GetAttr("example_attr", &example_attr));
    VLOG(tensorflow::INFO) << "Received example attribute with value: "
                           << example_attr;

    // each kernel is responsible for defining all its outputs
    // this needs to be called before context->DefineNode() in order to connect
    // output 0 to TPC
    OP_REQUIRES_OK(context, context->DefineOutputInfo(
                                0, habana::TensorInfo{in0.dtype, in0.shape}));

    // This is the core part of the kernel implementation.
    // TPC is defined with 'div_fwd_' name with appropriate suffix.
    // There are no custom parameters associated with this kernel.
    // TPC will be connected to input indices 0, 1. NOTE: Order of indices
    // matters, if it's {1, 0}, TPC would divide input2/input1 (see input names
    // in OP_REGISTER) TPC will be connected to output index 0.
#ifdef USE_CUSTOM_KERNEL
    OP_REQUIRES_OK(context, context->DefineNode(
                                habana::NodeDesc{.guid = "customdiv_fwd_f32",
                                                 .user_params = nullptr,
                                                 .params_size = 0,
                                                 .name = "custom_div"},
                                std::vector<int>{0, 1}, std::vector<int>{0}));
#else
    std::string tpc_suffix =
        (in0.dtype == tensorflow::DataType::DT_FLOAT) ? "f32" : "bf16";
    OP_REQUIRES_OK(
        context,
        context->DefineNode(habana::NodeDesc{.guid = "div_fwd_" + tpc_suffix,
                                             .user_params = nullptr,
                                             .params_size = 0,
                                             .name = "custom_div"},
                            std::vector<int>{0, 1}, std::vector<int>{0}));
#endif
  }
};

REGISTER_KERNEL_BUILDER(
    ForHpuWithName("CustomDivOp").TypeConstraint<float>("T"),
    habana::HpuKernel<CustomDivFunctor<float>>);
#ifndef USE_CUSTOM_KERNEL
REGISTER_KERNEL_BUILDER(
    ForHpuWithName("CustomDivOp").TypeConstraint<tensorflow::bfloat16>("T"),
    habana::HpuKernel<CustomDivFunctor<tensorflow::bfloat16>>);
#endif
