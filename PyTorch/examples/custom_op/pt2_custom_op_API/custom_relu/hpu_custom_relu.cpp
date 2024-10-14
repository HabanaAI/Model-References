/*******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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
#include <torch/extension.h>
#include "hpu_custom_op_pt2.h"
#include <perf_lib_layer_params.h>

// Callback describing output tensors parameters.
static habana::PartialOutputMetaDataVector output_meta(
    const at::Stack& inputs) {
  habana::PartialOutputMetaData meta_output{
      c10::ScalarType::Float, inputs[0].toTensor().sizes().vec()};
  return {meta_output};
}

// Callback describing TPC kernel parameters.
static std::shared_ptr<void> fill_params(
    const at::Stack& inputs,
    size_t& size) {
  HPU_PARAMS_STUB(ns_ReluKernel::Params);
    params->threshold.f = 0.0;
    return params;
}

// Registration of custom_relu in Habana registry.
bool register_custom_relu() {
  habana::custom_op::registerUserCustomOp(
      "custom_op::custom_relu", "custom_relu_fwd_f32_gaudi2", output_meta, fill_params);
  habana::custom_op::registerUserCustomOp(
      "custom_op::custom_relu_backward", "custom_relu_bwd_f32_gaudi2", output_meta, nullptr);
  return true;
}

// Call to the registered operator.
at::Tensor custom_relu(at::Tensor input_a) {
  TORCH_CHECK(input_a.scalar_type() == c10::ScalarType::Float, "Input input_a expected to be Float tensor");

  // Get custom op descriptor from Habana registry.
  auto op_desc =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          "custom_op::custom_relu");
  std::vector<c10::IValue> inputs{input_a};
  // Actual call for op execution.
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  // Output is always a vector of at::Tensor.
  return output[0];
}

// Call to the registered operator.
at::Tensor custom_relu_backward(at::Tensor input_a,
    at::Tensor input_b,
    c10::Scalar threshold) {
  TORCH_CHECK(input_a.scalar_type() == c10::ScalarType::Float, "Input input_a expected to be Float tensor");
  TORCH_CHECK(input_b.scalar_type() == c10::ScalarType::Float, "Input input_b expected to be Float tensor");
  TORCH_CHECK(threshold.to<float>() == 0.0, "Threshold values other than 0 are not supported")

  // Get custom op descriptor from Habana registry.
  auto op_desc =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          "custom_op::custom_relu_backward");
  std::vector<c10::IValue> inputs{input_a, input_b, threshold};
  // Actual call for op execution.
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  // Output is always a vector of at::Tensor.
  return output[0];
}

TORCH_LIBRARY(custom_op, m) {
  m.def("custom_relu(Tensor self) -> Tensor");
  m.def("custom_relu_backward(Tensor grad, Tensor self, Scalar threshold) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("custom_relu", custom_relu);
  m.impl("custom_relu_backward", custom_relu_backward);
}

// Meta kernel implementation describing output tensors.
// Needed only for torch.compile support.
at::Tensor custom_relu_meta(at::Tensor input_a) {
  return input_a.new_empty(input_a.sizes().vec(), c10::ScalarType::Float);
}
at::Tensor custom_relu_backward_meta(at::Tensor input_a,
    at::Tensor,
    c10::Scalar) {
  return input_a.new_empty(input_a.sizes().vec(), c10::ScalarType::Float);
}

TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
  m.impl("custom_relu", custom_relu_meta);
  m.impl("custom_relu_backward", custom_relu_backward_meta);
}

// Registration of custom_relu in Habana registry.
static const auto& KernelReg = register_custom_relu();
