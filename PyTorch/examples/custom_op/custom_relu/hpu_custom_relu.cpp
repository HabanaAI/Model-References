/******************************************************************************
###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################
*******************************************************************************/

#include "hpu_custom_op.h"
#include <torch/extension.h>
#include <perf_lib_layer_params.h>

bool register_custom_relu() {
    // Registering custom_op::custom_add
    // inputs desc
    habana::custom_op::InputDesc input_a_desc{
        habana::custom_op::input_type::TENSOR, 0};

    std::vector<habana::custom_op::InputDesc> inputs_desc{
        input_a_desc};

    // output desc
    // output shape callback
    auto output_size_lambda =
        [](const at::Stack& inputs) -> std::vector<int64_t> {
      auto self = inputs[0].toTensor(); // input
      std::vector<int64_t> result_sizes = self.sizes().vec();
      return result_sizes;
    };

    habana::custom_op::OutputDesc output_desc{
        0, c10::ScalarType::Float, output_size_lambda};

    std::vector<habana::custom_op::OutputDesc> outputs_desc{
        output_desc};

    // user param callback
    auto user_params_lambda = [](const at::Stack& inputs, size_t& size) {
      HPU_PARAMS_STUB(ns_ReluKernel::Params);
      params->threshold.f = 0.0;
      return params;
    };

    // actual register
    REGISTER_CUSTOM_OP_ATTRIBUTES(
        "custom_op::custom_relu", //schema name
        "relu_fwd_f32", // guid
        inputs_desc,
        outputs_desc,
        user_params_lambda);
    std::cout << "cpp registered custom_op::custom_relu\n";
    return true;
}

bool register_custom_relu_backward() {
    // Registering custom_op::custom_add
    // inputs desc
    habana::custom_op::InputDesc input_a_desc{
        habana::custom_op::input_type::TENSOR, 0};
    habana::custom_op::InputDesc input_b_desc{
        habana::custom_op::input_type::TENSOR, 0};
    habana::custom_op::InputDesc input_c_desc{
        habana::custom_op::input_type::SCALAR, 0};

    std::vector<habana::custom_op::InputDesc> inputs_desc{
        input_a_desc, input_b_desc, input_c_desc};

    // output desc
    // output shape callback
    auto output_size_lambda =
        [](const at::Stack& inputs) -> std::vector<int64_t> {
      auto self = inputs[0].toTensor(); // input
      std::vector<int64_t> result_sizes = self.sizes().vec();
      return result_sizes;
    };

    habana::custom_op::OutputDesc output_desc{
        0, c10::ScalarType::Float, output_size_lambda};

    std::vector<habana::custom_op::OutputDesc> outputs_desc{
        output_desc};

    // actual register
    REGISTER_CUSTOM_OP_ATTRIBUTES(
        "custom_op::custom_relu_backward", //schema name
        "relu_bwd_f32", // guid
        inputs_desc,
        outputs_desc,
        nullptr);
    std::cout << "cpp registered custom_op::custom_relu_backward\n";
    return true;
}

at::Tensor custom_relu_execute(
    torch::Tensor input_a) {
  TORCH_CHECK(input_a.scalar_type() == c10::ScalarType::Float, "Input input_a expected to be Float tensor");
  // Registering the custom op, need to be called only once
  static bool registered = register_custom_relu();
  TORCH_CHECK(registered, "custom_relu kernel not registered" );
  std::vector<c10::IValue> inputs{input_a};
  // Get custom op descriptor from registry
  auto op_desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::custom_relu");
  // Actual call for op execution
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  // op_desc.execute will always return a vector
  return output[0];
}

at::Tensor custom_relu_backward_execute(
    torch::Tensor input_a,
    torch::Tensor input_b,
    c10::Scalar threshold) {
  TORCH_CHECK(input_a.scalar_type() == c10::ScalarType::Float, "Input input_a expected to be Float tensor");
  TORCH_CHECK(input_b.scalar_type() == c10::ScalarType::Float, "Input input_b expected to be Float tensor");
  TORCH_CHECK(threshold.to<float>() == 0.0, "Threshold values other than 0 are not supported")
  // Registering the custom op, need to be called only once
  static bool registered = register_custom_relu_backward();
  TORCH_CHECK(registered, "custom_relu_backward kernel not registered" );
  std::vector<c10::IValue> inputs{input_a, input_b, threshold};
  // Get custom op descriptor from registry
  auto op_desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::custom_relu_backward");
  // Actual call for op execution
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  // op_desc.execute will always return a vector
  return output[0];
}

TORCH_LIBRARY(custom_op, m) {
  m.def("custom_relu(Tensor self) -> Tensor");
  m.def("custom_relu_backward(Tensor grad, Tensor self, Scalar threshold) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("custom_relu", custom_relu_execute);
  m.impl("custom_relu_backward", custom_relu_backward_execute);
}

