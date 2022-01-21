
// TODO: import exposed file
#include "hpu_custom_op.h"
#include <torch/extension.h>
#include <synapse_common_types.hpp>

bool register_custom_topk() {
    // Registering ustom_op::custom_add
    // inputs desc
    habana::custom_op::InputDesc input_a_desc{
        habana::custom_op::input_type::TENSOR, 0};
    habana::custom_op::InputDesc input_b_desc{
        habana::custom_op::input_type::USER_PARAMS, 1};
    habana::custom_op::InputDesc input_c_desc{
        habana::custom_op::input_type::USER_PARAMS, 2};
    habana::custom_op::InputDesc input_d_desc{
        habana::custom_op::input_type::USER_PARAMS, 3};
    std::vector<habana::custom_op::InputDesc> inputs_desc{
        input_a_desc, input_b_desc, input_c_desc, input_d_desc};

    // output desc
    // output shape callback
    auto output_size_lambda =
        [](const at::Stack& inputs) -> std::vector<int64_t> {
      auto self = inputs[0].toTensor(); // input
      auto k = inputs[1].toInt(); // k
      auto dim = inputs[2].toInt(); // dim
      std::vector<int64_t> result_sizes = self.sizes().vec();
      if (result_sizes.size() > 0) {
        result_sizes[dim] = k;
      }
      return result_sizes;
    };
    habana::custom_op::OutputDesc output_desc{
        0, c10::ScalarType::Float, output_size_lambda};
    habana::custom_op::OutputDesc output_desc_indices{
        1, c10::ScalarType::Long, output_size_lambda};
    std::vector<habana::custom_op::OutputDesc> outputs_desc{
        output_desc, output_desc_indices};

    // user param callback
    auto user_params_lambda = [](const at::Stack& inputs, size_t& size) {
      HPU_PARAMS_STUB(synBeamParams);
      auto self = inputs[0].toTensor(); // input
      params->bsw = inputs[1].toInt(); // k
      auto dim = inputs[2].toInt(); // axis
      params->axis = self.dim() - dim - 1;
      params->bottomK = inputs[3].toBool(); // bottom
      return params;
    };

    // acctual register
    REGISTER_CUSTOM_OP_ATTRIBUTES(
        "custom_op::custom_topk", //schema name
        "topk", // guid
        inputs_desc,
        outputs_desc,
        user_params_lambda);
    std::cout << "cpp registered custom_op::custom_topk\n";
    return true;
}

std::tuple<at::Tensor, at::Tensor> custom_topk_execute(
    torch::Tensor input_a,
    at::Scalar k,
    at::Scalar axis,
    bool bottom) {
  // Registering the custom op, need to be called only once
  static bool registered = register_custom_topk();
  std::vector<c10::IValue> inputs{input_a, k, axis, bottom};
  // Get custom op descriptor from registry
  auto op_desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::custom_topk");
  // Actual call for op execution
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  // op_desc.execute will always return a vector
  return {output[0], output[1]};
}

TORCH_LIBRARY(custom_op, m) {
  m.def("custom_topk(Tensor self, Scalar k, Scalar axis, bool bottom) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("custom_topk", custom_topk_execute);
}
