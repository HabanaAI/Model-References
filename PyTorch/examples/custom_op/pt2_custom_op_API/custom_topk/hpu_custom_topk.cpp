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
#include <synapse_common_types.hpp>

// Callback describing output tensors parameters.
static habana::PartialOutputMetaDataVector output_meta(
    const at::Stack& inputs) {
  auto self = inputs[0].toTensor();
  auto k = inputs[1].toInt();
  auto dim = inputs[2].toInt();
  std::vector<int64_t> output_shape = self.sizes().vec();
  if (output_shape.size() > 0) {
    output_shape[dim] = k;
  }
  habana::PartialOutputMetaData meta_output{
      c10::ScalarType::Float, output_shape};
  habana::PartialOutputMetaData meta_indices{
      c10::ScalarType::Long, output_shape};
  return {meta_output, meta_indices};
}

// Callback describing TPC kernel parameters.
static std::shared_ptr<void> fill_params(
    const at::Stack& inputs,
    size_t& size) {
  HPU_PARAMS_STUB(synBeamParams);
  auto self = inputs[0].toTensor();
  params->bsw = inputs[1].toInt();
  auto dim = inputs[2].toInt();
  params->axis = self.dim() - dim - 1;
  params->bottomK = inputs[3].toBool();
  return params;
}

// Registration of custom_topk in Habana registry.
bool register_custom_topk() {
  habana::custom_op::registerUserCustomOp(
      "custom_op::custom_topk", "topk", output_meta, fill_params);
  return true;
}

// Call to the registered operator.
std::tuple<at::Tensor, at::Tensor> custom_topk(
    at::Tensor input_a,
    at::Scalar k,
    at::Scalar axis,
    bool bottom) {
  // Get custom op descriptor from Habana registry.
  auto op_desc =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          "custom_op::custom_topk");
  std::vector<c10::IValue> inputs{input_a, k, axis, bottom};
  // Actual call for op execution.
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  // Output is always a vector of at::Tensor.
  return {output[0], output[1]};
}

// Registration of custom_topk in PyTorch registry.
TORCH_LIBRARY(custom_op, m) {
  m.def(
      "custom_topk(Tensor self, Scalar k, Scalar axis, bool bottom) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("custom_topk", custom_topk);
}

// Meta kernel implementation describing output tensors.
// Needed only for torch.compile support.
std::tuple<at::Tensor, at::Tensor> custom_topk_meta(
    at::Tensor input_a,
    at::Scalar k,
    at::Scalar axis,
    bool bottom) {
  auto output_shape = input_a.sizes().vec();
  if (output_shape.size() > 0) {
    output_shape[axis.toInt()] = k.toInt();
  }
  auto output = input_a.new_empty(output_shape, c10::ScalarType::Float);
  auto indices = input_a.new_empty(output_shape, c10::ScalarType::Long);
  return {output, indices};
}

TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
  m.impl("custom_topk", &custom_topk_meta);
}

// Registration of custom_topk in Habana registry.
static const auto& KernelReg = register_custom_topk();
