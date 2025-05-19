# ******************************************************************************
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ******************************************************************************

import os; os.environ["PT_HPU_LAZY_MODE"] = "1"
import torch
from custom_relu import CustomReLU

def test_custom_relu_op_function():
    print(torch.ops.custom_op.custom_relu)
    print(torch.ops.custom_op.custom_relu_backward)
    input = torch.randn(3, 5, requires_grad=True)
    input_hpu = input.to('hpu').detach()
    input_hpu.requires_grad = True
    relu = torch.nn.ReLU(inplace=False)
    output_cpu = relu(input)
    out = torch.ones_like(output_cpu)
    out_hpu = out.to('hpu')
    output_cpu.backward(out)
    out_bwd_cpu = input.grad
    custom_relu = CustomReLU()
    output_hpu = custom_relu(input_hpu);
    output_hpu.backward(out_hpu)
    out_bwd_hpu = input_hpu.grad
    assert(torch.equal(output_hpu.detach().cpu(), output_cpu.detach()))
    assert(torch.equal(out_bwd_hpu.detach().cpu(), out_bwd_cpu.detach()))

test_custom_relu_op_function()

