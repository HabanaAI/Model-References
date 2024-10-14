# ******************************************************************************
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ******************************************************************************

import torch
import os
import habana_frameworks.torch.core
from pathlib import Path

my_dir = os.path.realpath(__file__)
my_len = my_dir.rfind("/")
base_dir = my_dir[:my_len]

custom_relu_op_lib_path = str(
    next(
        Path(
            next(Path(os.path.join(base_dir, "build")).glob("lib.linux-x86_64-*"))
        ).glob("hpu_custom_relu.cpython-*-x86_64-linux-gnu.so")
    )
)
torch.ops.load_library(custom_relu_op_lib_path)


class CustomReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        # ctx is a context object that can be used to stash information
        # for backward computation
        tensor = torch.ops.custom_op.custom_relu(inputs)
        ctx.tensor = tensor
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None
        tensor = ctx.tensor
        ctx.tensor = None  # to free the memory
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        result = torch.ops.custom_op.custom_relu_backward(grad_output, tensor, 0.0)
        return result


class CustomReLU(torch.nn.Module):
    def __init__(self):
        super(CustomReLU, self).__init__()

    def forward(self, input):
        return CustomReLUFunction.apply(input)

    def extra_repr(self):
        return "CustomReLU for float32 only"
