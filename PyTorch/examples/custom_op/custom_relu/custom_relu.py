###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import torch
import os
from habana_frameworks.torch.utils.library_loader import load_habana_module
load_habana_module()

custom_relu_op_lib_path = "./build/lib.linux-x86_64-3.8/hpu_custom_relu.cpython-38-x86_64-linux-gnu.so"
my_dir = os.path.realpath(__file__)
my_len = my_dir.rfind('/')
base_dir = my_dir[:my_len]
torch.ops.load_library(os.path.join(base_dir, custom_relu_op_lib_path))

class CustomReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        # ctx is a context object that can be used to stash information
        # for backward computation
        tensor = torch.ops.custom_op.custom_relu(inputs);
        ctx.tensor = tensor
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None
        tensor = ctx.tensor
        ctx.tensor = None # to free the memory
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        result = torch.ops.custom_op.custom_relu_backward(grad_output, tensor, 0.0);
        return result

class CustomReLU(torch.nn.Module):
    def __init__(self):
        super(CustomReLU, self).__init__()

    def forward(self, input):
        return CustomReLUFunction.apply(input)

    def extra_repr(self):
        return 'CustomReLU for float32 only'

