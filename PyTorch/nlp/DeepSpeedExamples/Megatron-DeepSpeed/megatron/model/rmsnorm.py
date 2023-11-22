# coding=utf-8
# Copyright (c) 2023, Habana Labs, Ltd. an Intel Company

import torch
from torch.nn import init
from torch.nn.parameter import Parameter
from megatron import get_args
from megatron.global_vars import get_current_device


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6, sequence_parallel=False):
        super().__init__()
        args = get_args()
        self.epsilon = eps
        # Create weight parameter on device otherwise 'sequence_parallel' attribute is not passed to deepspeed
        self.weight = Parameter(torch.empty(dim, device=get_current_device(), dtype=args.params_dtype))
        init.ones_(self.weight)

        if sequence_parallel:
            # set sequence parallelism flag on weight parameter
            setattr(self.weight, 'sequence_parallel', True)

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        norm = torch.mean(x**2, -1, keepdim=True)
        norm = x.mul(norm.add_(self.epsilon).rsqrt_())
        return self.weight * norm.to(dtype)
