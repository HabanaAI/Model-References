# coding=utf-8
# Copyright (c) 2023, Habana Labs, Ltd. an Intel Company

import torch
from torch.nn import init
from torch.nn.parameter import Parameter

class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6, sequence_parallel=False):
        super().__init__()
        self.epsilon = eps
        self.weight = Parameter(torch.Tensor(dim))
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
