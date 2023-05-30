###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################
import torch


class Sampler(object):
    def __init__(self, **kwargs):
        super().__init__()

    @torch.no_grad()
    def compile(self, S, shape, **kwargs):
        pass

    def run_model(self, x, c_in, sigma):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        uncond, cond = self.model_wrap(x_in, sigma_in, cond=c_in).chunk(2)
        return uncond + (cond - uncond) * self.cond_scale
