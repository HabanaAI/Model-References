###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################
"""SAMPLING ONLY."""

import torch
import k_diffusion as K
import torch.nn as nn

import habana_compat
import habana_frameworks.torch as ht
import hpu_graph_utils


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale

class KDiffusionSampler(object):
    def __init__(self, model, sampler_name, v_mode = False, **kwargs):
        super().__init__()
        self.model = model
        if v_mode:
            self.model_wrap = K.external.CompVisVDenoiser(model)
        else:
            self.model_wrap = K.external.CompVisDenoiser(model)
        self.sampler_name = sampler_name

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None,
               ucg_schedule=None,
               use_hpu_graph=False,
               **kwargs
               ):
        sigmas = self.model_wrap.get_sigmas(S)
        x_T = torch.randn([batch_size, *shape], device=torch.device('cpu')) * sigmas[0]
        x_T = torch.tensor(x_T, device=self.model.betas.device).clone().detach()
        model_wrap_cfg = CFGDenoiser(self.model_wrap)
        if use_hpu_graph:
            model_wrap_cfg = ht.hpu.wrap_in_hpu_graph_func(model_wrap_cfg)
        extra_args = {'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': unconditional_guidance_scale}
        samples = K.sampling.__dict__[f'sample_{self.sampler_name}'](model_wrap_cfg, x_T, sigmas, extra_args=extra_args)
        habana_compat.mark_step()

        return samples, None
