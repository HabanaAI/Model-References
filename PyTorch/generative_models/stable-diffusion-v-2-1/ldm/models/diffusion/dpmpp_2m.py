###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################
import torch
import k_diffusion as K
import numpy as np

from ldm.models.diffusion.sampler import Sampler


class DPMPP2M_Sampler(Sampler):
    def __init__(self, model, v_mode, **kwargs):
        super().__init__()
        self.model = model
        if v_mode:
            self.model_wrap = K.external.CompVisVDenoiser(model)
        else:
            self.model_wrap = K.external.CompVisDenoiser(model)

    def generate_params(self, sigmas):
        """DPM-Solver++(2M)."""
        # Based on https://github.com/crowsonkb/k-diffusion/blob/v0.0.14/k_diffusion/sampling.py#L585
        device = sigmas.device
        sigmas = sigmas.cpu()
        def sigma_fn(t): return t.neg().exp()
        def t_fn(sigma): return sigma.log().neg()
        params = []
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            a = sigma_fn(t_next) / sigma_fn(t)
            if i == 0 or sigmas[i + 1] == 0:
                b = 1.0
                c = 0.0
            else:
                h_last = t - t_fn(sigmas[i - 1])
                r = h_last / h
                b = 1 + 1 / (2 * r)
                c = 1 / (2 * r)
            b *= - (-h).expm1()
            c *= (-h).expm1()
            p = np.array([a.numpy(), b.numpy(), c.numpy(), sigma.numpy()])
            params.append(p)
        params = torch.Tensor(np.stack(params, axis=0)
                              ).transpose(0, 1).to(device)
        return params

    @torch.no_grad()
    def compile(self,
                S,
                shape,
                unconditional_guidance_scale=1.,
                batch_size=1,
                **kwargs
                ):
        self.sigmas = self.model_wrap.get_sigmas(S)
        self.params = self.generate_params(self.sigmas)
        self.cond_scale = unconditional_guidance_scale
        self.old_denoised_zeros = self.sigmas.new_zeros([batch_size] + shape)
        self.rand_scale = self.params[3, 0].to(torch.float32).cpu()
        self.batch_size = batch_size

    def one_step(self, x, c_in, old_denoised, param_t):
        a, b, c, sigma = param_t.chunk(4)
        sigma = sigma.broadcast_to((self.batch_size)).contiguous()
        denoised = self.run_model(x, c_in, sigma)
        x = a * x + b * denoised + c * old_denoised
        return x, denoised

    def sampler_step(self, arg):
        x, c_in, params, old_denoised = arg
        x, denoised = self.one_step(x, c_in, old_denoised, params[:, 0])
        params = torch.roll(params, shifts=-1, dims=1)
        return [x, c_in, params, denoised]

    def init_loop(self, x, c_in):
        return [x, c_in, self.params.clone(), self.old_denoised_zeros.clone()]
