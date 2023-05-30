###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################
import torch
import numpy as np

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
from ldm.models.diffusion.sampler import Sampler


class DDIMSampler(Sampler):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.model_wrap = model.apply_model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.rand_scale = 1.0

    def register_buffer(self, name, attr):
        if self.model.device == "cuda":
            if type(attr) == torch.Tensor:
                if attr.device != torch.device("cuda"):
                    attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'

        def to_torch(x): return x.clone().detach().to(
            torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(
            self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(
            np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(
            np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta, verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas',
                             np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps',
                             sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def compile(self,
                S,
                shape,
                batch_size=1,
                eta=0.,
                temperature=1.,
                verbose=False,
                unconditional_guidance_scale=1.,
                use_original_steps=False,
                **kwargs
                ):

        self.steps = S
        self.batch_size = batch_size
        self.shape = shape
        self.eta = eta
        self.temperature = temperature
        self.cond_scale = unconditional_guidance_scale
        self.x_shape = (self.batch_size,
                        self.shape[0], self.shape[1], self.shape[2])

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        self.ts_list = torch.Tensor(
            np.expand_dims(self.ddim_timesteps, axis=0))
        self.ts_list = self.ts_list.fliplr().to(torch.int32).to(self.model.device)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        alphas_prev = torch.Tensor(alphas_prev)

        self.params_init = [
            ('alpha', alphas),
            ('alpha_prev', alphas_prev),
            ('rsqrt(alpha)', alphas.rsqrt()),
            ('sqrt(alpha_prev)', alphas_prev.sqrt()),
            ('sqrt(1-alpha)', sqrt_one_minus_alphas),
            ('sigma', torch.Tensor(sigmas)),
            ('dir', torch.sqrt(1. - alphas_prev - sigmas**2))
        ]

        self.params = torch.stack(list(map(lambda x: x[1], self.params_init)))
        self.params = self.params.fliplr().to(
            self.model.betas.dtype).to(self.model.device)

    def one_step(self, x, c_in, ts_t, param_t):
        ts = ts_t[0].broadcast_to((self.batch_size)).contiguous()

        param = {}
        for idx, val in enumerate(self.params_init):
            param[val[0]] = param_t[idx].broadcast_to(
                (self.batch_size, 1, 1, 1)).contiguous()

        model_output = self.run_model(x, c_in, ts)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, ts, model_output)
        else:
            e_t = model_output

        # current prediction for x_0
        if self.model.parameterization != "v":
            pred_x0 = (x - param['sqrt(1-alpha)'] *
                       e_t) * param['rsqrt(alpha)']
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(
                x, ts, model_output)

        # direction pointing to x_t
        dir_xt = param['dir'] * e_t
        noise = param['sigma'] * \
            noise_like(x.shape, self.model.device, False) * self.temperature
        x = param['sqrt(alpha_prev)'] * pred_x0 + dir_xt + noise
        return x

    def sampler_step(self, arg):
        x, c_in, ts, params = arg
        x = self.one_step(x, c_in, ts[:, 0], params[:, 0])
        ts = torch.roll(ts, shifts=-1, dims=1)
        params = torch.roll(params, shifts=-1, dims=1)
        return [x, c_in, ts, params]

    def init_loop(self, x, c_in):
        return [x, c_in, self.ts_list.clone(), self.params.clone()]
