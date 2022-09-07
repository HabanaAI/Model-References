#!/usr/bin/env python3

###############################################################################
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
###############################################################################

"""Classifier-free guidance sampling from a diffusion model."""

import os
import argparse
from functools import partial
from pathlib import Path

import habana_frameworks.torch.core as htcore
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import trange
import time

from CLIP import clip
from diffusion import get_model, get_models, sampling, utils

import habana_compat

MODULE_DIR = Path(__file__).resolve().parent


def parse_prompt(prompt, default_weight=3.):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', default_weight][len(vals):]
    return vals[0], float(vals[1])


def resize_and_center_crop(image, size):
    fac = max(size[0] / image.size[0], size[1] / image.size[1])
    image = image.resize((int(fac * image.size[0]), int(fac * image.size[1])), Image.LANCZOS)
    return TF.center_crop(image, size[::-1])


def main():
    os.environ['PT_HPU_ENABLE_SPLIT_INFERENCE'] = '1'

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('prompts', type=str, default=[], nargs='*',
                   help='the text prompts to use')
    p.add_argument('--images', type=str, default=[], nargs='*', metavar='IMAGE',
                   help='the image prompts')
    p.add_argument('--batch-size', '-bs', type=int, default=1,
                   help='the number of images per batch')
    p.add_argument('--checkpoint', type=str,
                   help='the checkpoint to use')
    p.add_argument('--device', type=str,
                   help='the device to use')
    p.add_argument('--eta', type=float, default=0.,
                   help='the amount of noise to add during sampling (0-1)')
    p.add_argument('--init', type=str,
                   help='the init image')
    p.add_argument('--method', type=str, default='plms',
                   choices=['ddpm', 'ddim', 'prk', 'plms', 'pie', 'plms2'],
                   help='the sampling method to use')
    p.add_argument('--model', type=str, default='cc12m_1_cfg', choices=['cc12m_1_cfg'],
                   help='the model to use')
    p.add_argument('-n', type=int, default=1,
                   help='the number of images to sample')
    p.add_argument('--seed', type=int, default=0,
                   help='the random seed')
    p.add_argument('--size', type=int, nargs=2,
                   help='the output image size')
    p.add_argument('--starting-timestep', '-st', type=float, default=0.9,
                   help='the timestep to start at (used with init images)')
    p.add_argument('--steps', type=int, default=50,
                   help='the number of timesteps')
    # HPU
    p.add_argument('--lazy_mode', default='True', type=lambda x: x.lower() == 'true',
                   help="""Whether to run model in lazy execution mode (enabled by default).
                   This feature is supported only on HPU device.
                   Any value other than True (case insensitive) disables lazy mode.""")
    # HPU mixed precision
    p.add_argument('--hmp', dest='use_hmp', action='store_true', help='Enable Habana Mixed Precision mode')
    p.add_argument('--hmp-bf16', default='ops_bf16.txt', help='Path to bf16 ops list in hmp O1 mode')
    p.add_argument('--hmp-fp32', default='ops_fp32.txt', help='Path to fp32 ops list in hmp O1 mode')
    p.add_argument('--hmp-opt-level', default='O1', help='Choose optimization level for hmp')
    p.add_argument('--hmp-verbose', action='store_true', help='Enable verbose mode for hmp')
    args = p.parse_args()

    habana_compat.setup_hpu(args)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = get_model(args.model)()
    _, side_y, side_x = model.shape
    if args.size:
        side_x, side_y = args.size
    checkpoint = args.checkpoint
    if not checkpoint:
        checkpoint = MODULE_DIR / f'checkpoints/{args.model}.pth'
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    if device.type == 'cuda':
        model = model.half()
    model = model.to(device).eval().requires_grad_(False)
    clip_model_name = model.clip_model if hasattr(model, 'clip_model') else 'ViT-B/16'
    clip_model = clip.load(clip_model_name, jit=False, device=device)[0]
    clip_model.eval().requires_grad_(False)
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    if args.init:
        init = Image.open(utils.fetch(args.init)).convert('RGB')
        init = resize_and_center_crop(init, (side_x, side_y))
        init = utils.from_pil_image(init).to(device)[None].repeat([args.n, 1, 1, 1])

    zero_embed = torch.zeros([1, clip_model.visual.output_dim], device=device)
    target_embeds, weights = [zero_embed], []

    for prompt in args.prompts:
        txt, weight = parse_prompt(prompt)
        target_embeds.append(clip_model.encode_text(clip.tokenize(txt).to(device)).float())
        weights.append(weight)

    for prompt in args.images:
        path, weight = parse_prompt(prompt)
        img = Image.open(utils.fetch(path)).convert('RGB')
        clip_size = clip_model.visual.input_resolution
        img = resize_and_center_crop(img, (clip_size, clip_size))
        batch = TF.to_tensor(img)[None].to(device)
        embed = F.normalize(clip_model.encode_image(normalize(batch)).float(), dim=-1)
        target_embeds.append(embed)
        weights.append(weight)

    weights = torch.tensor([1 - sum(weights), *weights], device=device)

    torch.manual_seed(args.seed)

    def cfg_model_fn(x, t):
        n = x.shape[0]
        n_conds = len(target_embeds)
        x_in = x.repeat([n_conds, 1, 1, 1])
        t_in = t.repeat([n_conds])
        clip_embed_in = torch.cat([*target_embeds])
        if n>1:
            clip_embed_in = clip_embed_in.repeat_interleave(n, 0, output_size=clip_embed_in.dim() * n)
        vs = model(x_in, t_in, clip_embed_in).view([n_conds, n, *x.shape[1:]])
        v = vs.mul(weights[:, None, None, None, None]).sum(0)
        return v

    def run(x, steps):
        if args.method == 'ddpm':
            return sampling.sample(cfg_model_fn, x, steps, 1., {})
        if args.method == 'ddim':
            return sampling.sample(cfg_model_fn, x, steps, args.eta, {})
        if args.method == 'prk':
            return sampling.prk_sample(cfg_model_fn, x, steps, {})
        if args.method == 'plms':
            return sampling.plms_sample(cfg_model_fn, x, steps, {})
        if args.method == 'pie':
            return sampling.pie_sample(cfg_model_fn, x, steps, {})
        if args.method == 'plms2':
            return sampling.plms2_sample(cfg_model_fn, x, steps, {})
        assert False

    def run_all(n, batch_size):
        # torch.randn is broken on HPU so running it on CPU
        t = torch.linspace(1, 0, args.steps + 1, device=device)[:-1]
        steps = utils.get_spliced_ddpm_cosine_schedule(t)
        if args.init:
            steps = steps[steps < args.starting_timestep]
            alpha, sigma = utils.t_to_alpha_sigma(steps[0])
            x = init * alpha + x * sigma
        num_batches = n // batch_size
        habana_compat.mark_step()
        for batch in range(num_batches):
            print("Batch {}/{}".format(batch+1, num_batches))
            start_time = time.time()
            x = torch.randn([batch_size, 3, side_y, side_x], device=torch.device('cpu'))
            x = torch.tensor(x, device=device).clone().detach()
            outs = run(x, steps).cpu()
            print("took {:.2f} sec".format(time.time() - start_time))
            for j, out in enumerate(outs):
                utils.to_pil_image(out).save(f'out_{batch*batch_size + j:05}.png')
    try:
        run_all(args.n, args.batch_size)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
