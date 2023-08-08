###############################################################################
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
###############################################################################
import argparse
import os
import time

# default config for autocast, must be set before importing torch
if "LOWER_LIST" not in os.environ:
    os.environ['LOWER_LIST'] = os.path.dirname(__file__) + "/../ops_bf16.txt"
if "FP32_LIST" not in os.environ:
    os.environ['FP32_LIST'] = os.path.dirname(__file__) + "/../ops_fp32.txt"

import torch
import numpy as np
import open_clip
from omegaconf import OmegaConf
from PIL import Image
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpmpp_2m import DPMPP2M_Sampler

torch.set_grad_enabled(False)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, device, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if device == "cuda":
        model.cuda()
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a professional photograph of an astronaut riding a horse",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=3,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file, separated by newlines",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v2-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--k_sampler",
        type=str,
        choices=["dpmpp_2m"],
        default=""
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="repeat each prompt in file this often",
    )
    parser.add_argument(
        '--device',
        type=str,
        default="hpu",
        help='the device to use',
        choices=['cpu', 'cuda', 'hpu'])
    parser.add_argument(
        '--use_hpu_graph',
        action='store_true',
        help="use HPU Graphs"
    )
    parser.add_argument(
        "--softmax_flavor",
        type=str,
        default="1",
        help="only on HPU: 0 is equivalent to nn.Softmax while 1 is its optimized version",
    )
    opt = parser.parse_args()
    return opt


class DeviceRunner(object):
    def __init__(self, device, use_hpu_graph = True):
        super().__init__()
        self.hpu_cache = {}
        self.device = device
        if device.type != "hpu":
            use_hpu_graph = False
        else:
            import habana_frameworks.torch.hpu.graphs as hpu_graphs
            self.copy_to = hpu_graphs.copy_to
            self.CachedParams = hpu_graphs.CachedParams
        self.use_hpu_graph = use_hpu_graph

    def run(self, func, arg):
        if self.use_hpu_graph:
            func_id = hash(func)
            if func_id in self.hpu_cache:
                self.copy_to(self.hpu_cache[func_id].graph_inputs, arg)
                self.hpu_cache[func_id].graph.replay()
                return self.hpu_cache[func_id].graph_outputs
            str = "Compiling HPU graph {:26s} ".format(func.__name__)
            print(str)
            t_start = time.time()
            import habana_frameworks.torch.hpu as hpu
            graph = hpu.HPUGraph()
            graph.capture_begin()
            out = func(arg)
            graph.capture_end()
            self.hpu_cache[func_id] = self.CachedParams(arg, out, graph)
            print("{} took {:6.2f} sec".format(str, time.time() - t_start))
            return out
        elif self.device.type == "hpu":
            import habana_frameworks.torch.core as core
            core.mark_step()
        return func(arg)


def main(opt):
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}", opt.device)

    if opt.device == "hpu" and opt.precision == "autocast":
        for _, param in model.named_parameters():
            param.data = param.data.to(torch.bfloat16)
        os.environ["CUSTOM_SOFTMAX_FLAVOR"] = opt.softmax_flavor

    device = torch.device(opt.device)
    runner = DeviceRunner(device, opt.use_hpu_graph)
    model = model.to(device)

    if opt.k_sampler:
        try:
            v_mode = config["model"]["params"]["parameterization"] == 'v'
        except KeyError:
            v_mode = False
        sampler = DPMPP2M_Sampler(model, v_mode)
    else:
        sampler = DDIMSampler(model)
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = [p for p in data for i in range(opt.repeat)]
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    sample_count = 0
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    precision_scope = autocast if opt.precision == "autocast" else nullcontext

    sampler.compile(opt.steps, [opt.C, opt.H // opt.f, opt.W // opt.f],
                    batch_size=opt.n_samples, eta=opt.ddim_eta, unconditional_guidance_scale=opt.scale
                    )

    data = opt.n_iter * data
    data.append(data[-1])  # due to pipelining we need to run loop N+1 times
    x_out_cpu = torch.Tensor()
    all_samples = list()
    cpu_device = torch.device('cpu')
    x_shape = [opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f]

    t_start = list()
    with torch.no_grad(), precision_scope(opt.device), model.ema_scope():
        for i, prompts in enumerate(data):
            t_start.append(time.time())
            tokens = open_clip.tokenize(batch_size * [""] + prompts).to(device)
            x = torch.randn(x_shape, device=cpu_device) * sampler.rand_scale
            x = x.clone().to(device)
            c_in = runner.run(model.cond_stage_model.encode_with_transformer, tokens)
            vars = sampler.init_loop(x, c_in)
            for _ in range(opt.steps):
                vars = runner.run(sampler.sampler_step, vars)
            x_out = runner.run(model.decode_first_stage, vars[0])

            # The code below is pipelined with device
            x_out_cpu = x_out_cpu.to(torch.float32)
            x_out_cpu = torch.clamp((x_out_cpu + 1.0) / 2.0, min=0.0, max=1.0)
            x_out_cpu *= 255.0
            for x_sample in x_out_cpu:
                x_sample = rearrange(x_sample.numpy(), 'c h w -> h w c')
                img = Image.fromarray(x_sample.astype(np.uint8))
                path = os.path.join(sample_path, f"{base_count:05}.png")
                img.save(path)
                base_count += 1
                sample_count += 1
            if len(x_out_cpu) > 0:
                all_samples.append(x_out_cpu)
            x_out_cpu = x_out.cpu()  # synchronization point CPU & device

            print("Batch {:3d} took {:9.1f} ms".format(
                i, (time.time() - t_start[i]) * 1000))
    t_end = time.time()

    # additionally, save as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=n_rows)
    # to image
    grid = rearrange(grid, 'c h w -> h w c').to(torch.float32).numpy()
    grid = Image.fromarray(grid.astype(np.uint8))
    grid.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
    grid_count += 1

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")

    # We skip first two batches (graph compilation and initialization)
    if len(data) > 3:
        t = t_end - t_start[2]
        n = (len(t_start) - 2)
        print("Initialization & first two batches took {:.2f} sec".format(
            t_start[2]-t_start[0]))
        print("Generated {} images in {:.2f} sec".format(n * batch_size, t))
        print("PERFORMANCE: batch_size = {}, throughput = {:.3f} images / sec, latency = {:.2f} ms".format(
            batch_size, n * batch_size / t, t/n*1000.0))

    if opt.device == "hpu":
        from habana_frameworks.torch.hpu.memory import max_memory_allocated, max_memory_reserved
        GB = 1024**3
        max_in_use = max_memory_allocated(device) / GB
        limit = max_memory_reserved(device) / GB

        print("HPU memory usage: {:.1f} GB / {:.1f} GB ({:.0f}%)".format(
            max_in_use, limit, max_in_use / limit * 100.0))


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
