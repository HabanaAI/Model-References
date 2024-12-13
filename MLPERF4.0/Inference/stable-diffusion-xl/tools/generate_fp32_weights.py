###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################
# Changes:
# - save fp32 weights locally to avoid double quantization

import os
import habana_frameworks.torch.core
import argparse

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
)
from diffusers.models.lora import LoRACompatibleConv
from diffusers.models.lora import LoRACompatibleLinear

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", help="Path to model weights")
    args = parser.parse_args()
    return args

# To avoid double quantization, we ensure that linear and conv weights which are quantized
# to FP8-143 are quantized directly from FP32, and not from BFloat16.
# To do this, the below function loads the model with dtype FP32, and temporarily saves the linear
# and conv weights. In evaluation we load the model with dtype BFloat16, but replace linear and conv weights
# with these temporarily saved FP32 weights.
def get_unet_weights(model_path):
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    scheduler = EulerDiscreteScheduler.from_pretrained(
                    os.path.join(model_path, "checkpoint_scheduler"),
                    subfolder="scheduler",
                )

    pipe = StableDiffusionXLPipeline.from_pretrained(os.path.join(model_path, "checkpoint_pipe"),
                                                         scheduler=scheduler,
                                                         safety_checker=None,
                                                         add_watermarker=False,
                                                         variant=None,
                                                         torch_dtype=torch.float32)

    quant_model = pipe.unet
    pipe = pipe.to(device)

    temp_dict = {}
    for name, module in quant_model.named_modules():
        mod_type=module.__class__.__name__
        if mod_type in ["Linear","Conv2d","LoRACompatibleLinear","LoRACompatibleConv"]:
            temp_dict.update({name: module.weight.to('cpu')})
    return temp_dict

if __name__ == "__main__":
    args = parse_args()

    params_dict = get_unet_weights(argsi.model_path)
    torch.save(params_dict, 'tools/fp32_weights.pt')
    print('dumped weights')
