# Diffusion Model Details

This directory contains four stable diffusion projects and one v-diffusion project. Each folder contains detailed instructions on how to use them. The stable-diffusion-v-1-5, stable-diffusion-v-2-1 and v-diffusion projects are specifically designed for inference, the stable-diffusion project is suitable for both inference and training and stable-diffusion-finetuning is for only training.

Since Stable-diffusion-v-2-1 is the latest version, it is recommended to consider it as the option for running `inference`. To run `training`, you can use the stable-diffusion project.

### Overview:

* stable-diffusion-v-1-5: inference code is based on https://github.com/CompVis/stable-diffusion/tree/69ae4b35e0a0f6ee1af8bb9a5d0016ccb27e36dc.
* stable-diffusion-v-2-1: is the latest version of stable diffusion (v2.1) and is based on https://github.com/Stability-AI/stablediffusion/tree/d55bcd4d31d0316fcbdf552f2fd2628fdc812500.
* stable-diffusion: is designed for both inference and training, based on the first version of stable diffusion https://github.com/pesser/stable-diffusion/tree/a166aa7fbf578f41f855efeab2e14001d6732563.
* stable-diffusion-finetuning: is designed for training on stable diffusion (v2.1) and is based on https://github.com/cloneofsimo/lora/tree/bdd51b04c49fa90a88919a19850ec3b4cf3c5ecd

Stable diffusion v1.5 utilizes CLIP, while stable-diffusion v2.1 ultilizes openCLIP. In addition, stable-diffusion is based on pretrained weights from ommer-lab, whereas stable-diffusion-v-1-5 uses huggingface.co.

### Supported Configuration
| Project  | SynapseAI Version | Mode |
|:---------|-------------------|-------|
| stable-diffusion-v-1-5  | 1.8.0             | Inference |
| stable-diffusion-v-2-1  | 1.10.0             | Inference |
| stable-diffusion        | 1.10.0             | Training  |
| stable-diffusion        | 1.7.1             | Inference |
| stable-diffusion-finetuning | 1.10.0        | Training  |
| v-diffusion | 1.7.1                         | Inference |
