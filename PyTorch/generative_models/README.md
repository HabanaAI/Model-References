# Diffusion Model Details

This directory contains two stable diffusion projects. Each folder contains detailed instructions on how to use them. The stable-diffusion and stable-diffusion-finetuning projects are suitable for training only. 

To run `training`, you can use the stable-diffusion project.

### Overview:

* stable-diffusion: is designed for training, based on the first version of stable diffusion https://github.com/pesser/stable-diffusion/tree/a166aa7fbf578f41f855efeab2e14001d6732563.
* stable-diffusion-finetuning: is designed for training on stable diffusion (v2.1) and is based on https://github.com/cloneofsimo/lora/tree/bdd51b04c49fa90a88919a19850ec3b4cf3c5ecd

### Supported Configuration
| Project  | Intel Gaudi Software Version | Mode | Device |
|:---------|-------------------|-------|-------|
| stable-diffusion        | 1.17.0             | Training  | Gaudi 2 |
| stable-diffusion-finetuning | 1.17.0        | Training  | Gaudi 2 |
