# Table of Contents
- [Image Classification for Pytorch](#Image-Classification-for-pytorch)

# Image Classification for Pytorch
This folder contains scripts to run Image Classification models on Habana Gaudi<sup>TM</sup> device to achieve state-of-the-art accuracy. Please visit [this page](https://developer.habana.ai/resources/habana-training-models/#performance) for performance information.

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

The demos included in this release are as follows:

## Models for image classification
- ResNet50, ResNet152, ResNext101, MobileNet_V2 and GoogLeNet models from torchvision package

- Swin-B from swin_transformer package

The demo script is a wrapper for respective python training scripts. Additional environment variables are used in training scripts in order to achieve optimal results for each workload.