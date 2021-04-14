## PyTorch Vision Models for Gaudi

For more information about training deep learning models on Gaudi, visit
[developer.habana.ai](https://developer.habana.ai/resources/).

This page contains a general description on how to use and optimize Vision
models for PyTorch on Gaudi. The supported model is Resnet50 with Imagenet
dataset in FP32 and BF16, in Eager mode and Lazy mode. Multinode support is for 8x Gaudis.
Resnet50 demo included in the official docker image is based on the torch vision
implementation of ResNet50 from https://github.com/pytorch/vision.git
(tag:0.5.0).

The demo_resnet is a wrapper script for train.py script. The train.py (located
at Model-References/PyTorch/computer_vision/ImageClassification/ResNet) is
modified to add support for Habana devices.
