## PyTorch Vision Models for Gaudi

For more information about training deep learning models on Gaudi, visit
[developer.habana.ai](https://developer.habana.ai/resources/).

This page will contain a general description on how to use and optmize Vision
models for PyTorch on Gaudi. The supported model are ResNet50 and ResNext101 with Imagenet
dataset in FP32 and BF16, in Eager mode & Lazy mode. Multinode support is for 1 HLS & 2 HLS/HLS1H.
ResNet50 & ResNext101 demo included in the official docker image is based on the torch vision
implementation from https://github.com/pytorch/vision.git
(tag:0.8.0).

The demo_resnet.py is a wrapper script for train.py script. The train.py (located
at Model-References/PyTorch/computer-vision/ImageClassification/ResNet) is
modified to add support for Habana devices.
