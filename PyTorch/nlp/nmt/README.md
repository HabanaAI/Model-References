# Table of Contents
- [NMT for Pytorch](#NMT-for-pytorch)

# NMT for Pytorch
This folder contains scripts to run NMT models on Habana Gaudi<sup>TM</sup> device to achieve state-of-the-art accuracy. Please visit [this page](https://developer.habana.ai/resources/habana-training-models/#performance) for performance information.

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

The demos included in this release are as follows:

## Transformer based NMT
- NMT for english to german language translation task using the fairseq transformers package

The demo script is a wrapper for respective python training scripts. Additional environment variables are used in training scripts in order to achieve optimal results for each workload.