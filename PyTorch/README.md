# PyTorch Models for Gaudi

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

This directory contains PyTorch deep learning model examples enabled on Gaudi. SynapseAI currently supports PyTorch v1.5, and all models are based on this version.

There are 3 modes of operation demonstrated in these examples:
1. Eager mode refers to op-by-op execution as defined in standard PyTorch eager mode scripts
2. Graph mode refers to Torchscript based execution as defined in PyTorch
3. Lazy mode refers to deferred execution of graphs, comprising of ops delivered from script op by op like eager mode. It gives eager mode experience with performance on Gaudi.

This directory contains PyTorch deep learning model examples enabled on Gaudi. All supported models are based on PyTorch v1.7.

There are 3 modes of operation demonstrated in these examples:
1. Eager mode refers to op-by-op execution as defined in standard PyTorch eager mode scripts
2. Graph mode refers to Torchscript based execution as defined in PyTorch
3. Lazy mode refers to deferred execution of graphs, comprising of ops delivered from script op by op like eager mode. It gives eager mode experience with performance on Gaudi.

## ResNet50
This model is based on the ResNet model available at https://github.com/pytorch/vision.git.

## ResNext101
This model is based on the ResNext model available at https://github.com/pytorch/vision.git.

## DLRM
This model is based on Facebook's [DLRM](https://github.com/facebookresearch/dlrm)

## BERT
The pretraining model scripts based on https://github.com/NVIDIA/DeepLearningExamples and the fine tuning model scrips are based on https://github.com/huggingface/transformers.