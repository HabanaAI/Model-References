## PyTorch NLP Models for Gaudi

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

This page will contain a general description on how to use and optmize NLP models for PyTorch on Gaudi.
The NLP models supported are BERT base eager mode for FP32 finetuning, and BERT Large fine tuning and pretraining for FP32 and BF16 mixed precision.
The Pretraining modeling scripts are derived from a clone of https://github.com/NVIDIA/DeepLearningExamples.git and the fine tuning is based on https://github.com/huggingface/transformers.git.