# Table of Contents
- [NLP Finetuning for PyTorch](#NLP-for-pytorch)

# NLP Finetuning for PyTorch

This folder contains scripts to fine tune language models on Habana Gaudi<sup>TM</sup> device to achieve state-of-the-art accuracy. Please visit [this page](https://developer.habana.ai/resources/habana-training-models/#performance) for performance information.

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

The demos included in this release are as follows:

## BERT Fine Tuning
- BERT Large and BERT base fine tuning for downstream tasks like SQUAD and MRPC using the hugging face transformers package

## DistilBERT Fine Tuning
- DistilBERT base fine tuning for downstream task like SQUAD using the hugging face transformers package

## RoBERTa Fine Tuning
- RoBERTa Large and RoBERTa base fine tuning for downstream task like SQUAD using the hugging face transformers package

## ALBERT Fine Tuning
- ALBERT Large and ALBERT XXLarge fine tuning for downstream task like SQUAD using the hugging face transformers package

The demo script is a wrapper for respective python training scripts. Additional environment variables are used in training scripts in order to achieve optimal results for each workload.
