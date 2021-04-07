# TensorFlow Models for Gaudi

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

This is the overview of the TensorFlow Model References on Gaudi. The current TensorFlow version supported is 2.2. Users need to convert their models to TensorFlow2 if they are currently based on TensorFlow V1.x, or run in compatibility mode.  Users can refer to the [TensorFlow User Guide](https://docs.habana.ai/en/latest/Tensorflow_User_Guide/Tensorflow_User_Guide.html) to learn how to migrate existing models to run on Gaudi.

## ResNets
Both ResNet v1.5 and ResNeXt models are a modified version of the original
ResNet v1 model. It supports layers 50, 101, and 152.

## SSD ResNet 34
SSD ResNet 34 is based on the MLPerf training 0.6 implementation by Google.

## Mask R-CNN
The Mask R-CNN model for Tensorflow is an optimized version of the
implementation in Matterport for Gaudi

## DenseNet
The DenseNet 121 model is a modified version of https://github.com/jkjung-avt/keras_imagenet.

## BERT
There are two distinct sets of scripts for exercising BERT training:
* Pre-training is based on a modified version of the BERT model from NVIDIA Deep Learning Examples.
* Fine-tuning is based on a modified version of Google BERT
