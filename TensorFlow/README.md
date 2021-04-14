# TensorFlow Models for Gaudi

This directory contains TensorFlow deep learning model examples enabled on Gaudi.  SynapseAI currently supports TensorFlow version 2.2. Some of the models here are based on TensorFlow2, while others are based on TensorFlow V1.x and run in compatibility mode.


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
Pre-training is based on modified version of the NVidia implementation of BERT. Fine-tuning is based on Google BERT. Both BERT-Base and BERT_Large models are enabled.
