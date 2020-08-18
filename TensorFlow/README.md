# TensorFlow Models for Gaudi

This is the overview of the TensorFlow Model References on Gaudi. The current TensorFlow version supported is 2.2. Users need to convert their models to TensorFlow2 if they are currently based on TensorFlow V1.x, or run in compatibility mode.  Users can refer to the [TensorFlow User Guide](https://docs.habana.ai/projects/SynapeAI-Gaudi/en/latest/Tensorflow_User_Guide/Tensorflow_User_Guide.html) to learn how to migrate existing models to run on Gaudi.

## ResNets
Both ResNet v1.5 and ResNeXt models are a modified version of the original ResNet v1 model. It supports layers 50, 101, and 152.  SSD-Resnet34 is a Single Shot Detection (SSD) (Liu et al., 2016) with backbone ResNet-34 trained with COCO2017 dataset on Gaudi. It is based on the MLPerf training 0.6 implementation by Google. The model provides output as bounding boxes.

## Mask R-CNN
The Mask R-CNN model for Tensorflow is an optimized version of the implementation in Matterport for Gaudi

## BERT Base  (Fine tuning)
BERT base Fine Tuning is verified on both MRPC and SQuAD dataset using BF16 data type.

## BERT Large  (Fine tuning)
BERT Large Fine Tuning is run with two datasets:
* Microsoft Research Paraphrase Corpus (MRPC) is a paraphrase identification dataset, where systems aim to identify if two sentences are paraphrases of each other;
* Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.

This model Uses AdamW ("ADAM with Weight Decay Regularization") optimizer and is based on model weights trained with pretraining.

## BERT Large Pre-training
The BERT Large Pre-training uses optimizer: LAMB ("Layer-wise Adaptive Moments optimizer for Batch training") and Consists of 2 phases:
Phase 1 - Masked Language Model - where given a sentence, a randomly chosen word is guessed.
Phase 2 - Next Sentence Prediction - where the model guesses whether sentence B comes after sentence A
The resulting (trained) model weights are language-specific (here: english) and has to be further "fitted" to do a specific task (with finetuning).