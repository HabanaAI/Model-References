# Habana Deep Learning Examples for Training and Inference

- [Habana Deep Learning Examples for Training and Inference](#habana-deep-learning-examples-for-training-and-inference)
  - [Model List and Performance Data](#model-list-and-performance-data)
  - [Computer Vision](#computer-vision)
  - [Natural Language Processing](#natural-language-processing)
  - [Audio](#audio)
  - [Generative Models](#generative-models)
  - [MLPerf™ 3.0](#mlperf-30)
  - [Reporting Bugs/Feature Requests](#reporting-bugsfeature-requests)
- [Community](#community)
  - [Hugging Face](#hugging-face)
  - [Fairseq](#fairseq)

## Model List and Performance Data

Please visit [this page](https://developer.habana.ai/resources/habana-training-models/#performance) for performance information.

This repository is a collection of models that have been ported to run on Habana Gaudi AI accelerator. They are intended as examples, and will be reasonably optimized for performance while still being easy to read.

## Computer Vision
| Models                                                                             | Framework         | Validated on Gaudi  | Validated on Gaudi2 |
| ---------------------------------------------------------------------------------- | ----------------- | ------------------- | ------------------- |
| [ResNet50, ResNeXt101](PyTorch/computer_vision/classification/torchvision)         | PyTorch           | Training            | Training, Inference |
| [ResNet50 for PyTorch Lightning](PyTorch/computer_vision/classification/lightning) | PyTorch Lightning | Training            | Training            |
| [ResNet152](PyTorch/computer_vision/classification/torchvision)                    | PyTorch           | Training            | -                   |
| [MobileNetV2](PyTorch/computer_vision/classification/torchvision)                  | PyTorch           | Training            | -                   |
| [UNet 2D, Unet3D](PyTorch/computer_vision/segmentation/Unet)                       | PyTorch Lightning | Training, Inference | Training, Inference |
| [SSD](PyTorch/computer_vision/detection/mlcommons/SSD/ssd)                         | PyTorch           | Training            | Training            |
| [GoogLeNet](PyTorch/computer_vision/classification/torchvision)                    | PyTorch           | Training            | -                   |
| [Vision Transformer](PyTorch/computer_vision/classification/ViT)                   | PyTorch           | Training            | -                   |
| [DINO](PyTorch/computer_vision/classification/dino)                                | PyTorch           | Training            | -                   |
| [YOLOX](PyTorch/computer_vision/detection/yolox)                                   | PyTorch           | Training            | -                   |
| [YOLOV3](PyTorch/computer_vision/detection/openmmlab_detection)                    | PyTorch           | Training            | -                   |
| [ResNet50 Keras](TensorFlow/computer_vision/Resnets/resnet_keras)                  | TensorFlow        | Training            | Training            |
| [ResNeXt101](TensorFlow/computer_vision/Resnets/ResNeXt)                           | TensorFlow        | Training            | Training            |
| [SSD](TensorFlow/computer_vision/SSD_ResNet34)                                     | TensorFlow        | Training            | Training            |
| [Mask R-CNN](TensorFlow/computer_vision/maskrcnn)                                  | TensorFlow        | Training            | Training            |
| [UNet 2D](TensorFlow/computer_vision/Unet2D)                                       | TensorFlow        | Training            | Training            |
| [UNet 3D](TensorFlow/computer_vision/UNet3D)                                       | TensorFlow        | Training            | Training            |
| [DenseNet](TensorFlow/computer_vision/densenet)                                    | TensorFlow        | Training            | -                   |
| [Vision Transformer](TensorFlow/computer_vision/VisionTransformer)                 | TensorFlow        | Training            | -                   |

## Natural Language Processing
| Models                                                                           | Framework  | Validated on Gaudi  | Validated on Gaudi2 |
|----------------------------------------------------------------------------------| ---------- | ------------------- | ------------------- |
| [BERT Pretraining and Finetuning](PyTorch/nlp/bert)                              | PyTorch    | Training, Inference | Training, Inference |
| [DeepSpeed BERT-1.5B, BERT-5B](PyTorch/nlp/DeepSpeedExamples/deepspeed-bert)     | PyTorch    | Training            | -                   |
| [BART](PyTorch/nlp/BART/simpletransformers)                                      | PyTorch    | Training            | -                   |
| [HuggingFace BLOOM](PyTorch/nlp/bloom)                                           | PyTorch    | Inference           | Inference           |
| [Megatron-DeepSpeed BLOOM 13B](PyTorch/nlp/DeepSpeedExamples/Megatron-DeepSpeed) | PyTorch    | -                   | Training            |
| [Megatron-DeepSpeed LLaMA 13B](PyTorch/nlp/DeepSpeedExamples/Megatron-DeepSpeed) | PyTorch    | -                   | Training            |
| [DeepSpeed-Chat](PyTorch/nlp/DeepSpeedExamples/DeepSpeed-Chat)                   | PyTorch    | -                   | Training            |
| [BERT](TensorFlow/nlp/bert)                                                      | TensorFlow | Training            | Training            |
| [Transformer](TensorFlow/nlp/transformer)                                        | TensorFlow | Training            | Training            |


## Audio
| Models                                             | Framework | Validated on Gaudi | Validated on Gaudi2 |
| -------------------------------------------------- | --------- | ------------------ | ------------------- |
| [Wav2Vec2ForCTC](PyTorch/audio/wav2vec2/inference) | PyTorch   | Inference          | Inference           |
| [Hubert](PyTorch/audio/hubert)                     | PyTorch   | -                  | Training            |

## Generative Models
| Models                                                                               | Framework         | Validated on Gaudi  | Validated on Gaudi2 |
| ------------------------------------------------------------------------------------ | ----------------- | ------------------- | ------------------- |
| [V-Diffusion](PyTorch/generative_models/v-diffusion)                                 | PyTorch           | Inference           | -                   |
| [Stable Diffusion](PyTorch/generative_models/stable-diffusion)                       | PyTorch Lightning | Training, Inference | Training, Inference |
| [Stable Diffusion FineTuning](PyTorch/generative_models/stable-diffusion-finetuning) | PyTorch           | Training            | Training            |
| [Stable Diffusion v1.5](PyTorch/generative_models/stable-diffusion-v-1-5)            | PyTorch           | Inference           | Inference           |
| [Stable Diffusion v2.1](PyTorch/generative_models/stable-diffusion-v-2-1)            | PyTorch           | Inference           | Inference           |

## MLPerf&trade; 3.0
| Models                                  | Framework  | Validated on Gaudi | Validated on Gaudi2 |
| --------------------------------------- | ---------- | ------------------ | ------------------- |
| [GPT3](MLPERF3.0/Habana/benchmarks)     | PyTorch    | -                  | Training            |
| [ResNet50](MLPERF3.0/Habana/benchmarks) | PyTorch    | -                  | Training            |
| [BERT](MLPERF3.0/Habana/benchmarks)     | PyTorch    | -                  | Training            |
| [Unet3D](MLPERF3.0/Habana/benchmarks)   | PyTorch    | -                  | Training            |
| [ResNet50](MLPERF3.0/Habana/benchmarks) | TensorFlow | -                  | Training            |
| [BERT](MLPERF3.0/Habana/benchmarks)     | TensorFlow | -                  | Training            |

MLPerf™ is a trademark and service mark of MLCommons Association in the United States and other countries. All rights reserved. Unauthorized use is strictly prohibited.

## Reporting Bugs/Feature Requests

We welcome you to use the [GitHub issue tracker](https://github.com/HabanaAI/Model-References/issues) to report bugs or suggest features.

When filing an issue, please check existing open, or recently closed, issues to make sure somebody else hasn't already
reported the issue. Please try to include as much information as you can. Details like these are incredibly useful:

* A reproducible test case or series of steps
* The version of our code being used
* Any modifications you've made relevant to the bug
* Anything unusual about your environment or deployment

# Community
## Hugging Face
* [ALBERT Large](https://huggingface.co/Habana/albert-large-v2)
* [ALBERT XXLarge](https://huggingface.co/Habana/albert-xxlarge-v1)
* [BERT base](https://huggingface.co/Habana/bert-base-uncased)
* [BERT large](https://huggingface.co/Habana/bert-large-uncased-whole-word-masking)
* [DistilBERT](https://huggingface.co/Habana/distilbert-base-uncased)
* [GPT2](https://huggingface.co/Habana/gpt2)
* [RoBERTa](https://huggingface.co/Habana/roberta-base)
* [RoBERTa large](https://huggingface.co/Habana/roberta-large)
* [Swin Transformer](https://huggingface.co/Habana/swin)
* [T5](https://huggingface.co/Habana/t5)
* [Vision Transformer (ViT)](https://huggingface.co/Habana/vit)
## Fairseq
* [Wav2Vec 2.0](https://github.com/HabanaAI/fairseq)
* [Transformer](https://github.com/HabanaAI/fairseq)
