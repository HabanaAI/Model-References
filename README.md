# Habana Deep Learning Examples for Training and Inference

- [Habana Deep Learning Examples for Training and Inference](#habana-deep-learning-examples-for-training-and-inference)
  - [Model List and Performance Data](#model-list-and-performance-data)
  - [Computer Vision](#computer-vision)
  - [Natural Language Processing](#natural-language-processing)
  - [Recommender Systems](#recommender-systems)
  - [Audio](#audio)
  - [Generative Models](#generative-models)
  - [MLPerf™ 2.1](#mlperf-21)
  - [Reporting Bugs/Feature Requests](#reporting-bugsfeature-requests)
- [Community](#community)
  - [Hugging Face](#hugging-face)

## Model List and Performance Data

Please visit [this page](https://developer.habana.ai/resources/habana-training-models/#performance) for performance information.

This repository is a collection of models that have been ported to run on Habana Gaudi AI accelerator. They are intended as examples, and will be reasonably optimized for performance while still being easy to read.

## Computer Vision
| Models  | Framework | Validated on Gaudi | Validated on Gaudi2 |
| ------- | --------- | ----- | ------ |
| [ResNet50, ResNeXt101](PyTorch/computer_vision/classification/torchvision) | PyTorch | ✔ | ✔ |
| [ResNet152](PyTorch/computer_vision/classification/torchvision) | PyTorch | ✔ | |
| [MobileNetV2](PyTorch/computer_vision/classification/torchvision) | PyTorch | ✔ | |
| [UNet 2D, Unet 3D](PyTorch/computer_vision/segmentation/Unet)  | PyTorch | ✔ | ✔ |
| [SSD](PyTorch/computer_vision/detection/mlcommons/SSD/ssd) | PyTorch | ✔ | ✔ |
| [GoogLeNet](PyTorch/computer_vision/classification/torchvision) | PyTorch | ✔ | |
| [Vision Transformer](PyTorch/computer_vision/classification/ViT) | PyTorch | ✔ | |
| [DINO](PyTorch/computer_vision/classification/dino) | PyTorch | ✔ | |
| [YOLOX](PyTorch/computer_vision/detection/yolox) | PyTorch | ✔ | |
| [YOLOV3](PyTorch/computer_vision/detection/openmmlab_detection) | PyTorch | ✔ | |
| [ResNet50 Keras](TensorFlow/computer_vision/Resnets/resnet_keras) | TensorFlow | ✔ | ✔ |
| [ResNeXt101](TensorFlow/computer_vision/Resnets/ResNeXt) |TensorFlow | ✔ | ✔ |
| [SSD](TensorFlow/computer_vision/SSD_ResNet34) |TensorFlow | ✔ | ✔ |
| [Mask R-CNN](TensorFlow/computer_vision/maskrcnn) |TensorFlow | ✔ | ✔ |
| [UNet 2D](TensorFlow/computer_vision/Unet2D) | TensorFlow | ✔ | ✔ |
| [UNet 3D](TensorFlow/computer_vision/UNet3D) | TensorFlow | ✔ | ✔ |
| [UNet Industrial](TensorFlow/computer_vision/UNet_Industrial) | TensorFlow | ✔ | |
| [DenseNet](TensorFlow/computer_vision/densenet) |TensorFlow | ✔ | |
| [EfficientDet](TensorFlow/computer_vision/efficientdet) | TensorFlow | ✔ | |
| [SegNet](TensorFlow/computer_vision/Segnet) | TensorFlow | ✔ | |
| [Vision Transformer](TensorFlow/computer_vision/VisionTransformer) | TensorFlow | ✔ | |


## Natural Language Processing
| Models  | Framework | Validated on Gaudi | Validated on Gaudi2 |
| ------- | --------- | ----- | ------ |
| [BERT Pretraining](PyTorch/nlp/pretraining/bert) | PyTorch | ✔ | ✔ |
| [BERT Finetuning](PyTorch/nlp/finetuning/huggingface/bert) | PyTorch | ✔ | ✔ |
| [DeepSpeed BERT-1.5B, BERT-5B](PyTorch/nlp/pretraining/deepspeed-bert) | PyTorch | ✔ | |
| [Transformer](PyTorch/nlp/nmt/fairseq) | PyTorch | ✔ | ✔ |
| [BART](PyTorch/nlp/BART/simpletransformers) | PyTorch | ✔ | |
| [HuggingFace BLOOM](PyTorch/nlp/bloom) | PyTorch | ✔ | ✔ |
| [Megatron-DeepSpeed BLOOM 13B](PyTorch/nlp/DeepSpeedExamples/Megatron-DeepSpeed) | PyTorch | | ✔ |
| [BERT](TensorFlow/nlp/bert) | TensorFlow | ✔ | ✔ |
| [DistilBERT](TensorFlow/nlp/distilbert) | TensorFlow | ✔ | |
| [Transformer](TensorFlow/nlp/transformer) | TensorFlow | ✔ | ✔ |
| [Electra](TensorFlow/nlp/electra) | TensorFlow | ✔ | |

## Recommender Systems
| Models  | Framework | Validated on Gaudi | Validated on Gaudi2 |
| ------- | --------- | ----- | ------ |
| [Wide & Deep](TensorFlow/recommendation/WideAndDeep) | TensorFlow | ✔ | |

## Audio
| Models  | Framework | Validated on Gaudi | Validated on Gaudi2 |
| ------- | --------- | ----- | ------ |
| [Wav2vec 2.0](PyTorch/audio/wav2vec2/fairseq) | PyTorch | ✔ | ✔ |
| [Wav2Vec2ForCTC](PyTorch/audio/wav2vec2/inference) | PyTorch | ✔ | ✔ |

## Generative Models
| Models  | Framework | Validated on Gaudi | Validated on Gaudi2 |
| ------- | --------- | ----- | ------ |
| [V-Diffusion](PyTorch/generative_models/v-diffusion) | PyTorch | ✔ | |
| [Stable Diffusion](PyTorch/generative_models/stable-diffusion) | PyTorch | ✔ | ✔ |
| [Stable Diffusion Training](PyTorch/generative_models/stable-diffusion-training) | PyTorch | ✔ | |
| [Stable Diffusion v1.5](PyTorch/generative_models/stable-diffusion-v-1-5) | PyTorch | ✔ | ✔ |
| [Stable Diffusion v2.1](PyTorch/generative_models/stable-diffusion-v-2-1) | PyTorch | ✔ | |
| [CycleGAN](TensorFlow/computer_vision/CycleGAN) | TensorFlow | ✔ | |

## MLPerf™ 2.1
| Models  | Framework | Validated on Gaudi | Validated on Gaudi2 |
| ------- | --------- | ----- | ------ |
| [ResNet50 Keras](MLPERF2.1/Habana/benchmarks) | PyTorch | | ✔ |
| [BERT](MLPERF2.1/Habana/benchmarks) | PyTorch | | ✔ |
| [ResNet50 Keras](MLPERF2.1/Habana/benchmarks) | TensorFlow | | ✔ |
| [BERT](MLPERF2.1/Habana/benchmarks) | TensorFlow | | ✔ |

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
