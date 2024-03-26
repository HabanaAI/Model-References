# Intel® Gaudi® AI Accelerator Examples for Training and Inference

- [Intel® Gaudi® AI Accelerator Examples for Training and Inference](#intel-gaudi-ai-accelerator-examples-for-training-and-inference)
  - [Model List and Performance Data](#model-list-and-performance-data)
  - [Computer Vision](#computer-vision)
  - [Natural Language Processing](#natural-language-processing)
  - [Audio](#audio)
  - [Generative Models](#generative-models)
  - [MLPerf™ Training 3.1](#mlperf-training-31)
  - [MLPerf™ Inference 3.1](#mlperf-inference-31)
  - [Reporting Bugs/Feature Requests](#reporting-bugsfeature-requests)
- [Community](#community)
  - [Hugging Face](#hugging-face)
  - [Megatron-DeepSpeed](#megatron-deepspeed)
  - [DeepSpeed-Chat](#deepspeed-chat)
  - [Fairseq](#fairseq)

## Model List and Performance Data

Please visit [this page](https://developer.habana.ai/resources/habana-training-models/#performance) for performance information.

This repository is a collection of models that have been ported to run on Intel®️ Gaudi®️ AI accelerator. They are intended as examples, and will be reasonably optimized for performance while still being easy to read.

## Computer Vision
| Models                                                                             | Framework         | Validated on Gaudi  | Validated on Gaudi2 |
| ---------------------------------------------------------------------------------- | ----------------- | ------------------- | ------------------- |
| [ResNet50, ResNeXt101](PyTorch/computer_vision/classification/torchvision)         | PyTorch           | Training            | Training, Inference |
| [ResNet152](PyTorch/computer_vision/classification/torchvision)                    | PyTorch           | Training            | -                   |
| [MobileNetV2](PyTorch/computer_vision/classification/torchvision)                  | PyTorch           | Training            | -                   |
| [UNet 2D, Unet3D](PyTorch/computer_vision/segmentation/Unet)                       | PyTorch Lightning | Training, Inference | Training, Inference |
| [SSD](PyTorch/computer_vision/detection/mlcommons/SSD/ssd)                         | PyTorch           | Training            | Training            |
| [GoogLeNet](PyTorch/computer_vision/classification/torchvision)                    | PyTorch           | Training            | -                   |
| [Vision Transformer](PyTorch/computer_vision/classification/ViT)                   | PyTorch           | Training            | -                   |
| [DINO](PyTorch/computer_vision/classification/dino)                                | PyTorch           | Training            | -                   |
| [YOLOX](PyTorch/computer_vision/detection/yolox)                                   | PyTorch           | Training            | -                   |


## Natural Language Processing
| Models                                                                             | Framework  | Validated on Gaudi  | Validated on Gaudi2 |
|------------------------------------------------------------------------------------| ---------- | ------------------- | ------------------- |
| [BERT Pretraining and Finetuning](PyTorch/nlp/bert)                                | PyTorch    | Training, Inference | Training, Inference |
| [DeepSpeed BERT-1.5B, BERT-5B](PyTorch/nlp/DeepSpeedExamples/deepspeed-bert)       | PyTorch    | Training            | -                   |
| [BART](PyTorch/nlp/BART/simpletransformers)                                        | PyTorch    | Training            | -                   |
| [HuggingFace BLOOM](PyTorch/nlp/bloom)                                             | PyTorch    | Inference           | Inference           |


## Audio
| Models                                             | Framework | Validated on Gaudi | Validated on Gaudi2 |
| -------------------------------------------------- | --------- | ------------------ | ------------------- |
| [Wav2Vec2ForCTC](PyTorch/audio/wav2vec2/inference) | PyTorch   | Inference          | Inference           |

## Generative Models
| Models                                                                               | Framework         | Validated on Gaudi  | Validated on Gaudi2 |
| ------------------------------------------------------------------------------------ | ----------------- | ------------------- | ------------------- |
| [Stable Diffusion](PyTorch/generative_models/stable-diffusion)                       | PyTorch Lightning | Training            | Training            |
| [Stable Diffusion FineTuning](PyTorch/generative_models/stable-diffusion-finetuning) | PyTorch           | Training            | Training            |
| [Stable Diffusion v2.1](PyTorch/generative_models/stable-diffusion-v-2-1)            | PyTorch           | Inference           | Inference           |

## MLPerf&trade; Training 3.1
| Models                                  | Framework  | Validated on Gaudi | Validated on Gaudi2 |
| --------------------------------------- | ---------- | ------------------ | ------------------- |
| [GPT3](MLPERF3.1/Training/benchmarks)     | PyTorch    | -                | Training            |
| [ResNet50](MLPERF3.1/Training/benchmarks) | PyTorch    | -                | Training            |
| [BERT](MLPERF3.1/Training/benchmarks)     | PyTorch    | -                | Training            |

## MLPerf&trade; Inference 3.1
| Models                                  | Framework  | Validated on Gaudi | Validated on Gaudi2 |
| --------------------------------------- | ---------- | ------------------ | ------------------- |
| [GPT-J](MLPERF3.1/Inference/code/gpt-j) | PyTorch    | -                  | Inference           |

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
* All supported models are available in Optimum Habana project https://github.com/huggingface/optimum-habana/ and as model cards at https://huggingface.co/Habana.

## Megatron-DeepSpeed
* Megatron-DeepSpeed was moved to a new GitHub repository [HabanaAI/Megatron-DeepSpeed](https://github.com/HabanaAI/Megatron-DeepSpeed).

## DeepSpeed-Chat
* This model was moved to a new GitHub repository [HabanaAI/DeepSpeedExample](https://github.com/HabanaAI/DeepSpeedExamples/tree/main/applications/DeepSpeed-Chat).

## Fairseq
* [Transformer](https://github.com/HabanaAI/fairseq)
