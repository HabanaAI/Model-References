# Intel® Gaudi® AI Accelerator Examples for Training and Inference

- [Intel® Gaudi® AI Accelerator Examples for Training and Inference](#intel-gaudi-ai-accelerator-examples-for-training-and-inference)
  - [Model List and Performance Data](#model-list-and-performance-data)
  - [Computer Vision](#computer-vision)
  - [Natural Language Processing](#natural-language-processing)
  - [Audio](#audio)
  - [Generative Models](#generative-models)
  - [MLPerf™ Training 4.0](#mlperf-training-40)
  - [MLPerf™ Inference 4.0](#mlperf-inference-40)
  - [Reporting Bugs/Feature Requests](#reporting-bugsfeature-requests)
- [Community](#community)
  - [Hugging Face](#hugging-face)
  - [Megatron-DeepSpeed](#megatron-deepspeed)
  - [Fairseq](#fairseq)

## Model List and Performance Data

Please visit [this page](https://developer.habana.ai/resources/habana-training-models/#performance) for performance information.

This repository is a collection of models that have been ported to run on Intel Gaudi AI accelerator. They are intended as examples, and will be reasonably optimized for performance while still being easy to read.

## Computer Vision
| Models                                                                     | Framework         | Validated on Gaudi                      | Validated on Gaudi 2                     | Validated on Gaudi 3                    |
| -------------------------------------------------------------------------- | ----------------- | --------------------------------------- | ---------------------------------------- | --------------------------------------- |
| [ResNet50](PyTorch/computer_vision/classification/torchvision)             | PyTorch           | Training (compile)                      | Training (compile), Inference (compile)  | Inference (compile)                     |
| [ResNeXt101](PyTorch/computer_vision/classification/torchvision)           | PyTorch           | -                                       | Training (compile)                       | Training (compile)                      |
| [ResNet152](PyTorch/computer_vision/classification/torchvision)            | PyTorch           | Training                                | -                                        | -                                       |
| [MobileNetV2](PyTorch/computer_vision/classification/torchvision)          | PyTorch           | Training                                | -                                        | -                                       |
| [UNet2D](PyTorch/computer_vision/segmentation/Unet)                        | PyTorch Lightning | Training (compile), Inference (compile) | Training (compile), Inference (compile)  | -                                       |
| [Unet3D](PyTorch/computer_vision/segmentation/Unet)                        | PyTorch Lightning | Training (compile), Inference (compile) | Training (compile), Inference (compile)  | Training (compile)*                     |
| [SSD](PyTorch/computer_vision/detection/mlcommons/SSD/ssd)                 | PyTorch           | Training                                | Training                                 | -                                       |
| [GoogLeNet](PyTorch/computer_vision/classification/torchvision)            | PyTorch           | Training                                | -                                        | -                                       |
| [Vision Transformer](PyTorch/computer_vision/classification/ViT)           | PyTorch           | Training                                | -                                        | -                                       |
| [DINO](PyTorch/computer_vision/classification/dino)                        | PyTorch           | Training                                | -                                        | -                                       |
| [YOLOX](PyTorch/computer_vision/detection/yolox)                           | PyTorch           | Training                                | -                                        | -                                       |

*Disclaimer: only functional checks done

## Natural Language Processing
| Models                                                                             | Framework  | Validated on Gaudi            | Validated on Gaudi 2           | Validated on Gaudi 3  |
|------------------------------------------------------------------------------------| ---------- | ----------------------------- | ------------------------------ | --------------------- |
| [BERT Pretraining](PyTorch/nlp/bert)                                               | PyTorch    | Training (compile), Inference | Training (compile), Inference  | -                     |
| [BERT Finetuning](PyTorch/nlp/bert)                                                | PyTorch    | Training, Inference           | Training, Inference (compile)  | Inference (compile)*  |
| [DeepSpeed BERT-1.5B, BERT-5B](PyTorch/nlp/DeepSpeedExamples/deepspeed-bert)       | PyTorch    | Training                      | Training (compile)             | -                     |
| [BART](PyTorch/nlp/BART/simpletransformers)                                        | PyTorch    | Training                      | -                              | -                     |

*Disclaimer: Only bf16

## Audio
| Models                                             | Framework | Validated on Gaudi | Validated on Gaudi 2 | Validated on Gaudi 3 |
| -------------------------------------------------- | --------- | ------------------ | -------------------- | -------------------- |
| [Wav2Vec2ForCTC](PyTorch/audio/wav2vec2/inference) | PyTorch   | Inference          | Inference            | -                    |

## Generative Models
| Models                                                                               | Framework         | Validated on Gaudi  | Validated on Gaudi 2 | Validated on Gaudi 3 |
| ------------------------------------------------------------------------------------ | ----------------- | ------------------- | -------------------- | -------------------- |
| [Stable Diffusion](PyTorch/generative_models/stable-diffusion)                       | PyTorch Lightning | Training            | Training             | -                    |
| [Stable Diffusion FineTuning](PyTorch/generative_models/stable-diffusion-finetuning) | PyTorch           | Training            | Training             | -                    |

## MLPerf&trade; Training 4.0
| Models                                                       | Framework | Validated on Gaudi | Validated on Gaudi 2 | Validated on Gaudi 3 |
| ------------------------------------------------------------ | --------- | ------------------ | -------------------- | -------------------- |
| [GPT3](MLPERF4.0/Training/benchmarks/gpt3)                   | PyTorch   | -                  | Training             | -                    |
| [Llama 70B LoRA](MLPERF4.0/Training/benchmarks/llm_finetune) | PyTorch   | -                  | Training             | -                    |

## MLPerf&trade; Inference 4.0
| Models                                                          | Framework | Validated on Gaudi | Validated on Gaudi 2 | Validated on Gaudi 3 |
| --------------------------------------------------------------- | --------- | ------------------ | -------------------- | -------------------- |
| [Llama 70B](MLPERF4.0/Inference/llama/)                         | PyTorch   | -                  | Inference            | -                    |
| [Stable Diffusion XL](MLPERF4.0/Inference/stable-diffusion-xl/) | PyTorch   | -                  | Inference            | -                    |

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

## Fairseq
* [Transformer](https://github.com/HabanaAI/fairseq)
