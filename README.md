# Intel® Gaudi® AI Accelerators Examples for Training and Inference

- [Intel® Gaudi® AI Accelerators Examples for Training and Inference](#intel-gaudi-ai-accelerators-examples-for-training-and-inference)
  - [Model List and Performance Data](#model-list-and-performance-data)
  - [Computer Vision](#computer-vision)
  - [Natural Language Processing](#natural-language-processing)
  - [Audio](#audio)
  - [Generative Models](#generative-models)
  - [MLPerf™ Training 4.0](#mlperf-training-40)
  - [MLPerf™ Inference 4.0](#mlperf-inference-40)
  - [Reporting Bugs/Feature Requests](#reporting-bugsfeature-requests)
- [Community](#community)
  - [Optimum Habana](#optimum-habana)
  - [vLLM](#vllm)
  - [Megatron-LM](#megatron-lm)
  - [Megatron-DeepSpeed](#megatron-deepspeed)

## Model List and Performance Data

Please visit [this page](https://developer.habana.ai/resources/habana-training-models/#performance) for performance information.

This repository is a collection of models that have been ported to run on Intel Gaudi AI accelerators. They are intended as examples, and will be reasonably optimized for performance while still being easy to read.

See [Community](#community) section for other projects with models ported and optimized for Intel Gaudi AI accelerators.

## Computer Vision
| Models                                                                     | Framework         | Validated on Gaudi                      | Validated on Gaudi 2                     | Validated on Gaudi 3                     |
| -------------------------------------------------------------------------- | ----------------- | --------------------------------------- | ---------------------------------------- | ---------------------------------------- |
| [ResNet50](PyTorch/computer_vision/classification/torchvision)             | PyTorch           | Training (compile)                      | Training (compile), Inference (compile)  | Training (compile)*, Inference (compile) |
| [ResNeXt101](PyTorch/computer_vision/classification/torchvision)           | PyTorch           | -                                       | Training (compile)                       | Training (compile)                       |
| [UNet2D](PyTorch/computer_vision/segmentation/Unet)                        | PyTorch Lightning | Training (compile), Inference (compile) | Training (compile), Inference (compile)  | -                                        |
| [Unet3D](PyTorch/computer_vision/segmentation/Unet)                        | PyTorch Lightning | Training (compile), Inference (compile) | Training (compile), Inference (compile)  | Training (compile)**                     |
| [Vision Transformer](PyTorch/computer_vision/classification/ViT)           | PyTorch           | Training                                | -                                        | -                                        |
| [YOLOX](PyTorch/computer_vision/detection/yolox)                           | PyTorch           | Training                                | Inference                                | Inference                                |

*Disclaimer: only on 8x
**Disclaimer: only functional checks done

## Natural Language Processing
| Models                                                                             | Framework  | Validated on Gaudi            | Validated on Gaudi 2           | Validated on Gaudi 3  |
|------------------------------------------------------------------------------------| ---------- | ----------------------------- | ------------------------------ | --------------------- |
| [BERT Pretraining](PyTorch/nlp/bert)                                               | PyTorch    | Training (compile)            | Training (compile)             | -                     |
| [BERT Finetuning](PyTorch/nlp/bert)                                                | PyTorch    | Training, Inference           | Training, Inference (compile)  | Inference (compile)*  |
| [DeepSpeed BERT-1.5B, BERT-5B](PyTorch/nlp/DeepSpeedExamples/deepspeed-bert)       | PyTorch    | Training                      | Training (compile)             | -                     |

*Disclaimer: only bf16

## Audio
| Models                                             | Framework | Validated on Gaudi | Validated on Gaudi 2 | Validated on Gaudi 3 |
| -------------------------------------------------- | --------- | ------------------ | -------------------- | -------------------- |
| [Wav2Vec2ForCTC](PyTorch/audio/wav2vec2/inference) | PyTorch   | Inference          | Inference            | -                    |

## Generative Models
| Models                                                                               | Framework         | Validated on Gaudi  | Validated on Gaudi 2 | Validated on Gaudi 3 |
| ------------------------------------------------------------------------------------ | ----------------- | ------------------- | -------------------- | -------------------- |
| [Stable Diffusion](PyTorch/generative_models/stable-diffusion)                       | PyTorch Lightning | Training            | Training             | -                    |

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
Other  projects with models ported and optimized for Intel Gaudi AI accelerators.

## Optimum Habana
HuggingFace models for finetuning and inference are available in Optimum Habana project https://github.com/huggingface/optimum-habana/ and as model cards at https://huggingface.co/Habana.

## vLLM
Models optimized for inferece with vLLM are available in [HabanaAI/vllm-fork](https://github.com/HabanaAI/vllm-fork).

## Megatron-LM
LLM training models like Llama or Mixtral are available in Megatron-LM's fork: [HabanaAI/Megatron-LM](https://github.com/HabanaAI/Megatron-LM).

## Megatron-DeepSpeed
⚠️Note that this project will be deprecated and replaced with Megatron-LM.

LLM training models like Llama or Mixtral are available in Megatron-DeepSpeed's fork: [HabanaAI/Megatron-DeepSpeed](https://github.com/HabanaAI/Megatron-DeepSpeed).

