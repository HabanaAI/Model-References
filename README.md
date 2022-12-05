# Habana Deep Learning Examples for Training

## Model List and Performance Data

Please visit [this page](https://developer.habana.ai/resources/habana-training-models/#performance) for performance information.

This repository is a collection of models that have been ported to run on Habana Gaudi training accelerators. They are intended as examples, and will be reasonably optimized for performance while still being easy to read.

## Computer Vision
| Models  | Framework | Gaudi | Gaudi2 |
| ------- | --------- | ----- | ------ |
| [ResNet50 Keras](TensorFlow/computer_vision/Resnets/resnet_keras) | TensorFlow | ✔ | ✔ |
| [ResNeXt101](TensorFlow/computer_vision/Resnets/ResNeXt) |TensorFlow | ✔ | ✔ |
| [SSD](TensorFlow/computer_vision/SSD_ResNet34) |TensorFlow | ✔ | ✔ |
| [Mask R-CNN](TensorFlow/computer_vision/maskrcnn) |TensorFlow | ✔ | ✔ |
| [UNet 2D](TensorFlow/computer_vision/Unet2D) | TensorFlow | ✔ | ✔ |
| [UNet 3D](TensorFlow/computer_vision/UNet3D) | TensorFlow | ✔ | ✔ |
| [UNet Industrial](TensorFlow/computer_vision/UNet_Industrial) | TensorFlow | ✔ | |
| [DenseNet](TensorFlow/computer_vision/densenet) |TensorFlow | ✔ | |
| [EfficientDet](TensorFlow/computer_vision/efficientdet) | TensorFlow | ✔ | |
| [RetinaNet](TensorFlow/computer_vision/RetinaNet) | TensorFlow | ✔ | |
| [SegNet](TensorFlow/computer_vision/Segnet) | TensorFlow | ✔ | |
| [Vision Transformer](TensorFlow/computer_vision/VisionTransformer) | TensorFlow | ✔ | |
| [MobileNetV2](TensorFlow/computer_vision/mobilenetv2) | TensorFlow | ✔ | |
| [ResNet50, ResNeXt101](PyTorch/computer_vision/classification/torchvision) | PyTorch | ✔ | ✔ |
| [ResNet152](PyTorch/computer_vision/classification/torchvision) | PyTorch | ✔ | |
| [MobileNetV2](PyTorch/computer_vision/classification/torchvision) | PyTorch | ✔ | |
| [UNet 2D, Unet 3D](PyTorch/computer_vision/segmentation/Unet)  | PyTorch | ✔ | ✔ |
| [SSD](PyTorch/computer_vision/detection/mlcommons/SSD/ssd) | PyTorch | ✔ | ✔ |
| [GoogLeNet](PyTorch/computer_vision/classification/torchvision) | PyTorch | ✔ | |
| [Vision Transformer](PyTorch/computer_vision/classification/ViT) | PyTorch | ✔ | |
| [Swin Transformer](PyTorch/computer_vision/classification/swin_transformer) | PyTorch | ✔ | |
| [DINO](PyTorch/computer_vision/classification/dino) | PyTorch | ✔ | |
| [YOLOX](PyTorch/computer_vision/detection/yolox) | PyTorch | ✔ | |
| [YOLOV3](PyTorch/computer_vision/detection/openmmlab_detection) | PyTorch | ✔ | |

## Natural Language Processing
| Models  | Framework | Gaudi | Gaudi2 |
| ------- | --------- | ----- | ------ |
| [BERT](TensorFlow/nlp/bert) | TensorFlow | ✔ | ✔ |
| [DistilBERT](TensorFlow/nlp/distilbert) | TensorFlow | ✔ | |
| [Transformer](TensorFlow/nlp/transformer) | TensorFlow | ✔ | ✔ |
| [T5 Base](TensorFlow/nlp/T5-base) | TensorFlow | ✔ | |
| [Electra](TensorFlow/nlp/electra) | TensorFlow | ✔ | |
| [BERT Pretraining](PyTorch/nlp/pretraining/bert) | PyTorch | ✔ | ✔ |
| [BERT Finetuning](PyTorch/nlp/finetuning/huggingface/bert) | PyTorch | ✔ | ✔ |
| [DeepSpeed BERT-1.5B, BERT-5B](PyTorch/nlp/pretraining/deepspeed-bert) | PyTorch | ✔ | |
| [RoBERTa](PyTorch/nlp/finetuning/huggingface/bert) | PyTorch | ✔ | |
| [ALBERT](PyTorch/nlp/finetuning/huggingface/bert) | PyTorch | ✔ | |
| [DistilBERT](PyTorch/nlp/finetuning/huggingface/distilbert) | PyTorch | ✔ | |
| [Electra](PyTorch/nlp/finetuning/huggingface/bert) | PyTorch | ✔ | |
| [Transformer](PyTorch/nlp/nmt/fairseq) | PyTorch | ✔ | ✔ |
| [BART](PyTorch/nlp/BART/simpletransformers) | PyTorch | ✔ | |
| [BLOOM](PyTorch/nlp/bloom) | PyTorch | ✔ | ✔ |

## Recommender Systems
| Models  | Framework | Gaudi | Gaudi2 |
| ------- | --------- | ----- | ------ |
| [Wide & Deep](TensorFlow/recommendation/WideAndDeep) | TensorFlow | ✔ | |

## Audio
| Models  | Framework | Gaudi | Gaudi2 |
| ------- | --------- | ----- | ------ |
| [Wav2vec 2.0](PyTorch/audio/wav2vec2/fairseq) | PyTorch | ✔ | |

## Generative Models
| Models  | Framework | Gaudi | Gaudi2 |
| ------- | --------- | ----- | ------ |
| [CycleGAN](TensorFlow/computer_vision/CycleGAN) | TensorFlow | ✔ | |
| [V-Diffusion](PyTorch/generative_models/v-diffusion) | PyTorch | ✔ | |
| [Stable Diffusion](PyTorch/generative_models/stable-diffusion) | PyTorch | ✔ | ✔ |

## MLPerf™ 2.0
| Models  | Framework | Gaudi | Gaudi2 |
| ------- | --------- | ----- | ------ |
| [ResNet50 Keras](MLPERF2.0/Habana/benchmarks) | TensorFlow | ✔ | |
| [BERT](MLPERF2.0/Habana/benchmarks) | TensorFlow | ✔ | |

## MLPerf™ 2.1
| Models  | Framework | Gaudi | Gaudi2 |
| ------- | --------- | ----- | ------ |
| [ResNet50 Keras](MLPERF2.1/Habana/benchmarks) | TensorFlow | | ✔ |
| [BERT](MLPERF2.1/Habana/benchmarks) | TensorFlow | | ✔ |
| [ResNet50 Keras](MLPERF2.1/Habana/benchmarks) | PyTorch | | ✔ |
| [BERT](MLPERF2.1/Habana/benchmarks) | PyTorch | | ✔ |

MLPerf™ is a trademark and service mark of MLCommons Association in the United States and other countries. All rights reserved. Unauthorized use is strictly prohibited.

## Reporting Bugs/Feature Requests

We welcome you to use the [GitHub issue tracker](https://github.com/HabanaAI/Model-References/issues) to report bugs or suggest features.

When filing an issue, please check existing open, or recently closed, issues to make sure somebody else hasn't already
reported the issue. Please try to include as much information as you can. Details like these are incredibly useful:

* A reproducible test case or series of steps
* The version of our code being used
* Any modifications you've made relevant to the bug
* Anything unusual about your environment or deployment
