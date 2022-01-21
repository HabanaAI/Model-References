# Habana Deep Learning Examples for Training

## Model List and Performance Data

Please visit [this page](https://developer.habana.ai/resources/habana-training-models/#performance) for performance information.

This repository is a collection of models that have been ported to run on Habana Gaudi training accelerators. They are intended as examples, and will be reasonably optimized for performance while still being easy to read.

## Computer Vision
| Models  | Framework |
| ------------- | ------------- |
| [ResNet 50 Keras](TensorFlow/computer_vision/Resnets/resnet_keras) | TensorFlow |
| [ResNeXt 50, 101, 152](TensorFlow/computer_vision/Resnets/ResNeXt)  |TensorFlow |
| [SSD](TensorFlow/computer_vision/SSD_ResNet34) |TensorFlow |
| [Mask R-CNN](TensorFlow/computer_vision/maskrcnn) |TensorFlow |
| [DenseNet](TensorFlow/computer_vision/densenet) |TensorFlow |
| [UNet 2D](TensorFlow/computer_vision/Unet2D) | TensorFlow |
| [UNet 3D](TensorFlow/computer_vision/UNet3D) | TensorFlow |
| [CycleGAN](TensorFlow/computer_vision/CycleGAN) | TensorFlow |
| [EfficientDet](TensorFlow/computer_vision/efficientdet) | TensorFlow |
| [RetinaNet](TensorFlow/computer_vision/RetinaNet) | TensorFlow |
| [SegNet](TensorFlow/computer_vision/Segnet) | TensorFlow |
| [MobileNet V2](staging/TensorFlow/computer_vision/mobilenetv2/research/slim) | TensorFlow |
| [ResNet50, ResNet152, ResNext101](PyTorch/computer_vision/classification/torchvision)  | PyTorch |
| [MobileNet V2](PyTorch/computer_vision/classification/torchvision)  | PyTorch |
| [UNet 2D](PyTorch/computer_vision/segmentation/Unet)  | PyTorch |
| [UNet 3D](PyTorch/computer_vision/segmentation/Unet)  | PyTorch |
| [SSD](PyTorch/computer_vision/detection/mlcommons/SSD/ssd) |PyTorch |
| [GoogLeNet](PyTorch/computer_vision/classification/torchvision)  | PyTorch |

## Natural Language Processing
| Models  | Framework |
| ------------- | ------------- |
| [BERT](TensorFlow/nlp/bert) | TensorFlow |
| [ALBERT](TensorFlow/nlp/albert) | TensorFlow |
| [Transformer](TensorFlow/nlp/transformer) | TensorFlow |
| [T5 Base](TensorFlow/nlp/T5-base) | TensorFlow |
| [BERT Pretraining](PyTorch/nlp/pretraining/bert) | PyTorch |
| [BERT Finetuning](PyTorch/nlp/finetuning/huggingface/bert) | PyTorch |
| [RoBERTa](PyTorch/nlp/finetuning/huggingface/bert) | PyTorch |
| [ALBERT](PyTorch/nlp/finetuning/huggingface/bert) | PyTorch |
| [DistilBERT](PyTorch/nlp/finetuning/huggingface/distilbert) | PyTorch |
| [Transformer](PyTorch/nlp/nmt/fairseq) | PyTorch |
| [BART](staging/PyTorch/nlp/BART/simpletransformers) | PyTorch |

## Reporting Bugs/Feature Requests

We welcome you to use the [GitHub issue tracker](https://github.com/HabanaAI/Model-References/issues) to report bugs or suggest features.

When filing an issue, please check existing open, or recently closed, issues to make sure somebody else hasn't already
reported the issue. Please try to include as much information as you can. Details like these are incredibly useful:

* A reproducible test case or series of steps
* The version of our code being used
* Any modifications you've made relevant to the bug
* Anything unusual about your environment or deployment
