# Habana Deep Learning Examples for Training

## Model List and Performance Data

Please visit [this page](https://developer.habana.ai/resources/habana-training-models/#performance) for performance information.

This repository is a collection of models that have been ported to run on Habana Gaudi training accelerators. They are intended as examples, and will be reasonably optimized for performance while still being easy to read.

**NEW**: TensorFlow [ResNet50 Keras](TensorFlow/computer_vision/Resnets/resnet_keras), TensorFlow [BERT](TensorFlow/nlp/bert) and PyTorch [ResNet50](PyTorch/computer_vision/classification/torchvision) are enabled on Gaudi2. Instructions are available in the respective README files for these models.

## Computer Vision
| Models  | Framework |
| ------- | --------- |
| [ResNet50 Keras](TensorFlow/computer_vision/Resnets/resnet_keras) | TensorFlow |
| [ResNeXt101](TensorFlow/computer_vision/Resnets/ResNeXt) |TensorFlow |
| [SSD](TensorFlow/computer_vision/SSD_ResNet34) |TensorFlow |
| [Mask R-CNN](TensorFlow/computer_vision/maskrcnn) |TensorFlow |
| [DenseNet](TensorFlow/computer_vision/densenet) |TensorFlow |
| [UNet 2D](TensorFlow/computer_vision/Unet2D) | TensorFlow |
| [UNet 3D](TensorFlow/computer_vision/UNet3D) | TensorFlow |
| [UNet Industrial](TensorFlow/computer_vision/UNet_Industrial) | TensorFlow |
| [CycleGAN](TensorFlow/computer_vision/CycleGAN) | TensorFlow |
| [EfficientDet](TensorFlow/computer_vision/efficientdet) | TensorFlow |
| [RetinaNet](TensorFlow/computer_vision/RetinaNet) | TensorFlow |
| [SegNet](TensorFlow/computer_vision/Segnet) | TensorFlow |
| [Vision Transformer](TensorFlow/computer_vision/VisionTransformer) | TensorFlow |
| [MobileNet V2](TensorFlow/computer_vision/mobilenetv2) | TensorFlow |
| [ResNet50, ResNet152, ResNext101](PyTorch/computer_vision/classification/torchvision) | PyTorch |
| [MobileNet V2](PyTorch/computer_vision/classification/torchvision) | PyTorch |
| [UNet 2D, Unet 3D](PyTorch/computer_vision/segmentation/Unet)  | PyTorch |
| [SSD](PyTorch/computer_vision/detection/mlcommons/SSD/ssd) | PyTorch |
| [GoogLeNet](PyTorch/computer_vision/classification/torchvision) | PyTorch |
| [Vision Transformer](PyTorch/computer_vision/classification/ViT) | PyTorch |

## Natural Language Processing
| Models  | Framework |
| ------- | --------- |
| [BERT](TensorFlow/nlp/bert) | TensorFlow |
| [DistilBERT](TensorFlow/nlp/distilbert) | TensorFlow |
| [ALBERT](TensorFlow/nlp/albert) | TensorFlow |
| [Transformer](TensorFlow/nlp/transformer) | TensorFlow |
| [T5 Base](TensorFlow/nlp/T5-base) | TensorFlow |
| [Electra](TensorFlow/nlp/electra) | TensorFlow |
| [BERT Pretraining](PyTorch/nlp/pretraining/bert) | PyTorch |
| [BERT Finetuning](PyTorch/nlp/finetuning/huggingface/bert) | PyTorch |
| [RoBERTa](PyTorch/nlp/finetuning/huggingface/bert) | PyTorch |
| [ALBERT](PyTorch/nlp/finetuning/huggingface/bert) | PyTorch |
| [DistilBERT](PyTorch/nlp/finetuning/huggingface/distilbert) | PyTorch |
| [Electra](PyTorch/nlp/finetuning/huggingface/bert) | PyTorch |
| [Transformer](PyTorch/nlp/nmt/fairseq) | PyTorch |
| [BART](PyTorch/nlp/BART/simpletransformers) | PyTorch |
| [GPT2](PyTorch/nlp/GPT2) | PyTorch |

## Recommender Systems
| Models  | Framework |
| ------- | --------- |
| [Wide & Deep](TensorFlow/recommendation/WideAndDeep) | TensorFlow |
## Reporting Bugs/Feature Requests

We welcome you to use the [GitHub issue tracker](https://github.com/HabanaAI/Model-References/issues) to report bugs or suggest features.

When filing an issue, please check existing open, or recently closed, issues to make sure somebody else hasn't already
reported the issue. Please try to include as much information as you can. Details like these are incredibly useful:

* A reproducible test case or series of steps
* The version of our code being used
* Any modifications you've made relevant to the bug
* Anything unusual about your environment or deployment
