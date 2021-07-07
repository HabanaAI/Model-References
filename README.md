# Habana Deep Learning Examples for Training

## Model List and Performance Data

Please visit [Habana Training Models and Performance](https://developer.habana.ai/resources/habana-training-models/#performance) to see the latest performance data on our reference models.

This repository is a collection of models that have been ported to run on Habana Gaudi training accelerators. They are intended as examples, and will be reasonably optimized for performance while still being easy to read.

## Computer Vision
| Models  | Framework |
| ------------- | ------------- |
| [ResNet 50 Keras](TensorFlow/computer_vision/Resnets/resnet_keras) | TensorFlow |
| [ResNeXt 50, 101, 152](TensorFlow/computer_vision/Resnets)  |TensorFlow |
| [SSD](TensorFlow/computer_vision/SSD_ResNet34) |TensorFlow |
| [Mask R-CNN](TensorFlow/computer_vision/maskrcnn) |TensorFlow |
| [DenseNet Keras](TensorFlow/computer_vision/densenet_keras) |TensorFlow |
| [ResNet50, ResNeXt101](PyTorch/computer_vision/ImageClassification/ResNet)  | PyTorch |
| [UNet 2D](TensorFlow/computer_vision/Unet2D) | TensorFlow |

## Natural Language Processing
| Models  | Framework |
| ------------- | ------------- |
| [BERT](TensorFlow/nlp/bert) |TensorFlow |
| [ALBERT](TensorFlow/nlp/albert) | TensorFlow |
| [BERT](PyTorch/nlp/bert) |PyTorch |
| [Transformer](TensorFlow/nlp/transformer) | TensorFlow |

## Recommender Systems
| Models  | Framework |
| ------------- | ------------- |
| [DLRM](PyTorch/recommendation/dlrm) |PyTorch |

## Reporting Bugs/Feature Requests

We welcome you to use the [GitHub issue tracker](https://github.com/HabanaAI/Model-References/issues) to report bugs or suggest features.

When filing an issue, please check existing open, or recently closed, issues to make sure somebody else hasn't already
reported the issue. Please try to include as much information as you can. Details like these are incredibly useful:

* A reproducible test case or series of steps
* The version of our code being used
* Any modifications you've made relevant to the bug
* Anything unusual about your environment or deployment
