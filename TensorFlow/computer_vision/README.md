# Resnet-family Convolutional Neural Networks for Image Classification in Tensorflow

In this repository you will find implementation of Resnet and its variations for image
classification

## Table Of Contents

* [Models](#models)
* [Validation accuracy results](#validation-accuracy-results)
* [Training performance results](#training-performance-results)
* [Release notes](#release-notes)
  * [Changelog](#changelog)


## Models

The following table provides links to where you can find additional information on each model:

| **Model** | **Link**|
|-----------|---------|
| resnet50/101/152 | [README](./resnet50v1.5/README.md) |
| ResNext101 | [README](./ResNext101/README.md) |
| SSD-ResNet34 | [README](./SSD-ResNet34/README.md) |

## Validation accuracy results

Our results were obtained by running the applicable training scripts **TBD container** 
on **TBD**. The specific training script that was run is documented in the corresponding model's README.

The following table shows the validation accuracy results of the 
three classification models side-by-side.


| **arch** | **AMP Top1** | **AMP Top5** | **FP32 Top1** | **FP32 Top5** |
|:-:|:-:|:-:|:-:|:-:|
| Resnet50/101/152    | 78.35 | 94.21 | 78.34 | 94.21 |
| ResNext101          | 80.21 | 95.00 | 80.21 | 94.99 |
| SSD-ResNet34        | 80.87 | 95.35 | 80.84 | 95.37 |

## Training performance results
### Training performance: NVIDIA DGX-1 (8x V100 16G)

Our results were obtained by running the applicable 
training scripts in the **TBD** container 
on **TBD** (8x Habana Gaudi(R)) acclerators
Performance numbers (in images per second) 
were averaged over an entire training epoch.
The specific training script that was run is documented 
in the corresponding model's README.

The following table shows the training accuracy results of the 
three classification models side-by-side.


| **arch** | **BF16** | **FP32** | **BF16 Speedup** |
|:-:|:-:|:-:|:-:|
| Resnet50/101/152            | - img/s | - img/s | -x |
| ResNext101                  | - img/s | - img/s | -x |
| SSD-ResNet34                | - img/s | - img/s | -x |

## Release notes

### Changelog
Nov 2020
  - Inception
