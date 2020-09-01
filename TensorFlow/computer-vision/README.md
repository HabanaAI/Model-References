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
| resnet50 | [README](./resnet50v1.5/README.md) |
| resnet "B" | [README](./resnext101-32x4d/README.md) |
| resnet "C" | [README](./se-resnext101-32x4d/README.md) |

## Validation accuracy results

Our results were obtained by running the applicable training scripts **TBD container** 
on **TBD**. The specific training script that was run is documented in the corresponding model's README.

The following table shows the validation accuracy results of the 
three classification models side-by-side.


| **arch** | **AMP Top1** | **AMP Top5** | **FP32 Top1** | **FP32 Top5** |
|:-:|:-:|:-:|:-:|:-:|
| resnet50            | 78.35 | 94.21 | 78.34 | 94.21 |
| resnet "B"          | 80.21 | 95.00 | 80.21 | 94.99 |
| resnet "C"          | 80.87 | 95.35 | 80.84 | 95.37 |

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
| resnet50            | - img/s | - img/s | -x |
| resnext101-32x4d    | - img/s | - img/s | -x |
| se-resnext101-32x4d | - img/s | - img/s | -x |

## Release notes

### Changelog
Oct 2020
  - Inception
