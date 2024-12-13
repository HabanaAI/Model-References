# GPU Migration
GPU Migration is currently an experimental feature that facilitates porting models using CUDA® API to Intel® Gaudi® AI accelerator.
Many community models are designed to perform well on GPUs and use CUDA natively.
The goal of GPU Migration is to make such models functional on Gaudi by adding just one environment variable:
```bash
PT_HPU_GPU_MIGRATION=1
```

For more details, refer to [GPU Migration Toolkit documentation](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Model_Porting/GPU_Migration_Toolkit/GPU_Migration_Toolkit.html).

On this page, you will find simple examples as well as fully functional models enabled with GPU Migration.

## Simple Examples
| Models  | Validated on Gaudi | Validated on Gaudi 2 |
| ------- | ----- | ------ |
| [MNIST Example](simple_examples/mnist) | | ✔ |

## Generative Models
| Models  | Validated on Gaudi | Validated on Gaudi 2 |
| ------- | ----- | ------ |
| [Stable Diffusion](generative_models/stable-diffusion) | ✔ | ✔ |

## Computer Vision
| Models  | Validated on Gaudi | Validated on Gaudi 2 |
| ------- | ----- | ------ |
| [ResNet50](computer_vision/classification/torchvision) | ✔ | ✔ |

## Natural Language Processing
| Models  | Validated on Gaudi | Validated on Gaudi 2 |
| ------- | ----- | ------ |
| [BERT](nlp/bert/) | ✔ | ✔ |
| [BLOOM-13B](nlp/DeepSpeedExamples/Megatron-DeepSpeed/) |   | ✔ |
