# GPU Migration
GPU Migration is currently an experimental feature that facilitates porting models using CUDA® API to Habana Gaudi.
Many community models are designed to perform well on GPUs and use CUDA natively.
The goal of GPU Migration is to make such models functional on Gaudi by adding just one line of code:

```python
import habana_frameworks.torch.gpu_migration
```

For more details, refer to [Migrating PyTorch Models from GPU to HPU](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Model_Porting/Migrating_PyTorch_Models_to_HPU.html) documentation.

On this page, you will find simple examples as well as fully functional models enabled with GPU Migration.

## Simple Examples
| Models  | Validated on Gaudi | Validated on Gaudi2 |
| ------- | ----- | ------ |
| [MNIST Example](simple_examples/mnist) | | ✔ |

## Generative Models
| Models  | Validated on Gaudi | Validated on Gaudi2 |
| ------- | ----- | ------ |
| [Stable Diffusion](generative_models/stable-diffusion) | ✔ | ✔ |

## Computer Vision
| Models  | Validated on Gaudi | Validated on Gaudi2 |
| ------- | ----- | ------ |
| [ResNet50](computer_vision/classification/torchvision) | ✔ | ✔ |

## Natural Language Processing
| Models  | Validated on Gaudi | Validated on Gaudi2 |
| ------- | ----- | ------ |
| [BERT](nlp/bert/) | ✔ | ✔ |