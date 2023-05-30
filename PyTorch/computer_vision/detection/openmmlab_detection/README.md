# YOLOV3(OpenMMLab Detection) for PyTorch
This repository provides scripts to train OpenMMLab Detection YOLOV3 model on Habana Gaudi device to achieve state-of-the-art accuracy. To obtain model performance data, refer to the [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).
For more information about training deep learning models using Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

The YOLOV3 demo in this release includes single-device partial train for eager/lazy mode and multi-devices full train for lazy mode with FP32 and BF16 mixed precision.

## Table of Contents
- [Model-References](../../../../README.md)
- [Model Overview](#model-overview)
- [Setup](#setup)
- [Training Examples](#training-examples)
- [Inference](#inference)
- [Supported Configurations](#supported-configurations)
- [Changelog](#changelog)


## Model Overview
- YOLOV3 model in this demo is based on [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolo)

- `mmcv` folder is based on  [open-mmlab/mmcv](https://github.com/open-mmlab/mmcv/tree/v1.4.4). `mmdetection` folder is based on [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection/tree/v2.21.0). We modified these to support HPU training and performance improvement. For further details, refer to the [Changelog](#changelog) section.`yolov3` folder contains configuration files for YOLOV3 training on HPU.

### mmcv and mmdet Version
To use `mmdet`, `mmcv-full`is required instead of `mmcv lite build`. In addition, CPU build is required where the training is supported only from `mmdet=2.21.0`, and the corresponding `mmcv` is `1.4.4`. Therefore, the versions are `mmcv=1.4.4` and `mmdet==2.21.0`.

## Setup
Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) 
to set up the environment including the `$PYTHON` environment variable. To achieve the best performance, please follow the methods outlined in the [Optimizing Training Platform guide](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html).
The guides will walk you through the process of setting up your system to run the model on Gaudi.  

### Clone Habana Model-References
In the docker container, clone this repository and switch to the branch that
matches your SynapseAI version. You can run the
[`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the SynapseAI version

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```

Go to PyTorch Openmmlab detection directory:
```bash
cd Model-References/PyTorch/computer_vision/detection/openmmlab_detection
```

### Install Model Requirements
- Install mmcv
```
cd mmcv
pip install -r requirements.txt
MMCV_WITH_OPS=1 MAX_JOBS=8 python setup.py build_ext --user
MMCV_WITH_OPS=1 MAX_JOBS=8 python setup.py develop --user
# you can use develop instead of install when developing
```

- Install mmdet
```
cd mmdetection
python setup.py develop --user
```

### Training Data
Download COCO 2017 dataset from http://cocodataset.org.

You can set the dataset location in a base configfile as below.
```bash
yolov3/yolov3_d53_mstrain-608-273e_coco.py
data_root = '/data/pytorch/coco2017/'
```

Alternatively, you can pass the COCO dataset location to the '--cfg-options data_root' argument of the training commands.
```bash
--cfg-options data_root='/data/pytorch/coco2017/'
```

## Training Examples
### Single Card Training Examples
- Run training on 1 HPU with 2 ephocs - Eager mode:
```
python mmdetection/tools/train.py yolov3/yolov3_d53_320_273e_coco.py --hpu=1  --batch-size=32 --eval-batch-size=64 --deterministic --log-interval 400 --epoch 2
```

- Run training on 1 HPU with 2 ephocs - Eager autocast:
```
python mmdetection/tools/train.py yolov3/yolov3_d53_320_273e_coco.py --hpu=1 --autocast  --deterministic --batch-size 32 --eval-batch-size 64 --log-interval 400 --epoch 2
```

- Run training on 1 HPU with 2 ephocs - Lazy mode:
```
python mmdetection/tools/train.py yolov3/yolov3_d53_320_273e_coco.py --hpu=1 --lazy --batch-size=16 --eval-batch-size=64 --deterministic --log-interval 800 --epoch 2
```

- Run training on 1 HPU with 2 ephocs - Lazy autocast:
```
python mmdetection/tools/train.py yolov3/yolov3_d53_320_273e_coco.py --hpu=1 --lazy --autocast --deterministic --batch-size 32 --eval-batch-size 64 --log-interval 400 --epoch 2
```

### Multi-Card Training Examples
- Run training on 8 HPUs - Lazy mode:
```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 mmdetection/tools/train.py yolov3/yolov3_d53_320_273e_coco.py --launcher pytorch  --hpu=1 --cfg-options checkpoint_config.interval=30 optimizer.lr=0.0015 lr_config.warmup_iters=1000 lr_config.step="[228,256]" --lazy --batch-size=16 --eval-batch-size=64 --deterministic --log-interval=96
```

- Run training on 8 HPUs - Lazy autocast:
```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 mmdetection/tools/train.py yolov3/yolov3_d53_320_273e_coco.py --launcher pytorch  --hpu=1 --autocast  --cfg-options checkpoint_config.interval=30 optimizer.lr=0.002 lr_config.warmup_iters=400 lr_config.step="[228,256]" --lazy --batch-size=32 --eval-batch-size=64 --deterministic --log-interval=48
```

## Inference
- Run inference on 1 HPU with saved checkpoint:
```
python mmdetection/tools/test.py yolov3/yolov3_d53_320_273e_coco.py <path to checkpoint.pth> --eval bbox
```

## Supported Configurations
| Device | SynapseAI Version | PyTorch Version |
|--------|-------------------|-----------------|
| Gaudi  | 1.7.1            | 1.13.0          |

## Changelog
### Training Script Modifications
- Modified mmcv mmcv/runner/hooks/epoch_based_runner.py for HPU migration and additional training statistics.
- Modified mmcv mmcv/parallel/data_parallel.py, distributed.py, mmcv/runner/dist_utils.py to
support single and distributed training on HPU.
- Modified mmdetection mmdet/apis/train.py to support DistributedDataParallel on HPU.
- Mofidied mmdetection tools/train.py to add extra arguments for training on HPU.
- Modified mmcv mmcv/runner/epoch_based_runner.py, log_buffer.py, mmdetection mmdet/models/detectors/base.py
to log training result based on logging interval for performance improvement.
- Added mmcv/mmcv/utils/hpu.py and modified mmcv/utils/__init__.py to include all utility functions on HPU.
- Modified mmcv mmcv/runner/hooks/optimizer.py for HPU enablement and performance/accuracy improvement.
- Modified mmdetection mmdet/models/dense_heads/yolo_head.py to remove pos_and_neg_mask check in loss_single
check and move loss function prior_generator for target/neg map list to CPU for performance improvement.
- Modified mmdetection mmdet/models/dense_heads/yolo_head.py get_bboxes for performance improvement.
- Modified mmdet/models/dense_heads/base_dense_head.py to improve performance by device/host parallism
- Modified mmcv mmcv/runner/hooks/evaluation.py, mmdetection tools/test.py, mmdet/models/dense_heads/yolo_head.py
for inference on HPU.
