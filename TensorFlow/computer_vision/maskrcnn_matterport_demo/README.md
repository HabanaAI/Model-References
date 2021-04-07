# Mask R-CNN for Object Detection and Segmentation

**Table of Contents**
* [Model overview](#model-overview)
* [Setup](#setup)
* [Training the Model](#training-the-model)
* [Known issues](#known-issues)
* [Changes in the model](#changes-in-the-model)

## Model overview
This repository provides a script to train the Mask R-CNN model for Tensorflow on Habana
Gaudi<sup>TM</sup>, and is an optimized version of the implementation in [Matterport](https://github.com/matterport/Mask_RCNN "Matterport")
for Gaudi<sup>TM</sup>. Compared to the reference implementation, the differences in this optimized
version are mixed precision support and higher training performance using Habana specific kernels.
The model architecture in this repo has been modified to have no dynamic shape. For the details on what was changed in the model, please see the [Changes in the model](#changes-in-the-model).

### Model architecture
- Mask R-CNN built on Feature Pyramid Network (FPN) and ResNet50 backbone
- Region proposal network (RPN)
- ROI align
- Bounding box and classification box
- Mask head

### Default configuration
It reads the default model configuration from the `MrcnnConfig` class in `mrcnn/config.py`:
- Model parameters
    - Images are resized and padded with zeros to get a square image of size [1024, 1024]
    - RPN ahchor stride set to 1
    - RPN anchor sizes set to (32, 64, 128, 256, 512)
    - Number of ROIs per image set to 256
    - 6000 ROIs kept before non-maximum suppression(NMS)
    - 2000 ROIs kept after non-maximum suppression(NMS) for training
    - 1000 ROIs kept after non-maximum suppression(NMS) for inference
    - RPN NMS threshold set to 0.7

- Hyperparameters
    - Momentum: 0.9
    - Learning rate: 0.005
    - Batch size per card: 2
    - Warmup: 1000 steps

## Setup
This model is tested with the Habana TensorFlow docker container 0.13.0-380 for Ubuntu 18.04 and
some dependencies contained within it.

### Requirements
- Docker version 19.03.12 or newer
- Sudo access to install required drivers/firmware

### Install Drivers
Follow steps in the [Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) to install the driver.

### Download the COCO 2017 dataset and unzip files
```
mkdir ./data && cd ./data
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip && rm train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip && rm val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
unzip test2017.zip && rm test2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip && rm annotations_trainval2017.zip
```
If you want to use your own dataset, read [Training on your own dataset](https://github.com/matterport/Mask_RCNN#training-on-your-own-dataset "Training on your own dataset").

In summary, to train the model on your own dataset you'll need to extend two classes:
- `Config`: This class contains the default configuration. Subclass it and modify the attributes you
need to change.
- `Dataset`: This class provides a consistent way to work with any dataset. It allows you to use new
datasets for training without having to change the code of the model. It also supports loading
multiple datasets at the same time, which is useful if the objects you want to detect are not all
available in one dataset.


## Training the Model
1. Stop running dockers
```
docker stop $(docker ps -a -q)
```

2. Download docker
```
docker pull vault.habana.ai/gaudi-docker/0.13.0/ubuntu18.04/habanalabs/tensorflow-installer:0.13.0-380
```

3. Run docker
```
docker run -it -v /dev:/dev --device=/dev:/dev -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice  -v /sys/kernel/debug:/sys/kernel/debug --net=host vault.habana.ai/gaudi-docker/0.13.0/ubuntu18.04/habanalabs/tensorflow-installer:0.13.0-380
```

4. Clone the repository
```
git clone https://github.com/HabanaAI/Model-References
cd Model-References/TensorFlow/computer_vision/maskrcnn_matterport_demo/
```

5. Run training
Add your Model-References repo path to PYTHONPATH.
```
export PYTHONPATH=<MODEL_REFERENCES_ROOT_PATH>:$PYTHONPATH
```
The `demo_mask_rcnn.py` is a wrapper file for the `samples/coco/coco.py` to execute training and evaluation.
Run BF16 single-card training using the `demo_mask_rcnn.py`:
```
python3 demo_mask_rcnn.py --command train \
        --dataset ./data/ \
        --model keras \
        --backbone kapp_ResNet50 \
        --epochs 1 \
        --dtype bf16 \
        --device HPU
```
The other way to run BF16 single-card training:
```
TF_ENABLE_BF16_CONVERSION=1 python3 samples/coco/coco.py train \
        --backbone kapp_ResNet50 \
        --model keras \
        --dataset ./data/ \
        --epochs 1 \
        --device HPU
```
Run BF16 8-cards training:
```
python3 demo_mask_rcnn.py --command train \
        --dataset ./data/ \
        --model keras \
        --backbone kapp_ResNet50 \
        --epochs 1 \
        --device HPU \
        --dtype bf16 \
        --hvd_workers 8
```
Run BF16 evaluation:
```
TF_ENABLE_BF16_CONVERSION=1 python3 samples/coco/coco.py evaluate \
        --backbone kapp_ResNet50 \
        --model <PATH_TO_SAVED_MODEL> \
        --short \
        --dataset ./data/
```

### Parameters
You can modify the training behavior through the various flags in the `demo_mask_rcnn.py` script and the `samples/coco/coco.py`. Flags in the `demo_mask_rcnn.py` script are as follows:
- `--command`: Run a `train` or `evaluate` on MS COCO
- `--dataset`: Dataset directory
- `--model`: Use `keras` to load backbone weights automatically from Keras Applications.
`last` finds the last checkpoint file of the trained model in the log directory (./logs).
`imagenet` downloads ResNet50 weights trained on ImageNet in .h5 format.
- `--backbone`: `kapp_ResNet50` or `resnet101`.
`kapp_ResNet50` instantiates the ResNet50 architecture from Keras Applications. The pre-trained
weights are downloaded automatically and the model is built upon instantiation. When the
`--backbone` is `kapp_ResNet50`, the `--model` should be `keras`.
`resnet101` builds a fresh ResNet101 graph.
- `--limit`: Number of images to use for evaluation. This is not used during training.
- `--epochs`: Number of epochs
- `--steps_per_epoch`: Number of steps per single epoch
- `--dtype`: Data type, `fp32` or `bf16`
`bf16` automatically converts the appropriate ops to the bfloat16 format. This approach is similar
to Automatic Mixed Precision of TensorFlow, which can reduce memory requirements and speed up
training.
- `--device`: Device selection, `HPU` or `CPU` or `GPU`

### Examples
| Command |
| ------- |
| `python3 demo_mask_rcnn.py --command train --dataset ./data/ --model keras --backbone kapp_ResNet50 --epochs 1 --dtype bf16 --device HPU` |
| `python3 demo_mask_rcnn.py --command train --dataset ./data/ --model keras --backbone kapp_ResNet50 --epochs 1 --device HPU` |
| `RUN_TPC_FUSER=false TF_ENABLE_BF16_CONVERSION=1 python3 samples/coco/coco.py train --backbone kapp_ResNet50 --model keras --short --dataset ./data/ --epochs 1 --train-layers 3+` |

## Changes in the model
The following changes were made to the original model to make the model functional on Gaudi or to improve the model's training performance:
* Support for Habana device was added.
* Horovod support.
* Runnable from TensorFlow 2.2, with keras replaced by tf.keras
* Int64 and assert workarounds.
* Script modified to offload dynamic shapes to CPU or use padding.
* Script improvements:
	* Improved root directory path detection in coco.py
	* Coco script is runnable as a module
	* Improved pool of parameters (selectable number of epochs and steps, TensorFlow timeline dump, post-training validation can be disabled, more deterministic run)
* Added support for tf.image.combined_non_max_suppression.
* Added custom Pyramid Roi Align functional blocks – fuses multiple tf.image.crop_and_resize and tf.nn.avg_pool2d.
* Backbone changed to “kapp_ResNet50”.
* SGD_with_colocate_grad – fix BW to use the same device as FW.
* CocoScheduler – learning rate regime.

## Known issues
- Currently we only support SGD optimizer for training.
- The Evaluation command in the [Training the Model](#training-the-model) works with the environmental variable `RUN_TPC_FUSER=false`.
- Disabling custom roi_align op (`samples/coco/coco.py --custom_roi 0`) fails during training.
- Training crashes when the use of tf.image.combined_non_max_supression is disabled (`samples/coco/coco.py --combined_nms 0`).

## Preview of TensorFlow MaskRCNN Python scripts with yaml configuration of parameters
For single card (in the future multi-node workloads will be supported as well) you can use model runners that are written in Python as opposed to bash.

You can run the following script: **model_garden/TensorFlow/habana_model_runner.py** which accepts two arguments:
- --model *model_name*
- --hb_config *path_to_yaml_config_file*

Example of config files can be found in the **model_garden/TensorFlow/computer_vision/maskrcnn_matterport_demo/maskrcnn_default.yaml**.

You can use these scripts as such:
> cd model_garden/TensorFlow/computer_vision/maskrcnn_matterport_demo
  >
  > python3 ../../habana_model_runner.py --model maskrcnn --hb_config *path_to_yaml_config_file*
  >
  > **Example:**
  >
  > python3 ../../habana_model_runner.py --model maskrcnn --hb_config maskrcnn_default.yaml