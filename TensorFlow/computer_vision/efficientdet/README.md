# EfficientDet

## Table of Contents
* [Model overview](#Model-overview)
* [Setup](#Setup)
* [Training the Model](#Training-the-model)

## Model Overview
This Repository implements training EfficientDet on multiple HPUs by using horovod, supporting synchronized batch norm.
External source for this code is: https://github.com/itsliupeng/automl
[1] Mingxing Tan, Ruoming Pang, Quoc V. Le. EfficientDet: Scalable and Efficient Object Detection. CVPR 2020. Arxiv link: https://arxiv.org/abs/1911.09070

## Setup

Please follow the instructions given in the following link for setting up the
environment including the `$PYTHON` environment variable: [Gaudi Setup and
Installation Guide](https://github.com/HabanaAI/Setup_and_Install). Please
answer the questions in the guide according to your preferences. This guide will
walk you through the process of setting up your system to run the model on
Gaudi.

### Clone the repo

In the docker container, clone this repository and switch to the branch that
matches your SynapseAI version. (Run the
[`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.)

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```
Go to the EfficientDet directory:
```bash
cd Model-References/TensorFlow/computer_vision/efficientdet
```

### Prepare COCO 2017 Dataset
#### Download COCO 2017 dataset
```
cd dataset
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip && rm train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip && rm val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
unzip test2017.zip && rm test2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip && rm annotations_trainval2017.zip
```

#### Convert COCO 2017 data to tfrecord
```
mkdir tfrecord
# training data
PYTHONPATH=".:$PYTHONPATH" python3 ./dataset/create_coco_tfrecord.py \
  --image_dir=./dataset/train2017 \
  --caption_annotations_file=./dataset/annotations/captions_train2017.json \
  --output_file_prefix=tfrecord/train \
  --num_shards=32
# validation data
PYTHONPATH=".:$PYTHONPATH" python3 ./dataset/create_coco_tfrecord.py \
  --image_dir=./dataset/val2017 \
  --caption_annotations_file=./dataset/annotations/captions_val2017.json \
  --output_file_prefix=tfrecord/val \
  --num_shards=32
```

### Download the backbone checkpoint
```
cd backbone
wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b0.tar.gz
tar -xvzf efficientnet-b0.tar.gz
```

## Training the Model
Add the Model-References path to PYTHONPATH
```
export PYTHONPATH=/path/to/Model-References:$PYTHONPATH
```

demo_efficientdet.py is a distributed launcher for main.py. It accepts the same arguments as main.py, it runs 'main.py [ARGS] --use_horovod' via mpirun with generated HCL config."

run on single HPU
```
python demo_efficientdet.py [options]
```
run on multiple HPU
```
python demo_efficientdet.py --use_horovod [options]
```

Available options, none is mandatory:

demo_efficientdet.py
  `--use_horovod`             | Use horovod for distributed run, pass amount of Horovod workers
  `--hls_type` {HLS1,HLS1-H}  | HLS Type
  `--kubernetes_run` KUBERNETES_RUN | Kubernetes run
In addition demo_efficientdet.py gets all main.py flags mentioned below.

main.py
  `--backbone_ckpt`: Location of the ResNet50 checkpoint to use for model initialization.
  `--ckpt`: Start training from this EfficientDet checkpoint.
  `--cp_every_n_steps`: Number of iterations after which checkpoint is saved.
  `--deterministic`: Deterministic input data.
  `--eval_after_training`: Run one eval after the training finishes.
  `--eval_batch_size`: evaluation batch size
  `--eval_master`: GRPC URL of the eval master. Set to an appropriate value when running on CPU/GPU
  `--eval_samples`: The number of samples for evaluation.
  `--eval_timeout`: Maximum seconds between checkpoints before evaluation terminates.
  `--gcp_project`: Project name for the Cloud TPU-enabled project. If not specified, we will attempt to automatically detect the GCE project from metadata.
  `--hparams`: Comma separated k=v pairs of hyperparameters.
  `--input_partition_dims`: A list that describes the partition dims for all the tensors.;
  `--iterations_per_loop`: Number of iterations per TPU training loop
  `--log_every_n_steps`: Number of iterations after which training parameters are logged.
  `--min_eval_interval`: Minimum seconds between evaluations.
  `--mode`: Mode to run: train or eval (default: train)
  `--model_dir`: Location of model_dir
  `--model_name`: Model name: retinanet or efficientdet
  `--no_hpu`: Do not load Habana modules = train on CPU/GPU
  `--num_cores`: Number of TPU cores for training
  `--num_cores_per_replica`: Number of TPU cores perreplica when using spatial partition.
  `--num_epochs`: Number of epochs for training
  `--num_examples_per_epoch`: Number of examples in one epoch
  `--sbs_test`: Config topology run for sbs testing.
  `--testdev_dir`: COCO testdev dir. If true, ignorer val_json_file.
  `--tpu`: The Cloud TPU to use for training. This should be either the name used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.
  `--tpu_zone`: GCE zone where the Cloud TPU is located in. If not specified, we will attempt to automatically detect the GCE project from metadata.
  `--train_batch_size`: training batch size
  `--training_file_pattern`: Glob for training data files (e.g., COCO train - minival set)
  `--use_amp`: Use AMP
  `--use_fake_data`: Use fake input.
  `--use_horovod`: Use Horovod for distributed training
  `--use_spatial_partition`: Use spatial partition.
  `--use_tpu`: Use TPUs rather than CPUs/GPUs
  `--use_xla`: Use XLA
  `--val_json_file`: COCO validation JSON containing golden bounding boxes.
  `--validation_file_pattern`: Glob for evaluation tfrecords (e.g., COCO val2017 set)

## Examples

run on single HPU, with batch size of 8, full training of 300 epochs, on COCO dataset
```bash
python demo_efficientdet.py --mode=train --train_batch_size 8 --num_epochs 300 --dataset_dir "/software/data/tf/coco2017/tf_records" --backbone_ckpt "/software/data/tf/data/efficientdet/backbones/efficientnet-b0"
```

run on 8 HPUs, with batch size of 8, full training of 300 epochs, on COCO dataset
```bash
python3 demo_efficientdet.py --backbone_ckpt=/root/tensorflow_datasets/efficientdet/backbones/efficientnet-b0/ --training_file_pattern=/root/tensorflow_datasets/coco2017/tf_records/train-* --model_dir /tmp/efficientdet --use_horovod 8 --keep_checkpoint_max=300
```

## Known Issues
* Currently TPC fuser disabled.
* Currently maximum batch size is 32. Batch size 64 will be available soon.
* Only d0 variant is avalable. Variants d1 to d7 will be available in the future.
* Only single HPU training is available. Multiple HPU will be available soon.
