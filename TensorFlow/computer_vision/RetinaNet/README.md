# RetinaNet

This repository provides scripts to train RetinaNet model for Tensorflow on Habana
Gaudi<sup>TM</sup> device.
For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

Please visit [this page](https://developer.habana.ai/resources/habana-training-models/#performance) for performance information.

## Table of Contents
*  [Model Overview](#model-overview)
*  [Setup](#Setup)
*  [Training the Model](#training-the-model)
*  [Examples](#examples)
*  [Known Issues](#known-issues)

## Model Overview

RetinaNet is a one-stage object detection model that utilizes a focal loss function to address class imbalance during training. 
Focal loss applies a modulating term to the cross entropy loss in order to focus learning on hard negative examples. 
RetinaNet is a single, unified network composed of a backbone network and two task-specific subnetworks. 
The backbone is responsible for computing a convolutional feature map over an entire input image and 
is an off-the-self convolutional network. The first subnet performs convolutional object classification on the 
backbone's output; the second subnet performs convolutional bounding box regression. The two subnetworks feature a simple 
design that the authors propose specifically for one-stage, dense detection. 
The complete description of RetinaNet model can be found in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) paper.


This repository implements training RetinaNet on a single HPU. The implementation is based on source code of:  
https://github.com/tensorflow/models/tree/03944bb4c9138672392f4ac52fd68bc17b5f5792, and tested with TensorFlow 2.5.0

### Changes to the original model
* Enabled HPU
* Disabled dynamic shapes
* Disabled eager to function rewriter
* Applied workaround for high dimensionality tensor until N-Dims support is available
* Removed most of the code not used by RetinaNet


## Setup

Please follow the instructions given in the following link for setting up the
environment including the `$PYTHON` environment variable: [Gaudi Setup and
Installation Guide](https://github.com/HabanaAI/Setup_and_Install). Please
answer the questions in the guide according to your preferences. This guide will
walk you through the process of setting up your system to run the model on
Gaudi.

In the docker container, clone this repository and switch to the branch that
matches your SynapseAI version. (Run the
[`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.) For example, if your SynapseAI
version is 1.0.1,

```bash
git clone -b 1.0.1 https://github.com/HabanaAI/Model-References
cd Model-References/TensorFlow/computer_vision/RetinaNet
```

### Generate Dataset
These steps will guide you to download and prepare COCO 2017 dataset from https://cocodataset.org/#download

#### Download COCO 2017 dataset
```
pushd official/vision/beta/data/
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip && rm train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip && rm val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
unzip test2017.zip && rm test2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip && rm annotations_trainval2017.zip
popd
```

#### Convert COCO 2017 data to tfrecord
```
pushd official/vision/beta/data/
mkdir tfrecord
# training data
PYTHONPATH="../../../../:$PYTHONPATH" $PYTHON ./create_coco_tf_record.py \
--image_dir=./train2017 \
--caption_annotations_file=./annotations/captions_train2017.json \
--output_file_prefix=tfrecord/train \
--num_shards=32
# validation data
PYTHONPATH="../../../../:$PYTHONPATH" $PYTHON ./create_coco_tf_record.py \
--image_dir=./val2017 \
--caption_annotations_file=./annotations/captions_val2017.json \
--output_file_prefix=tfrecord/val \
--num_shards=32
popd
```

### Download the backbone checkpoint
Backbone model is ResNet-50, RetinaNet starts with a pre-trained checkpoint of the backbone. A pre-trained checkpoint can be obtained with the following commands:
```
mkdir backbone
pushd backbone
wget https://storage.googleapis.com/cloud-tpu-checkpoints/vision-2.0/resnet50_imagenet/checkpoint
wget https://storage.googleapis.com/cloud-tpu-checkpoints/vision-2.0/resnet50_imagenet/ckpt-28080.data-00000-of-00001
wget https://storage.googleapis.com/cloud-tpu-checkpoints/vision-2.0/resnet50_imagenet/ckpt-28080.index
popd
```

## Training the Model
Add the model root path to PYTHONPATH
```
export PYTHONPATH=<path-to-Model-References>/TensorFlow/computer_vision/RetinaNet:$PYTHONPATH
```

Run on single HPU
```
cd official/vision/beta
$PYTHON train.py [options]
```

Available options
```
  `--experiment`        | The experiment type registered
  `--mode`              | Mode to run: `train`, `eval`, `train_and_eval`, `continuous_eval`, `continuous_train_and_eval`
  `--model_dir`         | The directory where the model and training/evaluation summaries
  `--config_file`       | YAML/JSON files which specifies overrides
  `--params_override`   | YAML/JSON string or a YAML file which specifies additional overrides over the default parameters and those specified in `--config_file`
  `--gin_file`          | List of paths to the config files
  `--gin_params`        | Newline separated list of Gin parameter bindings
  `--tpu`               | The Cloud TPU to use for training
  `--tf_data_service`   | The tf.data service address
```

Full config JSON
```json
{
    'runtime': {
        'all_reduce_alg': None,
        'batchnorm_spatial_persistent': False,
        'dataset_num_private_threads': None,
        'default_shard_dim': -1,
        'distribution_strategy': 'one_device',
        'dump_config': None,
        'enable_xla': False,
        'gpu_thread_mode': None,
        'loss_scale': None,
        'mixed_precision_dtype': 'float32',
        'num_cores_per_replica': 1,
        'num_gpus': 0,
        'num_hpus': 1,
        'num_packs': 1,
        'per_gpu_thread_count': 0,
        'run_eagerly': False,
        'task_index': -1,
        'tpu': None,
        'tpu_enable_xla_dynamic_padder': None,
        'worker_hosts': None
    },
    'task': {
        'annotation_file': '',
        'init_checkpoint': 'gs://cloud-tpu-checkpoints/vision-2.0/resnet50_imagenet/ckpt-28080',
        'init_checkpoint_modules': 'backbone',
        'losses': {
            'box_loss_weight': 50,
            'focal_loss_alpha': 0.25,
            'focal_loss_gamma': 1.5,
            'huber_loss_delta': 0.1,
            'l2_weight_decay': 0.0001
        },
        'model': {
            'anchor': {
                'anchor_size': 4.0,
                'aspect_ratios': [0.5, 1.0, 2.0],
                'num_scales': 3
            },
            'backbone': {
                'resnet': {
                    'depth_multiplier': 1.0,
                    'model_id': 50,
                    'replace_stem_max_pool': False,
                    'resnetd_shortcut': False,
                    'se_ratio': 0.0,
                    'stem_type': 'v0',
                    'stochastic_depth_drop_rate': 0.0
                },
                'type': 'resnet'
            },
            'decoder': {
                'fpn': {
                    'num_filters': 256,
                    'use_separable_conv': False
                },
                'type': 'fpn'
            },
            'detection_generator': {
                'apply_nms': True,
                'max_num_detections': 100,
                'nms_iou_threshold': 0.5,
                'pre_nms_score_threshold': 0.05,
                'pre_nms_top_k': 5000,
                'use_batched_nms': False
            },
            'head': {
                'attribute_heads': None,
                'num_convs': 4,
                'num_filters': 256,
                'use_separable_conv': False
            },
            'input_size': [640, 640, 3],
            'max_level': 7,
            'min_level': 3,
            'norm_activation': {
                'activation': 'relu',
                'norm_epsilon': 0.001,
                'norm_momentum': 0.99,
                'use_sync_bn': True
            },
            'num_classes': 91
        },
        'per_category_metrics': False,
        'train_data': {
            'block_length': 1,
            'cache': False,
            'cycle_length': None,
            'decoder': {
                'simple_decoder': {
                    'regenerate_source_id': False
                },
                'type': 'simple_decoder'
            },
            'deterministic': None,
            'drop_remainder': True,
            'dtype': 'bfloat16',
            'enable_tf_data_service': False,
            'file_type': 'tfrecord',
            'global_batch_size': 8,
            'input_path': 'data/tfrecord/train*',
            'is_training': True,
            'parser': {
                'aug_rand_hflip': True,
                'aug_scale_max': 1.2,
                'aug_scale_min': 0.8,
                'match_threshold': 0.5,
                'max_num_instances': 100,
                'num_channels': 3,
                'skip_crowd_during_training': True,
                'unmatched_threshold': 0.5
            },
            'seed': None,
            'sharding': True,
            'shuffle_buffer_size': 1000,
            'tf_data_service_address': None,
            'tf_data_service_job_name': None,
            'tfds_as_supervised': False,
            'tfds_data_dir': '',
            'tfds_name': '',
            'tfds_skip_decoding_feature': '',
            'tfds_split': ''
        },
        'validation_data': {
            'block_length': 1,
            'cache': False,
            'cycle_length': None,
            'decoder': {
                'simple_decoder': {
                    'regenerate_source_id': False
                },
                'type': 'simple_decoder'
            },
            'deterministic': None,
            'drop_remainder': True,
            'dtype': 'bfloat16',
            'enable_tf_data_service': False,
            'file_type': 'tfrecord',
            'global_batch_size': 8,
            'input_path': 'data/tfrecord/val*',
            'is_training': False,
            'parser': {
                'aug_rand_hflip': False,
                'aug_scale_max': 1.0,
                'aug_scale_min': 1.0,
                'match_threshold': 0.5,
                'max_num_instances': 100,
                'num_channels': 3,
                'skip_crowd_during_training': True,
                'unmatched_threshold': 0.5
            },
            'seed': None,
            'sharding': True,
            'shuffle_buffer_size': 10000,
            'tf_data_service_address': None,
            'tf_data_service_job_name': None,
            'tfds_as_supervised': False,
            'tfds_data_dir': '',
            'tfds_name': '',
            'tfds_skip_decoding_feature': '',
            'tfds_split': ''
        }
    },
    'trainer': {
        'allow_tpu_summary': False,
        'best_checkpoint_eval_metric': '',
        'best_checkpoint_export_subdir': '',
        'best_checkpoint_metric_comp': 'higher',
        'checkpoint_interval': 462,
        'continuous_eval_timeout': 3600,
        'eval_tf_function': True,
        'eval_tf_while_loop': False,
        'loss_upper_bound': 1000000.0,
        'max_to_keep': 5,
        'optimizer_config': {
            'ema': None,
            'learning_rate': {
                'stepwise': {
                    'boundaries': [26334, 30954],
                    'name': 'PiecewiseConstantDecay',
                    'values': [0.01,
                        0.001,
                        0.0001
                    ]
                },
                'type': 'stepwise'
            },
            'optimizer': {
                'sgd': {
                    'clipnorm': None,
                    'clipvalue': None,
                    'decay': 0.0,
                    'global_clipnorm': None,
                    'momentum': 0.9,
                    'name': 'SGD',
                    'nesterov': False
                },
                'type': 'sgd'
            },
            'warmup': {
                'linear': {
                    'name': 'linear',
                    'warmup_learning_rate': 0.0067,
                    'warmup_steps': 500
                },
                'type': 'linear'
            }
        },
        'recovery_begin_steps': 0,
        'recovery_max_trials': 0,
        'steps_per_loop': 462,
        'summary_interval': 462,
        'train_steps': 33264,
        'train_tf_function': True,
        'train_tf_while_loop': True,
        'validation_interval': 462,
        'validation_steps': 625,
        'validation_summary_subdir': 'validation'
    }
}
```

## Examples
Train on single HPU with batch size of 8 on COCO dataset
```
$PYTHON train.py --experiment=retinanet_resnetfpn_coco --model_dir=/tmp/retina_train --mode=train_and_eval --config_file=configs/experiments/retinanet/config_beta_retinanet_1_hpu_batch_8.yaml
```

Train on single HPU, train 3 steps, evaluate 5 steps, use COCO dataset from /datasets/coco2017, load backbone checkpoint from /data/init_checkpoint
```
$PYTHON train.py --experiment=retinanet_resnetfpn_coco --model_dir=/tmp/retina_train --mode=train_and_eval --config_file=configs/experiments/retinanet/config_beta_retinanet_1_hpu_batch_8.yaml --params_override="{'task': {'init_checkpoint': '/data/init_checkpoint/ckpt-28080', 'train_data':{'input_path': '/datasets/coco2017/tf_records/train*'}, 'validation_data': {'input_path': '/datasets/coco2017/tf_records/val*'}  }, 'trainer': { 'train_steps' : 3, 'steps_per_loop': 3, 'validation_steps': 5 }}"
```

## Known Issues
* Only single HPU training is available. Multiple HPU will be available soon.