# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file has been modified by HabanaLabs

"""Factory methods to build models."""

# Import libraries

import tensorflow as tf

from official.vision.beta.configs import retinanet as retinanet_cfg
from official.vision.beta.modeling import backbones
from official.vision.beta.modeling import retinanet_model
from official.vision.beta.modeling.decoders import factory as decoder_factory
from official.vision.beta.modeling.heads import dense_prediction_heads
from official.vision.beta.modeling.layers import detection_generator

def build_retinanet(
    input_specs: tf.keras.layers.InputSpec,
    model_config: retinanet_cfg.RetinaNet,
    l2_regularizer: tf.keras.regularizers.Regularizer = None) -> tf.keras.Model:
  """Builds RetinaNet model."""
  norm_activation_config = model_config.norm_activation
  backbone = backbones.factory.build_backbone(
      input_specs=input_specs,
      backbone_config=model_config.backbone,
      norm_activation_config=norm_activation_config,
      l2_regularizer=l2_regularizer)
  backbone(tf.keras.Input(input_specs.shape[1:]))

  decoder = decoder_factory.build_decoder(
      input_specs=backbone.output_specs,
      model_config=model_config,
      l2_regularizer=l2_regularizer)

  head_config = model_config.head
  generator_config = model_config.detection_generator
  num_anchors_per_location = (
      len(model_config.anchor.aspect_ratios) * model_config.anchor.num_scales)

  head = dense_prediction_heads.RetinaNetHead(
      min_level=model_config.min_level,
      max_level=model_config.max_level,
      num_classes=model_config.num_classes,
      num_anchors_per_location=num_anchors_per_location,
      num_convs=head_config.num_convs,
      num_filters=head_config.num_filters,
      attribute_heads=[
          cfg.as_dict() for cfg in (head_config.attribute_heads or [])
      ],
      use_separable_conv=head_config.use_separable_conv,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer)

  detection_generator_obj = detection_generator.MultilevelDetectionGenerator(
      apply_nms=generator_config.apply_nms,
      pre_nms_top_k=generator_config.pre_nms_top_k,
      pre_nms_score_threshold=generator_config.pre_nms_score_threshold,
      nms_iou_threshold=generator_config.nms_iou_threshold,
      max_num_detections=generator_config.max_num_detections,
      use_batched_nms=generator_config.use_batched_nms)

  model = retinanet_model.RetinaNetModel(
      backbone,
      decoder,
      head,
      detection_generator_obj,
      min_level=model_config.min_level,
      max_level=model_config.max_level,
      num_scales=model_config.anchor.num_scales,
      aspect_ratios=model_config.anchor.aspect_ratios,
      anchor_size=model_config.anchor.anchor_size)
  return model