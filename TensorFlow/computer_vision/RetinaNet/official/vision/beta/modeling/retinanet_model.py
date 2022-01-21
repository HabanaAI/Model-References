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

"""RetinaNet."""
from typing import Any, Mapping, List, Optional, Union

# Import libraries
import tensorflow as tf

from official.vision.beta.ops import anchor


@tf.keras.utils.register_keras_serializable(package='Vision')
class RetinaNetModel(tf.keras.Model):
  """The RetinaNet model class."""

  def __init__(self,
               backbone: tf.keras.Model,
               decoder: tf.keras.Model,
               head: tf.keras.layers.Layer,
               detection_generator: tf.keras.layers.Layer,
               min_level: Optional[int] = None,
               max_level: Optional[int] = None,
               num_scales: Optional[int] = None,
               aspect_ratios: Optional[List[float]] = None,
               anchor_size: Optional[float] = None,
               **kwargs):
    """Classification initialization function.

    Args:
      backbone: `tf.keras.Model` a backbone network.
      decoder: `tf.keras.Model` a decoder network.
      head: `RetinaNetHead`, the RetinaNet head.
      detection_generator: the detection generator.
      min_level: Minimum level in output feature maps.
      max_level: Maximum level in output feature maps.
      num_scales: A number representing intermediate scales added
        on each level. For instances, num_scales=2 adds one additional
        intermediate anchor scales [2^0, 2^0.5] on each level.
      aspect_ratios: A list representing the aspect raito
        anchors added on each level. The number indicates the ratio of width to
        height. For instances, aspect_ratios=[1.0, 2.0, 0.5] adds three anchors
        on each scale level.
      anchor_size: A number representing the scale of size of the base
        anchor to the feature stride 2^level.
      **kwargs: keyword arguments to be passed.
    """
    super(RetinaNetModel, self).__init__(**kwargs)
    self._config_dict = {
        'backbone': backbone,
        'decoder': decoder,
        'head': head,
        'detection_generator': detection_generator,
        'min_level': min_level,
        'max_level': max_level,
        'num_scales': num_scales,
        'aspect_ratios': aspect_ratios,
        'anchor_size': anchor_size,
    }
    self._backbone = backbone
    self._decoder = decoder
    self._head = head
    self._detection_generator = detection_generator

  def call(self,
           images: tf.Tensor,
           image_shape: Optional[tf.Tensor] = None,
           anchor_boxes: Optional[Mapping[str, tf.Tensor]] = None,
           training: bool = None) -> Mapping[str, tf.Tensor]:
    """Forward pass of the RetinaNet model.

    Args:
      images: `Tensor`, the input batched images, whose shape is
        [batch, height, width, 3].
      image_shape: `Tensor`, the actual shape of the input images, whose shape
        is [batch, 2] where the last dimension is [height, width]. Note that
        this is the actual image shape excluding paddings. For example, images
        in the batch may be resized into different shapes before padding to the
        fixed size.
      anchor_boxes: a dict of tensors which includes multilevel anchors.
        - key: `str`, the level of the multilevel predictions.
        - values: `Tensor`, the anchor coordinates of a particular feature
            level, whose shape is [height_l, width_l, num_anchors_per_location].
      training: `bool`, indicating whether it is in training mode.

    Returns:
      scores: a dict of tensors which includes scores of the predictions.
        - key: `str`, the level of the multilevel predictions.
        - values: `Tensor`, the box scores predicted from a particular feature
            level, whose shape is
            [batch, height_l, width_l, num_classes * num_anchors_per_location].
      boxes: a dict of tensors which includes coordinates of the predictions.
        - key: `str`, the level of the multilevel predictions.
        - values: `Tensor`, the box coordinates predicted from a particular
            feature level, whose shape is
            [batch, height_l, width_l, 4 * num_anchors_per_location].
      attributes: a dict of (attribute_name, attribute_predictions). Each
        attribute prediction is a dict that includes:
        - key: `str`, the level of the multilevel predictions.
        - values: `Tensor`, the attribute predictions from a particular
            feature level, whose shape is
            [batch, height_l, width_l, att_size * num_anchors_per_location].
    """
    # Feature extraction.
    features = self.backbone(images)
    if self.decoder:
      features = self.decoder(features)

    # Dense prediction. `raw_attributes` can be empty.
    raw_scores, raw_boxes, raw_attributes = self.head(features)

    if training:
      outputs = {
          'cls_outputs': raw_scores,
          'box_outputs': raw_boxes,
      }
      if raw_attributes:
        outputs.update({'attribute_outputs': raw_attributes})
      return outputs
    else:
      # Generate anchor boxes for this batch if not provided.
      if anchor_boxes is None:
        _, image_height, image_width, _ = images.get_shape().as_list()
        anchor_boxes = anchor.Anchor(
            min_level=self._config_dict['min_level'],
            max_level=self._config_dict['max_level'],
            num_scales=self._config_dict['num_scales'],
            aspect_ratios=self._config_dict['aspect_ratios'],
            anchor_size=self._config_dict['anchor_size'],
            image_size=(image_height, image_width)).multilevel_boxes
        for l in anchor_boxes:
          anchor_boxes[l] = tf.tile(
              tf.expand_dims(anchor_boxes[l], axis=0),
              [tf.shape(images)[0], 1, 1, 1])

      # Post-processing.
      final_results = self.detection_generator(
          raw_boxes, raw_scores, anchor_boxes, image_shape, raw_attributes)
      outputs = {
          'cls_outputs': raw_scores,
          'box_outputs': raw_boxes,
      }
      if self.detection_generator.get_config()['apply_nms']:
        outputs.update({
            'detection_boxes': final_results['detection_boxes'],
            'detection_scores': final_results['detection_scores'],
            'detection_classes': final_results['detection_classes'],
            'num_detections': final_results['num_detections']
        })
      else:
        outputs.update({
            'decoded_boxes': final_results['decoded_boxes'],
            'decoded_box_scores': final_results['decoded_box_scores']
        })

      if raw_attributes:
        outputs.update({
            'attribute_outputs': raw_attributes,
            'detection_attributes': final_results['detection_attributes'],
        })
      return outputs

  @property
  def checkpoint_items(
      self) -> Mapping[str, Union[tf.keras.Model, tf.keras.layers.Layer]]:
    """Returns a dictionary of items to be additionally checkpointed."""
    items = dict(backbone=self.backbone, head=self.head)
    if self.decoder is not None:
      items.update(decoder=self.decoder)

    return items

  @property
  def backbone(self) -> tf.keras.Model:
    return self._backbone

  @property
  def decoder(self) -> tf.keras.Model:
    return self._decoder

  @property
  def head(self) -> tf.keras.layers.Layer:
    return self._head

  @property
  def detection_generator(self) -> tf.keras.layers.Layer:
    return self._detection_generator

  def get_config(self) -> Mapping[str, Any]:
    return self._config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)
