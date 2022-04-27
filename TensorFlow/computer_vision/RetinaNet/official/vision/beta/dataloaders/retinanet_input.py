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

"""Data parser and processing for RetinaNet.

Parse image and ground truths in a dataset to training targets and package them
into (image, labels) tuple for RetinaNet.
"""

# Import libraries
import tensorflow as tf

from official.vision.beta.dataloaders import parser
from official.vision.beta.dataloaders import utils
from official.vision.beta.ops import anchor
from official.vision.beta.ops import box_ops
from official.vision.beta.ops import preprocess_ops


class Parser(parser.Parser):
  """Parser to parse an image and its annotations into a dictionary of tensors."""

  def __init__(self,
               output_size,
               min_level,
               max_level,
               num_scales,
               aspect_ratios,
               anchor_size,
               match_threshold=0.5,
               unmatched_threshold=0.5,
               aug_rand_hflip=False,
               aug_scale_min=1.0,
               aug_scale_max=1.0,
               use_autoaugment=False,
               autoaugment_policy_name='v0',
               skip_crowd_during_training=True,
               max_num_instances=100,
               dtype='bfloat16',
               mode=None):
    """Initializes parameters for parsing annotations in the dataset.

    Args:
      output_size: `Tensor` or `list` for [height, width] of output image. The
        output_size should be divided by the largest feature stride 2^max_level.
      min_level: `int` number of minimum level of the output feature pyramid.
      max_level: `int` number of maximum level of the output feature pyramid.
      num_scales: `int` number representing intermediate scales added on each
        level. For instances, num_scales=2 adds one additional intermediate
        anchor scales [2^0, 2^0.5] on each level.
      aspect_ratios: `list` of float numbers representing the aspect raito
        anchors added on each level. The number indicates the ratio of width to
        height. For instances, aspect_ratios=[1.0, 2.0, 0.5] adds three anchors
        on each scale level.
      anchor_size: `float` number representing the scale of size of the base
        anchor to the feature stride 2^level.
      match_threshold: `float` number between 0 and 1 representing the
        lower-bound threshold to assign positive labels for anchors. An anchor
        with a score over the threshold is labeled positive.
      unmatched_threshold: `float` number between 0 and 1 representing the
        upper-bound threshold to assign negative labels for anchors. An anchor
        with a score below the threshold is labeled negative.
      aug_rand_hflip: `bool`, if True, augment training with random horizontal
        flip.
      aug_scale_min: `float`, the minimum scale applied to `output_size` for
        data augmentation during training.
      aug_scale_max: `float`, the maximum scale applied to `output_size` for
        data augmentation during training.
      use_autoaugment: `bool`, if True, use the AutoAugment augmentation policy
        during training.
      autoaugment_policy_name: `string` that specifies the name of the
        AutoAugment policy that will be used during training.
      skip_crowd_during_training: `bool`, if True, skip annotations labeled with
        `is_crowd` equals to 1.
      max_num_instances: `int` number of maximum number of instances in an
        image. The groundtruth data will be padded to `max_num_instances`.
      dtype: `str`, data type. One of {`bfloat16`, `float32`, `float16`}.
      mode: a ModeKeys. Specifies if this is training, evaluation, prediction or
        prediction with groundtruths in the outputs.
    """
    self._mode = mode
    self._max_num_instances = max_num_instances
    self._skip_crowd_during_training = skip_crowd_during_training

    # Anchor.
    self._output_size = output_size
    self._min_level = min_level
    self._max_level = max_level
    self._num_scales = num_scales
    self._aspect_ratios = aspect_ratios
    self._anchor_size = anchor_size
    self._match_threshold = match_threshold
    self._unmatched_threshold = unmatched_threshold

    # Data augmentation.
    self._aug_rand_hflip = aug_rand_hflip
    self._aug_scale_min = aug_scale_min
    self._aug_scale_max = aug_scale_max

    # Data Augmentation with AutoAugment.
    self._use_autoaugment = use_autoaugment
    self._autoaugment_policy_name = autoaugment_policy_name

    # Data type.
    self._dtype = dtype

  def _parse_train_data(self, data):
    """Parses data for training and evaluation."""
    classes = data['groundtruth_classes']
    boxes = data['groundtruth_boxes']
    # If not empty, `attributes` is a dict of (name, ground_truth) pairs.
    # `ground_gruth` of attributes is assumed in shape [N, attribute_size].
    # TODO(xianzhi): support parsing attributes weights.
    attributes = data.get('groundtruth_attributes', {})
    is_crowds = data['groundtruth_is_crowd']

    # Skips annotations with `is_crowd` = True.
    if self._skip_crowd_during_training:
      num_groundtrtuhs = tf.shape(input=classes)[0]
      with tf.control_dependencies([num_groundtrtuhs, is_crowds]):
        indices = tf.cond(
            pred=tf.greater(tf.size(input=is_crowds), 0),
            true_fn=lambda: tf.where(tf.logical_not(is_crowds))[:, 0],
            false_fn=lambda: tf.cast(tf.range(num_groundtrtuhs), tf.int64))
      classes = tf.gather(classes, indices)
      boxes = tf.gather(boxes, indices)
      for k, v in attributes.items():
        attributes[k] = tf.gather(v, indices)

    # Gets original image and its size.
    image = data['image']

    image_shape = tf.shape(input=image)[0:2]

    # Normalizes image with mean and std pixel values.
    image = preprocess_ops.normalize_image(image)

    # Flips image randomly during training.
    if self._aug_rand_hflip:
      image, boxes, _ = preprocess_ops.random_horizontal_flip(image, boxes)

    # Converts boxes from normalized coordinates to pixel coordinates.
    boxes = box_ops.denormalize_boxes(boxes, image_shape)

    # Resizes and crops image.
    image, image_info = preprocess_ops.resize_and_crop_image(
        image,
        self._output_size,
        padded_size=preprocess_ops.compute_padded_size(self._output_size,
                                                       2**self._max_level),
        aug_scale_min=self._aug_scale_min,
        aug_scale_max=self._aug_scale_max)
    image_height, image_width, _ = image.get_shape().as_list()

    # Resizes and crops boxes.
    image_scale = image_info[2, :]
    offset = image_info[3, :]
    boxes = preprocess_ops.resize_and_crop_boxes(boxes, image_scale,
                                                 image_info[1, :], offset)
    # Filters out ground truth boxes that are all zeros.
    indices = box_ops.get_non_empty_box_indices(boxes)
    boxes = tf.gather(boxes, indices)
    classes = tf.gather(classes, indices)
    for k, v in attributes.items():
      attributes[k] = tf.gather(v, indices)

    # Assigns anchors.
    input_anchor = anchor.build_anchor_generator(
        min_level=self._min_level,
        max_level=self._max_level,
        num_scales=self._num_scales,
        aspect_ratios=self._aspect_ratios,
        anchor_size=self._anchor_size)
    anchor_boxes = input_anchor(image_size=(image_height, image_width))
    anchor_labeler = anchor.AnchorLabeler(self._match_threshold,
                                          self._unmatched_threshold)
    (cls_targets, box_targets, att_targets, cls_weights,
     box_weights) = anchor_labeler.label_anchors(
         anchor_boxes, boxes, tf.expand_dims(classes, axis=1), attributes)

    # Casts input image to desired data type.
    image = tf.cast(image, dtype=self._dtype)

    # Packs labels for model_fn outputs.
    labels = {
        'cls_targets': cls_targets,
        'box_targets': box_targets,
        'anchor_boxes': anchor_boxes,
        'cls_weights': cls_weights,
        'box_weights': box_weights,
        'image_info': image_info,
    }
    if att_targets:
      labels['attribute_targets'] = att_targets
    return image, labels

  def _parse_eval_data(self, data):
    """Parses data for training and evaluation."""
    groundtruths = {}
    classes = data['groundtruth_classes']
    boxes = data['groundtruth_boxes']
    # If not empty, `attributes` is a dict of (name, ground_truth) pairs.
    # `ground_gruth` of attributes is assumed in shape [N, attribute_size].
    # TODO(xianzhi): support parsing attributes weights.
    attributes = data.get('groundtruth_attributes', {})

    # Gets original image and its size.
    image = data['image']
    image_shape = tf.shape(input=image)[0:2]

    # Normalizes image with mean and std pixel values.
    image = preprocess_ops.normalize_image(image)

    # Converts boxes from normalized coordinates to pixel coordinates.
    boxes = box_ops.denormalize_boxes(boxes, image_shape)

    # Resizes and crops image.
    image, image_info = preprocess_ops.resize_and_crop_image(
        image,
        self._output_size,
        padded_size=preprocess_ops.compute_padded_size(self._output_size,
                                                       2**self._max_level),
        aug_scale_min=1.0,
        aug_scale_max=1.0)
    image_height, image_width, _ = image.get_shape().as_list()

    # Resizes and crops boxes.
    image_scale = image_info[2, :]
    offset = image_info[3, :]
    boxes = preprocess_ops.resize_and_crop_boxes(boxes, image_scale,
                                                 image_info[1, :], offset)
    # Filters out ground truth boxes that are all zeros.
    indices = box_ops.get_non_empty_box_indices(boxes)
    boxes = tf.gather(boxes, indices)
    classes = tf.gather(classes, indices)
    for k, v in attributes.items():
      attributes[k] = tf.gather(v, indices)

    # Assigns anchors.
    input_anchor = anchor.build_anchor_generator(
        min_level=self._min_level,
        max_level=self._max_level,
        num_scales=self._num_scales,
        aspect_ratios=self._aspect_ratios,
        anchor_size=self._anchor_size)
    anchor_boxes = input_anchor(image_size=(image_height, image_width))
    anchor_labeler = anchor.AnchorLabeler(self._match_threshold,
                                          self._unmatched_threshold)
    (cls_targets, box_targets, att_targets, cls_weights,
     box_weights) = anchor_labeler.label_anchors(
         anchor_boxes, boxes, tf.expand_dims(classes, axis=1), attributes)

    # Casts input image to desired data type.
    image = tf.cast(image, dtype=self._dtype)

    # Sets up groundtruth data for evaluation.
    groundtruths = {
        'source_id': data['source_id'],
        'height': data['height'],
        'width': data['width'],
        'num_detections': tf.shape(data['groundtruth_classes']),
        'image_info': image_info,
        'boxes': box_ops.denormalize_boxes(
            data['groundtruth_boxes'], image_shape),
        'classes': data['groundtruth_classes'],
        'areas': data['groundtruth_area'],
        'is_crowds': tf.cast(data['groundtruth_is_crowd'], tf.int32),
    }
    if 'groundtruth_attributes' in data:
      groundtruths['attributes'] = data['groundtruth_attributes']
    groundtruths['source_id'] = utils.process_source_id(
        groundtruths['source_id'])
    groundtruths = utils.pad_groundtruths_to_fixed_size(
        groundtruths, self._max_num_instances)

    # Packs labels for model_fn outputs.
    labels = {
        'cls_targets': cls_targets,
        'box_targets': box_targets,
        'anchor_boxes': anchor_boxes,
        'cls_weights': cls_weights,
        'box_weights': box_weights,
        'image_info': image_info,
        'groundtruths': groundtruths,
    }
    if att_targets:
      labels['attribute_targets'] = att_targets
    return image, labels
