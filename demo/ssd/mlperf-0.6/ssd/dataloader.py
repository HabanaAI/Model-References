# Copyright 2018 Google. All Rights Reserved.
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
# ==============================================================================
"""Data loader and processing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools as it
import math

import numpy as np
import tensorflow as tf

from object_detection import argmax_matcher
from object_detection import box_list
from object_detection import faster_rcnn_box_coder
from object_detection import preprocessor
from object_detection import region_similarity_calculator
from object_detection import target_assigner
from object_detection import tf_example_decoder
from mlp_log import mlp_log
import ssd_constants


class DefaultBoxes(object):
  """Default bounding boxes for 300x300 5 layer SSD.

  Default bounding boxes generation follows the order of (W, H, anchor_sizes).
  Therefore, the tensor converted from DefaultBoxes has a shape of
  [anchor_sizes, H, W, 4]. The last dimension is the box coordinates; 'ltrb'
  is [ymin, xmin, ymax, xmax] while 'xywh' is [cy, cx, h, w].
  """

  def __init__(self):
    fk = ssd_constants.IMAGE_SIZE / np.array(ssd_constants.STEPS)

    self.default_boxes = []
    # size of feature and number of feature
    for idx, feature_size in enumerate(ssd_constants.FEATURE_SIZES):
      sk1 = ssd_constants.SCALES[idx] / ssd_constants.IMAGE_SIZE
      sk2 = ssd_constants.SCALES[idx+1] / ssd_constants.IMAGE_SIZE
      sk3 = math.sqrt(sk1*sk2)
      all_sizes = [(sk1, sk1), (sk3, sk3)]

      for alpha in ssd_constants.ASPECT_RATIOS[idx]:
        w, h = sk1 * math.sqrt(alpha), sk1 / math.sqrt(alpha)
        all_sizes.append((w, h))
        all_sizes.append((h, w))

      assert len(all_sizes) == ssd_constants.NUM_DEFAULTS[idx]

      for w, h in all_sizes:
        for i, j in it.product(range(feature_size), repeat=2):
          cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
          box = tuple(np.clip(k, 0, 1) for k in (cy, cx, h, w))
          self.default_boxes.append(box)

    assert len(self.default_boxes) == ssd_constants.NUM_SSD_BOXES
    mlp_log.mlperf_print('max_samples', ssd_constants.NUM_SSD_BOXES)

    def to_ltrb(cy, cx, h, w):
      return cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2

    # For IoU calculation
    self.default_boxes_ltrb = tuple(to_ltrb(*i) for i in self.default_boxes)

  def __call__(self, order='ltrb'):
    if order == 'ltrb': return self.default_boxes_ltrb
    if order == 'xywh': return self.default_boxes


def calc_iou_tensor(box1, box2):
  """ Calculation of IoU based on two boxes tensor,
      Reference to https://github.com/kuangliu/pytorch-ssd
      input:
          box1 (N, 4)
          box2 (M, 4)
      output:
          IoU (N, M)
  """
  N = tf.shape(box1)[0]
  M = tf.shape(box2)[0]

  be1 = tf.tile(tf.expand_dims(box1, axis=1), (1, M, 1))
  be2 = tf.tile(tf.expand_dims(box2, axis=0), (N, 1, 1))

  # Left Top & Right Bottom
  lt = tf.maximum(be1[:,:,:2], be2[:,:,:2])

  rb = tf.minimum(be1[:,:,2:], be2[:,:,2:])

  delta = tf.maximum(rb - lt, 0)

  intersect = delta[:,:,0]*delta[:,:,1]

  delta1 = be1[:,:,2:] - be1[:,:,:2]
  area1 = delta1[:,:,0]*delta1[:,:,1]
  delta2 = be2[:,:,2:] - be2[:,:,:2]
  area2 = delta2[:,:,0]*delta2[:,:,1]

  iou = intersect/(area1 + area2 - intersect)
  return iou


def ssd_crop(image, boxes, classes):
  """IoU biassed random crop.

  Reference: https://github.com/chauhan-utk/ssd.DomainAdaptation
  """

  num_boxes = tf.shape(boxes)[0]

  def no_crop_check():
    return (tf.random_uniform(shape=(), minval=0, maxval=1, dtype=tf.float32)
            < ssd_constants.P_NO_CROP_PER_PASS)

  def no_crop_proposal():
    return (
        tf.ones((), tf.bool),
        tf.convert_to_tensor([0, 0, 1, 1], dtype=tf.float32),
        tf.ones((num_boxes,), tf.bool),
    )

  def crop_proposal():
    rand_vec = lambda minval, maxval: tf.random_uniform(
        shape=(ssd_constants.NUM_CROP_PASSES, 1), minval=minval, maxval=maxval,
        dtype=tf.float32)

    width, height = rand_vec(0.3, 1), rand_vec(0.3, 1)
    left, top = rand_vec(0, 1-width), rand_vec(0, 1-height)

    right = left + width
    bottom = top + height

    ltrb = tf.concat([left, top, right, bottom], axis=1)

    min_iou = tf.random_shuffle(ssd_constants.CROP_MIN_IOU_CHOICES)[0]
    ious = calc_iou_tensor(ltrb, boxes)

    # discard any bboxes whose center not in the cropped image
    xc, yc = [tf.tile(0.5 * (boxes[:, i + 0] + boxes[:, i + 2])[tf.newaxis, :],
                      (ssd_constants.NUM_CROP_PASSES, 1)) for i in range(2)]

    masks = tf.reduce_all(tf.stack([
        tf.greater(xc, tf.tile(left, (1, num_boxes))),
        tf.less(xc, tf.tile(right, (1, num_boxes))),
        tf.greater(yc, tf.tile(top, (1, num_boxes))),
        tf.less(yc, tf.tile(bottom, (1, num_boxes))),
    ], axis=2), axis=2)

    # Checks of whether a crop is valid.
    valid_aspect = tf.logical_and(tf.less(height/width, 2),
                                  tf.less(width/height, 2))
    valid_ious = tf.reduce_all(tf.greater(ious, min_iou), axis=1, keepdims=True)
    valid_masks = tf.reduce_any(masks, axis=1, keepdims=True)

    valid_all = tf.cast(tf.reduce_all(tf.concat(
        [valid_aspect, valid_ious, valid_masks], axis=1), axis=1), tf.int32)

    # One indexed, as zero is needed for the case of no matches.
    index = tf.range(1, 1 + ssd_constants.NUM_CROP_PASSES, dtype=tf.int32)

    # Either one-hot, or zeros if there is no valid crop.
    selection = tf.equal(tf.reduce_max(index * valid_all), index)

    use_crop = tf.reduce_any(selection)
    output_ltrb = tf.reduce_sum(tf.multiply(ltrb, tf.tile(tf.cast(
        selection, tf.float32)[:, tf.newaxis], (1, 4))), axis=0)
    output_masks = tf.reduce_any(tf.logical_and(masks, tf.tile(
        selection[:, tf.newaxis], (1, num_boxes))), axis=0)

    return use_crop, output_ltrb, output_masks

  def proposal(*args):
    return tf.cond(
        pred=no_crop_check(),
        true_fn=no_crop_proposal,
        false_fn=crop_proposal,
    )

  _, crop_bounds, box_masks = tf.while_loop(
      cond=lambda x, *_: tf.logical_not(x),
      body=proposal,
      loop_vars=[tf.zeros((), tf.bool), tf.zeros((4,), tf.float32), tf.zeros((num_boxes,), tf.bool)],
  )

  filtered_boxes = tf.boolean_mask(boxes, box_masks, axis=0)

  # Clip boxes to the cropped region.
  filtered_boxes = tf.stack([
      tf.maximum(filtered_boxes[:, 0], crop_bounds[0]),
      tf.maximum(filtered_boxes[:, 1], crop_bounds[1]),
      tf.minimum(filtered_boxes[:, 2], crop_bounds[2]),
      tf.minimum(filtered_boxes[:, 3], crop_bounds[3]),
  ], axis=1)

  left = crop_bounds[0]
  top = crop_bounds[1]
  width = crop_bounds[2] - left
  height = crop_bounds[3] - top

  cropped_boxes = tf.stack([
      (filtered_boxes[:, 0] - left) / width,
      (filtered_boxes[:, 1] - top) / height,
      (filtered_boxes[:, 2] - left) / width,
      (filtered_boxes[:, 3] - top) / height,
  ], axis=1)

  cropped_image = tf.image.crop_and_resize(
      image=image[tf.newaxis, :, :, :],
      boxes=crop_bounds[tf.newaxis, :],
      box_ind=tf.zeros((1,), tf.int32),
      crop_size=(ssd_constants.IMAGE_SIZE, ssd_constants.IMAGE_SIZE),
  )[0, :, :, :]

  cropped_classes = tf.boolean_mask(classes, box_masks, axis=0)

  return cropped_image, cropped_boxes, cropped_classes


def color_jitter(image, brightness=0, contrast=0, saturation=0, hue=0):
  """Distorts the color of the image.

  Args:
    image: The input image tensor.
    brightness: A float, specifying the brightness for color jitter.
    contrast: A float, specifying the contrast for color jitter.
    saturation: A float, specifying the saturation for color jitter.
    hue: A float, specifying the hue for color jitter.

  Returns:
    The distorted image tensor.
  """
  with tf.name_scope('distort_color'):
    if brightness > 0:
      image = tf.image.random_brightness(image, max_delta=brightness)
    if contrast > 0:
      image = tf.image.random_contrast(
          image, lower=1-contrast, upper=1+contrast)
    if saturation > 0:
      image = tf.image.random_saturation(
          image, lower=1-saturation, upper=1+saturation)
    if hue > 0:
      image = tf.image.random_hue(image, max_delta=hue)
    return image


def encode_labels(gt_boxes, gt_labels):
  """Labels anchors with ground truth inputs.

  Args:
    gt_boxes: A float tensor with shape [N, 4] representing groundtruth boxes.
      For each row, it stores [y0, x0, y1, x1] for four corners of a box.
    gt_labels: A integer tensor with shape [N, 1] representing groundtruth
      classes.
  Returns:
    encoded_classes: a tensor with shape [num_anchors, 1].
    encoded_boxes: a tensor with shape [num_anchors, 4].
    num_positives: scalar tensor storing number of positives in an image.
  """
  similarity_calc = region_similarity_calculator.IouSimilarity()
  matcher = argmax_matcher.ArgMaxMatcher(
      matched_threshold=ssd_constants.MATCH_THRESHOLD,
      unmatched_threshold=ssd_constants.MATCH_THRESHOLD,
      negatives_lower_than_unmatched=True,
      force_match_for_each_row=True)

  box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder(
      scale_factors=ssd_constants.BOX_CODER_SCALES)

  default_boxes = box_list.BoxList(tf.convert_to_tensor(DefaultBoxes()('ltrb')))
  target_boxes = box_list.BoxList(gt_boxes)

  assigner = target_assigner.TargetAssigner(
      similarity_calc, matcher, box_coder)

  encoded_classes, _, encoded_boxes, _, matches = assigner.assign(
      default_boxes, target_boxes, gt_labels)
  num_matched_boxes = tf.reduce_sum(
      tf.cast(tf.not_equal(matches.match_results, -1), tf.float32))
  return encoded_classes, encoded_boxes, num_matched_boxes


def fused_transpose_and_space_to_depth(
    images,
    block_size=ssd_constants.SPACE_TO_DEPTH_BLOCK_SIZE,
    transpose_input=True):
  """Fuses space-to-depth and transpose.

  Space-to-depth performas the following permutation, which is equivalent to
  tf.nn.space_to_depth.

  images = tf.reshape(images, [batch, h // block_size, block_size,
                               w // block_size, block_size, c])
  images = tf.transpose(images, [0, 1, 3, 2, 4, 5])
  images = tf.reshape(images, [batch, h // block_size, w // block_size,
                               c * (block_size ** 2)])

  Args:
    images: A tensor with a shape of [batch_size, h, w, c] as the images. The h
      and w can be dynamic sizes.
    block_size: A integer for space-to-depth block size.
    transpose_input: A boolean to indicate if the images tensor should be
      transposed.

  Returns:
    A transformed images tensor.

  """
  batch_size, h, w, c = images.get_shape().as_list()
  images = tf.reshape(
      images,
      [batch_size, h // block_size, block_size, w // block_size, block_size, c])
  if transpose_input:
    if batch_size > 8:
      # HWCN
      images = tf.transpose(images, [1, 3, 2, 4, 5, 0])
      images = tf.reshape(
          images,
          [h // block_size, w // block_size, c * (block_size**2), batch_size])
    else:
      # HWNC
      images = tf.transpose(images, [1, 3, 0, 2, 4, 5])
      images = tf.reshape(
          images,
          [h // block_size, w // block_size, batch_size, c * (block_size**2)])
  else:
    images = tf.transpose(images, [0, 1, 3, 2, 4, 5])
    images = tf.reshape(
        images,
        [batch_size, h // block_size, w // block_size, c * (block_size**2)])
  return images


class SSDInputReader(object):
  """Input reader for dataset."""

  def __init__(self,
               file_pattern,
               transpose_input=False,
               is_training=False,
               use_fake_data=False,
               distributed_eval=False,
               count=-1):
    self._file_pattern = file_pattern
    self._transpose_input = transpose_input
    self._is_training = is_training
    self._use_fake_data = use_fake_data
    self._distributed_eval = distributed_eval
    self._count = count

  def __call__(self, params):
    example_decoder = tf_example_decoder.TfExampleDecoder()

    def _parse_example(data):
      with tf.name_scope('augmentation'):
        source_id = data['source_id']
        image = data['image']  # dtype uint8
        raw_shape = tf.shape(image)
        boxes = data['groundtruth_boxes']
        classes = tf.reshape(data['groundtruth_classes'], [-1, 1])

        # Only 80 of the 90 COCO classes are used.
        class_map = tf.convert_to_tensor(ssd_constants.CLASS_MAP)
        classes = tf.gather(class_map, classes)
        classes = tf.cast(classes, dtype=tf.float32)

        if self._is_training:
          image, boxes, classes = ssd_crop(image, boxes, classes)
          # ssd_crop resizes and returns image of dtype float32 and does not
          # change its range (i.e., value in between 0--255). Divide by 255.
          # converts it to [0, 1] range. Not doing this before cropping to
          # avoid dtype cast (which incurs additional memory copy).
          image /= 255.0

          # random_horizontal_flip() is hard coded to flip with 50% chance.
          image, boxes = preprocessor.random_horizontal_flip(
              image=image, boxes=boxes)

          # TODO(shibow): Investigate the parameters for color jitter.
          image = color_jitter(
              image, brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05)

          if params['use_bfloat16']:
            image = tf.cast(image, dtype=tf.bfloat16)

          encoded_classes, encoded_boxes, num_matched_boxes = encode_labels(
              boxes, classes)

          # TODO(taylorrobie): Check that this cast is valid.
          encoded_classes = tf.cast(encoded_classes, tf.int32)

          labels = {
              ssd_constants.NUM_MATCHED_BOXES: num_matched_boxes,
              ssd_constants.BOXES: encoded_boxes,
              ssd_constants.CLASSES: tf.squeeze(encoded_classes, axis=1),
          }
          # This is for dataloader visualization; actual model doesn't use this.
          if params['visualize_dataloader']:
            box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder(
                scale_factors=ssd_constants.BOX_CODER_SCALES)
            decoded_boxes = tf.expand_dims(box_coder.decode(
                rel_codes=tf.squeeze(encoded_boxes),
                anchors=box_list.BoxList(
                    tf.convert_to_tensor(DefaultBoxes()('ltrb')))
            ).get(), axis=0)
            labels['decoded_boxes'] = tf.squeeze(decoded_boxes)

          return image, labels

        else:
          image = tf.image.resize_images(
              image, size=(ssd_constants.IMAGE_SIZE, ssd_constants.IMAGE_SIZE))
          # resize_image returns image of dtype float32 and does not change its
          # range. Divide by 255 to convert image to [0, 1] range.
          image /= 255.

          if params['use_bfloat16']:
            image = tf.cast(image, dtype=tf.bfloat16)

          def trim_and_pad(inp_tensor, dim_1):
            """Limit the number of boxes, and pad if necessary."""
            inp_tensor = inp_tensor[:ssd_constants.MAX_NUM_EVAL_BOXES]
            num_pad = ssd_constants.MAX_NUM_EVAL_BOXES - tf.shape(inp_tensor)[0]
            inp_tensor = tf.pad(inp_tensor, [[0, num_pad], [0, 0]])
            return tf.reshape(
                inp_tensor, [ssd_constants.MAX_NUM_EVAL_BOXES, dim_1])

          boxes, classes = trim_and_pad(boxes, 4), trim_and_pad(classes, 1)

          sample = {
              ssd_constants.IMAGE: image,
              ssd_constants.BOXES: boxes,
              ssd_constants.CLASSES: classes,
              ssd_constants.SOURCE_ID: tf.string_to_number(source_id, tf.int32),
              ssd_constants.RAW_SHAPE: raw_shape,
          }

          if not self._is_training and self._count > params['eval_samples']:
            sample[ssd_constants.IS_PADDED] = data[ssd_constants.IS_PADDED]
          return sample

    batch_size = params['batch_size']
    dataset = tf.data.Dataset.list_files(self._file_pattern, shuffle=False)

    if self._is_training or self._distributed_eval:
      if 'context' in params:
        dataset = dataset.shard(
            params['context'].num_hosts,
            params['context'].current_input_fn_deployment()[1])
        if self._is_training:
          dataset = dataset.shuffle(
              tf.to_int64(256 / params['context'].num_hosts))
      else:
        dataset = dataset.shard(params['dataset_num_shards'],
                                params['dataset_index'])
        if self._is_training:
          dataset = dataset.shuffle(
              tf.to_int64(256 / params['dataset_num_shards']))

    # Prefetch data from files.
    def _prefetch_dataset(filename):
      dataset = tf.data.TFRecordDataset(filename).prefetch(1)
      return dataset
    dataset = dataset.apply(
        tf.data.experimental.parallel_interleave(
            _prefetch_dataset, cycle_length=32, sloppy=self._is_training))

    # Parse the fetched records to input tensors for model function.
    dataset = dataset.map(example_decoder.decode, num_parallel_calls=64)

    def _mark_is_padded(data):
      sample = data
      sample[ssd_constants.IS_PADDED] = tf.constant(True, dtype=tf.bool)
      return sample

    def _mark_is_not_padded(data):
      sample = data
      sample[ssd_constants.IS_PADDED] = tf.constant(False, dtype=tf.bool)
      return sample

    # Pad dataset to the desired size and mark if the data is padded.
    # During eval/predict, if local_batch_size * num_shards > 5000,
    # original dataset size won't be fit for computations on that number
    # of shards. In this case, will take
    # (local_batch_size - 5000 / num_shards) data from the original dataset
    # on each shard and mark the padded data as `is_padded`.
    # Also mark the original data as `not_padded`.
    # Append the padded data to the original dataset.
    if not self._is_training and self._count > params['eval_samples']:
      padded_dataset = dataset.map(_mark_is_padded)
      dataset = dataset.map(_mark_is_not_padded)
      dataset = dataset.concatenate(padded_dataset).take(
          self._count // params['dataset_num_shards'])

    if self._is_training:
      dataset = dataset.map(
          # pylint: disable=g-long-lambda
          lambda data: (data,
                        tf.greater(tf.shape(data['groundtruth_boxes'])[0], 0)),
          num_parallel_calls=64)
      dataset = dataset.filter(lambda data, pred: pred)
      # Prefetching and caching increases the memory usage, so disable when
      # using fake data.
      if not self._use_fake_data:
        dataset = dataset.cache().shuffle(64).repeat()
      dataset = dataset.map(
          lambda data, _: _parse_example(data), num_parallel_calls=64)
      dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    else:
      dataset = dataset.prefetch(batch_size * 64)
      dataset = dataset.map(_parse_example, num_parallel_calls=64)
      dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

    if params['conv0_space_to_depth']:
      def _space_to_depth_training_fn(images, labels):
        images = fused_transpose_and_space_to_depth(
            images,
            block_size=ssd_constants.SPACE_TO_DEPTH_BLOCK_SIZE,
            transpose_input=self._transpose_input)
        if self._transpose_input and batch_size > 8:
          labels[ssd_constants.BOXES] = tf.transpose(
              labels[ssd_constants.BOXES], [1, 2, 0])
        return images, labels

      def _space_to_depth_eval_fn(labels):
        images = labels[ssd_constants.IMAGE]
        labels[ssd_constants.IMAGE] = fused_transpose_and_space_to_depth(
            images,
            block_size=ssd_constants.SPACE_TO_DEPTH_BLOCK_SIZE,
            transpose_input=False)
        return labels

      if self._is_training:
        space_to_depth_fn = _space_to_depth_training_fn
      else:
        space_to_depth_fn = _space_to_depth_eval_fn
      dataset = dataset.map(space_to_depth_fn, num_parallel_calls=64)
    elif self._transpose_input and self._is_training:
      # Manually apply the double transpose trick for training data.
      def _transpose_dataset(image, labels):
        if batch_size > 8:
          image = tf.transpose(image, [1, 2, 3, 0])
          labels[ssd_constants.BOXES] = tf.transpose(
              labels[ssd_constants.BOXES], [1, 2, 0])
        else:
          image = tf.transpose(image, [1, 2, 0, 3])
        return image, labels

      dataset = dataset.map(_transpose_dataset, num_parallel_calls=64)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    options = tf.data.Options()
    options.experimental_threading.max_intra_op_parallelism = 1
    options.experimental_threading.private_threadpool_size = 48
    dataset = dataset.with_options(options)

    if self._use_fake_data:
      dataset = dataset.take(1).cache().repeat()

    return dataset
