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
###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################
# Changes:
# - Ported to TF 2.2
# - Removed mlp_log
# - Renamed ssd_constants to constants
# - Removed fused_transpose_and_space_to_depth
# - Added transpose_labels function
# - Added multi-node training on Horovod
# - Use dtype parameter insted of use_bfloat16
# - Additional info logs added
# - experimental_deterministic option set when not training
# - Removed support for conv0_space_to_depth
# - AUTOTUNE flag used in dataset prefetching
# - Conditionally disabled dataset caching
# - Formatted with autopep8
# - Removed __future__ imports
# - Added absolute paths in imports

"""Data loader and processing."""

import itertools as it
import glob
import math

import numpy as np
import tensorflow.compat.v1 as tf

from TensorFlow.computer_vision.SSD_ResNet34.object_detection import argmax_matcher
from TensorFlow.computer_vision.SSD_ResNet34.object_detection import box_list
from TensorFlow.computer_vision.SSD_ResNet34.object_detection import faster_rcnn_box_coder
from TensorFlow.computer_vision.SSD_ResNet34.object_detection import preprocessor
from TensorFlow.computer_vision.SSD_ResNet34.object_detection import region_similarity_calculator
from TensorFlow.computer_vision.SSD_ResNet34.object_detection import target_assigner
from TensorFlow.computer_vision.SSD_ResNet34.object_detection import tf_example_decoder
from TensorFlow.computer_vision.SSD_ResNet34 import constants


class DefaultBoxes(object):
    """Default bounding boxes for 300x300 5 layer SSD.

    Default bounding boxes generation follows the order of (W, H, anchor_sizes).
    Therefore, the tensor converted from DefaultBoxes has a shape of
    [anchor_sizes, H, W, 4]. The last dimension is the box coordinates; 'ltrb'
    is [ymin, xmin, ymax, xmax] while 'xywh' is [cy, cx, h, w].
    """

    def __init__(self):
        fk = constants.IMAGE_SIZE / np.array(constants.STEPS)

        self.default_boxes = []
        # size of feature and number of feature
        for idx, feature_size in enumerate(constants.FEATURE_SIZES):
            sk1 = constants.SCALES[idx] / constants.IMAGE_SIZE
            sk2 = constants.SCALES[idx+1] / constants.IMAGE_SIZE
            sk3 = math.sqrt(sk1*sk2)
            all_sizes = [(sk1, sk1), (sk3, sk3)]

            for alpha in constants.ASPECT_RATIOS[idx]:
                w, h = sk1 * math.sqrt(alpha), sk1 / math.sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))

            assert len(all_sizes) == constants.NUM_DEFAULTS[idx]

            for w, h in all_sizes:
                for i, j in it.product(range(feature_size), repeat=2):
                    cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
                    box = tuple(np.clip(k, 0, 1) for k in (cy, cx, h, w))
                    self.default_boxes.append(box)

        assert len(self.default_boxes) == constants.NUM_SSD_BOXES

        def to_ltrb(cy, cx, h, w):
            return cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2

        # For IoU calculation
        self.default_boxes_ltrb = tuple(to_ltrb(*i)
                                        for i in self.default_boxes)

    def __call__(self, order='ltrb'):
        if order == 'ltrb':
            return self.default_boxes_ltrb
        if order == 'xywh':
            return self.default_boxes


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
    lt = tf.maximum(be1[:, :, :2], be2[:, :, :2])

    rb = tf.minimum(be1[:, :, 2:], be2[:, :, 2:])

    delta = tf.maximum(rb - lt, 0)

    intersect = delta[:, :, 0]*delta[:, :, 1]

    delta1 = be1[:, :, 2:] - be1[:, :, :2]
    area1 = delta1[:, :, 0]*delta1[:, :, 1]
    delta2 = be2[:, :, 2:] - be2[:, :, :2]
    area2 = delta2[:, :, 0]*delta2[:, :, 1]

    iou = intersect/(area1 + area2 - intersect)
    return iou


def ssd_crop(image, boxes, classes):
    """IoU biassed random crop.

    Reference: https://github.com/chauhan-utk/ssd.DomainAdaptation
    """

    num_boxes = tf.shape(boxes)[0]

    def no_crop_check():
        return (tf.random_uniform(shape=(), minval=0, maxval=1, dtype=tf.float32)
                < constants.P_NO_CROP_PER_PASS)

    def no_crop_proposal():
        return (
            tf.ones((), tf.bool),
            tf.convert_to_tensor([0, 0, 1, 1], dtype=tf.float32),
            tf.ones((num_boxes,), tf.bool),
        )

    def crop_proposal():
        def rand_vec(minval, maxval): return tf.random_uniform(
            shape=(constants.NUM_CROP_PASSES, 1), minval=minval, maxval=maxval,
            dtype=tf.float32)

        width, height = rand_vec(0.3, 1), rand_vec(0.3, 1)
        left, top = rand_vec(0, 1-width), rand_vec(0, 1-height)

        right = left + width
        bottom = top + height

        ltrb = tf.concat([left, top, right, bottom], axis=1)

        min_iou = tf.random_shuffle(constants.CROP_MIN_IOU_CHOICES)[0]
        ious = calc_iou_tensor(ltrb, boxes)

        # discard any bboxes whose center not in the cropped image
        xc, yc = [tf.tile(0.5 * (boxes[:, i + 0] + boxes[:, i + 2])[tf.newaxis, :],
                          (constants.NUM_CROP_PASSES, 1)) for i in range(2)]

        masks = tf.reduce_all(tf.stack([
            tf.greater(xc, tf.tile(left, (1, num_boxes))),
            tf.less(xc, tf.tile(right, (1, num_boxes))),
            tf.greater(yc, tf.tile(top, (1, num_boxes))),
            tf.less(yc, tf.tile(bottom, (1, num_boxes))),
        ], axis=2), axis=2)

        # Checks of whether a crop is valid.
        valid_aspect = tf.logical_and(tf.less(height/width, 2),
                                      tf.less(width/height, 2))
        valid_ious = tf.reduce_all(tf.greater(
            ious, min_iou), axis=1, keepdims=True)
        valid_masks = tf.reduce_any(masks, axis=1, keepdims=True)

        valid_all = tf.cast(tf.reduce_all(tf.concat(
            [valid_aspect, valid_ious, valid_masks], axis=1), axis=1), tf.int32)

        # One indexed, as zero is needed for the case of no matches.
        index = tf.range(1, 1 + constants.NUM_CROP_PASSES, dtype=tf.int32)

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
        loop_vars=[tf.zeros((), tf.bool), tf.zeros(
            (4,), tf.float32), tf.zeros((num_boxes,), tf.bool)],
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
        box_indices=tf.zeros((1,), tf.int32),
        crop_size=(constants.IMAGE_SIZE, constants.IMAGE_SIZE),
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
        matched_threshold=constants.MATCH_THRESHOLD,
        unmatched_threshold=constants.MATCH_THRESHOLD,
        negatives_lower_than_unmatched=True,
        force_match_for_each_row=True)

    box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder(
        scale_factors=constants.BOX_CODER_SCALES)

    default_boxes = box_list.BoxList(
        tf.convert_to_tensor(DefaultBoxes()('ltrb')))
    target_boxes = box_list.BoxList(gt_boxes)

    assigner = target_assigner.TargetAssigner(
        similarity_calc, matcher, box_coder)

    encoded_classes, _, encoded_boxes, _, matches = assigner.assign(
        default_boxes, target_boxes, gt_labels)
    num_matched_boxes = tf.reduce_sum(
        tf.cast(tf.not_equal(matches.match_results, -1), tf.float32))
    return encoded_classes, encoded_boxes, num_matched_boxes


def transpose_labels(classes, boxes):
    sizes = [constants.FEATURE_SIZES[i] ** 2 * constants.NUM_DEFAULTS[i]
             for i in range(len(constants.FEATURE_SIZES))]

    classes_split = tf.split(classes, sizes, axis=0)
    boxes_split = tf.split(boxes, sizes, axis=0)

    classes_t = []
    boxes_t = []
    for i in range(len(constants.FEATURE_SIZES)):
        cls_i = tf.reshape(classes_split[i], [
                           constants.NUM_DEFAULTS[i], constants.FEATURE_SIZES[i], constants.FEATURE_SIZES[i], 1])
        cls_i = tf.transpose(cls_i, (1, 2, 0, 3))
        cls_i = tf.reshape(cls_i, [-1, 1])
        classes_t.append(cls_i)
        box_i = tf.reshape(boxes_split[i], [
                           constants.NUM_DEFAULTS[i], constants.FEATURE_SIZES[i], constants.FEATURE_SIZES[i], 4])
        box_i = tf.transpose(box_i, (1, 2, 0, 3))
        box_i = tf.reshape(box_i, [-1, 4])
        boxes_t.append(box_i)

    classes = tf.concat(classes_t, axis=0)
    boxes = tf.concat(boxes_t, axis=0)

    return classes, boxes


class SSDInputReader(object):
    """Input reader for dataset."""

    def __init__(self,
                 file_pattern,
                 is_training=False,
                 use_fake_data=False,
                 count=-1):
        self._file_pattern = file_pattern
        self._is_training = is_training
        self._use_fake_data = use_fake_data
        self._count = count

    def __call__(self, params):
        example_decoder = tf_example_decoder.TfExampleDecoder()

        def normalize(img):
            img -= tf.constant(
                constants.NORMALIZATION_MEAN, shape=[1, 1, 3], dtype=img.dtype)
            COEF_STD = 1.0 / tf.constant(
                constants.NORMALIZATION_STD, shape=[1, 1, 3], dtype=img.dtype)
            img *= COEF_STD
            return img

        def _parse_example(data):
            with tf.name_scope('augmentation'):
                source_id = data['source_id']
                image = data['image']  # dtype uint8
                raw_shape = tf.shape(image)
                boxes = data['groundtruth_boxes']
                classes = tf.reshape(data['groundtruth_classes'], [-1, 1])

                # Only 80 of the 90 COCO classes are used.
                class_map = tf.convert_to_tensor(constants.CLASS_MAP)
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

                    image = normalize(image)

                    if params['dtype'] == 'bf16':
                        image = tf.cast(image, dtype=tf.bfloat16)

                    encoded_classes, encoded_boxes, num_matched_boxes = encode_labels(
                        boxes, classes)

                    # We transpose in dataloader instead of in the topology to save time
                    encoded_classes, encoded_boxes = transpose_labels(
                        encoded_classes, encoded_boxes)

                    encoded_classes = tf.cast(encoded_classes, tf.int32)

                    labels = {
                        constants.NUM_MATCHED_BOXES: num_matched_boxes,
                        constants.BOXES: encoded_boxes,
                        constants.CLASSES: tf.squeeze(encoded_classes, axis=1),
                    }
                    # This is for dataloader visualization; actual model doesn't use this.
                    if params['visualize_dataloader']:
                        box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder(
                            scale_factors=constants.BOX_CODER_SCALES)
                        decoded_boxes = tf.expand_dims(box_coder.decode(
                            rel_codes=tf.squeeze(encoded_boxes),
                            anchors=box_list.BoxList(
                                tf.convert_to_tensor(DefaultBoxes()('ltrb')))
                        ).get(), axis=0)
                        labels['decoded_boxes'] = tf.squeeze(decoded_boxes)

                    return image, labels

                else:
                    image = tf.image.resize_images(
                        image, size=(constants.IMAGE_SIZE, constants.IMAGE_SIZE))
                    # resize_image returns image of dtype float32 and does not change its
                    # range. Divide by 255 to convert image to [0, 1] range.
                    image /= 255.

                    image = normalize(image)

                    if params['dtype'] == 'bf16':
                        image = tf.cast(image, dtype=tf.bfloat16)

                    def trim_and_pad(inp_tensor, dim_1):
                        """Limit the number of boxes, and pad if necessary."""
                        inp_tensor = inp_tensor[:constants.MAX_NUM_EVAL_BOXES]
                        num_pad = constants.MAX_NUM_EVAL_BOXES - \
                            tf.shape(inp_tensor)[0]
                        inp_tensor = tf.pad(inp_tensor, [[0, num_pad], [0, 0]])
                        return tf.reshape(
                            inp_tensor, [constants.MAX_NUM_EVAL_BOXES, dim_1])

                    boxes, classes = trim_and_pad(
                        boxes, 4), trim_and_pad(classes, 1)

                    sample = {
                        constants.IMAGE: image,
                        constants.BOXES: boxes,
                        constants.CLASSES: classes,
                        constants.SOURCE_ID: tf.string_to_number(source_id, tf.int32),
                        constants.RAW_SHAPE: raw_shape,
                    }

                    if not self._is_training and self._count > params['eval_samples']:
                        sample[constants.IS_PADDED] = data[constants.IS_PADDED]
                    return sample

        batch_size = params['batch_size']
        dataset = tf.data.Dataset.list_files(self._file_pattern, shuffle=False)
        tf.logging.info("Dataset file pattern '%s': found %d files.",
                        self._file_pattern, len(glob.glob(self._file_pattern)))

        if self._is_training:
            dataset_num_shards = params['num_shards']
            dataset_shard_index = params['shard_index']

            dataset = dataset.shard(dataset_num_shards,
                                    dataset_shard_index)
            if self._is_training:
                dataset = dataset.shuffle(
                    tf.cast(256 / dataset_num_shards, tf.int64))

        # Prefetch data from files.
        def _prefetch_dataset(filename):
            dataset = tf.data.TFRecordDataset(filename).prefetch(1)
            return dataset

        options = tf.data.Options()
        options.experimental_deterministic = not self._is_training
        dataset = dataset.interleave(
            map_func=_prefetch_dataset,
            cycle_length=32,
            block_length=1,
            num_parallel_calls=tf.data.experimental.AUTOTUNE).with_options(options)

        # Parse the fetched records to input tensors for model function.
        dataset = dataset.map(example_decoder.decode, num_parallel_calls=64)

        def _mark_is_padded(data):
            sample = data
            sample[constants.IS_PADDED] = tf.constant(True, dtype=tf.bool)
            return sample

        def _mark_is_not_padded(data):
            sample = data
            sample[constants.IS_PADDED] = tf.constant(False, dtype=tf.bool)
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
                self._count)

        if self._is_training:
            dataset = dataset.map(
                # pylint: disable=g-long-lambda
                lambda data: (data,
                              tf.greater(tf.shape(data['groundtruth_boxes'])[0], 0)),
                num_parallel_calls=64)
            dataset = dataset.filter(lambda data, pred: pred)
            # Prefetching and caching increases the memory usage, so disable when
            # using fake data.
            meminfo = dict((i.split()[0].rstrip(':'), int(i.split()[1]))
                           for i in open('/proc/meminfo').readlines())
            mem_kib = meminfo['MemTotal']

            # rough approx. 1 GiB per tf-record
            caching_mem_kib = len(glob.glob(self._file_pattern)) * 1000000

            if not self._use_fake_data:
                if caching_mem_kib > mem_kib:
                    dataset = dataset.shuffle(64).repeat()
                    tf.logging.info(
                        "Dataset cache OFF because MemTotal = %d KiB! It may decrease performance.", mem_kib)
                else:
                    dataset = dataset.cache().shuffle(64).repeat()
                    tf.logging.info("Dataset cache ON")

            dataset = dataset.map(lambda data, _: _parse_example(
                data), num_parallel_calls=params['num_parallel_calls'])
            dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

        else:
            dataset = dataset.prefetch(batch_size * 64)
            dataset = dataset.map(_parse_example, num_parallel_calls=params['num_parallel_calls'])
            dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

        if self._use_fake_data:
            dataset = dataset.take(1).cache().repeat()
        else:
            options = tf.data.Options()
            options.threading.private_threadpool_size = params['threadpool_size']
            dataset = dataset.with_options(options)

        if len(tf.config.list_logical_devices('HPU')) > 0:
            device = "/device:HPU:0"
            with tf.device(device):
                dataset = dataset.apply(tf.data.experimental.prefetch_to_device(device))
        else:
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset
