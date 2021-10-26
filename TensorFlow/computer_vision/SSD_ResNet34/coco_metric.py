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
# - Changed COCO import to import from pycocotools
# - Renamed ssd_constants to constants
# - Removed support for TPU
# - Formatted with autopep8
# - Removed __future__ imports
# - Added absolute paths in imports

"""COCO-style evaluation metrics.

Implements the interface of COCO API and metric_fn in tf.TPUEstimator.

COCO API: github.com/cocodataset/cocoapi/
"""

import atexit
import tempfile
import time

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import six

import tensorflow.compat.v1 as tf

from TensorFlow.computer_vision.SSD_ResNet34 import constants


# https://github.com/cocodataset/cocoapi/issues/49
if six.PY3:
    COCO.unicode = str


def create_coco(val_json_file, use_cpp_extension=True):
    """Creates Microsoft COCO helper class object and return it."""
    if val_json_file.startswith('gs://'):
        _, local_val_json = tempfile.mkstemp(suffix='.json')
        tf.gfile.Remove(local_val_json)

        tf.gfile.Copy(val_json_file, local_val_json)
        atexit.register(tf.gfile.Remove, local_val_json)
    else:
        local_val_json = val_json_file

    if use_cpp_extension:
        coco_gt = COCO(local_val_json, False)
    else:
        coco_gt = COCO(local_val_json)
    return coco_gt


def compute_map(labels_and_predictions,
                coco_gt,
                use_cpp_extension=True):
    """Use model predictions to compute mAP.

    The evaluation code is largely copied from the MLPerf reference
    implementation. While it is possible to write the evaluation as a tensor
    metric and use Estimator.evaluate(), this approach was selected for simplicity
    and ease of duck testing.

    Args:
      labels_and_predictions: A map from TPU predict method.
      coco_gt: ground truch COCO object.
      use_cpp_extension: use cocoeval C++ library.
    Returns:
      Evaluation result.
    """

    predictions = []
    tic = time.time()

    for example in labels_and_predictions:
        if constants.IS_PADDED in example and example[
                constants.IS_PADDED]:
            continue

        htot, wtot, _ = example[constants.RAW_SHAPE]
        pred_box = example['pred_box']
        pred_scores = example['pred_scores']
        indices = example['indices']
        loc, label, prob = decode_single(
            pred_box, pred_scores, indices, constants.OVERLAP_CRITERIA,
            constants.MAX_NUM_EVAL_BOXES, constants.MAX_NUM_EVAL_BOXES)

        for loc_, label_, prob_ in zip(loc, label, prob):
            # Ordering convention differs, hence [1], [0] rather than [0], [1]
            predictions.append([
                int(example[constants.SOURCE_ID]),
                loc_[1] * wtot, loc_[0] * htot, (loc_[3] - loc_[1]) * wtot,
                (loc_[2] - loc_[0]) * htot, prob_,
                constants.CLASS_INV_MAP[label_]
            ])

    toc = time.time()
    tf.logging.info('Prepare predictions DONE (t={:0.2f}s).'.format(toc - tic))

    if use_cpp_extension:
        coco_dt = coco_gt.LoadRes(np.array(predictions, dtype=np.float32))
        coco_eval = COCOeval(coco_gt, coco_dt, iou_type='bbox')
        coco_eval.Evaluate()
        coco_eval.Accumulate()
        coco_eval.Summarize()
        stats = coco_eval.GetStats()

    else:
        coco_dt = coco_gt.loadRes(np.array(predictions))

        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats

    print('Current AP: {:.5f}'.format(stats[0]))
    metric_names = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'ARmax1',
                    'ARmax10', 'ARmax100', 'ARs', 'ARm', 'ARl']
    coco_time = time.time()
    tf.logging.info('COCO eval DONE (t={:0.2f}s).'.format(coco_time - toc))

    # Prefix with "COCO" to group in TensorBoard.
    return {'COCO/' + key: value for key, value in zip(metric_names, stats)}


def calc_iou(target, candidates):
    target_tiled = np.tile(target[np.newaxis, :], (candidates.shape[0], 1))
    # Left Top & Right Bottom
    lt = np.maximum(target_tiled[:, :2], candidates[:, :2])

    rb = np.minimum(target_tiled[:, 2:], candidates[:, 2:])

    delta = np.maximum(rb - lt, 0)

    intersect = delta[:, 0] * delta[:, 1]

    delta1 = target_tiled[:, 2:] - target_tiled[:, :2]
    area1 = delta1[:, 0] * delta1[:, 1]
    delta2 = candidates[:, 2:] - candidates[:, :2]
    area2 = delta2[:, 0] * delta2[:, 1]

    iou = intersect/(area1 + area2 - intersect)
    return iou


def decode_single(bboxes_in,
                  scores_in,
                  indices,
                  criteria,
                  max_output,
                  max_num=200):
    """Implement Non-maximum suppression.

      Reference to https://github.com/amdegroot/ssd.pytorch

    Args:
      bboxes_in: a Tensor with shape [N, 4], which stacks box regression outputs
        on all feature levels. The N is the number of total anchors on all levels.
      scores_in: a Tensor with shape [constants.MAX_NUM_EVAL_BOXES,
        num_classes]. The top constants.MAX_NUM_EVAL_BOXES box scores for each
        class.
      indices: a Tensor with shape [constants.MAX_NUM_EVAL_BOXES,
        num_classes]. The indices for these top boxes for each class.
      criteria: a float number to specify the threshold of NMS.
      max_output: maximum output length.
      max_num: maximum number of boxes before NMS.

    Returns:
      boxes, labels and scores after NMS.
    """

    bboxes_out = []
    scores_out = []
    labels_out = []

    for i, score in enumerate(np.split(scores_in, scores_in.shape[1], 1)):
        class_indices = indices[:, i]
        bboxes = bboxes_in[class_indices, :]
        score = np.squeeze(score, 1)

        # skip background
        if i == 0:
            continue

        mask = score > constants.MIN_SCORE
        if not np.any(mask):
            continue

        bboxes, score = bboxes[mask, :], score[mask]

        score_idx_sorted = np.argsort(score)
        score_sorted = score[score_idx_sorted]

        score_idx_sorted = score_idx_sorted[-max_num:]
        candidates = []

        # perform non-maximum suppression
        while len(score_idx_sorted):
            idx = score_idx_sorted[-1]
            bboxes_sorted = bboxes[score_idx_sorted, :]
            bboxes_idx = bboxes[idx, :]
            iou = calc_iou(bboxes_idx, bboxes_sorted)

            score_idx_sorted = score_idx_sorted[iou < criteria]
            candidates.append(idx)

        bboxes_out.append(bboxes[candidates, :])
        scores_out.append(score[candidates])
        labels_out.extend([i]*len(candidates))

    if len(scores_out) == 0:
        tf.logging.info("No objects detected. Returning dummy values.")
        return (
            np.zeros(shape=(1, 4), dtype=np.float32),
            np.zeros(shape=(1,), dtype=np.int32),
            np.ones(shape=(1,), dtype=np.float32) * constants.DUMMY_SCORE,
        )

    bboxes_out = np.concatenate(bboxes_out, axis=0)
    scores_out = np.concatenate(scores_out, axis=0)
    labels_out = np.array(labels_out)

    max_ids = np.argsort(scores_out)[-max_output:]

    return bboxes_out[max_ids, :], labels_out[max_ids], scores_out[max_ids]
