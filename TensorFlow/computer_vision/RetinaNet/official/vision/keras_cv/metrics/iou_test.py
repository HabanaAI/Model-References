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

"""Tests for Keras metrics functions."""

import tensorflow as tf

from official.vision.keras_cv.metrics import iou


class MeanIoUTest(tf.test.TestCase):

  def test_config(self):
    m_obj = iou.PerClassIoU(num_classes=2, name='per_class_iou')
    self.assertEqual(m_obj.name, 'per_class_iou')
    self.assertEqual(m_obj.num_classes, 2)

    m_obj2 = iou.PerClassIoU.from_config(m_obj.get_config())
    self.assertEqual(m_obj2.name, 'per_class_iou')
    self.assertEqual(m_obj2.num_classes, 2)

  def test_unweighted(self):
    y_pred = [0, 1, 0, 1]
    y_true = [0, 0, 1, 1]

    m_obj = iou.PerClassIoU(num_classes=2)

    result = m_obj(y_true, y_pred)

    # cm = [[1, 1],
    #       [1, 1]]
    # sum_row = [2, 2], sum_col = [2, 2], true_positives = [1, 1]
    # iou = true_positives / (sum_row + sum_col - true_positives))
    expected_result = [1 / (2 + 2 - 1), 1 / (2 + 2 - 1)]
    self.assertAllClose(expected_result, result, atol=1e-3)

  def test_weighted(self):
    y_pred = tf.constant([0, 1, 0, 1], dtype=tf.float32)
    y_true = tf.constant([0, 0, 1, 1])
    sample_weight = tf.constant([0.2, 0.3, 0.4, 0.1])

    m_obj = iou.PerClassIoU(num_classes=2)

    result = m_obj(y_true, y_pred, sample_weight=sample_weight)

    # cm = [[0.2, 0.3],
    #       [0.4, 0.1]]
    # sum_row = [0.6, 0.4], sum_col = [0.5, 0.5], true_positives = [0.2, 0.1]
    # iou = true_positives / (sum_row + sum_col - true_positives))
    expected_result = [0.2 / (0.6 + 0.5 - 0.2), 0.1 / (0.4 + 0.5 - 0.1)]
    self.assertAllClose(expected_result, result, atol=1e-3)

  def test_multi_dim_input(self):
    y_pred = tf.constant([[0, 1], [0, 1]], dtype=tf.float32)
    y_true = tf.constant([[0, 0], [1, 1]])
    sample_weight = tf.constant([[0.2, 0.3], [0.4, 0.1]])

    m_obj = iou.PerClassIoU(num_classes=2)

    result = m_obj(y_true, y_pred, sample_weight=sample_weight)

    # cm = [[0.2, 0.3],
    #       [0.4, 0.1]]
    # sum_row = [0.6, 0.4], sum_col = [0.5, 0.5], true_positives = [0.2, 0.1]
    # iou = true_positives / (sum_row + sum_col - true_positives))
    expected_result = [0.2 / (0.6 + 0.5 - 0.2), 0.1 / (0.4 + 0.5 - 0.1)]
    self.assertAllClose(expected_result, result, atol=1e-3)

  def test_zero_valid_entries(self):
    m_obj = iou.PerClassIoU(num_classes=2)
    self.assertAllClose(m_obj.result(), [0, 0], atol=1e-3)

  def test_zero_and_non_zero_entries(self):
    y_pred = tf.constant([1], dtype=tf.float32)
    y_true = tf.constant([1])

    m_obj = iou.PerClassIoU(num_classes=2)
    result = m_obj(y_true, y_pred)

    # cm = [[0, 0],
    #       [0, 1]]
    # sum_row = [0, 1], sum_col = [0, 1], true_positives = [0, 1]
    # iou = true_positives / (sum_row + sum_col - true_positives))
    expected_result = [0, 1 / (1 + 1 - 1)]
    self.assertAllClose(expected_result, result, atol=1e-3)

if __name__ == '__main__':
  tf.test.main()
