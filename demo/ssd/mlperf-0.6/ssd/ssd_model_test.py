"""Tests for ssd_model operators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf

from tensorflow.python.platform import test
import ssd_architecture
import ssd_model

tpu = tf.contrib.tpu


class SSDModelTest(test.TestCase):

  def setUp(self):
    super(SSDModelTest, self).setUp()
    self.boxes_data = [
        [[1.42864347e-01, 5.89409590e-01, 8.20568502e-01, 7.83712029e-01],
         [1.13038629e-01, 5.90494156e-01, 8.08443785e-01, 7.77864337e-01],
         [1.63048059e-01, 5.94742000e-01, 8.12249303e-01, 7.87663996e-01],
         [1.29777223e-01, 5.90861559e-01, 8.43621969e-01, 7.81312227e-01],
         [1.61663324e-01, 5.93474388e-01, 8.09598088e-01, 7.89204478e-01],
         [1.13868237e-01, 5.90315461e-01, 8.14452648e-01, 7.79579520e-01],
         [1.65083930e-01, 5.58505952e-01, 4.59944963e-01, 6.34872854e-01],
         [1.68551877e-01, 5.59545994e-01, 4.42642570e-01, 6.35466576e-01],
         [1.85653567e-01, 5.53336978e-01, 4.76624668e-01, 6.34629130e-01],
         [1.78655505e-01, 5.35557508e-01, 7.59687126e-01, 6.66503549e-01],
         [1.70532703e-01, 6.18388474e-01, 4.50028300e-01, 7.84762323e-01],
         [1.70998275e-01, 6.68687463e-01, 4.39314961e-01, 7.72987127e-01]]
    ]
    self.scores_data = [[
        0.99056089, 0.91968429, 0.91328019, 0.86525613, 0.75137419, 0.74802309,
        0.37547377, 0.26630139, 0.25483057, 0.18991362, 0.03, 0.01
    ]]

  def testIOU(self):

    boxes_np = np.array(self.boxes_data, dtype=np.float32)

    def iou_fn(boxes):
      iou = ssd_architecture._bbox_overlap(boxes, boxes)
      return iou

    with tf.Session() as sess:
      boxes = tf.placeholder(boxes_np.dtype, shape=boxes_np.shape)
      tpu_model_fn = tpu.rewrite(iou_fn, [boxes])
      sess.run(tpu.initialize_system())
      sess.run(tf.global_variables_initializer())
      inputs = {
          boxes: boxes_np,
      }
      iou = sess.run(tpu_model_fn, inputs)
      sess.run(tpu.shutdown_system())
    iou_output = np.array(iou)
    self.assertAllClose(
        iou_output,
        [[[[
            0.99999988, 0.90851283, 0.91384727, 0.93148685, 0.9114396,
            0.92665923, 0.09521391, 0.09025703, 0.09254451, 0.27486169,
            0.35012585, 0.21252605
        ],
           [
               0.90851283, 0.99999988, 0.86043209, 0.91082579, 0.86508727,
               0.98035324, 0.09364863, 0.08882008, 0.09100712, 0.27224982,
               0.3370952, 0.21477918
           ],
           [
               0.91384727, 0.86043209, 0.99999994, 0.86468625, 0.97959578,
               0.87052649, 0.08705246, 0.0827493, 0.08453463, 0.26119694,
               0.37127772, 0.22344439
           ],
           [
               0.93148685, 0.91082579, 0.86468625, 0.99999994, 0.8627646,
               0.92732346, 0.08919499, 0.08458695, 0.08670953, 0.26147613,
               0.33258566, 0.20584686
           ],
           [
               0.9114396, 0.86508727, 0.97959578, 0.8627646, 0.99999982,
               0.86856151, 0.08901545, 0.08455542, 0.08646145, 0.26442167,
               0.36666632, 0.22066914
           ],
           [
               0.92665923, 0.98035324, 0.87052649, 0.92732346, 0.86856151,
               0.99999988, 0.09253918, 0.08775141, 0.08994443, 0.26924923,
               0.33609992, 0.21105805
           ],
           [
               0.09521391, 0.09364863, 0.08705246, 0.08919499, 0.08901545,
               0.09253918, 0.99999958, 0.9103201, 0.82557553, 0.27854183,
               0.0715298, 0.
           ],
           [
               0.09025703, 0.08882008, 0.0827493, 0.08458695, 0.08455542,
               0.08775141, 0.9103201, 0.99999952, 0.76669204, 0.26079145,
               0.07416078, 0.
           ],
           [
               0.09254451, 0.09100712, 0.08453463, 0.08670953, 0.08646145,
               0.08994443, 0.82557553, 0.76669204, 0.99999958, 0.31088972,
               0.06519231, 0.
           ],
           [
               0.27486169, 0.27224982, 0.26119694, 0.26147613, 0.26442167,
               0.26924923, 0.27854183, 0.26079145, 0.31088972, 0.99999994,
               0.11921325, 0.
           ],
           [
               0.35012585, 0.3370952, 0.37127772, 0.33258566, 0.36666632,
               0.33609992, 0.0715298, 0.07416078, 0.06519231, 0.11921325,
               0.9999997, 0.6018253
           ],
           [
               0.21252605, 0.21477918, 0.22344439, 0.20584686, 0.22066914,
               0.21105805, 0., 0., 0., 0., 0.6018253, 0.99999964
           ]]]])

  def testNonMaxSuppressionOp(self):
    boxes_np = np.array(self.boxes_data, dtype=np.float32)
    scores_np = np.array(self.scores_data, dtype=np.float32)

    def nms_fn(scores, boxes):
      (scores, boxes) = ssd_model._filter_scores(scores, boxes)
      (scores, boxes) = ssd_architecture.non_max_suppression_padded(
          scores=tf.to_float(scores),
          boxes=tf.to_float(boxes),
          max_output_size=12,
          iou_threshold=0.5)

      return scores, boxes

    with tf.Session() as sess:
      boxes = tf.placeholder(boxes_np.dtype, shape=boxes_np.shape)
      scores = tf.placeholder(scores_np.dtype, shape=scores_np.shape)
      tpu_model_fn = tpu.rewrite(nms_fn, [scores, boxes])
      sess.run(tpu.initialize_system())
      sess.run(tf.global_variables_initializer())
      inputs = {
          scores: scores_np,
          boxes: boxes_np,
      }
      (scores, boxes) = sess.run(tpu_model_fn, inputs)
      sess.run(tpu.shutdown_system())
    scores_output = np.array(scores)
    boxes_output = np.array(boxes)
    self.assertAllClose(
        boxes_output,
        [[[0.14286435, 0.58940959, 0.8205685, 0.78371203],
          [0.16508393, 0.55850595, 0.45994496, 0.63487285],
          [0.178656, 0.535558, 0.759687, 0.666504], [0, 0, 0, 0], [0, 0, 0, 0],
          [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
          [0, 0, 0, 0], [0, 0, 0, 0]]])
    self.assertAllClose(scores_output, [[
        0.990561, 0.375474, 0.189914, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0
    ]])


if __name__ == "__main__":
  tf.test.main()
