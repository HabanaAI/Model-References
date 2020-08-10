"""Tests for topk_mask implementation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import googletest
import topk_mask


# Reference implementation with the same expected accuracy as the fast
# implementation using tf.nn.top_k and tf.cast.
def refernce_topk_mask(score, k):
  values, _ = tf.nn.top_k(score, k=k)
  kth = tf.slice(values,
                 [0] * (values.shape.rank - 1) + [k - 1],
                 [-1] * values.shape.rank)
  return tf.cast(score >= kth, dtype=tf.float32)


class TopkMaskTest(tf.test.TestCase):

  def testR1AllPositive(self):
    inputs = tf.placeholder(tf.float32, shape=(5,))
    mask = topk_mask.topk_mask(inputs, 2)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      mask_value = sess.run(
          mask, feed_dict={inputs: [0.12, 0.33, 0.05, 1, 0.088]})
      self.assertAllEqual(mask_value, [0, 1, 0, 1, 0])

  def testR1AllNegative(self):
    inputs = tf.placeholder(tf.float32, shape=(5,))
    mask = topk_mask.topk_mask(inputs, 2)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      mask_value = sess.run(
          mask, feed_dict={inputs: [-0.12, -0.33, -0.05, -1, -0.088]})
      self.assertAllEqual(mask_value, [0, 0, 1, 0, 1])

  def testR1PositiveSplit(self):
    inputs = tf.placeholder(tf.float32, shape=(5,))
    mask = topk_mask.topk_mask(inputs, 2)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      mask_value = sess.run(
          mask, feed_dict={inputs: [-0.12, 0.33, 0.05, -1, 0.088]})
      self.assertAllEqual(mask_value, [0, 1, 0, 0, 1])

  def testR2WithDuplicate(self):
    inputs = tf.placeholder(tf.float32, shape=(2, 5))
    mask = topk_mask.topk_mask(inputs, 2)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      mask_value = sess.run(
          mask,
          feed_dict={
              inputs: [[0.1, 0.1, 0.1, 0.10001, 0.1],
                       [0.01, 0.000001, 0.2, 0.1, 0.1]]
          })
      self.assertAllEqual(mask_value, [[1, 0, 0, 1, 0], [0, 0, 1, 1, 0]])

  def testR1WithDuplicate(self):
    inputs = tf.placeholder(tf.float32, shape=(5,))
    mask = topk_mask.topk_mask(inputs, 2)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      mask_value = sess.run(mask, feed_dict={inputs: [0.1, 0.1, 0.2, 0.1, 0.1]})
      self.assertAllEqual(mask_value, [1, 0, 1, 0, 0])

  def testR1SameValues(self):
    inputs = tf.placeholder(tf.float32, shape=(5,))
    mask = topk_mask.topk_mask(inputs, 2)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      mask_value = sess.run(
          mask, feed_dict={inputs: [-0.1, -0.1, -0.1, -0.1, -0.1]})
      self.assertAllEqual(mask_value, [1, 1, 0, 0, 0])

  def testR1NegativeSplit(self):
    inputs = tf.placeholder(tf.float32, shape=(5,))
    mask = topk_mask.topk_mask(inputs, 2)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      mask_value = sess.run(
          mask, feed_dict={inputs: [-0.12, -0.33, -0.05, -1, 0.088]})
      self.assertAllEqual(mask_value, [0, 0, 1, 0, 1])

  def testR2MixedSplit(self):
    inputs = tf.placeholder(tf.float32, shape=(2, 5))
    mask = topk_mask.topk_mask(inputs, 2)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      mask_value = sess.run(
          mask, feed_dict={inputs: [[-0.12, 0.33, 0.05, -1, 0.088],
                                    [-0.12, -0.33, -0.05, -1, 0.088]]})
      self.assertAllEqual(mask_value, [[0, 1, 0, 0, 1], [0, 0, 1, 0, 1]])

  def testR3Large(self):
    data = np.random.randn(33, 55, 77)

    inputs = tf.placeholder(tf.float32, shape=(33, 55, 77))
    mask = topk_mask.topk_mask(inputs, 37)
    refernce_mask = refernce_topk_mask(inputs, 37)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      mask_value = sess.run(mask, feed_dict={inputs: data})
      refernce_mask_value = sess.run(refernce_mask, feed_dict={inputs: data})
      self.assertAllEqual(mask_value, refernce_mask_value)


if __name__ == '__main__':
  googletest.main()
