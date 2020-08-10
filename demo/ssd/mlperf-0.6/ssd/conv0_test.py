"""Tests for conv0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf

from tensorflow.python.platform import test
import dataloader
import ssd_architecture


class SSDConv0Test(test.TestCase):

  def testConv0(self):
    batch_size = 4
    height = 300
    width = 300
    channel = 3
    random_input = np.random.normal(0.0, 1.0,
                                    [batch_size, height, width, channel])

    with tf.Session() as sess:
      with tf.variable_scope("conv"):
        inputs = tf.placeholder(
            tf.float32, shape=(batch_size, height, width, channel))
        space_to_depth_input = dataloader.fused_transpose_and_space_to_depth(
            inputs, block_size=2, transpose_input=False)
        conv0_output = ssd_architecture.conv0_space_to_depth(
            space_to_depth_input)
      with tf.variable_scope("conv", reuse=True):
        output = ssd_architecture.conv2d_fixed_padding(
            inputs=inputs, filters=64, kernel_size=7, strides=2)

      init = tf.global_variables_initializer()
      sess.run(init)

      conv0_result = sess.run(conv0_output, {inputs: random_input})
      result = sess.run(output, {inputs: random_input})

    self.assertAllClose(conv0_result, result, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  tf.test.main()
