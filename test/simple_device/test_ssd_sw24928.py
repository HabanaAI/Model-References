###############################################################################
# Copyright (C) 2020 HabanaLabs, Ltd.
# All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
#
###############################################################################

import os
import numpy as np
import pytest
import tensorflow as tf
import time

from test.test_helpers.test_utils import (CPU,
                                          habana_device,
                                          format_tc,
                                          load_habana_module)
tf.compat.v1.disable_eager_execution()

# This test compares old and new implementation of input normalization in SSD
# It prints execution time so we can see whether the issue is solved.


@pytest.mark.parametrize("dtype", [tf.float32], ids=format_tc)
@pytest.mark.parametrize("shape", [(64, 300, 300, 3)], ids=format_tc)
def test_ssd_sw24928(shape, dtype):
    load_habana_module()

    np.random.seed(123)
    features_data = np.random.uniform(size=shape)

    NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
    NORMALIZATION_STD = (0.229, 0.224, 0.225)

    def create_old_graph(device=habana_device, dtype=dtype):
        with tf.device(device):
            input_batch = tf.compat.v1.placeholder(dtype, shape=shape)
            features = input_batch

            features -= tf.constant(
                NORMALIZATION_MEAN, shape=[1, 1, 3], dtype=features.dtype)
            features /= tf.constant(
                NORMALIZATION_STD, shape=[1, 1, 3], dtype=features.dtype)

            feed_dict = {input_batch: features_data}

            return feed_dict, features

    def create_new_graph(device=habana_device, dtype=dtype):
        with tf.device(device):
            input_batch = tf.compat.v1.placeholder(dtype, shape=shape)
            features = input_batch

            COEF_MEAN = tf.constant(
                NORMALIZATION_MEAN, shape=[1, 1, 1, 3], dtype=features.dtype)
            features -= tf.tile(COEF_MEAN, tf.shape(features)-[0, 0, 0, 2])

            COEF_STD = 1.0 / tf.constant(
                NORMALIZATION_STD, shape=[1, 1, 1, 3], dtype=features.dtype)
            features *= tf.tile(COEF_STD, tf.shape(features)-[0, 0, 0, 2])

            feed_dict = {input_batch: features_data}

            return feed_dict, features

    tf.compat.v1.reset_default_graph()
    old_graph = create_old_graph(device=habana_device, dtype=dtype)
    new_graph = create_new_graph(device=habana_device, dtype=dtype)

    with tf.compat.v1.Session() as sess:
        print("Old normalization:")
        start = time.time()
        old_res = sess.run(old_graph[1], feed_dict=old_graph[0])
        end = time.time()
        print("1st run took {:.2f} ms".format((end - start)*1000))

        start = time.time()
        old_res = sess.run(old_graph[1], feed_dict=old_graph[0])
        end = time.time()
        print("2nd run took {:.2f} ms".format((end - start)*1000))

        start = time.time()
        old_res = sess.run(old_graph[1], feed_dict=old_graph[0])
        end = time.time()
        print("3rd run took {:.2f} ms".format((end - start)*1000))

        print("New normalization:")

        start = time.time()
        new_res = sess.run(new_graph[1], feed_dict=new_graph[0])
        end = time.time()
        print("1st run took {:.2f} ms".format((end - start)*1000))

        start = time.time()
        new_res = sess.run(new_graph[1], feed_dict=new_graph[0])
        end = time.time()
        print("2nd run took {:.2f} ms".format((end - start)*1000))

        start = time.time()
        new_res = sess.run(new_graph[1], feed_dict=new_graph[0])
        end = time.time()
        print("3rd run took {:.2f} ms".format((end - start)*1000))

        np.testing.assert_allclose(old_res, new_res)


if __name__ == "__main__":
    test_ssd_sw24928([64, 300, 300, 3], tf.float32)
