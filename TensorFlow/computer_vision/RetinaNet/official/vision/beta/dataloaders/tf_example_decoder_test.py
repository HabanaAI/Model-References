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

"""Tests for tf_example_decoder.py."""

import io
# Import libraries
from absl.testing import parameterized
import numpy as np
from PIL import Image
import tensorflow as tf

from official.vision.beta.dataloaders import tf_example_decoder


DUMP_SOURCE_ID = b'123'


def _encode_image(image_array, fmt):
  image = Image.fromarray(image_array)
  with io.BytesIO() as output:
    image.save(output, format=fmt)
    return output.getvalue()


class TfExampleDecoderTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      (100, 100, 0, True),
      (100, 100, 1, True),
      (100, 100, 2, True),
      (100, 100, 0, False),
      (100, 100, 1, False),
      (100, 100, 2, False),
  )
  def test_result_shape(self,
                        image_height,
                        image_width,
                        num_instances,
                        regenerate_source_id):
    decoder = tf_example_decoder.TfExampleDecoder(
        include_mask=True, regenerate_source_id=regenerate_source_id)

    image = _encode_image(
        np.uint8(np.random.rand(image_height, image_width, 3) * 255),
        fmt='JPEG')
    if num_instances == 0:
      xmins = []
      xmaxs = []
      ymins = []
      ymaxs = []
      labels = []
      areas = []
      is_crowds = []
      masks = []
    else:
      xmins = list(np.random.rand(num_instances))
      xmaxs = list(np.random.rand(num_instances))
      ymins = list(np.random.rand(num_instances))
      ymaxs = list(np.random.rand(num_instances))
      labels = list(np.random.randint(100, size=num_instances))
      areas = [(xmax - xmin) * (ymax - ymin) * image_height * image_width
               for xmin, xmax, ymin, ymax in zip(xmins, xmaxs, ymins, ymaxs)]
      is_crowds = [0] * num_instances
      masks = []
      for _ in range(num_instances):
        mask = _encode_image(
            np.uint8(np.random.rand(image_height, image_width) * 255),
            fmt='PNG')
        masks.append(mask)
    serialized_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/encoded': (
                    tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[image]))),
                'image/source_id': (
                    tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[DUMP_SOURCE_ID]))),
                'image/height': (
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[image_height]))),
                'image/width': (
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[image_width]))),
                'image/object/bbox/xmin': (
                    tf.train.Feature(
                        float_list=tf.train.FloatList(value=xmins))),
                'image/object/bbox/xmax': (
                    tf.train.Feature(
                        float_list=tf.train.FloatList(value=xmaxs))),
                'image/object/bbox/ymin': (
                    tf.train.Feature(
                        float_list=tf.train.FloatList(value=ymins))),
                'image/object/bbox/ymax': (
                    tf.train.Feature(
                        float_list=tf.train.FloatList(value=ymaxs))),
                'image/object/class/label': (
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(value=labels))),
                'image/object/is_crowd': (
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(value=is_crowds))),
                'image/object/area': (
                    tf.train.Feature(
                        float_list=tf.train.FloatList(value=areas))),
                'image/object/mask': (
                    tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=masks))),
            })).SerializeToString()
    decoded_tensors = decoder.decode(
        tf.convert_to_tensor(value=serialized_example))

    results = tf.nest.map_structure(lambda x: x.numpy(), decoded_tensors)

    self.assertAllEqual(
        (image_height, image_width, 3), results['image'].shape)
    if not regenerate_source_id:
      self.assertEqual(DUMP_SOURCE_ID, results['source_id'])
    self.assertEqual(image_height, results['height'])
    self.assertEqual(image_width, results['width'])
    self.assertAllEqual(
        (num_instances,), results['groundtruth_classes'].shape)
    self.assertAllEqual(
        (num_instances,), results['groundtruth_is_crowd'].shape)
    self.assertAllEqual(
        (num_instances,), results['groundtruth_area'].shape)
    self.assertAllEqual(
        (num_instances, 4), results['groundtruth_boxes'].shape)
    self.assertAllEqual(
        (num_instances, image_height, image_width),
        results['groundtruth_instance_masks'].shape)
    self.assertAllEqual(
        (num_instances,), results['groundtruth_instance_masks_png'].shape)

  def test_result_content(self):
    decoder = tf_example_decoder.TfExampleDecoder(include_mask=True)

    image_content = [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                     [[0, 0, 0], [255, 255, 255], [255, 255, 255], [0, 0, 0]],
                     [[0, 0, 0], [255, 255, 255], [255, 255, 255], [0, 0, 0]],
                     [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]]
    image = _encode_image(np.uint8(image_content), fmt='PNG')
    image_height = 4
    image_width = 4
    num_instances = 2
    xmins = [0, 0.25]
    xmaxs = [0.5, 1.0]
    ymins = [0, 0]
    ymaxs = [0.5, 1.0]
    labels = [3, 1]
    areas = [
        0.25 * image_height * image_width, 0.75 * image_height * image_width
    ]
    is_crowds = [1, 0]
    mask_content = [[[255, 255, 0, 0],
                     [255, 255, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]],
                    [[0, 255, 255, 255],
                     [0, 255, 255, 255],
                     [0, 255, 255, 255],
                     [0, 255, 255, 255]]]
    masks = [_encode_image(np.uint8(m), fmt='PNG') for m in list(mask_content)]
    serialized_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/encoded': (
                    tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[image]))),
                'image/source_id': (
                    tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[DUMP_SOURCE_ID]))),
                'image/height': (
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[image_height]))),
                'image/width': (
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[image_width]))),
                'image/object/bbox/xmin': (
                    tf.train.Feature(
                        float_list=tf.train.FloatList(value=xmins))),
                'image/object/bbox/xmax': (
                    tf.train.Feature(
                        float_list=tf.train.FloatList(value=xmaxs))),
                'image/object/bbox/ymin': (
                    tf.train.Feature(
                        float_list=tf.train.FloatList(value=ymins))),
                'image/object/bbox/ymax': (
                    tf.train.Feature(
                        float_list=tf.train.FloatList(value=ymaxs))),
                'image/object/class/label': (
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(value=labels))),
                'image/object/is_crowd': (
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(value=is_crowds))),
                'image/object/area': (
                    tf.train.Feature(
                        float_list=tf.train.FloatList(value=areas))),
                'image/object/mask': (
                    tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=masks))),
            })).SerializeToString()
    decoded_tensors = decoder.decode(
        tf.convert_to_tensor(value=serialized_example))

    results = tf.nest.map_structure(lambda x: x.numpy(), decoded_tensors)

    self.assertAllEqual(
        (image_height, image_width, 3), results['image'].shape)
    self.assertAllEqual(image_content, results['image'])
    self.assertEqual(DUMP_SOURCE_ID, results['source_id'])
    self.assertEqual(image_height, results['height'])
    self.assertEqual(image_width, results['width'])
    self.assertAllEqual(
        (num_instances,), results['groundtruth_classes'].shape)
    self.assertAllEqual(
        (num_instances,), results['groundtruth_is_crowd'].shape)
    self.assertAllEqual(
        (num_instances,), results['groundtruth_area'].shape)
    self.assertAllEqual(
        (num_instances, 4), results['groundtruth_boxes'].shape)
    self.assertAllEqual(
        (num_instances, image_height, image_width),
        results['groundtruth_instance_masks'].shape)
    self.assertAllEqual(
        (num_instances,), results['groundtruth_instance_masks_png'].shape)
    self.assertAllEqual(
        [3, 1], results['groundtruth_classes'])
    self.assertAllEqual(
        [True, False], results['groundtruth_is_crowd'])
    self.assertNDArrayNear(
        [0.25 * image_height * image_width, 0.75 * image_height * image_width],
        results['groundtruth_area'], 1e-4)
    self.assertNDArrayNear(
        [[0, 0, 0.5, 0.5], [0, 0.25, 1.0, 1.0]],
        results['groundtruth_boxes'], 1e-4)
    self.assertNDArrayNear(
        mask_content, results['groundtruth_instance_masks'], 1e-4)
    self.assertAllEqual(
        masks, results['groundtruth_instance_masks_png'])

  def test_handling_missing_fields(self):
    decoder = tf_example_decoder.TfExampleDecoder(include_mask=True)

    image_content = [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                     [[0, 0, 0], [255, 255, 255], [255, 255, 255], [0, 0, 0]],
                     [[0, 0, 0], [255, 255, 255], [255, 255, 255], [0, 0, 0]],
                     [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]]
    image = _encode_image(np.uint8(image_content), fmt='PNG')
    image_height = 4
    image_width = 4
    num_instances = 2
    xmins = [0, 0.25]
    xmaxs = [0.5, 1.0]
    ymins = [0, 0]
    ymaxs = [0.5, 1.0]
    labels = [3, 1]
    mask_content = [[[255, 255, 0, 0],
                     [255, 255, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]],
                    [[0, 255, 255, 255],
                     [0, 255, 255, 255],
                     [0, 255, 255, 255],
                     [0, 255, 255, 255]]]
    masks = [_encode_image(np.uint8(m), fmt='PNG') for m in list(mask_content)]
    serialized_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/encoded': (
                    tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[image]))),
                'image/source_id': (
                    tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[DUMP_SOURCE_ID]))),
                'image/height': (
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[image_height]))),
                'image/width': (
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[image_width]))),
                'image/object/bbox/xmin': (
                    tf.train.Feature(
                        float_list=tf.train.FloatList(value=xmins))),
                'image/object/bbox/xmax': (
                    tf.train.Feature(
                        float_list=tf.train.FloatList(value=xmaxs))),
                'image/object/bbox/ymin': (
                    tf.train.Feature(
                        float_list=tf.train.FloatList(value=ymins))),
                'image/object/bbox/ymax': (
                    tf.train.Feature(
                        float_list=tf.train.FloatList(value=ymaxs))),
                'image/object/class/label': (
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(value=labels))),
                'image/object/mask': (
                    tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=masks))),
            })).SerializeToString()
    decoded_tensors = decoder.decode(
        tf.convert_to_tensor(serialized_example))
    results = tf.nest.map_structure(lambda x: x.numpy(), decoded_tensors)

    self.assertAllEqual(
        (image_height, image_width, 3), results['image'].shape)
    self.assertAllEqual(image_content, results['image'])
    self.assertEqual(DUMP_SOURCE_ID, results['source_id'])
    self.assertEqual(image_height, results['height'])
    self.assertEqual(image_width, results['width'])
    self.assertAllEqual(
        (num_instances,), results['groundtruth_classes'].shape)
    self.assertAllEqual(
        (num_instances,), results['groundtruth_is_crowd'].shape)
    self.assertAllEqual(
        (num_instances,), results['groundtruth_area'].shape)
    self.assertAllEqual(
        (num_instances, 4), results['groundtruth_boxes'].shape)
    self.assertAllEqual(
        (num_instances, image_height, image_width),
        results['groundtruth_instance_masks'].shape)
    self.assertAllEqual(
        (num_instances,), results['groundtruth_instance_masks_png'].shape)
    self.assertAllEqual(
        [3, 1], results['groundtruth_classes'])
    self.assertAllEqual(
        [False, False], results['groundtruth_is_crowd'])
    self.assertNDArrayNear(
        [0.25 * image_height * image_width, 0.75 * image_height * image_width],
        results['groundtruth_area'], 1e-4)
    self.assertNDArrayNear(
        [[0, 0, 0.5, 0.5], [0, 0.25, 1.0, 1.0]],
        results['groundtruth_boxes'], 1e-4)
    self.assertNDArrayNear(
        mask_content, results['groundtruth_instance_masks'], 1e-4)
    self.assertAllEqual(
        masks, results['groundtruth_instance_masks_png'])


if __name__ == '__main__':
  tf.test.main()
