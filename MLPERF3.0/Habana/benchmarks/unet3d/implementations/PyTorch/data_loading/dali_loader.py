# Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved.
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
###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################
# Original file source: https://github.com/mlcommons/training_results_v2.0/blob/main/NVIDIA/benchmarks/unet3d/implementations/mxnet/data_loading/dali_loader.py
# Changes done to the original file:
# - convert code from mxnet to pytorch
# - remove redundant code
# - move all operations to CPU
# - cleanup

import math as m
import numpy as np

import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy


class InputIterator(object):
    def __init__(self, files: list, seed: int):
        self.files = files
        self.size = len(self.files)
        self.rng = np.random.default_rng(seed)
        self.order = [i for i in range(self.size)]
        self.start_from_idx = 0
        self.data = []
        for file_name in self.files:
            image = np.load(file_name)
            self.data.append(image)

    def __iter__(self):
        self.i = self.start_from_idx
        self.rng.shuffle(self.order)
        return self

    def __next__(self):
        self.i = (self.i + 1) % self.size
        return self.data[self.order[self.i]]

    def __len__(self):
        return self.size

    next = __next__


class BasicPipeline(Pipeline):
    def __init__(self, flags, batch_size: int, input_shape: list):
        super().__init__(batch_size=batch_size, num_threads=flags.num_workers, device_id=None, seed=flags.seed,
                         py_start_method="spawn", exec_pipelined=True, prefetch_queue_depth=2, exec_async=True)
        self.flags = flags
        self.internal_seed = flags.seed
        self.input_shape = input_shape

        self.crop_shape = types.Constant(self.input_shape, dtype=types.INT64)
        self.axis_names = "DHW"
        self.reshape = ops.Reshape(device="cpu", layout="CDHW")

    @staticmethod
    def random_augmentation(probability, augmented, original):
        condition = fn.cast(fn.random.coin_flip(probability=probability), dtype=types.DALIDataType.BOOL)
        neg_condition = condition ^ True
        return condition * augmented + neg_condition * original

    def reshape_fn(self, img, label):
        img = self.reshape(img)
        label = self.reshape(label)
        return img, label

    def random_flips_fn(self, img, label):
        hflip, vflip, dflip = [fn.random.coin_flip(probability=0.33) for _ in range(3)]
        flips = {"horizontal": hflip, "vertical": vflip, "depthwise": dflip, 'bytes_per_sample_hint': 8388608}
        return fn.flip(img, **flips), fn.flip(label, **flips)

    def gaussian_noise_fn(self, img):
        img_noised = img + fn.random.normal(img, stddev=0.1)
        return self.random_augmentation(0.1, img_noised, img)

    def brightness_fn(self, img):
        brightness_scale = self.random_augmentation(0.1, fn.random.uniform(range=(0.7, 1.3)), 1.0)
        return img * brightness_scale

    def gaussian_blur_fn(self, img):
        img_blured = fn.gaussian_blur(img, sigma=fn.random.uniform(range=(0.25, 1.5)))
        return self.random_augmentation(0.1, img_blured, img)

    @staticmethod
    def slice_fn(img, start_idx, length):
        return fn.slice(img, start_idx, length, axes=[0], out_of_bounds_policy="pad")

    def biased_crop_fn(self, img, label):
        roi_start, roi_end = fn.segmentation.random_object_bbox(label,
                                                                format='start_end',
                                                                foreground_prob=self.flags.oversampling,
                                                                classes=[1, 2],
                                                                k_largest=2,
                                                                seed=self.internal_seed,
                                                                cache_objects=True)

        anchor = fn.roi_random_crop(label,
                                    roi_start=roi_start,
                                    roi_end=roi_end,
                                    crop_shape=[1, *self.input_shape])
        anchor = fn.slice(anchor, 1, 3, axes=[0])
        img, label = fn.slice([img, label], anchor,
                              self.crop_shape, axis_names=self.axis_names,
                              out_of_bounds_policy="pad", device='cpu')

        return img, label


class TrainNumpyPipeline(BasicPipeline):
    def __init__(self, flags, image_iterator: InputIterator, label_iterator: InputIterator):
        super().__init__(flags=flags, batch_size=flags.batch_size, input_shape=flags.input_shape)
        self.image_iterator = image_iterator
        self.label_iterator = label_iterator

    def define_graph(self):
        image = fn.external_source(source=self.image_iterator, no_copy=True,
                                   name="ReaderX", layout='CDHW', batch=False)
        label = fn.external_source(source=self.label_iterator, no_copy=True,
                                   name="ReaderY", layout='CDHW', batch=False)

        image, label = self.biased_crop_fn(image, label)
        image, label = self.random_flips_fn(image, label)
        image = self.brightness_fn(image)
        image = self.gaussian_noise_fn(image)

        return image, label


class ValNumpyPipeline(BasicPipeline):
    def __init__(self, flags, image_list, label_list):
        super().__init__(flags=flags, batch_size=1, input_shape=flags.val_input_shape)
        self.input_x = ops.readers.Numpy(files=image_list,
                                         seed=flags.seed)
        self.input_y = ops.readers.Numpy(files=label_list,
                                         seed=flags.seed)

    def define_graph(self):
        image = self.input_x(name="ReaderX")
        label = self.input_y(name="ReaderY")

        image, label = self.reshape_fn(image, label)

        return image, label


class DaliIterator(DALIGenericIterator):
    def __init__(self, pipe: Pipeline, num_shards: int, train_mode: bool, dataset_len: int):
        super().__init__(pipelines=[pipe], output_map=["image", "label"],
                         auto_reset=True, size=-1, last_batch_padded=False,
                         last_batch_policy=LastBatchPolicy.FILL if train_mode else LastBatchPolicy.PARTIAL)

        self.dataset_size = dataset_len
        self.batch_size = pipe.max_batch_size
        self.num_shards = num_shards
        self.shard_size = m.ceil(self.dataset_size / num_shards)
        self.i = 0

    def __len__(self):
        return m.ceil(self.shard_size / self.batch_size)

    def __next__(self):
        if self.i >= len(self):
            self.i = 0
            raise StopIteration

        self.i += 1
        return super().__next__()[0]


def get_dali_loader(flags, image_list: list, label_list: list,
                    train_mode: bool, num_shards: int) -> DALIGenericIterator:
    if train_mode is True:
        image_iterator = InputIterator(files=image_list,
                                       seed=flags.seed)
        label_iterator = InputIterator(files=label_list,
                                       seed=flags.seed)
        pipe = TrainNumpyPipeline(flags,
                                  image_iterator=image_iterator,
                                  label_iterator=label_iterator)
    else:
        pipe = ValNumpyPipeline(flags,
                                image_list=image_list,
                                label_list=label_list)
    pipe.build()

    return DaliIterator(pipe=pipe,
                        num_shards=num_shards,
                        train_mode=train_mode,
                        dataset_len=len(image_list))
