# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
# Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
###############################################################################


import itertools
import os
import math as m
import numpy as np

DALI_AVAILABLE = None
try:
    import nvidia.dali.fn as fn
    import nvidia.dali.math as math
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False

if DALI_AVAILABLE:
    RAND_AUG_PROB = 0.15

    def get_numpy_reader(files, shard_id, num_shards, seed, shuffle):
        if shard_id is None:
            shard_id = int(os.getenv("LOCAL_RANK", "0"))
        return ops.readers.Numpy(
            seed=seed,
            files=files,
            device="cpu",
            read_ahead=True,
            shard_id=shard_id,
            pad_last_batch=True,
            num_shards=num_shards,
            shuffle_after_epoch=shuffle,
        )


    class TrainPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, **kwargs):
            super(TrainPipeline, self).__init__(batch_size, num_threads, device_id)
            self.dim = kwargs["dim"]
            self.oversampling = kwargs["oversampling"]
            self.input_x = get_numpy_reader(
                num_shards=kwargs["num_device"],
                files=kwargs["imgs"],
                seed=kwargs["seed"],
                shard_id=device_id,
                shuffle=True,
            )
            self.input_y = get_numpy_reader(
                num_shards=kwargs["num_device"],
                files=kwargs["lbls"],
                seed=kwargs["seed"],
                shard_id=device_id,
                shuffle=True,
            )
            self.patch_size = kwargs["patch_size"]
            if self.dim == 2:
                self.patch_size = [kwargs["batch_size_2d"]] + self.patch_size
            self.crop_shape = types.Constant(np.array(self.patch_size), dtype=types.INT64)
            self.crop_shape_float = types.Constant(np.array(self.patch_size), dtype=types.FLOAT)
            shard_id = int(os.getenv("LOCAL_RANK", "0"))
            if kwargs['set_aug_seed']:
                aug_seed = kwargs['seed'] + shard_id
                self.aug_seed_kwargs = {'seed': aug_seed}
                print("TrainPipeline augmentation seed: ", aug_seed)
            else:
                self.aug_seed_kwargs = {}
                print("TrainPipeline WO augmentation seed")
            self.augment = kwargs['augment']

        def load_data(self):
            img, lbl = self.input_x(name="ReaderX"), self.input_y(name="ReaderY")
            img, lbl = fn.reshape(img, layout="CDHW"), fn.reshape(lbl, layout="CDHW")
            return img, lbl

        def random_augmentation(self, probability, augmented, original):
            condition = fn.cast(fn.random.coin_flip(probability=probability, **self.aug_seed_kwargs), dtype=types.DALIDataType.BOOL)
            neg_condition = condition ^ True
            return condition * augmented + neg_condition * original

        @staticmethod
        def slice_fn(img):
            return fn.slice(img, 1, 3, axes=[0])

        def crop_fn(self, img, lbl):
            center = fn.segmentation.random_mask_pixel(lbl, foreground=fn.random.coin_flip(probability=self.oversampling, **self.aug_seed_kwargs),
                                                    **self.aug_seed_kwargs)
            crop_anchor = self.slice_fn(center) - self.crop_shape // 2
            adjusted_anchor = math.max(0, crop_anchor)
            max_anchor = self.slice_fn(fn.shapes(lbl)) - self.crop_shape
            crop_anchor = math.min(adjusted_anchor, max_anchor)
            img = fn.slice(img, crop_anchor, self.crop_shape, axis_names="DHW", out_of_bounds_policy="pad")
            lbl = fn.slice(lbl, crop_anchor, self.crop_shape, axis_names="DHW", out_of_bounds_policy="pad")
            return img, lbl

        def zoom_fn(self, img, lbl):
            resized_shape = self.crop_shape * self.random_augmentation(RAND_AUG_PROB, fn.random.uniform(range=(0.7, 1.0), **self.aug_seed_kwargs), 1.0)
            img, lbl = fn.crop(img, crop=resized_shape), fn.crop(lbl, crop=resized_shape)
            img = fn.resize(img, interp_type=types.DALIInterpType.INTERP_CUBIC, size=self.crop_shape_float)
            lbl = fn.resize(lbl, interp_type=types.DALIInterpType.INTERP_NN, size=self.crop_shape_float)
            return img, lbl

        def noise_fn(self, img):
            img_noised = img + fn.random.normal(img, stddev=fn.random.uniform(range=(0.0, 0.33), **self.aug_seed_kwargs), **self.aug_seed_kwargs)
            return self.random_augmentation(RAND_AUG_PROB, img_noised, img)

        def blur_fn(self, img):
            img_blured = fn.gaussian_blur(img, sigma=fn.random.uniform(range=(0.5, 1.5), **self.aug_seed_kwargs),**self.aug_seed_kwargs)
            return self.random_augmentation(RAND_AUG_PROB, img_blured, img)

        def brightness_fn(self, img):
            brightness_scale = self.random_augmentation(RAND_AUG_PROB, fn.random.uniform(range=(0.7, 1.3), **self.aug_seed_kwargs), 1.0)
            return img * brightness_scale

        def contrast_fn(self, img):
            min_, max_ = fn.reductions.min(img), fn.reductions.max(img)
            scale = self.random_augmentation(RAND_AUG_PROB, fn.random.uniform(range=(0.65, 1.5), **self.aug_seed_kwargs), 1.0)
            img = math.clamp(img * scale, min_, max_)
            return img

        def flips_fn(self, img, lbl):
            kwargs = {"horizontal": fn.random.coin_flip(probability=0.33, **self.aug_seed_kwargs),
                    "vertical": fn.random.coin_flip(probability=0.33, **self.aug_seed_kwargs)}
            if self.dim == 3:
                kwargs.update({"depthwise": fn.random.coin_flip(probability=0.33, **self.aug_seed_kwargs)})
            return fn.flip(img, **kwargs), fn.flip(lbl, **kwargs)

        def transpose_fn(self, img, lbl):
            img, lbl = fn.transpose(img, perm=(1, 0, 2, 3)), fn.transpose(lbl, perm=(1, 0, 2, 3))
            return img, lbl

        def define_graph(self):
            img, lbl = self.load_data()
            img, lbl = self.crop_fn(img, lbl)
            if self.augment:
                img, lbl = self.zoom_fn(img, lbl)
                img, lbl = self.flips_fn(img, lbl)
                img = self.noise_fn(img)
                img = self.blur_fn(img)
                img = self.brightness_fn(img)
                img = self.contrast_fn(img)
            if self.dim == 2:
                img, lbl = self.transpose_fn(img, lbl)
            return img, lbl


    class EvalPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, **kwargs):
            super(EvalPipeline, self).__init__(batch_size, num_threads, device_id)
            self.input_x = get_numpy_reader(
                files=kwargs["imgs"],
                shard_id=device_id,
                num_shards=kwargs["num_device"],
                seed=kwargs["seed"],
                shuffle=False,
            )
            self.input_y = get_numpy_reader(
                files=kwargs["lbls"],
                shard_id=device_id,
                num_shards=kwargs["num_device"],
                seed=kwargs["seed"],
                shuffle=False,
            )

        def define_graph(self):
            img, lbl = self.input_x(name="ReaderX"), self.input_y(name="ReaderY")
            img, lbl = fn.reshape(img, layout="CDHW"), fn.reshape(lbl, layout="CDHW")
            return img, lbl


    class TestPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, **kwargs):
            super(TestPipeline, self).__init__(batch_size, num_threads, device_id)
            self.input_x = get_numpy_reader(
                files=kwargs["imgs"],
                shard_id=device_id,
                num_shards=kwargs["num_device"],
                seed=kwargs["seed"],
                shuffle=False,
            )
            self.input_meta = get_numpy_reader(
                files=kwargs["meta"],
                shard_id=device_id,
                num_shards=kwargs["num_device"],
                seed=kwargs["seed"],
                shuffle=False,
            )

        def define_graph(self):
            img, meta = self.input_x(name="ReaderX"), self.input_meta(name="ReaderY")
            img = fn.reshape(img, layout="CDHW")
            return img, meta


    class BenchmarkPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, **kwargs):
            super(BenchmarkPipeline, self).__init__(batch_size, num_threads, device_id)
            self.input_x = get_numpy_reader(
                files=kwargs["imgs"],
                shard_id=device_id,
                seed=kwargs["seed"],
                num_shards=kwargs["num_device"],
                shuffle=False,
            )
            self.input_y = get_numpy_reader(
                files=kwargs["lbls"],
                shard_id=device_id,
                num_shards=kwargs["num_device"],
                seed=kwargs["seed"],
                shuffle=False,
            )
            self.dim = kwargs["dim"]
            self.patch_size = kwargs["patch_size"]
            if self.dim == 2:
                self.patch_size = [kwargs["batch_size_2d"]] + self.patch_size

        def load_data(self):
            img, lbl = self.input_x(name="ReaderX"), self.input_y(name="ReaderY")
            img, lbl = fn.reshape(img, layout="CDHW"), fn.reshape(lbl, layout="CDHW")
            return img, lbl

        def transpose_fn(self, img, lbl):
            img, lbl = fn.transpose(img, perm=(1, 0, 2, 3)), fn.transpose(lbl, perm=(1, 0, 2, 3))
            return img, lbl

        def crop_fn(self, img, lbl):
            img = fn.crop(img, crop=self.patch_size, out_of_bounds_policy="pad")
            lbl = fn.crop(lbl, crop=self.patch_size, out_of_bounds_policy="pad")
            return img, lbl

        def define_graph(self):
            img, lbl = self.load_data()
            img, lbl = self.crop_fn(img, lbl)
            if self.dim == 2:
                img, lbl = self.transpose_fn(img, lbl)
            return img, lbl


    class LightningWrapper(DALIGenericIterator):
        def __init__(self, pipe, num_shards: int, train_mode: str, dataset_len: int,  **kwargs):
            super().__init__(pipe,
                             last_batch_padded=False,
                             last_batch_policy=LastBatchPolicy.FILL if train_mode=="train" else LastBatchPolicy.PARTIAL,
                             **kwargs)

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

    def fetch_dali_loader(imgs, lbls, batch_size, mode, **kwargs):
        assert len(imgs) > 0, "Got empty list of images"
        if lbls is not None:
            assert len(imgs) == len(lbls), f"Got {len(imgs)} images but {len(lbls)} lables"

        if kwargs["benchmark"]:  # Just to make sure the number of examples is large enough for benchmark run.
            nbs = kwargs["test_batches"] if mode == "test" else kwargs["train_batches"]
            if kwargs["dim"] == 3:
                nbs *= batch_size
            imgs = list(itertools.chain(*(100 * [imgs])))[: nbs * kwargs["num_device"]]
            lbls = list(itertools.chain(*(100 * [lbls])))[: nbs * kwargs["num_device"]]
        if mode == "eval":
            reminder = len(imgs) % kwargs["num_device"]
            if reminder != 0:
                imgs = imgs[:-reminder]
                lbls = lbls[:-reminder]

        pipe_kwargs = {
            "imgs": imgs,
            "lbls": lbls,
            "dim": kwargs["dim"],
            "num_device": kwargs["num_device"],
            "seed": kwargs["seed"],
            "meta": kwargs["meta"],
            "patch_size": kwargs["patch_size"],
            "oversampling": kwargs["oversampling"],
        }
        if kwargs["benchmark"]:
            pipeline = BenchmarkPipeline
            output_map = ["image", "label"]
            dynamic_shape = False
            if kwargs["dim"] == 2:
                pipe_kwargs.update({"batch_size_2d": batch_size})
                batch_size = 1
        elif mode == "train":
            pipeline = TrainPipeline
            output_map = ["image", "label"]
            dynamic_shape = False
            if kwargs["dim"] == 2:
                pipe_kwargs.update({"batch_size_2d": batch_size // kwargs["nvol"]})
                batch_size = kwargs["nvol"]

            pipe_kwargs.update({'augment': kwargs['augment'], 'set_aug_seed': kwargs['set_aug_seed']})

        elif mode == "eval":
            pipeline = EvalPipeline
            output_map = ["image", "label"]
            dynamic_shape = True
        else:
            pipeline = TestPipeline
            output_map = ["image", "label"]
            dynamic_shape = True

        if kwargs["device"] == "gpu":
            device_id = int(os.getenv("LOCAL_RANK", "0"))
        else:
            device_id = None

        pipe = pipeline(batch_size, kwargs["num_workers"], device_id, **pipe_kwargs)
        return LightningWrapper(
            pipe,
            auto_reset=True,
            reader_name="ReaderX",
            output_map=output_map,
            dynamic_shape=dynamic_shape,
            train_mode=mode,
            num_shards=kwargs["num_device"],
            dataset_len=len(imgs)
        )
else:
    def fetch_dali_loader(imgs, lbls, batch_size, mode, **kwargs):
        pass
