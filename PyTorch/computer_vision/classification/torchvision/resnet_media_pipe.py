# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company

import time

import numpy as np
import media_pipe_settings as settings

try:
    from habana_frameworks.mediapipe import fn
    from habana_frameworks.mediapipe.mediapipe import MediaPipe
    from habana_frameworks.mediapipe.media_types import imgtype
    from habana_frameworks.mediapipe.media_types import dtype
    from habana_frameworks.mediapipe.media_types import ftype
    from habana_frameworks.mediapipe.media_types import randomCropType
    from habana_frameworks.mediapipe.media_types import decoderStage
    from habana_frameworks.mediapipe.operators.cpu_nodes.cpu_nodes import media_function
except:
    pass


class ResnetMediaPipe(MediaPipe):
    """
    Class defining resnet media pipe.

    """
    instance_count = 0

    def __init__(self, is_training=False, root=None, batch_size=1, shuffle=False, drop_last=True,
                 queue_depth=1, num_instances=1, instance_id=0, device=None, seed=None, num_threads=1):
        """
        :params is_training: True if ResnetMediaPipe handles training data, False in case of evaluation.
        :params root: path from which to load the images.
        :params batch_size: mediapipe output batch size.
        :params shuffle: whether images have to be shuffled.
        :params drop_last: whether to drop the last incomplete batch or round up.
        :params queue_depth: Number of preloaded image batches for every slice in mediapipe. <1/2/3>
        :params num_instances: number of devices.
        :params instance_id: instance id of current device.
        :params device: media device to run mediapipe on. <hpu/hpu:0>
        """
        self.is_training = is_training
        self.root = root
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.num_instances = num_instances
        self.instance_id = instance_id

        ResnetMediaPipe.instance_count += 1
        pipe_name = "{}:{}".format(
            self.__class__.__name__, ResnetMediaPipe.instance_count)
        pipe_name = str(pipe_name)

        super().__init__(device=device, batch_size=batch_size,
                         prefetch_depth=queue_depth, num_threads=num_threads, pipe_name=pipe_name)

        if seed == None:
            seed = int(time.time_ns() % (2**31 - 1))
        resize_dim = settings.TRAIN_RESIZE_DIM if self.is_training else settings.EVAL_RESIZE_DIM

        self.input = fn.ReadImageDatasetFromDir(dir=self.root, format="JPEG",
                                                seed=seed,
                                                shuffle=self.shuffle,
                                                drop_remainder=self.drop_last,
                                                label_dtype=dtype.INT32,
                                                num_slices=self.num_instances,
                                                slice_index=self.instance_id,
                                                device="cpu")

        if self.is_training == True:
            self.crop_window = fn.CropWindowGen(resize_width=resize_dim,
                                                resize_height=resize_dim,
                                                type=randomCropType.RANDOMIZED_AREA_AND_ASPECT_RATIO_CROP,
                                                scale_min=settings.DECODER_SCALE_MIN,
                                                scale_max=settings.DECODER_SCALE_MAX,
                                                ratio_min=settings.DECODER_RATIO_MIN,
                                                ratio_max=settings.DECODER_RATIO_MAX,
                                                seed=seed,
                                                device="cpu")
            self.decode = fn.ImageDecoder(output_format=imgtype.RGB_P,
                                          resize=[resize_dim, resize_dim],
                                          resampling_mode=ftype.BI_LINEAR,
                                          decoder_stage=decoderStage.ENABLE_ALL_STAGES,
                                          device="hpu")
        else:
            self.decode = fn.ImageDecoder(output_format=imgtype.RGB_P,
                                          resize=[resize_dim, resize_dim],
                                          resampling_mode=ftype.BI_LINEAR,
                                          decoder_stage=decoderStage.ENABLE_ALL_STAGES,
                                          device="hpu")

        if self.is_training == True:
            self.prob = fn.Constant(constant=settings.FLIP_PROBABILITY,
                                    dtype=dtype.FLOAT32,
                                    device='cpu')
            self.coin_flip = fn.CoinFlip(seed=seed,
                                         dtype=dtype.INT8,
                                         device='cpu')
            self.random_flip = fn.RandomFlip(
                horizontal=settings.USE_HORIZONTAL_FLIP,
                device='hpu')

        self.reshape = fn.Reshape(size=[batch_size],
                                  tensorDim=1,
                                  layout='',
                                  dtype=dtype.UINT8,
                                  device='hpu')

        normalized_mean = np.array([m * settings.RGB_MULTIPLIER for m in settings.RGB_MEAN_VALUES],
                                   dtype=np.float32)
        normalized_std = np.array([1 / (s * settings.RGB_MULTIPLIER) for s in settings.RGB_STD_VALUES],
                                  dtype=np.float32)

        # Define Constant tensors
        self.norm_mean = fn.MediaConst(data=normalized_mean,
                                       shape=[1, 1, normalized_mean.size],
                                       dtype=dtype.FLOAT32,
                                       device='cpu')
        self.norm_std = fn.MediaConst(data=normalized_std,
                                      shape=[1, 1, normalized_std.size],
                                      dtype=dtype.FLOAT32,
                                      device='cpu')

        if self.is_training == True:
            self.cmn = fn.CropMirrorNorm(crop_w=settings.CROP_DIM,
                                         crop_h=settings.CROP_DIM,
                                         crop_d=0,
                                         dtype=dtype.FLOAT32,
                                         device='hpu')
        else:
            self.cmn = fn.CropMirrorNorm(crop_w=settings.CROP_DIM,
                                         crop_h=settings.CROP_DIM,
                                         crop_d=0,
                                         crop_pos_x=settings.EVAL_CROP_X,
                                         crop_pos_y=settings.EVAL_CROP_Y,
                                         dtype=dtype.FLOAT32,
                                         device='hpu')
        self.reshape_lbs = fn.Reshape(size=[batch_size],
                                      tensorDim=1,
                                      layout='',
                                      dtype=dtype.INT32,
                                      device='hpu')

    def definegraph(self):
        """
        Method defines the media graph for Resnet.

        :returns : output images, labels
        """
        jpegs, data = self.input()

        if self.is_training == True:
            crop_window = self.crop_window(jpegs)
            images = self.decode(jpegs, crop_window)
            prob = self.prob()
            flip = self.coin_flip(prob)
            flip = self.reshape(flip)
            images = self.random_flip(images, flip)
        else:
            images = self.decode(jpegs)

        mean = self.norm_mean()
        std = self.norm_std()
        images = self.cmn(images, mean, std)
        data = self.reshape_lbs(data)

        return images, data
