# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company

import time

import numpy as np

import TensorFlow.computer_vision.common.media_pipe_settings as settings

try:
    from habana_frameworks.mediapipe import fn
    from habana_frameworks.mediapipe.mediapipe import MediaPipe
    from habana_frameworks.mediapipe.media_types import imgtype, dtype
    from habana_frameworks.mediapipe.operators.cpu_nodes.cpu_nodes import media_function
    from habana_frameworks.mediapipe.plugins.readers.tfrecord_reader_cpp import tfr_reader_cpp, tfr_reader_params
except:
    pass

class ResnetPipe(MediaPipe):
    """
    Class defining resnet media pipe.

    """
    instance_count = 0

    def __init__(self, device, queue_depth, batch_size,
                 channel, height, width, is_training, data_dir,
                 out_dtype, num_slices, slice_index,
                 random_crop_type):
        """
        :params device: device name. <hpu>
        :params queue_depth: Number of preloaded image batches for every slice in mediapipe. <1/2/3>
        :params channel: mediapipe image output channel size.
        :params height: mediapipe image output height.
        :params width: mediapipe image output width.
        :params is_training: bool value to state is training pipe or validation pipe.
        :params data_dir: dataset directory to be used by dataset reader.
        :params out_dtype: output image datatype.
        :params num_slices: Total number of slice to be performed on dataset.
        :params slice_index: Slice index to be used for this instance of mediapipe.
        """
        pipe_type = "Train" if is_training else "Eval"
        ResnetPipe.instance_count += 1
        pipe_name = "{}:{}".format(
            self.__class__.__name__ + pipe_type, ResnetPipe.instance_count)
        pipe_name = str(pipe_name)

        super(
            ResnetPipe,
            self).__init__(
            device=device,
            prefetch_depth=queue_depth,
            batch_size=batch_size,
            pipe_name=pipe_name)

        # Setting repeat count to -1 means there is no predefined limit on number of image batches
        # to be iterated through by MediaPipe. With this setting MediaPipe will iterate through
        # all batches allocated to the given slice.
        self.set_repeat_count(-1)

        self.is_training = is_training
        self.out_dtype = out_dtype

        mediapipe_seed = int(time.time_ns() % (2**31 - 1))
        print("media data loader {}/{} seed : {}".format(slice_index,
                                                         num_slices,
                                                         mediapipe_seed))
        print("media data loader: data_dir:", data_dir)

        # file reader Node
        self.input = fn.ReadImageDatasetFromDir(dir=data_dir, format="JPEG",
                                                seed=mediapipe_seed,
                                                shuffle=True,
                                                label_dtype=dtype.FLOAT32,
                                                num_slices=num_slices,
                                                slice_index=slice_index)

        # decoder node
        if is_training == True:
            self.decode = fn.ImageDecoder(device="Gaudi2",
                                          output_format=imgtype.RGB_P,
                                          random_crop_type=random_crop_type,
                                          resize=[height, width],
                                          scale_min=settings.DECODER_SCALE_MIN,
                                          scale_max=settings.DECODER_SCALE_MAX,
                                          ratio_min=settings.DECODER_RATIO_MIN,
                                          ratio_max=settings.DECODER_RATIO_MAX,
                                          seed=mediapipe_seed)
        else:
            self.decode = fn.ImageDecoder(device="Gaudi2",
                                          output_format=imgtype.RGB_P,
                                          random_crop_type=random_crop_type,
                                          resize=[settings.EVAL_RESIZE,
                                                  settings.EVAL_RESIZE])

            self.crop = fn.Crop(crop_w=width,
                                crop_h=height,
                                crop_pos_x=settings.EVAL_CROP_X,
                                crop_pos_y=settings.EVAL_CROP_Y)
        # Random Flip node
        self.random_flip_input = fn.MediaFunc(func=RandomFlipFunction,
                                              shape=[batch_size],
                                              dtype=dtype.UINT8,
                                              seed=mediapipe_seed)

        self.random_flip = fn.RandomFlip(horizontal=settings.USE_HORIZONTAL_FLIP)
        # cast data to f32 for subtraction
        self.cast_pre = fn.Cast(dtype=dtype.FLOAT32)
        # substract mean node
        mean_data = np.array(settings.RGB_MEAN_VALUES,
                             dtype=np.float32)

        self.mean_node = fn.MediaConst(data=mean_data,
                                       shape=[1, 1, mean_data.size],
                                       dtype=dtype.FLOAT32)

        self.sub = fn.Sub(dtype=dtype.FLOAT32)
        # cast to output datatype
        self.cast_pst = fn.Cast(dtype=out_dtype)
        # Transpose node
        self.pst_transp = fn.Transpose(permutation=settings.TRANSPOSE_DIMS,
                                       tensorDim=len(settings.TRANSPOSE_DIMS),
                                       dtype=out_dtype)

    def definegraph(self):
        """
        Method defining dataflow between nodes.

        :returns : output nodes of the graph defined.
        """
        jpegs, data = self.input()
        if self.is_training == True:
            images = self.decode(jpegs)
            random_flip_input = self.random_flip_input()
            images = self.random_flip(images, random_flip_input)
        else:
            images = self.decode(jpegs)
            images = self.crop(images)
        mean = self.mean_node()
        images = self.cast_pre(images)
        images = self.sub(images, mean)
        if self.out_dtype != dtype.FLOAT32:
            images = self.cast_pst(images)
        images = self.pst_transp(images)
        return images, data

class RandomFlipFunction(media_function):
    """
    Class defining the random flip implementation.

    """

    def __init__(self, params):
        """
        :params params: dictionary of params conatining
                        shape: output shape of this class.
                        dtype: output dtype of this class.
                        seed: seed to be used for randomization.
        """
        self.np_shape = params['shape'][::-1]
        self.np_dtype = params['dtype']
        self.seed = params['seed']
        self.rng = np.random.default_rng(self.seed)

    def __call__(self):
        """
        :returns : random flip values calculated per image.
        """
        probabilities = [1.0 - settings.FLIP_PROBABILITY,
                         settings.FLIP_PROBABILITY]
        a = self.rng.choice([0, 1], p=probabilities, size=self.np_shape)
        a = np.array(a, dtype=self.np_dtype)
        return a
