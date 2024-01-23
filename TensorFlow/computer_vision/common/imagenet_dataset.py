# Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company

import os

import tensorflow as tf

from habana_frameworks.tensorflow.multinode_helpers import comm_size, comm_rank

def media_loader_can_be_used(jpeg_data_dir):
    try:
        import habana_frameworks.medialoaders
        import habana_frameworks.mediapipe
    except:
        return False

    if jpeg_data_dir is None:
        return False

    from habana_frameworks.tensorflow import habana_device
    # Media loader is not available for first-gen Gaudi
    return habana_device.get_type().split()[0] != "GAUDI"

def habana_imagenet_dataset(is_training,
                            jpeg_data_dir,
                            batch_size,
                            num_channels,
                            img_size,
                            data_type,
                            use_distributed_eval,
                            use_pytorch_style_crop=False):
    """
    Function responsible for preparing TF dataset with media loader
    Args:
        is_training: A boolean denoting whether the input is for training.
        tf_data_dir: The directory containing the input data in tf_record format - used for fallback.
        jpeg_data_dir: The directory containing the input data in jpeg format - used for media loader.
        batch_size: The number of samples per batch.
        num_channels: Number of channels.
        img_size: Image size.
        data_type: Data type to use for images/features.
        use_distributed_eval: Whether or not to use distributed evaluation.
        use_pytorch_style_crop: Whether or not to use pytorch style crop function (using tf algorithm by default)
    Returns:
        A dataset that can be used for iteration.
    """

    from habana_frameworks.mediapipe.media_types import dtype
    if data_type == tf.float32:
        media_data_type = dtype.FLOAT32
    elif data_type == tf.bfloat16:
        media_data_type = dtype.BFLOAT16
    else:
        raise RuntimeError("Unsupported data type {}.".format(data_type))

    if comm_size() > 1 and (is_training or use_distributed_eval):
        num_slices = comm_size()
        slice_index = comm_rank()
    else:
        num_slices = 1
        slice_index = 0

    from habana_frameworks.mediapipe.media_types import randomCropType
    if not is_training:
        crop_type = randomCropType.CENTER_CROP
    elif use_pytorch_style_crop:
        crop_type = randomCropType.RANDOMIZED_AREA_AND_ASPECT_RATIO_CROP
    else:
        crop_type = randomCropType.RANDOMIZED_ASPECT_RATIO_CROP

    if jpeg_data_dir is not None:
        if is_training:
            data_dir = os.path.join(jpeg_data_dir, 'train')
        else:
            data_dir = os.path.join(jpeg_data_dir, 'val')
    else:
        raise RuntimeError("--jpeg_data_dir not provided")

    queue_depth = 3

    from TensorFlow.computer_vision.common.resnet_media_pipe import ResnetPipe
    pipe = ResnetPipe("hpu", queue_depth, batch_size, num_channels, img_size, img_size, is_training,
                      data_dir, media_data_type, num_slices, slice_index, crop_type)

    from habana_frameworks.tensorflow.media.habana_dataset import HabanaDataset
    dataset = HabanaDataset(output_shapes=[(batch_size,
                                            img_size,
                                            img_size,
                                            num_channels),
                                            (batch_size,)],
                            output_types=[data_type, tf.float32], pipeline=pipe)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset
