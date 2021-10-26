###############################################################################
# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
###############################################################################
"""ResNet50 model for Keras.

Related papers/blogs:
- https://arxiv.org/abs/1512.03385
- https://arxiv.org/pdf/1603.05027v2.pdf
- http://torch.ch/blog/2016/02/04/resnets.html

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras import models

from tensorflow.python.keras import backend
from TensorFlow.computer_vision.common import imagenet_preprocessing

layers = tf.keras.layers

def resnet(num_classes, model_type, batch_size=None, rescale_inputs=False):
    """Instantiates the ResNet50 architecture.

    Args:
        num_classes: `int` number of classes for image classification.
        batch_size: Size of the batches for each step.
        rescale_inputs: whether to rescale inputs from 0 to 1.

    Raises:
        ValueError if wrong model type is given

    Returns:
        A Keras model instance.
    """
    if model_type == "ResNet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
        from tensorflow.keras.applications import ResNet50 as ResNet
    elif model_type == "ResNet101":
        from tensorflow.keras.applications.resnet import preprocess_input
        from tensorflow.keras.applications import ResNet101 as ResNet
    elif model_type == "ResNet152":
        from tensorflow.keras.applications.resnet import preprocess_input
        from tensorflow.keras.applications import ResNet152 as ResNet
    elif model_type == "ResNet50V2":
        from tensorflow.keras.applications.resnet_v2 import preprocess_input
        from tensorflow.keras.applications import ResNet50V2 as ResNet
    elif model_type == "ResNet101V2":
        from tensorflow.keras.applications.resnet_v2 import preprocess_input
        from tensorflow.keras.applications import ResNet101V2 as ResNet
    elif model_type == "ResNet152V2":
        from tensorflow.keras.applications.resnet_v2 import preprocess_input
        from tensorflow.keras.applications import ResNet152V2 as ResNet
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported models: ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2")

    input_shape = (224, 224, 3)
    img_input = layers.Input(shape=input_shape, batch_size=batch_size)
    if rescale_inputs:
        # Hub image modules expect inputs in the range [0, 1]. This rescales these
        # inputs to the range expected by the trained model.
        x = layers.Lambda(
            lambda x: x * 255.0 - backend.constant(
                imagenet_preprocessing.CHANNEL_MEANS,
                shape=[1, 1, 3],
                dtype=x.dtype),
            name='rescale')(
                img_input)
    else:
        x = img_input

    x = preprocess_input(x)

    base_model = ResNet(input_tensor=x, weights=None, include_top=False)
    base_model.trainable = True
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes)(x)
    x = layers.Activation('softmax', dtype='float32')(x)

    return models.Model(img_input, x, name='resnet')