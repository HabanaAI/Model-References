#!/usr/bin/env python3

###############################################################################
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
###############################################################################

from os import environ
import tensorflow as tf

from absl import app
from absl import logging
from absl import flags

from TensorFlow.computer_vision.Resnets.resnet_keras.transfer_learning_demo.flowers_dataset import build_dataset, IMAGE_SIZE
from TensorFlow.computer_vision.Resnets.resnet_keras import resnet_model
from TensorFlow.computer_vision.Resnets.resnet_keras import common
from TensorFlow.computer_vision.Resnets.utils.optimizers.keras import lars_util
from TensorFlow.computer_vision.Resnets.resnet_keras.official.utils.flags._conventions import help_wrap
from TensorFlow.computer_vision.common import imagenet_preprocessing
from TensorFlow.common.tb_utils import TensorBoardWithHParamsV2

from habana_frameworks.tensorflow import load_habana_module
from habana_frameworks.tensorflow import backward_compatible_optimizers

load_habana_module()

FLAGS = flags.FLAGS
flags.DEFINE_string(
    name='checkpoint_path',
    default=None,
    help=help_wrap('Checkpoint path.'))
flags.DEFINE_string(
    name='saved_model_path',
    default=None,
    help=help_wrap('Path to the saved model.'))
flags.DEFINE_boolean(
    name='do_data_augmentation',
    default=False,
    help=help_wrap('Indicates whether data augmentation should be performed.'))
flags.DEFINE_integer(
    name='new_batch_size',
    short_name='nbs',
    default=16,
    help=help_wrap('Batch size used for transfer learning.'))
flags.DEFINE_float(
    name='initial_learning_rate',
    default=1e-4,
    help=help_wrap('Initial learning rate for cyclical learning rate optimizer'))
flags.DEFINE_float(
    name='maximal_learning_rate',
    default=4e-3,
    help=help_wrap('Maximal learning rate for cyclical learning rate optimizer'))
flags.DEFINE_integer(
    name='step_size',
    default=40,
    help=help_wrap('Step size for cyclical learning rate optimizer'))

def perform_data_augmentation(preprocessing_model):
    preprocessing_model.add(
        tf.keras.layers.RandomRotation(40))
    preprocessing_model.add(
        tf.keras.layers.RandomTranslation(0, 0.2))
    preprocessing_model.add(
        tf.keras.layers.RandomTranslation(0.2, 0))
    # Like the old tf.keras.preprocessing.image.ImageDataGenerator(),
    # image sizes are fixed when reading, and then a random zoom is applied.
    # If all training inputs are larger than image_size, one could also use
    # RandomCrop with a batch size of 1 and rebatch later.
    preprocessing_model.add(
        tf.keras.layers.RandomZoom(0.2, 0.2))
    preprocessing_model.add(
        tf.keras.layers.RandomFlip(mode="horizontal"))

def main(_):
    environ['TF_BF16_CONVERSION'] = environ.get('TF_BF16_CONVERSION', '1')

    callbacks = [
        TensorBoardWithHParamsV2(
            FLAGS.flag_values_dict(),
            write_graph=False,
            log_dir=FLAGS.model_dir,
            update_freq=FLAGS.log_steps)
    ]

    train_dataset = build_dataset("training")
    class_names = tuple(train_dataset.class_names)
    train_size = train_dataset.cardinality().numpy()
    train_dataset = train_dataset.unbatch().batch(FLAGS.new_batch_size)
    train_dataset = train_dataset.repeat()

    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    preprocessing_model = tf.keras.Sequential([normalization_layer])
    if FLAGS.do_data_augmentation:
        perform_data_augmentation(preprocessing_model)
    train_dataset = train_dataset.map(lambda images, labels:
                            (preprocessing_model(images), labels))

    val_ds = build_dataset("validation")
    valid_size = val_ds.cardinality().numpy()
    val_ds = val_ds.unbatch().batch(FLAGS.new_batch_size)
    val_ds = val_ds.map(lambda images, labels:
                        (normalization_layer(images), labels))

    from tensorflow_addons.optimizers import CyclicalLearningRate
    cyclical_learning_rate = CyclicalLearningRate(
        initial_learning_rate=FLAGS.initial_learning_rate,
        maximal_learning_rate=FLAGS.maximal_learning_rate,
        step_size=FLAGS.step_size,
        scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
        scale_mode='cycle')

    _optimizer = backward_compatible_optimizers.Adam(learning_rate=cyclical_learning_rate)

    is_checkpoint_path_given = FLAGS.checkpoint_path is not None
    is_model_path_given = FLAGS.saved_model_path is not None
    if is_checkpoint_path_given and is_model_path_given:
        raise RuntimeError("Both checkpoint_path and saved_model_path given. Please choose one.")
    elif is_checkpoint_path_given:
        pretrained_model = resnet_model.resnet50(
            num_classes=imagenet_preprocessing.NUM_CLASSES, rescale_inputs=False)
        ckpt = tf.train.Checkpoint(model=pretrained_model)
        ckpt.restore(FLAGS.checkpoint_path).expect_partial()
    elif is_model_path_given:
        pretrained_model = tf.keras.models.load_model(FLAGS.saved_model_path)
    else:
        raise RuntimeError("Please provide checkpoint_path or saved_model_path argument.")
    for layer in pretrained_model.layers:
        layer.trainable = False
    pretrained_model = tf.keras.Model(inputs=pretrained_model.input, outputs=pretrained_model.layers[-3].output)

    # Build the model. The first layer after the input is the pretrained ResNet50, and all we do is switch the last layer to output according to Flowers dataset class names.
    model = tf.keras.Sequential([
        # Explicitly define the input shape so the model can be properly
        # loaded by the TFLiteConverter
        tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
        tf.keras.layers.Lambda(
            lambda x: x * 255.0 - tf.keras.backend.constant(    # pylint: disable=g-long-lambda
                imagenet_preprocessing.CHANNEL_MEANS,
                shape=[1, 1, 3],
                dtype=x.dtype),
            name='rescale'),
        pretrained_model,
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(len(class_names),
                            kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    ])
    model.build((None,)+IMAGE_SIZE+(3,))
    model.summary()

    # Model compilation
    model.compile(
    optimizer=_optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
    metrics=['accuracy'])

    # Model training
    steps_per_epoch = train_size // FLAGS.new_batch_size
    validation_steps = valid_size // FLAGS.new_batch_size
    model.fit(
        train_dataset,
        epochs=20, steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks)

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    common.define_keras_flags()
    common.define_habana_flags()
    lars_util.define_lars_flags()
    app.run(main)
