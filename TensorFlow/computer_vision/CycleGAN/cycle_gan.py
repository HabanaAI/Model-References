#######################################################################################
# Title: CycleGAN
# Author: [A_K_Nain](https://twitter.com/A_K_Nain)
# Date created: 2020/08/12
# Last modified: 2020/08/12
# Description: Implementation of CycleGAN.
#
# CycleGAN
#
# CycleGAN is a model that aims to solve the image-to-image translation
# problem. The goal of the image-to-image translation problem is to learn the
# mapping between an input image and an output image using a training set of
# aligned image pairs. However, obtaining paired examples isn't always feasible.
# CycleGAN tries to learn this mapping without requiring paired input-output images,
# using cycle-consistent adversarial networks.
#
# - [Paper](https://arxiv.org/pdf/1703.10593.pdf)
# - [Original implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
#######################################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
#######################################################################################

import os
import re

from TensorFlow.common.horovod_helpers import hvd as horovod
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from tensorflow.data.experimental import AUTOTUNE as autotune

from TensorFlow.common.tb_utils import TensorBoardWithHParamsV2, ExamplesPerSecondKerasHookV2
from habana_frameworks.tensorflow.ops.instance_norm import HabanaInstanceNormalization
from arguments import CycleGANArgParser
from data import TrasformInputs
from modeling import get_discriminator, get_resnet_generator, CycleGan, TFPool
from monitoring import GANMonitor
from loss import get_adversarial_losses_fn
from scheduling import MultiOptimizerLR, CosineDecay


def is_master(hvd):
    return hvd is False or horovod.rank() == 0


def is_local_master(hvd):
    return hvd is False or horovod.local_rank() == 0


def train(args, cycle_gan_model, train_ds, test_ds, checkpoint=None):
    gen_X = cycle_gan_model.gen_X
    gen_Y = cycle_gan_model.gen_Y
    cycle_loss_fn = keras.losses.MeanAbsoluteError()
    id_loss_fn = keras.losses.MeanAbsoluteError()
    discriminator_loss_fn, generator_loss_fn = get_adversarial_losses_fn(
        'lsgan')

    lr_opts = dict(
        gen_optimizer=args.generator_lr,
        disc_optimizer=args.discriminator_lr,
    )

    if args.use_horovod:
        for k in lr_opts.keys():
            lr_opts[k] *= horovod.size()

    # Callbacks
    hooks = []
    if args.use_hooks and (args.log_all_workers or is_local_master(args.use_horovod)):

        hparams = {
            'batch_size': args.batch_size,
            'precision': args.data_type,
            'epochs': args.epochs,
            'logdir': args.logdir,
            'hvd_workers': args.hvd_workers
        }
        tb = TensorBoardWithHParamsV2(
            hparams, log_dir=os.path.join(args.logdir, "train"))
        examples_per_sec = ExamplesPerSecondKerasHookV2(
            output_dir=os.path.join(args.logdir, "train"), batch_size=args.batch_size)

        # Apply the preprocessing operations to the test data
        file_writer_imgs = tf.summary.create_file_writer(
            os.path.join(args.logdir, 'imgs'))
        plotter = GANMonitor(
            file_writer_imgs, test_ds[0], test_ds[1], freq=args.monitor_freq)
        steps_per_epoch = int(train_ds.reduce(0, lambda x, _: x+1).numpy()
                              ) if args.steps_per_epoch is None else args.steps_per_epoch
        save_every_n_steps = args.save_freq*steps_per_epoch
        checkpoint_filename = "cyclegan_checkpoints.{epoch:03d}"
        hooks += [plotter, tb, examples_per_sec,
                  keras.callbacks.ModelCheckpoint(filepath=os.path.join(args.logdir, checkpoint_filename), save_weights_only=True, save_freq=save_every_n_steps)]

    scheduler_hook = MultiOptimizerLR(initial_lr=lr_opts,
                                      multiplier=CosineDecay(steps=args.epochs - args.cosine_decay_delay,
                                                             clif=args.cosine_decay_delay))
    hooks += [scheduler_hook]

    start_epoch = 0
    if checkpoint:
        print(f'Resuming from {checkpoint}')
        start_epoch = int(re.search(r'[0-9]{3}', checkpoint)[0])
        cycle_gan_model.load_weights(checkpoint)
    else:
        print(f'Couldn\'t find checkpoint at {args.logdir}')

    pool_F = None
    pool_G = None
    if args.pool_size > 0:
        print('Populating pool')
        pool_F = []
        pool_G = []
        for i, (A, B) in enumerate(train_ds):
            if i >= args.pool_size // args.batch_size:
                break
            pool_F.append(gen_X(A))
            pool_G.append(gen_Y(B))
        pool_F = TFPool(tf.concat(pool_F, 0), batch_size=args.batch_size)
        pool_G = TFPool(tf.concat(pool_G, 0), batch_size=args.batch_size)
        print(
            f'Done, sample count- F: {pool_F.pool.shape[0]}, G: {pool_G.pool.shape[0]}')

    cycle_gan_model.compile(
        gen_optimizer=keras.optimizers.Adam(
            learning_rate=lr_opts["gen_optimizer"], beta_1=0.5),
        disc_optimizer=keras.optimizers.Adam(
            learning_rate=lr_opts["disc_optimizer"], beta_1=0.5),
        gen_loss_fn=generator_loss_fn, cycle_loss=cycle_loss_fn, id_loss=id_loss_fn,
        disc_loss_fn=discriminator_loss_fn, hvd=horovod if args.use_horovod else None, pool_f=pool_F, pool_g=pool_G)
    print('Model is compiled, setting hooks')
    if is_local_master(args.use_horovod):
        print('Saving initial checkpoint')
        cycle_gan_model.save_weights(os.path.join(
            args.logdir, f'init_checkpoint.{start_epoch:03d}'))
    if args.use_horovod:
        horovod.broadcast_variables(cycle_gan_model.variables, 0)
    print('Start model training')
    cycle_gan_model.fit(
        train_ds,
        epochs=args.epochs,
        initial_epoch=start_epoch,
        steps_per_epoch=args.steps_per_epoch,
        callbacks=hooks,
        verbose=is_master(args.use_horovod),
    )
    if is_local_master(args.use_horovod):
        print('Saving final checkpoint')
        cycle_gan_model.save_weights(os.path.join(
            args.logdir, f'final_checkpoint.{args.epochs:03d}'))


def eval(args, cycle_gan_model, test_ds, input_transformation, checkpoint=None):
    test_horses, test_zebras = test_ds
    # Load the checkpoints
    if not cycle_gan_model._is_compiled:
        cycle_gan_model.load_weights(checkpoint).expect_partial()
        print("Weights loaded successfully")
    test_horses = (
        test_horses.map(input_transformation.preprocess_test_image,
                        num_parallel_calls=autotune)
        .take(20).batch(1)
    )
    test_zebras = (
        test_zebras.map(input_transformation.preprocess_test_image,
                        num_parallel_calls=autotune)
        .take(20).batch(1)
    )
    print('Running horses to zebras')
    test_image_path = os.path.join(
        args.logdir, 'test_images', 'horses_to_zebras')
    os.makedirs(test_image_path, exist_ok=True)
    for i, img in enumerate(test_horses):
        prediction = cycle_gan_model.gen_Y(img, training=False)
        prediction = input_transformation.denormalizer(prediction)
        prediction = keras.preprocessing.image.array_to_img(prediction[0])
        prediction.save(os.path.join(
            test_image_path, f"predicted_img_{i}.png"))
    print('Running zebras to horses')
    test_image_path = os.path.join(
        args.logdir, 'test_images', 'zebras_to_horses')
    os.makedirs(test_image_path, exist_ok=True)
    for i, img in enumerate(test_zebras):
        prediction = cycle_gan_model.gen_X(img, training=False)
        prediction = input_transformation.denormalizer(prediction)
        prediction = keras.preprocessing.image.array_to_img(prediction[0])
        prediction.save(os.path.join(
            test_image_path, f"predicted_img_{i}.png"))


def main():
    parser = CycleGANArgParser(is_demo=False)
    args = parser.parse_args()
    if not args.no_hpu:
        from TensorFlow.common.library_loader import load_habana_module
        load_habana_module()
        args.habana_instance_norm = False
        os.environ['TF_REWRITERS_CONFIG_FILE'] = 'rewriters_config'
        if args.habana_instance_norm:
            tfa.layers.InstanceNormalization = HabanaInstanceNormalization
        if args.data_type == 'bf16':
            tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

    input_image_shape = (args.crop, args.crop, 3)
    input_transformation = TrasformInputs(orig_img_size=(
        args.resize, args.resize), input_img_size=(args.crop, args.crop))

    if args.use_horovod:
        horovod.init()
        if args.log_all_workers:
            args.logdir = os.path.join(args.logdir, f"worker_{horovod.rank()}")

    tfds.disable_progress_bar()
    # Load the horse-zebra dataset using tensorflow-datasets.
    if is_local_master(args.use_horovod):
        dataset, _ = tfds.load("cycle_gan/horse2zebra", data_dir=args.dataset_dir,
                               with_info=True, as_supervised=True, download=True)
        if args.use_horovod:
            horovod.broadcast(0, 0)  # nodes synchronization
    else:
        if args.use_horovod:
            horovod.broadcast(0, 0)
        dataset, _ = tfds.load(
            "cycle_gan/horse2zebra", data_dir=args.dataset_dir, with_info=True, as_supervised=True)

    train_horses, train_zebras = dataset["trainA"], dataset["trainB"]
    test_horses, test_zebras = dataset["testA"], dataset["testB"]

    # Apply the preprocessing operations to the training data
    train_horses = (
        train_horses.map(
            input_transformation.preprocess_train_image, num_parallel_calls=autotune)
        .cache()
        .shuffle(args.buffer)
        .batch(args.batch_size, drop_remainder=True)
    )
    train_zebras = (
        train_zebras.map(
            input_transformation.preprocess_train_image, num_parallel_calls=autotune)
        .cache()
        .shuffle(args.buffer)
        .batch(args.batch_size, drop_remainder=True)
    )
    train_ds = tf.data.Dataset.zip((train_horses, train_zebras))
    test_ds = test_horses, test_zebras

    disc_X = get_discriminator(input_image_shape, name="discriminator_X")
    disc_Y = get_discriminator(input_image_shape, name="discriminator_Y")
    gen_X = get_resnet_generator(input_image_shape, name="generator_X")
    gen_Y = get_resnet_generator(input_image_shape, name="generator_Y")

    # Create cycle gan model
    cycle_gan_model = CycleGan(
        generator_X=gen_X, generator_Y=gen_Y, discriminator_X=disc_X, discriminator_Y=disc_Y
    )

    latest = None
    if args.restore:
        print(f"Trying to restore checkpoint from {args.logdir}")
        latest = tf.train.latest_checkpoint(args.logdir)

    if args.train:
        train(args, cycle_gan_model, train_ds, test_ds, latest)

    if args.test and is_master(args.use_horovod):
        eval(args, cycle_gan_model, test_ds, input_transformation, latest)


if __name__ == '__main__':
    main()
