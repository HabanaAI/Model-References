###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import argparse
import os
import random

import numpy as np
import tensorflow as tf
from TensorFlow.common.debug import dump_callback
from TensorFlow.common.tb_utils import (ExamplesPerSecondKerasHookV2,
                                        TensorBoardWithHParamsV2)
from config import config
from models.models import get_lr_func, get_optimizer
from utils.distribution_utils import configure_cluster, comm_size, comm_rank
from utils.dataset import get_dataset
from vit_keras import vit


def tf_distribute_config(base_tf_server_port: int):
    """
    Generates a TensorFlow cluster information and sets it to TF_CONFIG environment variable.
    TF_CONFIG won't be altered if it was externally set.
    """
    hls_addresses = str(os.environ.get(
        'MULTI_HLS_IPS', '127.0.0.1')).split(',')
    rank = comm_rank()
    size = comm_size()

    worker_hosts = ",".join([",".join([address + ':' + str(base_tf_server_port + r)
                                       for r in range(size//len(hls_addresses))])
                            for address in hls_addresses])

    configure_cluster(worker_hosts, rank)
    print(os.environ['TF_CONFIG'])


DESCRIPTION = 'VisionTransformer training script.'


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--dataset', '--dataset_dir', metavar='PATH',
                        default=config.DEFAULT_DATASET_DIR, help='Dataset directory.')
    parser.add_argument('--optimizer', default='sgd',
                        choices=['sgd', 'adam', 'rmsprop'], help='Optimizer.')
    parser.add_argument('-d', '--dtype', default='fp32',
                        choices=['fp32', 'bf16'], help='Data type.')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='Global batch size.')
    parser.add_argument('--lr_sched', default='WarmupCosine', choices=[
                        'linear', 'exp', 'steps', 'constant', 'WarmupCosine'], help='Learning rate scheduler.')
    parser.add_argument('--initial_lr', type=float,
                        default=6e-2, help='Initial learning rate.')
    parser.add_argument('--final_lr', type=float,
                        default=1e-5, help='Final learning rate.')
    parser.add_argument('--warmup_steps', type=int,
                        default=4000, help='Warmup steps.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Total number of epochs for training.')
    parser.add_argument('--steps_per_epoch', type=int,
                        help='Number of steps for training per epoch, overrides default value.')
    parser.add_argument('--validation_steps', type=int,
                        help='Number of steps for validation, overrides default value.')
    parser.add_argument('--profile', type=str, default='0',
                        help='Profile the batch(es) to sample compute characteristics.'
                        'Must be an integer or a pair of comma-separated integers. For example: --profile 4,6')
    parser.add_argument('--model', default='ViT-B_16',
                        choices=['ViT-B_16', 'ViT-L_16', 'ViT-B_32', 'ViT-L_32'], help='Model.')
    parser.add_argument('--train_subset', default='train/train',
                        help='Pattern to detect train subset in dataset directory.')
    parser.add_argument('--val_subset', default='validation/validation',
                        help='Pattern to detect validation subset in dataset directory.')
    parser.add_argument('--grad_accum_steps', type=int,
                        default=8, help='Gradient accumulation steps.')
    parser.add_argument('--resume_from_checkpoint_path',
                        metavar='PATH', help='Path to checkpoint to start from.')
    parser.add_argument('--resume_from_epoch', metavar='EPOCH_INDEX',
                        type=int, default=0, help='Initial epoch index.')
    parser.add_argument('--evaluate_checkpoint_path', metavar='PATH',
                        help='Checkpoint path for evaluating the model on --val_subset')
    parser.add_argument('--weights_path', metavar='PATH',
                        help='Path to weights cache directory. ~/.keras is used if not set.')
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='Enable deterministic behavior, this will also disable data augmentation. --seed must be set.')
    parser.add_argument('--seed', type=int,
                        help='Seed to be used by random functions.')
    parser.add_argument('--device', default='HPU',
                        choices=['CPU', 'HPU'], help='Device type.')
    parser.add_argument('--distributed', action='store_true',
                        default=False, help='Enable distributed training.')
    parser.add_argument('--base_tf_server_port', type=int,
                        default=7850, help='Rank 0 port used by tf.distribute.')
    parser.add_argument('--save_summary_steps', type=int, default=0,
                        help='Steps between saving summaries to TensorBoard.')
    parser.add_argument('--recipe_cache', default='/tmp/vit_recipe_cache',
                        help='Path to recipe cache directory. Set to empty to disable recipe cache. Externally set \'TF_RECIPE_CACHE_PATH\' will override this setting.')
    parser.add_argument(
        '--dump_config', help='Side-by-side config file. Internal, do not use.')
    args = parser.parse_args()

    if args.weights_path is not None:
        config.WEIGHTS_DIR = args.weights_path

    if args.dtype == 'bf16':
        tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

    if args.device == 'HPU':
        if args.distributed:
            os.environ['TF_HCCL_MEMORY_ALLOWANCE_MB'] = '1000'
        from habana_frameworks.tensorflow import load_habana_module
        load_habana_module()

        # Handle recipe caching.
        recipe_cache = args.recipe_cache
        if 'TF_RECIPE_CACHE_PATH' not in os.environ.keys() and recipe_cache:
            os.environ['TF_RECIPE_CACHE_PATH'] = recipe_cache

        # Clear previous recipe cache.
        if not args.distributed or comm_rank() == 0:
            if os.path.exists(recipe_cache) and os.path.isdir(recipe_cache):
                import shutil
                shutil.rmtree(recipe_cache)
        # Wait for rank 0 to remove cache.
        if args.distributed:
            from mpi4py import MPI
            MPI.COMM_WORLD.Barrier()

        # Create separate log dir directory.
        if args.distributed:
            config.LOG_DIR = os.path.join(
                config.LOG_DIR, f'worker_{comm_rank()}')

    # Handle determinism.
    config.DETERMINISTIC = args.deterministic
    config.SEED = args.seed
    if args.deterministic:
        assert args.seed is not None, "Deterministic behavior require seed to be set."
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        config.DATA_AUGMENTATION = False
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    # Handle distribution strategy.
    if args.distributed:
        tf_distribute_config(args.base_tf_server_port)
        if args.device == 'HPU':
            from habana_frameworks.tensorflow.distribute import HPUStrategy
            strategy = HPUStrategy()
        else:
            strategy = tf.distribute.MultiWorkerMirroredStrategy()
    else:
        strategy = tf.distribute.OneDeviceStrategy(f'device:{args.device}:0')

    if not args.distributed or comm_rank() == 0:
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    num_classes = 1000
    batch_size = args.batch_size
    nb_epoch = args.epochs
    dataset = args.dataset
    resume_from_checkpoint_path = args.resume_from_checkpoint_path
    resume_from_epoch = args.resume_from_epoch
    optim_name = args.optimizer
    initial_lr = args.initial_lr
    final_lr = args.final_lr
    lr_sched = args.lr_sched
    warmup_steps = args.warmup_steps
    model_name = args.model
    grad_accum_steps = args.grad_accum_steps

    ds_train = get_dataset(dataset, args.train_subset, batch_size,
                           is_training=True, distributed=args.distributed)
    ds_valid = get_dataset(dataset, args.val_subset,
                           batch_size, False, distributed=args.distributed)

    if args.dump_config is not None:
        vit.CONFIG_B['dropout'] = 0.0
        vit.CONFIG_L['dropout'] = 0.0

    # Load our model
    with strategy.scope():
        image_size = 384
        if model_name == 'ViT-B_16':
            model = vit.vit_b16(
                image_size=image_size,
                activation='softmax',
                pretrained=True,
                include_top=True,
                pretrained_top=False,
                classes=num_classes,
                weights="imagenet21k")
        elif model_name == 'ViT-L_16':
            model = vit.vit_l16(
                image_size=image_size,
                activation='softmax',
                pretrained=True,
                include_top=True,
                pretrained_top=False,
                classes=num_classes,
                weights="imagenet21k")
        elif model_name == 'ViT-B_32':
            model = vit.vit_b32(
                image_size=image_size,
                activation='softmax',
                pretrained=True,
                include_top=True,
                pretrained_top=False,
                classes=num_classes,
                weights="imagenet21k")
        elif model_name == 'ViT-L_32':
            model = vit.vit_l32(
                image_size=image_size,
                activation='softmax',
                pretrained=True,
                include_top=True,
                pretrained_top=False,
                classes=num_classes,
                weights="imagenet21k")
        else:
            print(
                "Model is not supported, please use either ViT-B_16 or ViT-L_16 or ViT-B_32 or ViT-L_32")
            exit(0)

        optimizer = get_optimizer(
            optim_name, initial_lr, accumulation_steps=grad_accum_steps, epsilon=1e-2)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                      metrics=['accuracy'], run_eagerly=False)

        # Start training

        steps_per_epoch = 1281167 // batch_size
        if args.steps_per_epoch is not None:
            steps_per_epoch = args.steps_per_epoch
        validation_steps = 50000 // batch_size
        if args.validation_steps is not None:
            validation_steps = args.validation_steps

        total_steps = nb_epoch * steps_per_epoch
        resume_step = resume_from_epoch * steps_per_epoch

        lrate = get_lr_func(nb_epoch, lr_sched, initial_lr,
                            final_lr, warmup_steps, resume_step, total_steps)

        save_name = model_name if not model_name.endswith('.h5') else \
            os.path.split(model_name)[-1].split('.')[0].split('-')[0]
        model_ckpt = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(config.SAVE_DIR, save_name) + '-ckpt-{epoch:03d}.h5',
            monitor='train_loss')

        callbacks = [lrate, model_ckpt]

        profile_batch = 0
        if not args.distributed or comm_rank() == 0:
            profile_batch = tuple(int(i) for i in args.profile.split(','))
            if len(profile_batch) == 1:
                profile_batch = profile_batch[0]
        callbacks += [TensorBoardWithHParamsV2(
            vars(args), log_dir=config.LOG_DIR, update_freq=args.save_summary_steps, profile_batch=profile_batch)]

        if args.save_summary_steps > 0:
            callbacks += [ExamplesPerSecondKerasHookV2(
                output_dir=config.LOG_DIR, every_n_steps=args.save_summary_steps, batch_size=args.batch_size)]

        if (args.evaluate_checkpoint_path is not None):
            model.load_weights(args.evaluate_checkpoint_path)
            results = model.evaluate(x=ds_valid, steps=validation_steps)
            print("Test loss, Test acc:", results)
            exit()

        if ((resume_from_epoch is not None) and (resume_from_checkpoint_path is not None)):
            model.load_weights(resume_from_checkpoint_path)

        with dump_callback(args.dump_config):
            model.fit(x=ds_train, y=None,
                      steps_per_epoch=steps_per_epoch,
                      callbacks=callbacks,
                      initial_epoch=resume_from_epoch,
                      epochs=nb_epoch,
                      shuffle=not args.deterministic,
                      verbose=1 if not args.distributed else comm_rank() == 0,
                      validation_data=(ds_valid, None),
                      validation_steps=validation_steps,
                      )

        if not args.distributed or comm_rank() == 0:
            model.save(f'{config.SAVE_DIR}/{save_name}-model-final.h5')


if __name__ == '__main__':
    main()
