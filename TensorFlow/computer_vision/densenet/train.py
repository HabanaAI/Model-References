###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################
# List of changes:
# - Added evaluation from saved chekpoint via --evaluate_checkpoint_path flag
# - Added resuming from checkpoint via --resume_from_checkpoint_path and --resume_from_epoch  flags
# - Added seed via --seed flag
# - Added densent general configuration to support densent121, densenet161, densenet169 via --model flag
# - Added warmup epochs via --warmup_epochs flag
# - Added support for HPU training with --run_on_hpu flag
# - Added HPUStrategy() distributed training with --use_hpu_strategy
# - Added steps-based training duration specification via --steps_per_epoch and --validation_steps flags
# - Added StepLearningRateScheduleWithWarmup learning rate scheduler for warmup
# - Added deterministic mode

import tensorflow as tf
from tensorflow.keras.optimizers import SGD

from utils.dataset import get_dataset
from utils.arguments import DenseNetArgumentParser
from models.models import StepLearningRateScheduleWithWarmup
from models.models import get_optimizer
from config import config
from densenet import densenet_model

from TensorFlow.common.debug import dump_callback
from TensorFlow.common.library_loader import load_habana_module
from TensorFlow.common.multinode_helpers import comm_size, comm_rank
from TensorFlow.utils.misc import distribution_utils
from habana_frameworks.tensorflow.distribute import HPUStrategy
from TensorFlow.common.tb_utils import (
    TensorBoardWithHParamsV2, ExamplesPerSecondKerasHookV2)

import os
import random
import numpy as np

def set_deterministic():
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    np.random.seed(0)
    random.seed(0)
    tf.random.set_seed(0)

def main():
    parser = DenseNetArgumentParser(
        description=(
            "train.py is the main training/evaluation script for DenseNet. "
            "In order to run training on multiple Gaudi cards, use demo_densenet.py or run "
            "train.py with mpirun."))
    args, _ = parser.parse_known_args()

    strategy = None
    verbose = 1

    os.environ['RUN_TPC_FUSER'] = 'false'

    if args.deterministic:
        if args.inputs is None:
            raise ValueError("Must provide inputs for deterministic mode")
        if args.resume_from_checkpoint_path is None:
            raise ValueError("Must provide checkpoint for deterministic mode")

    if args.dtype == 'bf16':
        os.environ['TF_BF16_CONVERSION'] = '1'

    if args.run_on_hpu:
        log_info_devices = load_habana_module()
        print(f"Devices:\n {log_info_devices}")
        if args.use_hpu_strategy:
            hls_addresses = str(os.environ.get(
                "MULTI_HLS_IPS", "127.0.0.1")).split(",")
            TF_BASE_PORT = 2410
            mpi_rank = comm_rank()
            mpi_size = comm_size()
            if mpi_rank > 0:
                verbose = 0
            worker_hosts = ""
            for address in hls_addresses:
                # worker_hosts: comma-separated list of worker ip:port pairs.
                worker_hosts = worker_hosts + ",".join(
                    [address + ':' + str(TF_BASE_PORT + rank)
                     for rank in range(mpi_size//len(hls_addresses))])
            task_index = mpi_rank

            # Configures cluster spec for distribution strategy.
            _ = distribution_utils.configure_cluster(worker_hosts, task_index)
            strategy = HPUStrategy()
            print('Number of devices: {}'.format(
                strategy.num_replicas_in_sync))
    else:
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    if args.seed is not None:
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    img_rows, img_cols = 224, 224  # Resolution of inputs
    channel = 3
    num_classes = 1000
    batch_size = args.batch_size
    nb_epoch = args.epochs
    dataset_dir = args.dataset_dir
    resume_from_checkpoint_path = args.resume_from_checkpoint_path
    resume_from_epoch = args.resume_from_epoch
    dropout_rate = args.dropout_rate
    weight_decay = args.weight_decay
    optim_name = args.optimizer
    initial_lr = args.initial_lr
    model_name = args.model
    save_summary_steps = args.save_summary_steps

    if model_name == "densenet121":
        growth_rate = 32
        nb_filter = 64
        nb_layers = [6, 12, 24, 16]

    elif model_name == "densenet161":
        growth_rate = 48
        nb_filter = 96
        nb_layers = [6, 12, 36, 24]

    elif model_name == "densenet169":
        growth_rate = 32
        nb_filter = 64
        nb_layers = [6, 12, 32, 32]

    else:
        print("model is not supported")
        exit(1)

    # Load our model
    if strategy:
        with strategy.scope():
            model = densenet_model(img_rows=img_rows, img_cols=img_cols, color_type=channel,
                                   dropout_rate=dropout_rate, weight_decay=weight_decay, num_classes=num_classes,
                                   growth_rate=growth_rate, nb_filter=nb_filter, nb_layers=nb_layers)
            optimizer = get_optimizer(
                model_name, optim_name, initial_lr, epsilon=1e-2)
            model.compile(optimizer=optimizer,
                          loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        model = densenet_model(img_rows=img_rows, img_cols=img_cols, color_type=channel,
                               dropout_rate=dropout_rate, weight_decay=weight_decay, num_classes=num_classes,
                               growth_rate=growth_rate, nb_filter=nb_filter, nb_layers=nb_layers)
        optimizer = get_optimizer(
            model_name, optim_name, initial_lr, epsilon=1e-2)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy', metrics=['accuracy'])

    # Start training
    steps_per_epoch = 1281167 // batch_size
    if args.steps_per_epoch is not None:
        steps_per_epoch = args.steps_per_epoch
    validation_steps = 50000 // batch_size
    if args.validation_steps is not None:
        validation_steps = args.validation_steps
    warmup_steps = args.warmup_epochs * steps_per_epoch
    rescaled_lr = initial_lr * batch_size / 256
    lr_sched = {0: 1, 30: 0.1, 60: 0.01, 80: 0.001}
    lr_sched_steps = {
        epoch * steps_per_epoch: multiplier for (epoch, multiplier) in lr_sched.items()}

    lrate = StepLearningRateScheduleWithWarmup(initial_lr=rescaled_lr,
                                               initial_global_step=0,
                                               warmup_steps=warmup_steps,
                                               decay_schedule=lr_sched_steps,
                                               verbose=0)

    save_name = model_name if not model_name.endswith('.h5') else \
        os.path.split(model_name)[-1].split('.')[0].split('-')[0]

    model_ckpt = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(args.model_dir, config.SAVE_DIR,
                     save_name) + '-ckpt-{epoch:03d}.h5',
        monitor='train_loss')

    callbacks = [lrate, model_ckpt]

    if save_summary_steps is not None and save_summary_steps > 0:
        log_dir = os.path.join(args.model_dir, config.LOG_DIR)
        local_batch_size = batch_size
        
        if args.use_hpu_strategy:
            log_dir = os.path.join(log_dir, 'worker_' + str(comm_rank()))
            local_batch_size = batch_size // strategy.num_replicas_in_sync

        callbacks += [
            TensorBoardWithHParamsV2(
                args.__dict__, log_dir=log_dir,
                update_freq=save_summary_steps, profile_batch=0),
            ExamplesPerSecondKerasHookV2(
                save_summary_steps, output_dir=log_dir,
                batch_size=local_batch_size),
        ]

    if (args.evaluate_checkpoint_path is not None):
        model.load_weights(args.evaluate_checkpoint_path)
        results = model.evaluate(x=ds_valid, steps=validation_steps)
        print("Test loss, Test acc:", results)
        exit()

    if ((resume_from_epoch is not None) and (resume_from_checkpoint_path is not None)):
        model.load_weights(resume_from_checkpoint_path)

    if args.deterministic:
        set_deterministic()
        if not os.path.isfile(args.dump_config):
            raise FileNotFoundError("wrong dump config path")

        import pickle
        x_path = os.path.join(args.inputs, "input")
        y_path = os.path.join(args.inputs, "target")
        x = pickle.load(open(x_path, 'rb'))
        y = pickle.load(open(y_path, 'rb'))

        with dump_callback(args.dump_config):
          model.fit(x=x, y=y,
                  steps_per_epoch=steps_per_epoch,
                  callbacks=callbacks,
                  initial_epoch=resume_from_epoch,
                  epochs=nb_epoch,
                  shuffle=False,
                  verbose=verbose,
                  validation_data=None,
                  validation_steps=0,
                  )
    else:
      ds_train = get_dataset(dataset_dir, args.train_subset, batch_size)
      ds_valid = get_dataset(dataset_dir, args.val_subset, batch_size)

      model.fit(x=ds_train, y=None,
                steps_per_epoch=steps_per_epoch,
                callbacks=callbacks,
                initial_epoch=resume_from_epoch,
                epochs=nb_epoch,
                shuffle=True,
                verbose=verbose,
                validation_data=(ds_valid, None),
                validation_steps=validation_steps,
                validation_freq=1,
                )

if __name__ == '__main__':
    main()
