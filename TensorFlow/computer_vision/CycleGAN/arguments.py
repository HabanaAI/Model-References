###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import argparse
from pathlib import Path
import os


class CycleGANArgParser(argparse.ArgumentParser):
    def __init__(self, is_demo):
        if is_demo:
            description = "demo_cycle_gan.py is a distributed launcher for cycle_gan.py \
                It accepts the same arguments as cycle_gan.py. In case hvd_workers > 1, \
                it runs 'cycle_gan.py [ARGS] --use_horovod' via mpirun with generated HCL config."
        else:
            description = "cycle_gan.py is the main training/evaluation script for CycleGAN. \
                In order to run training on multiple Gaudi cards, use demo_cycle_gan.py or run \
                cycle_gan.py --use_horovod with mpirun."
        super().__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                         description=description)
        self.add_argument('--data_type', '-d', type=str, choices=[
                          'bf16', 'fp32'], default='bf16', help='Data type, possible values: fp32, bf16')
        self.add_argument('--batch_size', '-b', type=int,
                          default=2, help='Batch size')
        self.add_argument('--buffer', type=int, default=256,
                          help='Buffer size for shuffling dataset')
        self.add_argument('--epochs', '-e', type=int,
                          default=200, help='Epochs num')
        self.add_argument('--steps_per_epoch', type=int,
                          default=None, help='Steps per epoch')
        self.add_argument('--logdir', type=str, default='./model_checkpoints/exp_alt_pool_no_tv',
                          help='Path where all logs will be saved')
        self.add_argument('--log_all_workers', dest='log_all_workers',
                          action='store_true', help='If set, every worker will log training data')
        self.set_defaults(log_all_workers=False)
        self.add_argument('--dataset_dir', type=str, default='dataset',
                          help='Path to dataset. If dataset doesn\'t exist, it will be downloaded')
        self.add_argument('--use_hooks', dest='use_hooks',
                          action='store_true', help="Whether to use hooks during training")
        self.set_defaults(use_hooks=False)
        self.add_argument('--no_hpu', dest='no_hpu', action='store_true',
                          help='If set to True HPU won\'t be used')
        self.set_defaults(no_hpu=True if os.getenv(
            'IS_CPU_RUN') == '1' else False)
        self.add_argument('--hvd_workers', type=int, default=1,
                          help="Amount of Horovod workers" if is_demo else argparse.SUPPRESS)  # ignored by cycle_gan.py
        self.add_argument('--use_horovod', help=argparse.SUPPRESS if is_demo else 'Use Horovod for distributed training',
                          action='store_true')  # ignored by demo_cycle_gan.py
        self.add_argument("--hls_type", default="HLS1", type=str,
                          help="Type of HLS" if is_demo else argparse.SUPPRESS)  # ignored by cycle_gan.py
        self.add_argument("--kubernetes_run", default=False, type=bool,
                          help="Kubernetes run" if is_demo else argparse.SUPPRESS)  # ignored by cycle_gan.py
        self.add_argument('--use_tf_instance_norm', dest='habana_instance_norm', action='store_false',
                          help='If set tensorflow implementation of InstanceNormalization will be used')
        self.set_defaults(habana_instance_norm=True)
        self.add_argument('--resize', default=286, type=int,
                          help='Size to which images will be resized')
        self.add_argument('--crop', default=256, type=int,
                          help='Size to which images will be cropped')
        self.add_argument('--restore', dest='restore', action='store_true',
                          help='If set, model will be restored from checkpoint')
        self.add_argument('--no_train', dest='train', action='store_false',
                          help='If set, training will be skipped')
        self.add_argument('--no_test', dest='test',
                          action='store_false', help='If set, test will be skipped')
        self.add_argument('--generator_lr', default=4e-4,
                          type=float, help='Generator learning rate')
        self.add_argument('--discriminator_lr', default=2e-4,
                          type=float, help='Discriminator learning rate')
        self.add_argument('--monitor_freq', default=5, type=int,
                          help='How often transformed test images')
        self.add_argument('--save_freq', default=1, type=int,
                          help='How often save model')
        self.add_argument('--pool_size', default=50, type=int,
                          help='Discriminator images pool size')
        self.add_argument('--cosine_decay_delay', default=100, type=int,
                          help='After how many epoch start decaying learning rates')
