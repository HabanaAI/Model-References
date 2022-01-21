###############################################################################
# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import argparse
import os


class SSDArgParser(argparse.ArgumentParser):
    def __init__(self, is_demo):
        if is_demo:
            description = "demo_ssd.py is a distributed launcher for ssd.py \
                It accepts the same arguments as ssd.py. In case hvd_workers > 1, \
                it runs 'ssd.py [ARGS] --use_horovod' via mpirun with generated HCL config."
        else:
            description = "ssd.py is the main training/evaluation script for SSD_ResNet34. \
                In order to run training on multiple Gaudi cards, use demo_ssd.py or run \
                ssd.py --use_horovod with mpirun."

        super().__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                         description=description)

        self.add_argument('-d', '--dtype', metavar='bf16/fp32',
                          help='Data type: fp32 or bf16', type=str, choices=['fp32', 'bf16'], default='bf16')
        self.add_argument('-b', '--batch_size', metavar='N',
                          default=128, help='Batch size', type=int)
        self.add_argument('-e', '--epochs', metavar='N', default=50,
                          help='Number of epochs for training', type=float)
        self.add_argument('--mode', metavar='train/eval', help='\'train\' or \'eval\'',
                          type=str, choices=['train', 'eval'], default='train')
        self.add_argument("--hvd_workers", default=1, type=int,
                          help="Amount of Horovod workers" if is_demo else argparse.SUPPRESS) # ignored by ssd.py
        self.add_argument("--num_workers_per_hls", default=8, type=int,
                          help="Num workers per HLS" if is_demo else argparse.SUPPRESS) # ignored by ssd.py
        self.add_argument("--kubernetes_run", default=False, type=bool,
                          help="Kubernetes run" if is_demo else argparse.SUPPRESS) # ignored by ssd.py
        self.add_argument(
            '--use_horovod', help=argparse.SUPPRESS if is_demo else 'Use Horovod for distributed training', action='store_true') # ignored by demo_ssd.py
        self.add_argument('--inference', metavar='IMAGE', type=str,
                          help='path to image for inference (if set then mode is ignored)')
        self.add_argument(
            '--no_hpu', help='Do not load Habana modules = train on CPU/GPU', action='store_true')
        self.add_argument(
            '--distributed_bn', help='Use distributed batch norm', action='store_true')
        self.add_argument('--vis_dataloader',
                          help='Visualize dataloader', action='store_true')
        self.add_argument(
            '--profiling', help='turn on profiling (generate json every 20 steps)', action='store_true')
        self.add_argument('--use_cocoeval_cc',
                          help='use_cocoeval_cc', action='store_true')
        self.add_argument(
            '-f', '--use_fake_data', help='Use fake data to reduce the input preprocessing overhead (for unit tests)', action='store_true')
        self.add_argument('--lr_warmup_epoch', default=5.0, metavar='N',
                          help='numer of epochs for learning rate warmup', type=float)
        self.add_argument('--base_lr', default=3e-3,
                          metavar='BASE_LR', help='base learning rate', type=float)
        self.add_argument('--weight_decay', default=5e-4,
                          metavar='WD', help='L2 wight decay', type=float)
        self.add_argument('--k', default=0, help='k is an integer defining at which epochs the learning rate decays: '
                          '[40, 50] * (1 + k/10)', type=int)
        self.add_argument('--model_dir', metavar='<dir>', default='/tmp/ssd',
                          help='Location of model_dir', type=str)
        self.add_argument('--resnet_checkpoint', metavar='<PATH>', default='/data/ssd_r34-mlperf',
                          help='Location of the ResNet ckpt to use for model '
                          'init.', type=str)
        self.add_argument('--data_dir', metavar='<PATH>', default='/data/coco2017/ssd_tf_records',
                          help='Path to dataset', type=str)
        self.add_argument('--eval_samples', default=5000, metavar='N',
                          help='number of samples for evaluation.', type=int)
        self.add_argument('--num_examples_per_epoch', default=117266,
                          metavar='N', help='Number of examples in one epoch', type=int)
        self.add_argument('-s', '--steps', default=0,
                          help='Number of training steps (epochs and num_examples_per_epoch are ignored when set)', type=int)
        self.add_argument('-v', '--log_step_count_steps', default=1, metavar='STEPS',
                          help='How often print global_step/sec and loss', type=int)
        self.add_argument('-c', '--save_checkpoints_epochs', default=5.0,
                          metavar='EPOCHS', help='How often save checkpoints', type=float)
        self.add_argument('--keep_ckpt_max', default=20, metavar='N',
                          help='Maximum number of checkpoints to keep', type=int)
        self.add_argument('--save_summary_steps', default=1,
                          help='How often save summary', type=int)
        self.add_argument('--static', default=False, action='store_true',
                          help='Enables use of static dataloader')
        self.add_argument('--recipe_cache', default='/tmp/ssd_recipe_cache/',
                          help='Path to recipe cache directory. Set to empty to disable recipe cache. '
                               'Externally set "TF_RECIPE_CACHE_PATH" will override this settings.')
