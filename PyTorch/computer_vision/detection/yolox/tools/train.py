#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
###########################################################################
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
###########################################################################

import argparse
import random
import warnings
from loguru import logger

import torch
# import torch.backends.cudnn as cudnn

from yolox.core import Trainer, launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl, configure_omp, get_num_devices
import numpy as np

def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="Input the experiment description file",
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "--cache",
        dest="cache",
        default=False,
        action="store_true",
        help="Caching imgs to RAM for fast training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "-l",
        "--logger",
        type=str,
        help="Logger to be used for metrics",
        default="tensorboard"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    # Add Habana HPU related arguments
    parser.add_argument('--hpu', action='store_true', help='Use Habana HPU for training')
    parser.add_argument('--freeze', action='store_true', help='Freezing the backbone')
    parser.add_argument('--noresize', action='store_true', help='No image resizing')
    parser.add_argument("--use_lazy_mode",
                        default='True', type=lambda x: x.lower() == 'true',
                        help='run model in lazy or eager execution mode, default=True for lazy mode')
    parser.add_argument("--hmp", action="store_true", help="Enable HMP")
    parser.add_argument('--hmp-bf16', default='ops_bf16_yolox.txt', help='path to bf16 ops list in hmp O1 mode')
    parser.add_argument('--hmp-fp32', default='ops_fp32_yolox.txt', help='path to fp32 ops list in hmp O1 mode')
    parser.add_argument('--hmp-verbose', action='store_true', help='enable verbose mode for hmp')
    parser.add_argument(
        "--data_dir",
        default=None,
        help="custom location of data dir",
    )
    return parser


@logger.catch
def main(exp, args):
    exp.data_dir = args.data_dir

    if exp.seed is not None:
        random.seed(exp.seed)
        np.random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )
    if torch.cuda.is_available():
        import torch.backends.cudnn as cudnn
        cudnn.deterministic = True if exp.seed is not None else False
        cudnn.benchmark = True
 
        if args.n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)

        # set environment variables for distributed training
        configure_nccl()
        configure_omp()

    trainer = Trainer(exp, args)
    trainer.train()


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.use_lazy_mode:
        logger.error("YOLOX eager mode is not supported.")
        exit(1)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    if torch.cuda.is_available():
        num_gpu = get_num_devices() if args.devices is None else args.devices
        assert num_gpu <= get_num_devices()
    elif args.hpu:
        num_gpu = 0 if args.devices is None else args.devices
        args.dist_backend = "hccl"
    else:
        num_gpu = 0

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args),
    )
