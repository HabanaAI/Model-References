#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
###########################################################################
# Copyright (C) 2022-2024 Habana Labs, Ltd. an Intel Company
###########################################################################

import argparse
import os
import random
import warnings
from loguru import logger

WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
# if world size is greater than 8 we're using more than one machine
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count() / min(WORLD_SIZE, 8))

import torch
# import torch.backends.cudnn as cudnn
import torch.distributed as dist
import time


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
    mixed_precision_group = parser.add_mutually_exclusive_group()
    mixed_precision_group.add_argument("--autocast", dest='is_autocast', action="store_true", help="Enable autocast")
    parser.add_argument(
        "--data_dir",
        default=None,
        help="custom location of data dir",
    )
    return parser

def setup_distributed_hpu():
    #TBD : get seed from command line
    input_shape_seed = int(time.time())

    from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu

    world_size, global_rank, local_rank = initialize_distributed_hpu()
    print('| distributed init (rank {}) (world_size {})'.format(
        global_rank, world_size), flush=True)

    dist._DEFAULT_FIRST_BUCKET_BYTES = 200*1024*1024  # 200MB
    dist.init_process_group('hccl', rank=global_rank, world_size=world_size)

    random.seed(input_shape_seed)
    # torch.set_num_interop_threads(7)
    # torch.set_num_threads(7)

@logger.catch
def main(exp, args):

    if args.devices>1:
        setup_distributed_hpu()
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
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    hpu_mode = os.environ.get("PT_HPU_LAZY_MODE")

    if hpu_mode is not None and hpu_mode == str(2):
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

    main(exp, args)
