#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import random
import warnings
from loguru import logger

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import time
import numpy as np

import habana_frameworks.torch.core as htcore

from yolox.exp import get_exp
from yolox.utils import fuse_model, get_local_rank, get_model_info, setup_logger


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
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
        "-d", "--devices", default=1, type=int, help="device for training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        default=False,
        action="store_true",
        help="speed test only.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        '--hpu',
        action='store_true',
        help='Use Habana HPU for training'
    )

    mixed_precision_group = parser.add_mutually_exclusive_group()
    mixed_precision_group.add_argument("--autocast", dest='is_autocast', action="store_true", help="Enable autocast")

    parser.add_argument(
        "--data_dir",
        default=None,
        help="custom location of data dir",
    )
    parser.add_argument(
        "--data_num_workers",
        default=4, type=int,
        help="Number of workers for data processing"
    )

    parser.add_argument(
        "--warmup_steps",
        default=2, type=int,
        help="Number of first steps not taken into account in the performance statistic."

    )
    parser.add_argument(
        "--cpu-post-processing", dest="is_postproc_cpu", action="store_true",
        help="Offload post-processing on CPU."
    )

    return parser


def setup_distributed_hpu():
    from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu

    world_size, global_rank, local_rank = initialize_distributed_hpu()
    print('| distributed init (rank {}) (world_size {})'.format(
        global_rank, world_size), flush=True)

    dist.init_process_group('hccl', rank=global_rank, world_size=world_size)


def set_seed(exp, args):
    seed = int(time.time())
    seed = exp.seed if exp.seed is None else seed
    seed = args.seed if args.seed is None else seed

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


@logger.catch
def main(exp, args):
    set_seed(exp, args)

    is_distributed = False
    device=torch.device("cpu")
    if args.hpu: # Load Habana SW modules
        device = torch.device("hpu")
        os.environ["PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES"] = "0"
        if args.devices > 1:
            setup_distributed_hpu()
            is_distributed = True

    exp.data_dir = args.data_dir
    exp.data_num_workers = args.data_num_workers

    rank = get_local_rank()

    log_file_name = os.path.join(exp.output_dir, args.experiment_name)
    if rank == 0:
        os.makedirs(log_file_name, exist_ok=True)

    setup_logger(log_file_name, distributed_rank=rank, filename="val_log.txt", mode="a")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    # model creation
    model = exp.get_model(args.hpu)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    if not args.speed:
        if args.ckpt is None:
            ckpt_file = os.path.join(log_file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint from {}".format(ckpt_file))
        ckpt = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    model.to(device)

    if args.fuse:
        logger.info("\tFusing model...")
        with torch.no_grad():
            model = fuse_model(model)

    if is_distributed:
        logger.info("\tDistributing model...")
        model = DDP(model, broadcast_buffers=False)

    model.eval()

    evaluator = exp.get_evaluator(args.batch_size, is_distributed,
                                  args.test, args.legacy,
                                  use_hpu=args.hpu,
                                  cpu_post_processing=args.is_postproc_cpu,
                                  warmup_steps=args.warmup_steps
                                  )
    evaluator.per_class_AP = True
    evaluator.per_class_AR = True

    # start evaluate
    with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=args.is_autocast):
        *_, summary = exp.eval(
                model, evaluator, is_distributed
            )

    logger.info("\n" + summary)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if args.hpu:
        args.dist_backend = "hccl"

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    main(exp, args)

