#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
from loguru import logger

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from yolox.exp import get_exp
from yolox.utils import fuse_model, get_local_rank, get_model_info, setup_logger, configure_nccl


def make_parser():
    parser = argparse.ArgumentParser("YOLOx Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None,
                            help="Experiment name.")
    parser.add_argument( "-f", "--exp-file", default=None, type=str,
                            help="Your expriment description file path.")
    parser.add_argument("-n", "--model-name", type=str, default=None,
                            help="Model name.")
    parser.add_argument("-c", "--ckpt-path", default=None, type=str,
                            help="Checkpoint for eval.")
    parser.add_argument("--data-dir", default=None,
                            help="Custom location of data directory.")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                            help="Modify config options using the command-line.")

    parser.add_argument("-b", "--batch-size", type=int, default=64,
                            help="Batch size.")
    parser.add_argument("-d", "--devices", default=1, type=int,
                            help="Number of the devices for evaluation.")

    parser.add_argument("--legacy", dest="legacy", default=False, action="store_true",
                            help="To be compatible with older versions.")
    parser.add_argument("--conf-threshold", default=None, type=float,
                            help="Test b-box confidence.")
    parser.add_argument("--iou-threshold", default=None, type=float,
                            help="Test NMS (IoU) threshold.")
    parser.add_argument("--per-object-class-metrics", default=False, action="store_true",
                            help="Enables per object class metrics collection.")
    parser.add_argument("--post-processing", dest="post_proc_option", default='device',
                            choices=['off', 'device', 'cpu', 'cpu-async'],
                            help="Post-processing can be computed on the device, or can be offloaded on CPU.")

    parser.add_argument("--test", dest="test", default=False, action="store_true",
                            help="Evaluating on test-dev set.")
    parser.add_argument("--tsize", default=None, type=int,
                            help="Test image size.")

    parser.add_argument("--seed", default=None, type=int,
                            help="Eval seed.")

    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument('--cuda', action='store_true',
                            help='Use CUDA for evaluation')
    device_group.add_argument('--hpu', action='store_true',
                            help='Use Habana HPU for evaluation.')

    mixed_precision_group = parser.add_mutually_exclusive_group()
    mixed_precision_group.add_argument("--autocast", dest='is_autocast', action="store_true",
                            help="Enable autocast")
    mixed_precision_group.add_argument("--fp16", dest="fp16", default=False, action="store_true",
                            help="Adopting mix precision evaluating.")
    parser.add_argument("--disable-mediapipe", dest="disable_mediapipe", action="store_true",
                            help="Disable accelerated media processing with HPU MediaPipe.")

    parser.add_argument( "--fuse", dest="fuse", default=False, action="store_true",
                            help="Fuse convolution and banchnorm for testing.")

    parser.add_argument("--performance-test", dest="is_performance_test_only", default=False, action="store_true",
                            help="Performance test only.")
    parser.add_argument("--data-num-workers", default=4, type=int,
                            help="Number of workers for data processing.")
    parser.add_argument("--warmup-steps", default=4, type=int,
                            help="Number of warmup steps used to compile model. Inference will still run full dataset.")
    parser.add_argument("--repetitions", default=1, type=int,
                            help="Number of evaluation repetitions to run.")
    parser.add_argument("--export-performance-data", dest="csv_file", default=None, type=str,
                            help="Save performance data to .csv file. Append to existing file, or create if it does not exist.")
    return parser


def setup_distributed_hpu():
    from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu

    world_size, global_rank, local_rank = initialize_distributed_hpu()
    print('| distributed init (rank {}) (world_size {})'.format(
        global_rank, world_size), flush=True)

    dist.init_process_group('hccl', rank=global_rank, world_size=world_size)


def set_seed(exp, args):
    import time

    seed = int(time.time())
    seed = exp.seed if exp.seed is None else seed
    seed = args.seed if args.seed is None else seed

    if seed is not None:
        import numpy as np
        import random

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def append_to_csv(csv_file, perf_results):
    import csv

    perf_header = ['#', 'Data type', 'Batch size', 'Total throughput', 'Inference throughput', 'Total images']

    # create file if it does not exist, and add header row
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(perf_header)
            outfile.close()

    # append new row to existing file
    with open(csv_file, 'a', newline='') as outfile:
        writer = csv.writer(outfile)
        perf_results_fmt = []
        # save floats with 2 decimal points
        for i in perf_results:
            if isinstance(i, float):
                perf_results_fmt.append(f'{i:.2f}')
            else:
                perf_results_fmt.append(i)
        writer.writerow(perf_results_fmt)
        outfile.close()


@logger.catch
def main(exp, args):
    set_seed(exp, args)

    is_distributed = False
    device = torch.device("cpu")
    if args.hpu: # Load Habana SW modules
        import habana_frameworks.torch.core as htcore

        device = torch.device("hpu")
        os.environ["PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES"] = "0"
        os.environ["DECODER_MAX_RESIZE"] = "1024"   # must be >= dimensions of resized images (default = 640x640)
        if args.devices > 1:
            setup_distributed_hpu()
            is_distributed = True
    elif args.cuda:
        device = torch.device("cuda")
        if args.devices > 1:
            configure_nccl()
            is_distributed = True

    exp.data_dir = args.data_dir
    exp.data_num_workers = args.data_num_workers

    rank = get_local_rank()

    log_file_name = os.path.join(exp.output_dir, args.experiment_name)
    if rank == 0:
        os.makedirs(log_file_name, exist_ok=True)

    setup_logger(log_file_name, distributed_rank=rank, filename="val_log.txt", mode="a")

    logger.info("Args: {}".format(args))

    if args.conf_threshold is not None:
        exp.test_conf = args.conf_threshold
    if args.iou_threshold is not None:
        exp.nmsthre = args.iou_threshold
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    # model creation
    model = exp.get_model(args.hpu)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    if args.ckpt_path:
        ckpt_file = args.ckpt_path
        logger.info("loading checkpoint from {}".format(ckpt_file))
        ckpt = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.cuda:
        torch.cuda.set_device(rank)
        model.cuda(rank)
    else:
        model.to(device)
    model.eval()

    if args.fuse:
        logger.info("Fusing model...")
        with torch.no_grad():
            model = fuse_model(model)

    if is_distributed:
        logger.info("Distributing model...")
        if args.cuda:
            model = DDP(model, device_ids=[rank])
        else:
            model = DDP(model, broadcast_buffers=False)

    post_processing = args.post_proc_option
    if post_processing == 'device':
        if args.hpu:
            post_processing = 'hpu'
        else:
            post_processing = 'cuda'

    evaluator = exp.get_evaluator(args.batch_size, is_distributed,
                                  args.test, args.legacy,
                                  use_hpu=args.hpu,
                                  post_processing=post_processing,
                                  warmup_steps=args.warmup_steps,
                                  enable_mediapipe=(not args.disable_mediapipe),
                                  )
    evaluator.per_class_AP = args.per_object_class_metrics
    evaluator.per_class_AR = args.per_object_class_metrics

    logger.info("Starting eval...")
    repetitions = args.repetitions
    for repetition_number in range(0, repetitions):
        is_performance_test_only = False
        if args.is_performance_test_only or (repetition_number+1) < repetitions:
            is_performance_test_only = True

        # start evaluate
        if repetitions > 1:
            logger.info(f"Repetition #{(repetition_number + 1):d}.")
        if args.hpu:
            with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=args.is_autocast):
                *_, summary, perf_results = evaluator.evaluate(model, is_distributed,
                                                performance_test_only=is_performance_test_only,
                                                test_size=exp.test_size)
        else:
            *_, summary, perf_results = evaluator.evaluate(model, is_distributed,
                                            performance_test_only=is_performance_test_only,
                                            half=args.fp16,
                                            test_size=exp.test_size)

        if repetitions > 1:
            logger.info(f"Statistics for repetition #{(repetition_number + 1):d}:\n" + summary)
        else:
            logger.info(f"Statistics:\n" + summary)

        # write results to csv file
        if rank == 0 and args.csv_file:
            dt = 'fp32'
            if args.fp16:
                dt = 'fp16'
            elif args.is_autocast:
                dt = 'bf16'
            perf_results = [repetition_number, dt] + perf_results
            append_to_csv(args.csv_file, perf_results)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.model_name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    main(exp, args)
