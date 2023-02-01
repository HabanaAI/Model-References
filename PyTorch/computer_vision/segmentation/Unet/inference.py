###############################################################################
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
import argparse


def main():
    os.environ['framework'] = "PTL"
    from ptl import ptlrun

    parser = argparse.ArgumentParser(description="run benchmarks on Gaudi's devices")
    parser.add_argument("--mode",
                        action='store',
                        type=str,
                        required=True,
                        help='mode - lazy/trace/graphs')
    parser.add_argument("--bs",
                        action='store',
                        type=int,
                        required=True,
                        help='batch size')
    parser.add_argument("--dtype",
                        action='store',
                        type=str,
                        required=False,
                        default='bfloat16',
                        help='model dtype - bfloat16/float32')
    parser.add_argument("--ckpt",
                        action='store',
                        type=str,
                        required=False,
                        default='./pretrained_checkpoint/pretrained_checkpoint.pt',
                        help='path to pre-trained checkpoint')
    parser.add_argument('--accuracy',
                        action='store_true',
                        required=False,
                        help='compute accuracy metrics')
    parser.add_argument('--results',
                        action='store',
                        type=str,
                        required=False,
                        default='/tmp/Unet/results/fold_0/',
                        help='temporary path for results logging')
    parser.add_argument('--data',
                        action='store',
                        type=str,
                        required=False,
                        default='/data/01_2d/',
                        help='path to Unet2d dataset')

    args = parser.parse_args()
    metrics = ptlrun(mode=args.mode,
                     batch_size=args.bs,
                     precision=args.dtype,
                     ckpt_path=args.ckpt,
                     data_path=args.data,
                     res_path=args.results,
                     acc=args.accuracy if args.accuracy else False)
    print(metrics)


if __name__ == "__main__":
    main()
