#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
###############################################################################
import os
import subprocess
import sys



from keras_segmentation.cli_interface import get_arg_parser



parser = get_arg_parser(True)
args = parser.parse_args()
if args.train_engine == 'hpu':
    from TensorFlow.common.common import setup_jemalloc
    setup_jemalloc()

cmd = ["python3", "-m", "keras_segmentation"]
if args.num_workers_per_hls > 1:
    assert args.distributed
    assert args.train_engine != 'cpu'
    if args.train_engine == 'hpu':
        from central.training_run_config import TrainingRunHWConfig
        hw_config = TrainingRunHWConfig(
            scaleout=True,
            num_workers_per_hls=args.num_workers_per_hls,
            hls_type=args.hls_type,
            kubernetes_run=args.kubernetes_run,
            output_filename="demo_segnet"
        )
        cmd = hw_config.mpirun_cmd.split(" ") + cmd
    elif args.train_engine == 'gpu':
        cmd = ["horovodrun", "-np", "args.num_workers_per_hls"] + cmd
cmd += sys.argv[1:]
print(f"Running: {' '.join(map(str, cmd))}")
subprocess.run(cmd)

# single
# python demo_segnet.py train --train_images=dataset1/images_prepped_train/ --train_annotations=dataset1/annotations_prepped_train/ --n_classes=12 --model_name=vgg_segnet --val_images=dataset1/images_prepped_test/ --val_annotations=dataset1/annotations_prepped_test/ --epochs=5
# multi: add following to single command above
# --distributed --num_workers_per_hls=4
