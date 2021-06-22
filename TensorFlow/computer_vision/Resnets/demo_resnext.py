#!/usr/bin/env python3

###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import argparse
import os
import subprocess

from TensorFlow.common.common import setup_jemalloc, setup_preloading
from central.training_run_config import TrainingRunHWConfig

def main():
    parser = argparse.ArgumentParser(add_help=False, usage=argparse.SUPPRESS)
    parser.add_argument("--num_workers_per_hls", default=1, type=int)
    parser.add_argument("--hls_type", default="HLS1", type=str)
    parser.add_argument("--kubernetes_run", default=False, type=bool)
    args, unknown_args = parser.parse_known_args()
    script_to_run = str(os.path.abspath(os.path.join(os.path.dirname(__file__), "imagenet_main.py")))

    if '--help' in unknown_args or '-h' in unknown_args:
        print(
        """\ndemo_resnext.py is a distributed launcher for imagenet_main.py.
        \nusage: python demo_resnext.py [arguments]
        \noptional arguments:\n

        -dt <data_type>,   --dtype <data_type>                  Data type, possible values: fp32, bf16. Defaults to fp32
        -dlit <data_type>, --data_loader_image_type <data_type> Data loader images output. Should normally be set to the same data_type as the '--dtype' param
        -bs <batch_size>,  --batch_size <batch_size>            Batch size, defaults to 256
        -rs <size>,        --resnet_size <size>                 The size of the ResNet model to use. Defaults to 50
        -te <epochs>,      --train_epochs <epochs>              Number of training epochs, defaults to 1
        -dd <data_dir>,    --data_dir <data_dir>                Data dir, defaults to `/data/tensorflow_datasets/imagenet/tf_records/`.
                                                                Needs to be specified if the above does not exists.
        -md <model_dir>,   --model_dir <model_dir>              Model dir, defaults to /tmp/resnet
                           --clean                              If set, model_dir will be removed if it exists. Unset by default.
                                                                Important: --clean may return errors in distributed environments. If that happens, try again
        -mts <steps>,      --max_train_steps <steps>            Max train steps
                           --log_steps <steps>                  How often display step status, defaults to 100
        -ebe <epochs>      --epochs_between_evals <epochs>      Number of training epochs between evaluations, defaults to 1.
                           --experimental_preloading            Enables support for 'data.experimental.prefetch_to_device' TensorFlow operator. If set:
                                                                - loads extension dynpatch_prf_remote_call.so (via LD_PRELOAD)
                                                                - sets environment variable HBN_TF_REGISTER_DATASETOPS to 1
                                                                - this feature is experimental and works only with single node
                           --num_workers_per_hls <num_workers>  Number of Horovod workers per node. Defaults to 1.
                                                                In case num_workers_per_hls>1, it runs 'resnet_ctl_imagenet_main.py [ARGS] --use_horovod' via mpirun with generated HCL config.
                           --hls_type <hls_type>                HLS type: either HLS1 (8-cards) or HLS1-H (4-cards). Defaults to HLS1.
                           --kubernetes_run                     Setup kubernetes run for multi HLS training
        \nexamples:\n
        python demo_resnext.py -bs 64 -rs 50 --clean
        python demo_resnext.py -bs 128 -dt bf16 --experimental_preloading -te 90
        python demo_resnext.py -bs 128 -dt bf16 -dlit bf16 --experimental_preloading -te 90 --num_workers_per_hls 8
        \nIn order to see all possible arguments to imagenet_main.py, run "python imagenet_main.py --helpfull"
        """)
        exit(0)

    # libjemalloc for better allocations
    setup_jemalloc()

    if '--experimental_preloading' in unknown_args:
        setup_preloading()

    if args.num_workers_per_hls > 1:
        hw_config = TrainingRunHWConfig(scaleout=True, num_workers_per_hls=args.num_workers_per_hls,
                                        hls_type=args.hls_type, kubernetes_run=args.kubernetes_run,
                                        output_filename="demo_resnext_log")
        cmd = hw_config.mpirun_cmd.split(" ") + ["python3", script_to_run, "--use_horovod"]
    else:
        cmd = ["python3", script_to_run]

    cmd.extend(unknown_args)
    cmd_str = ' '.join(map(str, cmd))
    print(f"Running: {cmd_str}", flush=True)
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()