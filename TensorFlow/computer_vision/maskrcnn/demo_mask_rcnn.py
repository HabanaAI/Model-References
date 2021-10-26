#! /usr/bin/python3
###############################################################################
# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import argparse
import os
from pathlib import Path
import shutil
from central.generate_hcl_config import generate_hcl_config_r as generate_hcl_config

DATASET_LOCATIONS = [
        '/data/coco2017/tf_records',
        '/software/data/tf/coco2017/tf_records'
]

CHECKPOINT_LOCATIONS = [
        "weights/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603",
        "/software/data/resnet-model_ckp/model.ckpt-112603"
]

ANNOTATIONS_LOCATIONS = [
        "raw_data/annotations/instances_val2017.json",
        "../annotations/instances_val2017.json",
        "annotations/instances_val2017.json"
]

DEFAULT_MODEL_DIR='results/'
DEFAULT_SEED=12345

def find_first_existing(locations, prefix = '', suffix = ''):
    for l in locations:
        if os.path.exists(prefix + l + suffix):
            return l
    return None

def find_default_coco_dataset():
    return find_first_existing(DATASET_LOCATIONS)

def find_default_checkpoint():
    return find_first_existing(CHECKPOINT_LOCATIONS, suffix = '.index')

def find_default_annotations(dataset_dir):
    return find_first_existing(ANNOTATIONS_LOCATIONS, prefix = dataset_dir + "/")

def get_script_path():
    return os.path.dirname(os.path.realpath(__file__))

def create_command_str(args):
    def create_mpi_cmd(num_workers, map_by_socket):
        mpi_cmd_parts = []
        mpi_cmd_parts.append("mpirun")
        mpi_cmd_parts.append("--allow-run-as-root")
        if map_by_socket:
            mpi_pe = get_mpi_pe(num_workers)
            mpi_cmd_parts.append("--bind-to core --map-by socket:PE={}".format(mpi_pe))
        mpi_cmd_parts.append("--np {}".format(num_workers))
        return ' '.join(mpi_cmd_parts)

    def get_mpi_pe(num_workers):
        cpus = os.cpu_count()
        return cpus//num_workers//2

    cmd_parts = []

    assert args.dataset is not None, "Couldn't find default dataset!"
    assert args.checkpoint is not None, "Couldn't find default checkpoint!"

    val_json_file = find_default_annotations(args.dataset)
    assert val_json_file is not None, "Couldn't find validation json file!"

    # Normalize paths.
    args.dataset = os.path.normpath(args.dataset)
    args.checkpoint = os.path.normpath(args.checkpoint)
    val_json_file = os.path.normpath(val_json_file)
    args.model_dir = os.path.normpath(args.model_dir)

    # bfloat16 handling.
    bf16_config_directory = os.fspath(Path(os.path.realpath(__file__)).parents[2].joinpath("common/bf16_config"))
    bf16_flag = 'TF_BF16_CONVERSION'
    if args.dtype == 'fp32':
        cmd_parts.append(f'{bf16_flag}=0')
    elif args.dtype == 'bf16-basic':
        cmd_parts.append(f'{bf16_flag}={bf16_config_directory}/basic.json')
    elif args.dtype == 'bf16':
        cmd_parts.append(f'{bf16_flag}={bf16_config_directory}/full.json')
    else:
        raise Exception("data_type can only be \'bf16\', \'bf16-basic\' or \'fp32\'")

    num_workers = args.hvd_workers

    init_learning_rate = 0.005
    learning_rate_steps = [240000,320000]
    num_steps_per_eval = 29568
    total_steps = 360000

    if num_workers > 1:
        generate_hcl_config(get_script_path(), num_workers)
        cmd_parts.append(create_mpi_cmd(num_workers, args.map_by_socket))
        init_learning_rate *= num_workers
        learning_rate_steps = [int(x/num_workers) for x in learning_rate_steps]
        num_steps_per_eval = int(num_steps_per_eval/num_workers)
        total_steps = int(total_steps/num_workers)

    learning_rate_steps = ",".join(str(x) for x in learning_rate_steps)

    if args.total_steps is not None:
        total_steps = args.total_steps
    if args.num_steps_per_eval is not None:
        num_steps_per_eval = args.num_steps_per_eval
    if args.init_learning_rate is not None:
        init_learning_rate = args.init_learning_rate
    if args.learning_rate_steps is not None:
        learning_rate_steps = args.learning_rate_steps

    main_script = os.path.join(get_script_path(), 'mask_rcnn_main.py')
    cmd_parts.append(f'python3 {main_script}')
    cmd_parts.append(f'--mode={args.command}')
    cmd_parts.append(f'--checkpoint="{args.checkpoint}"')
    cmd_parts.append(f'--eval_samples={args.eval_samples}')
    cmd_parts.append(f'--init_learning_rate={init_learning_rate}')
    cmd_parts.append(f'--learning_rate_steps={learning_rate_steps}')
    cmd_parts.append(f'--model_dir="{args.model_dir}"')
    cmd_parts.append(f'--num_steps_per_eval={num_steps_per_eval}')
    cmd_parts.append(f'--total_steps={total_steps}')
    if args.train_batch_size is not None:
        cmd_parts.append(f'--train_batch_size={args.train_batch_size}')
    if args.eval_batch_size is not None:
        cmd_parts.append(f'--eval_batch_size={args.eval_batch_size}')
    cmd_parts.append(f'--training_file_pattern="{args.dataset}/train-*.tfrecord"')
    cmd_parts.append(f'--validation_file_pattern="{args.dataset}/val-*.tfrecord"')
    cmd_parts.append(f'--val_json_file="{args.dataset}/{val_json_file}"')
    if args.pyramid_roi_impl is not None:
        cmd_parts.append(f'--pyramid_roi_impl={args.pyramid_roi_impl}')
    if args.save_summary_steps:
        cmd_parts.append(f'--save_summary_steps={args.save_summary_steps}')
    if args.no_eval_after_training:
        cmd_parts.append(f'--noeval_after_training')
    if args.use_fake_data:
        cmd_parts.append(f'--use_fake_data')
    if args.deterministic:
        cmd_parts.append('--deterministic')
        cmd_parts.append(f'--seed={DEFAULT_SEED}')
    if args.device is not None:
        cmd_parts.append(f'--device={args.device}')
    if args.profile:
        cmd_parts.append('--profile')
    if args.dump_config is not None:
        cmd_parts.append(f'--dump_config={args.dump_config}')
    if args.recipe_cache is not None:
        cmd_parts.append(f'--recipe_cache={args.recipe_cache}')

    return ' '.join(cmd_parts)

def prepare_environment(args):
    if args.clean_model_dir:
        print(f"Removing old model in {args.model_dir}")
        shutil.rmtree(args.model_dir, ignore_errors=True)
        os.makedirs(args.model_dir, exist_ok=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Runs MaskRCNN demo.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('command', nargs='?', metavar='<command>', help='\'train\', \'train_and_eval\' or \'eval\' on MS COCO',
                        choices=['train', 'train_and_eval', 'eval'], default='train_and_eval')
    parser.add_argument('--dataset', metavar='<dataset>', help='Dataset directory', default=find_default_coco_dataset())
    parser.add_argument('--checkpoint', metavar='<checkpoint>', help='Model checkpoint', default=find_default_checkpoint())
    parser.add_argument('--model_dir', metavar='<model_dir>', help='Model directory', default=DEFAULT_MODEL_DIR)
    parser.add_argument('-s', '--total_steps', metavar='<total_steps>',
                        help='The number of steps to use for training. This flag'
                                ' should be adjusted according to the --train_batch_size flag.', type=int)
    parser.add_argument('-d', '--dtype', metavar='<data_type>',
                        help='Data type: fp32, bf16 or bf16-basic', choices=['fp32', 'bf16', 'bf16-basic'], default='bf16')
    parser.add_argument('--hvd_workers', metavar='<hvd_workers>', help='Number of Horovod workers, default 1 - Horovod disabled',
                        type=int, default=1)
    parser.add_argument('-b', '--train_batch_size', '--bs', metavar='<train_batch_size>', help='Batch size for training.', type=int)
    parser.add_argument('-t', '--no_eval_after_training', help='Disable evaluation step after training.', action='store_true', default=False)
    parser.add_argument('-c', '--clean_model_dir', help='Clean model directory before execution', action='store_true', default=False)
    parser.add_argument('-f', '--use_fake_data', help='Use fake input', action='store_true', default=False)
    parser.add_argument('--pyramid_roi_impl', metavar='<impl_name>', help='Implementation to use for PyramidRoiAlign.',
                        choices=['habana', 'habana_fp32', 'gather'])
    parser.add_argument('--eval_samples', metavar='<samples>', help='Number of eval samples. Number of steps will be divided by "eval_batch_size".',
                        type=int, default=5000)
    parser.add_argument('--eval_batch_size', metavar='<eval_batch_size>', help='Batch size for evaluation.', type=int)
    parser.add_argument('--num_steps_per_eval', metavar='<num_steps_per_eval>', help='Number of steps used for evaluation.', type=int)
    parser.add_argument('--deterministic', help='Enable deterministic behavior', action='store_true', default=False)
    parser.add_argument('--save_summary_steps', type=int, metavar='<summary_steps>',
                        help='Steps between saving summaries to TensorBoard.')
    parser.add_argument('--map_by_socket', help='MPI maps processes to sockets.', action='store_true', default=False)
    parser.add_argument('--device', metavar='<device>', help='Device type.', choices=['CPU', 'HPU'])
    parser.add_argument('--profile', help='Gather TensorBoard profiling data.', action='store_true', default=False)
    parser.add_argument('--dump_config', metavar='<path_to_config>', help='Side-by-side config file. Internal, do not use.')
    parser.add_argument('--init_learning_rate', metavar='<init_learning_rate>', help='Initial learning rate.', type=float)
    parser.add_argument('--learning_rate_steps', metavar='<learning_rate_steps>', help='Warmup learning rate decay factor. Expected format: "first_value,second_value".')
    parser.add_argument('--recipe_cache', metavar='<recipe_cache_path>',
                        help='Path to recipe cache directory. Set to empty to disable recipe cache. Externally set \'TF_RECIPE_CACHE_PATH\' will override this setting.'
                             ' Default: \'/tmp/maskrcnn_recipe_cache/\'.')


    args = parser.parse_args()

    command_str = create_command_str(args)
    prepare_environment(args)
    print(command_str)
    os.system(command_str)
