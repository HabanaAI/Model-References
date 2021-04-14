#! /usr/bin/python3

import argparse
import os
import pathlib
import shutil
import subprocess

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
    def find_common_path():
        current_path = pathlib.Path().absolute()
        root_dir = pathlib.Path()
        for p in current_path.parents:
            for d in p.iterdir():
                if d.match('TensorFlow'):
                    root_dir = p
                    break
        return root_dir.joinpath('TensorFlow', 'common').resolve()

    def create_mpi_cmd(num_workers, map_by_socket):
        mpi_cmd_parts = []
        mpi_cmd_parts.append("mpirun")
        mpi_cmd_parts.append("--allow-run-as-root")
        if map_by_socket:
            mpi_pe = get_mpi_pe(num_workers)
            mpi_cmd_parts.append("--bind-to core --map-by socket:PE={}".format(mpi_pe))
        mpi_cmd_parts.append("--np {}".format(num_workers))
        return ' '.join(mpi_cmd_parts)

    def generate_hcl_config(num_workers):
        common_path = find_common_path()
        common_sh = common_path.joinpath('common.sh')
        script_path = get_script_path()
        subprocess.Popen(['bash', '-c', 'source {}; generate_hcl_config {} {}'.format(common_sh, script_path, num_workers)])
        config_file_name = "config_{}.json".format(num_workers)
        os.environ['HCL_CONFIG_PATH'] = "{}/{}".format(script_path, config_file_name)
        os.environ['HABANA_USE_PREALLOC_BUFFER_FOR_ALLREDUCE'] = 'false'

    def get_mpi_pe(num_workers):
        cpus = os.cpu_count()
        return cpus//num_workers//2

    cmd_parts = []

    # Add path for 'library_loader'.
    common_path_env = f'PYTHONPATH=$PYTHONPATH:{find_common_path()}'
    cmd_parts.append(common_path_env)

    assert args.dataset is not None, "Couldn't find default dataset!"
    assert args.checkpoint is not None, "Couldn't find default checkpoint!"

    val_json_file = find_default_annotations(args.dataset)
    assert val_json_file is not None, "Couldn't find validation json file!"

    # bfloat16 handling.
    bf16_flag = 'TF_ENABLE_BF16_CONVERSION'
    if args.dtype == 'fp32':
        cmd_parts.append(f'{bf16_flag}=0')
    elif args.dtype == 'bf16':
        cmd_parts.append(f'{bf16_flag}=1')

    # TODO SW-38086: enable TPC fuser
    cmd_parts.append(f'RUN_TPC_FUSER=false')

    num_workers = args.hvd_workers

    init_learning_rate = 0.005
    learning_rate_steps = [240000,320000]
    num_steps_per_eval = 29568
    total_steps = 360000

    if num_workers > 1:
        generate_hcl_config(num_workers)
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
                        type=str, choices=['train', 'train_and_eval', 'eval'], default='train_and_eval')
    parser.add_argument('--dataset', metavar='<dataset>', help='Dataset directory', type=str, default=find_default_coco_dataset())
    parser.add_argument('--checkpoint', metavar='<checkpoint>', help='Model checkpoint', type=str, default=find_default_checkpoint())
    parser.add_argument('--model_dir', metavar='<model_dir>', help='Model directory', type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument('-s', '--total_steps', metavar='<total_steps>',
                        help='The number of steps to use for training. This flag'
                                ' should be adjusted according to the --train_batch_size flag.', type=int, default=None)
    parser.add_argument('-d', '--dtype', metavar='<data_type>',
                        help='Data type: fp32 or bf16', type=str, choices=['fp32', 'bf16'], default='bf16')
    parser.add_argument('--hvd_workers', metavar='<hvd_workers>', help='Number of Horovod workers, default 1 - Horovod disabled',
                        type=int, default=1)
    parser.add_argument('-b', '--train_batch_size', '--bs', metavar='<train_batch_size>', help='Batch size for training.',
                        type=int, default=None)
    parser.add_argument('-t', '--no_eval_after_training', help='Disable evaluation step after training.', action='store_true', default=False)
    parser.add_argument('-c', '--clean_model_dir', help='Clean model directory before execution', action='store_true', default=False)
    parser.add_argument('-f', '--use_fake_data', help='Use fake input', action='store_true', default=False)
    parser.add_argument('--pyramid_roi_impl', metavar='<impl_name>', help='Implementation to use for PyramidRoiAlign.',
                        type=str, choices=['habana', 'habana_bf16', 'gather'], default=None)
    parser.add_argument('--eval_samples', metavar='<samples>', help='Number of eval samples. Number of steps will be divided by "eval_batch_size".',
                        type=int, default=5000)
    parser.add_argument('--eval_batch_size', metavar='<eval_batch_size>', help='Batch size for evaluation.',
                        type=int, default=None)
    parser.add_argument('--num_steps_per_eval', metavar='<num_steps_per_eval>', help='Num of steps used for evaluation.',
                        type=int, default=None)
    parser.add_argument('--save_summary_steps', default=None, type=int, metavar='<summary_steps>',
                        help='Steps between saving summaries to TensorBoard.')
    parser.add_argument('--map_by_socket', help='MPI maps processes to sockets', action='store_true', default=False)
    parser.add_argument('--disable_recovery', help='Disable application recovery on crash.', action='store_true')

    args = parser.parse_args()

    command_str = create_command_str(args)
    prepare_environment(args)
    print(command_str)

    # Command invokation with recovery handling.
    run_training = True
    while run_training:
        result = subprocess.run(command_str, shell=True)
        if result.returncode == 0 or args.disable_recovery:
            run_training = False
            break
        print(f'Process finished with {result.returncode}. Restarting...')
