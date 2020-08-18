#! /usr/bin/python3

import argparse
import os
import pathlib
import subprocess

COCO_DATASET = '/software/data/tf/coco2017/'
LOCAL_COCO_DATASET = '/data/coco2017/'


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

    def create_mpi_cmd(num_workers):
        mpi_cmd_parts = []
        mpi_cmd_parts.append("mpirun")
        mpi_cmd_parts.append("--allow-run-as-root")
        mpi_pe = get_mpi_pe(num_workers)
        mpi_cmd_parts.append("--bind-to core --map-by socket:PE={}".format(mpi_pe))
        mpi_cmd_parts.append("--np {}".format(num_workers))

        return ' '.join(mpi_cmd_parts)

    def generate_hcl_config(num_workers):
        common_path = find_common_path()
        common_sh = common_path.joinpath('common.sh')
        script_path = os.path.dirname(os.path.realpath(__file__))
        subprocess.Popen(['bash', '-c', 'source {}; generate_hcl_config {} {}'.format(common_sh, script_path, num_workers)])
        config_file_name = "config_{}.json".format(num_workers)
        os.environ['HCL_CONFIG_PATH'] = "{}/{}".format(script_path, config_file_name)

    def get_mpi_pe(num_workers):
        lscpu = subprocess.check_output("lscpu", shell=True).strip().decode().split('\n')
        for line in lscpu:
            if line.startswith("CPU(s):"):
                cpus = int(line.split()[1])
                return cpus//num_workers//2

    cmd_parts = []

    # Add path for 'library_loader'.
    common_path_env = f'PYTHONPATH=$PYTHONPATH:{find_common_path()}'
    cmd_parts.append(common_path_env)

    # disable tpc_fuser for 0.13
    fuser_env = f'RUN_TPC_FUSER=false'
    cmd_parts.append(fuser_env)

    # bfloat16 handling.
    bf16_flag = 'TF_ENABLE_BF16_CONVERSION'
    if args.dtype == 'fp32':
        cmd_parts.append(f'{bf16_flag}=0')
    elif args.dtype == 'bf16-basic':
        cmd_parts.append(f'{bf16_flag}=basic')
    elif args.dtype == 'bf16':
        cmd_parts.append(f'{bf16_flag}=1')

    num_workers = args.hvd_workers

    if num_workers > 1:
        generate_hcl_config(num_workers)
        cmd_parts.append(create_mpi_cmd(num_workers))

    cmd_parts.append('python3 samples/coco/coco.py')

    cmd_parts.append(args.command)
    if args.command == 'evaluate':
        cmd_parts.append(f'--limit {args.limit}')

    cmd_parts.append(f'--backbone {args.backbone}')
    cmd_parts.append(f'--model {args.model}')
    cmd_parts.append('--short')
    cmd_parts.append(f'--dataset {args.dataset}')
    cmd_parts.append(f'--device {args.device}')

    if args.epochs is not None:
        cmd_parts.append(f'--epochs {args.epochs}')
    if args.steps_per_epoch is not None:
        cmd_parts.append(f'--steps_per_epoch {args.steps_per_epoch}')

    if args.disable_validation:
        cmd_parts.append('--disable_validation')
    if args.dump_tf_timeline:
        cmd_parts.append('--dump_tf_timeline')

    if num_workers > 1:
        cmd_parts.append('--using_horovod')

    cmd_parts.append(f'--custom_roi={args.custom_roi}')

    return ' '.join(cmd_parts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Runs MaskRCNN demo.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--command', metavar='<command>', help='\'train\' or \'evaluate\' on MS COCO',
                        type=str, choices=['train', 'evaluate'], default='train')
    parser.add_argument(
        '--local', help=f'Use local dataset, overrides default \'--dataset\' to {LOCAL_COCO_DATASET}', action='store_true')
    parser.add_argument('--dataset', metavar='<dataset>',
                        help='Dataset directory', type=str, default=COCO_DATASET)
    parser.add_argument('--model', metavar='<model>', help='Model',
                        type=str, choices=['keras'], default='keras')
    parser.add_argument('--backbone', metavar='<backbone>', help='Path to weights .h5 file or \'coco\'',
                        type=str, choices=['kapp_ResNet50', 'resnet101'], default='kapp_ResNet50')
    parser.add_argument('--limit', metavar='<image count>')
    parser.add_argument('-e', '--epochs', metavar='<epochs>',
                        help='Number of epochs', type=int)
    parser.add_argument('-s', '--steps_per_epoch', metavar='<steps per epoch>',
                        help='Steps per epoch', type=int)
    parser.add_argument('-d', '--dtype', metavar='<data type>',
                        help='Data type: fp32 or bf16', type=str, choices=['fp32', 'bf16'], default='fp32')
    parser.add_argument('--device', metavar='<device>', help='Device selection',
                        type=str, choices=['CPU', 'HPU', 'GPU'], default='HPU')
    parser.add_argument('--disable_validation', '--dv', help='Disables validation', action='store_true')
    parser.add_argument('--dump_tf_timeline',
                        help='Gathers additional data from last iteration', action='store_true')
    parser.add_argument('--hvd_workers', metavar='<hvd_workers>', help='Number of Horovod workers, default 1 - Horovod disabled',
                        type=int, default=1)
    parser.add_argument('--custom_roi', metavar='<custom_roi>', help='Enables use of custom op 1:habana_ops.pyramid_roi_align 2/3:GEMM-based',
                        choices=['0', '1', '2', '3'], default='3')

    args = parser.parse_args()
    if args.local:
        args.dataset = LOCAL_COCO_DATASET

    command_str = create_command_str(args)

    print(command_str)

    import os
    os.system(command_str)
