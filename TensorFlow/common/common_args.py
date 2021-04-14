# ******************************************************************************
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
# ******************************************************************************
import argparse
import os

def default_dataset_dir(dataset_dirs):
    for dir in dataset_dirs:
        if os.path.exists(dir):
            return dir
    return None

def check_path(label, dir, default_dirs):
    if not dir:
        print("{} dir not defined and neither default location was found:\n{}".format(label, default_dirs))
        return False
    elif not os.path.exists(dir):
        print("{} path does not exist:\n{}".format(label, dir))
        return False
    else:
        return True

def common_args(custom_args_fun = None, custom_bf16_fun = None, bf16_val = '1', dataset_dirs = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", "-dt", choices=['fp32', 'bf16'], default='bf16')
    parser.add_argument("--batch_size", "-bs", type=int)
    parser.add_argument("--num_epochs", "-ep", type=int)
    parser.add_argument("--num_steps", "-st", type=int)
    parser.add_argument("--log_every_n_steps", "-ls", type=int)
    parser.add_argument("--cp_every_n_steps", "-cs", type=int)
    parser.add_argument("--dataset_dir", "-dd", type=str,
                        default=default_dataset_dir(dataset_dirs if dataset_dirs else []))

    if custom_args_fun:
        custom_args_fun(parser)

    args = parser.parse_args()

    print("args = {}".format(args), flush=True)

    if not check_path("Dataset", args.dataset_dir, dataset_dirs):
        exit(1)

    if args.data_type == 'bf16':
        if custom_bf16_fun:
            custom_bf16_fun()
        else:
            os.environ["TF_ENABLE_BF16_CONVERSION"] = str(bf16_val)

    return args
