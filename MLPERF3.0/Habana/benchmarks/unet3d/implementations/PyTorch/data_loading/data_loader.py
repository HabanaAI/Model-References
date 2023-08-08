# Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################
# Copied functions `calculate_work`, `make_val_split_even` from:
# https://github.com/mlcommons/training_results_v2.0/blob/main/NVIDIA/benchmarks/unet3d/implementations/mxnet/data_loading/data_loader.py
# Changes done to the original functions:
# - refactor the code, rename some of variables
# - get rid of sharded eval mechanism

import os
import glob

import numpy as np

from data_loading.dali_loader import get_dali_loader
from data_loading.pytorch_loader import get_pytorch_loader
from runtime.logging import mllog_event
from habana_frameworks.mediapipe.plugins.iterator_pytorch import CPUUnetPytorchIterator
from habana_frameworks.medialoaders.torch.mediapipe_unet_cpp import UnetMediaPipe

def list_files_with_pattern(path, files_pattern):
    data = sorted(glob.glob(os.path.join(path, files_pattern)))
    assert len(data) > 0, f"Found no data at {path}"
    return data


def load_data(path, files_pattern):
    data = sorted(glob.glob(os.path.join(path, files_pattern)))
    assert len(data) > 0, f"Found no data at {path}"
    return data


def get_split(data, train_idx, val_idx):
    train = list(np.array(data)[train_idx])
    val = list(np.array(data)[val_idx])
    return train, val


def split_data(x_all, y_all, num_shards, shard_id):
    x = [a.tolist() for a in np.array_split(x_all, num_shards)]
    y = [a.tolist() for a in np.array_split(y_all, num_shards)]
    return x[shard_id], y[shard_id]


def calculate_work(file_path):
    image = np.load(file_path)
    image_shape = list(image.shape[1:])
    n_dim = len(image_shape)
    # Compute work to be done based on number of patches of sliding window in each dimension
    return np.prod([image_shape[i] // 64 - 1 + (1 if image_shape[i] % 64 >= 32 else 0) for i in range(n_dim)])


def make_val_split_even(x_val, y_val, num_shards, shard_id):
    work = np.array(list(map(calculate_work, y_val)))
    x_res = [[] for _ in range(num_shards)]
    y_res = [[] for _ in range(num_shards)]
    work_per_shard = np.zeros(shape=num_shards)

    x_val, y_val = np.array(x_val), np.array(y_val)
    sort_idx = np.argsort(work)[::-1]
    work = work[sort_idx]
    x_val, y_val = x_val[sort_idx], y_val[sort_idx]

    for w_idx, w in enumerate(work):
        idx = np.argmin(work_per_shard)
        work_per_shard[idx] += w
        x_res[idx].append(x_val[w_idx])
        y_res[idx].append(y_val[w_idx])

    return x_res[shard_id], y_res[shard_id]


def get_data_split(path: str, num_shards: int, shard_id: int, split_train_data: bool = False):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../evaluation_cases.txt"), "r") as f:
        val_cases_list = f.readlines()
    val_cases_list = [case.rstrip("\n") for case in val_cases_list]
    imgs = load_data(path, "*_x.npy")
    lbls = load_data(path, "*_y.npy")
    assert len(imgs) == len(lbls), f"Found {len(imgs)} volumes but {len(lbls)} corresponding masks"
    imgs_train, lbls_train, imgs_val, lbls_val = [], [], [], []
    for (case_img, case_lbl) in zip(imgs, lbls):
        if case_img.split("_")[-2] in val_cases_list:
            imgs_val.append(case_img)
            lbls_val.append(case_lbl)
        else:
            imgs_train.append(case_img)
            lbls_train.append(case_lbl)
    mllog_event(key='train_samples', value=len(imgs_train), sync=False)
    mllog_event(key='eval_samples', value=len(imgs_val), sync=False)

    if split_train_data is True:
        imgs_train, lbls_train = split_data(imgs_train, lbls_train, num_shards, shard_id)
    imgs_val, lbls_val = make_val_split_even(imgs_val, lbls_val, num_shards=num_shards, shard_id=shard_id)

    return imgs_train, imgs_val, lbls_train, lbls_val


def get_data_loaders(flags, num_shards, local_rank, warmup=False):
    if flags.loader == "synthetic" or warmup:
        train_dataloader = get_pytorch_loader(flags, train_mode=True,
                                              batch_size=flags.batch_size, num_shards=num_shards)
        val_dataloader = get_pytorch_loader(flags, train_mode=False,
                                            batch_size=1, num_shards=num_shards)

    elif flags.loader == "pytorch":
        x_train, x_val, y_train, y_val = get_data_split(flags.data_dir, num_shards, shard_id=local_rank)
        train_dataloader = get_pytorch_loader(flags, x_train, y_train, train_mode=True,
                                              batch_size=flags.batch_size, num_shards=num_shards)
        val_dataloader = get_pytorch_loader(flags, x_val, y_val, train_mode=False,
                                            batch_size=1, num_shards=num_shards)

    elif flags.loader == "dali":
        x_train, x_val, y_train, y_val = get_data_split(flags.data_dir, num_shards=num_shards,
                                                        shard_id=local_rank, split_train_data=True)
        train_dataloader = get_dali_loader(flags, x_train, y_train, train_mode=True, num_shards=1)
        val_dataloader = get_dali_loader(flags, x_val, y_val, train_mode=False, num_shards=1)

    elif flags.loader == "media":
        x_train, x_val, y_train, y_val = get_data_split(flags.data_dir, num_shards=num_shards,
                                                        shard_id=local_rank, split_train_data=True)

        train_pipe = UnetMediaPipe(device='cpu',
                                         queue_depth=2,
                                         batch_size=flags.batch_size,
                                         input_list=[x_train, y_train],
                                         patch_size=flags.input_shape,
                                         seed=flags.seed,
                                         drop_remainder=True,
                                         num_slices=1,
                                         slice_index=0,
                                         num_threads=7,
                                         is_testing=False)

        train_dataloader = CPUUnetPytorchIterator(train_pipe)

        val_pipe = UnetMediaPipe(device='cpu',
                                       queue_depth=2,
                                       batch_size=1,
                                       input_list=[x_val, y_val],
                                       patch_size=flags.val_input_shape,
                                       seed=flags.seed,
                                       drop_remainder=True,
                                       num_slices=1,
                                       slice_index=0,
                                       num_threads=2,
                                       is_testing=True)

        val_dataloader = CPUUnetPytorchIterator(val_pipe)

    return train_dataloader, val_dataloader
