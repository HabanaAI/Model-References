# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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


import glob
import os
import pickle
from subprocess import call
import numpy as np
import torch
from dllogger import JSONStreamBackend, Logger, StdOutBackend, Verbosity
from sklearn.model_selection import KFold
import random

from lightning_utilities import module_available
if module_available('lightning'):
    from lightning.pytorch import seed_everything
elif module_available('pytorch_lightning'):
    from pytorch_lightning import  seed_everything

try:
    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.utils.debug as htdebug
except ImportError:
    print(f"habana_frameworks could not be loaded")


def mark_step(is_lazy_mode):
    if is_lazy_mode:
        htcore.mark_step()


def get_device(args):
    return torch.device(get_device_str(args))

def is_main_process():
    return int(os.getenv("LOCAL_RANK", "0")) == 0


def get_device_str(args):
    if args.gpus:
        return 'cuda'
    elif args.hpus:
        return 'hpu'
    else:
        return 'cpu'


def get_device_data_type(args):
    if args.gpus:
        return torch.float16
    elif args.hpus:
        return torch.bfloat16
    else:
        return torch.float32


def set_cuda_devices(args):
    assert args.gpus <= torch.cuda.device_count(), f"Requested {args.gpus} gpus, available {torch.cuda.device_count()}."
    device_list = ",".join([str(i) for i in range(args.gpus)])
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", device_list)


def verify_ckpt_path(args):
    resume_path = os.path.join(args.results, "checkpoints", "last.ckpt")
    ckpt_path = resume_path if args.resume_training and os.path.exists(resume_path) else args.ckpt_path
    return ckpt_path


def get_task_code(args):
    return f"{args.task}_{args.dim}d"


def get_config_file(args):
    task_code = get_task_code(args)
    if args.data != "/data":
        path = os.path.join(args.data, "config.pkl")
    else:
        path = os.path.join(args.data, task_code, "config.pkl")
    return pickle.load(open(path, "rb"))


def get_dllogger(results):
    return Logger(
        backends=[
            JSONStreamBackend(Verbosity.VERBOSE, os.path.join(results, "logs.json")),
            StdOutBackend(Verbosity.VERBOSE, step_format=lambda step: f"Epoch: {step} "),
        ]
    )


def get_tta_flips(dim):
    if dim == 2:
        return [[2], [3], [2, 3]]
    return [[2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]


def make_empty_dir(path):
    call(["rm", "-rf", path])
    os.makedirs(path)


def flip(data, axis):
    return torch.flip(data, dims=axis)




def get_unet_params(args):
    config = get_config_file(args)
    patch_size, spacings = config["patch_size"], config["spacings"]
    strides, kernels, sizes = [], [], patch_size[:]
    while True:
        spacing_ratio = [spacing / min(spacings) for spacing in spacings]
        stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)
        if len(strides) == 5:
            break
    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return config["in_channels"], config["n_class"], kernels, strides, patch_size


def log(logname, dice, results="/results"):
    dllogger = Logger(
        backends=[
            JSONStreamBackend(Verbosity.VERBOSE, os.path.join(results, logname)),
            StdOutBackend(Verbosity.VERBOSE, step_format=lambda step: ""),
        ]
    )
    metrics = {}
    metrics.update({"Mean dice": round(dice.mean().item(), 2)})
    metrics.update({f"L{j+1}": round(m.item(), 2) for j, m in enumerate(dice)})
    dllogger.log(step=(), data=metrics)
    dllogger.flush()


def layout_2d(img, lbl):
    batch_size, depth, channels, height, weight = img.shape
    img = torch.reshape(img, (batch_size * depth, channels, height, weight))
    if lbl is not None:
        lbl = torch.reshape(lbl, (batch_size * depth, 1, height, weight))
        return img, lbl
    return img


def get_split(data, idx):
    return list(np.array(data)[idx])


def load_data(path, files_pattern):
    return sorted(glob.glob(os.path.join(path, files_pattern)))


def get_path(args):
    if args.data != "/data":
        if args.exec_mode == "predict" and not args.benchmark:
            return os.path.join(args.data, "test")
        return args.data
    data_path = os.path.join(args.data, get_task_code(args))
    if args.exec_mode == "predict" and not args.benchmark:
        data_path = os.path.join(data_path, "test")
    return data_path


def get_test_fnames(args, data_path, meta=None):
    kfold = KFold(n_splits=args.nfolds, shuffle=True, random_state=12345)
    test_imgs = load_data(data_path, "*_x.npy")

    if args.exec_mode == "predict" and "val" in data_path:
        _, val_idx = list(kfold.split(test_imgs))[args.fold]
        test_imgs = sorted(get_split(test_imgs, val_idx))
        if meta is not None:
            meta = sorted(get_split(meta, val_idx))

    return test_imgs, meta


def set_seed(seed):
    if seed is not None:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        seed_everything(seed)

