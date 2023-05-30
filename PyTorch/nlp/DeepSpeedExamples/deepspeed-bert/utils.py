# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
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

import torch
import random
import numpy as np
import torch.distributed as dist

from pathlib import Path


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    return get_rank() == 0


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def format_step(step):
    if isinstance(step, str):
        return step
    s = ""
    if len(step) > 0:
        s += "Training Epoch: {} ".format(step[0])
    if len(step) > 1:
        s += "Training Iteration: {} ".format(step[1])
    if len(step) > 2:
        s += "Validation Iteration: {} ".format(step[2])
    return s


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def mkdir_by_main_process(path):
    if is_main_process():
        mkdir(path)
    barrier()


def fix_tensor_numpy():
    def numpy_detached(self):
        return numpy_func(self.detach())
    numpy_func = getattr(torch.Tensor, 'numpy')
    setattr(torch.Tensor, 'numpy', numpy_detached)


def get_local_rng_state(with_cuda=False, with_hpu=False):
    assert not (with_hpu and with_cuda), 'Specify at most one of with_cuda/with_hpu'
    state = {
        'random': random.getstate(),
        'np': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if with_hpu:
        import habana_frameworks.torch.hpu.random as hpu_random
        state.update({'hpu': hpu_random.get_rng_state()})
    if with_cuda:
        state.update({'cuda': torch.cuda.get_rng_state()})
    return state


def set_local_rng_state(new_state, with_cuda=False, with_hpu=False):
    assert not (with_hpu and with_cuda), 'Specify at most one of with_cuda/with_hpu'
    random.setstate(new_state['random'])
    np.random.set_state(new_state['np'])
    torch.set_rng_state(new_state['torch'])
    if with_hpu:
        import habana_frameworks.torch.hpu.random as hpu_random
        hpu_rng_state = new_state.get('hpu', None)
        if hpu_rng_state is None:
            print("Warning: No hpu rng state found. Continuing without full reproducibility.")
        else:
            hpu_random.set_rng_state(hpu_rng_state)
    if with_cuda:
        cuda_rng_state = new_state.get('cuda', None)
        if cuda_rng_state is None:
            print("Warning: No cuda state found. Continuing without full reproducibility.")
        else:
            torch.cuda.set_rng_state(cuda_rng_state)
