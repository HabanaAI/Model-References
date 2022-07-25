# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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


# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch

import contextlib

import torch
from torch import _C
from torch.cuda import _lazy_call, device as device_ctx_manager
from torch.utils.checkpoint import detach_variable

from .initialize import get_data_parallel_rank
from .initialize import get_model_parallel_rank

from torch.utils.checkpoint import check_backward_validity, get_device_states, set_device_states, detach_variable

# Default name for the model parallel rng tracker.
_MODEL_PARALLEL_RNG_TRACKER_NAME = 'model-parallel-rng'


def _set_cuda_rng_state(new_state, device=-1):
    """Sets the random number generator state of the current GPU.

    Argumentss:
        new_state (torch.ByteTensor): The desired state
    This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 GPU cases.
    """
    if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):
        # older PyTorch
        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)
    else:
        # newer PyTorch
        if device == -1:
            device = torch.device('cuda')
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device('cuda', device)

        def cb():
            idx = device.index
            if idx is None:
                idx = torch.cuda.current_device()
            default_generator = torch.cuda.default_generators[idx]
            default_generator.set_state(new_state)

    _lazy_call(cb)


class CudaRNGStatesTracker:
    """Tracker for the cuda RNG states.

    Using the `add` method, a cuda rng state is initialized based on
    the input `seed` and is assigned to `name`. Later, by forking the
    rng state, we can perform operations and return to our starting
    cuda state.
    """
    def __init__(self):
        # Map from a string name to the cuda rng state.
        self.states_ = {}
        # Seeds are just for book keeping and ensure no seed is set twice.
        self.seeds_ = set()

    def reset(self):
        """Set to the initial state (no tracker)."""
        self.states_ = {}
        self.seeds_ = set()

    def get_states(self):
        """Get rng states. Copy the dictionary so we have direct
        pointers to the states, not just a pointer to the dictionary."""
        states = {}
        for name in self.states_:
            states[name] = self.states_[name]
        return states

    def set_states(self, states):
        """Set the rng states. For efficiency purposes, we do not check
        the size of seed for compatibility."""
        self.states_ = states

    def add(self, name, seed):
        """Track the rng state."""
        # Check seed is not already used.
        if seed in self.seeds_:
            raise Exception('seed {} already exists'.format(seed))
        self.seeds_.add(seed)
        # Check that state is not already defined.
        if name in self.states_:
            raise Exception('cuda rng state {} already exists'.format(name))
        # Get the current rng state.
        orig_rng_state = torch.cuda.get_rng_state()
        # Set the new state and store it.
        torch.cuda.manual_seed(seed)
        self.states_[name] = torch.cuda.get_rng_state()
        # Reset rng state to what it was.
        _set_cuda_rng_state(orig_rng_state)

    @contextlib.contextmanager
    def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
        """Fork the cuda rng state, perform operations, and exit with
        the original state."""
        # Check if we have added the state
        if name not in self.states_:
            raise Exception('cuda rng state {} is not added'.format(name))
        # Store current rng state.
        orig_cuda_rng_state = torch.cuda.get_rng_state()
        # Set rng state to the desired one
        _set_cuda_rng_state(self.states_[name])
        # Do the stuff we wanted to do.
        try:
            yield
        finally:
            # Update the current rng state for later use.
            self.states_[name] = torch.cuda.get_rng_state()
            # And set the state to the original state we started with.
            _set_cuda_rng_state(orig_cuda_rng_state)


def _set_hpu_rng_state(new_state, device=-1):
    """Sets the random number generator state of the current HPU.

    Argumentss:
        new_state (torch.ByteTensor): The desired state
    This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 HPU cases.
    """
    if device == -1:
        device = torch.device('hpu')
    elif isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device('hpu', device)

    def cb():
        idx = device.index
        if idx is None:
            idx = torch.current_device()
        default_generator = torch.default_generators[idx]
        default_generator.set_state(new_state)
    _lazy_call(cb)

class HpuRNGStatesTracker:
    """Tracker for the hpu RNG states.

    Using the `add` method, a hpu rng state is initialized based on
    the input `seed` and is assigned to `name`. Later, by forking the
    rng state, we can perform operations and return to our starting
    hpu state.
    """
    def __init__(self):
        # Map from a string name to the cuda rng state.
        self.states_ = {}
        # Seeds are just for book keeping and ensure no seed is set twice.
        self.seeds_ = set()

    def reset(self):
        """Set to the initial state (no tracker)."""
        self.states_ = {}
        self.seeds_ = set()

    def get_states(self):
        """Get rng states. Copy the dictionary so we have direct
        pointers to the states, not just a pointer to the dictionary."""
        states = {}
        for name in self.states_:
            states[name] = self.states_[name]
        return states

    def set_states(self, states):
        """Set the rng states. For efficiency purposes, we do not check
        the size of seed for compatibility."""
        self.states_ = states

    def add(self, name, seed):
        """Track the rng state."""
        # Check seed is not already used.
        if seed in self.seeds_:
            raise Exception('seed {} already exists'.format(seed))
        self.seeds_.add(seed)
        # Check that state is not already defined.
        if name in self.states_:
            raise Exception('hpu rng state {} already exists'.format(name))
        # Get the current rng state.
        orig_rng_state = torch.get_rng_state()
        # Set the new state and store it.
        torch.manual_seed(seed)
        self.states_[name] = torch.get_rng_state()
        # Reset rng state to what it was.
        _set_hpu_rng_state(orig_rng_state)

    @contextlib.contextmanager
    def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
        """Fork the hpu rng state, perform operations, and exit with
        the original state."""
        # Check if we have added the state
        if name not in self.states_:
            raise Exception('hpu rng state {} is not added'.format(name))
        # Store current rng state.
        orig_hpu_rng_state = torch.get_rng_state()
        # Set rng state to the desired one
        _set_hpu_rng_state(self.states_[name])
        # Do the stuff we wanted to do.
        try:
            yield
        finally:
            # Update the current rng state for later use.
            self.states_[name] = torch.get_rng_state()
            # And set the state to the original state we started with.
            _set_hpu_rng_state(orig_hpu_rng_state)

# RNG tracker object.
_CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()
_HPU_RNG_STATE_TRACKER = HpuRNGStatesTracker()

def get_cuda_rng_tracker():
    """Get cuda rng tracker."""
    return _CUDA_RNG_STATE_TRACKER

def get_hpu_rng_tracker():
    """Get hpu rng tracker."""
    return _HPU_RNG_STATE_TRACKER

def model_parallel_cuda_manual_seed(seed):
    """Initialize model parallel cuda seed.

    This function should be called after the model parallel is
    initialized. Also, no torch.cuda.manual_seed should be called
    after this function. Basically, this is replacement for that
    function.
    Two set of RNG states are tracked:
        default state: This is for data parallelism and is the same among a
                       set of model parallel GPUs but different across
                       different model paralle groups. This is used for
                       example for dropout in the non-model-parallel regions.
        model-parallel state: This state is different among a set of model
                              parallel GPUs, but the same across data parallel
                              groups. This is used for example for dropout in
                              model parallel regions.
    """
    # 2718 is just for fun and any POSITIVE value will work.
    offset = seed + 2718
    model_parallel_seed = offset + get_model_parallel_rank()
    # Data parallel gets the original sedd.
    data_parallel_seed = seed

    if torch.distributed.get_rank() == 0:
        print('> initializing model parallel cuda seeds on global rank {}, '
              'model parallel rank {}, and data parallel rank {} with '
              'model parallel seed: {} and data parallel seed: {}'.format(
                  torch.distributed.get_rank(), get_model_parallel_rank(),
                  get_data_parallel_rank(), model_parallel_seed,
                  data_parallel_seed), flush=True)
    _CUDA_RNG_STATE_TRACKER.reset()
    # Set the default state.
    torch.cuda.manual_seed(data_parallel_seed)
    # and model parallel state.
    _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME,
                                model_parallel_seed)

def model_parallel_hpu_manual_seed(seed):
    """Initialize model parallel cuda seed.

    This function should be called after the model parallel is
    initialized.
    Two set of RNG states are tracked:
        default state: This is for data parallelism and is the same among a
                       set of model parallel HPUs but different across
                       different model parallel groups. This is used for
                       example for dropout in the non-model-parallel regions.
        model-parallel state: This state is different among a set of model
                              parallel HPUs, but the same across data parallel
                              groups. This is used for example for dropout in
                              model parallel regions.
    """
    # 2709 is just for fun and any POSITIVE value will work.
    offset = seed + 2709
    model_parallel_seed = offset + get_model_parallel_rank()
    # Data parallel gets the original sedd.
    data_parallel_seed = seed

    if torch.distributed.get_rank() == 0:
        print('> initializing model parallel hpu seeds on global rank {}, '
              'model parallel rank {}, and data parallel rank {} with '
              'model parallel seed: {} and data parallel seed: {}'.format(
                  torch.distributed.get_rank(), get_model_parallel_rank(),
                  get_data_parallel_rank(), model_parallel_seed,
                  data_parallel_seed), flush=True)
    _HPU_RNG_STATE_TRACKER.reset()
    # Set the default state.
    torch.manual_seed(data_parallel_seed)
    # and model parallel state.
    _HPU_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME,
                                model_parallel_seed)

class CheckpointFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        check_backward_validity(args)
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        ctx.fwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            # Don't eagerly initialize the cuda context by accident.
            # (If the user intends that the context is initialized later, within their
            # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
            # we have no way to anticipate this will happen before we run the function.)
            ctx.had_cuda_in_fwd = False
            if torch.cuda._initialized:
                ctx.had_cuda_in_fwd = True
                ctx.fwd_gpu_devices, ctx.fwd_gpu_states = get_device_states(*args)
        ctx.save_for_backward(*args)
        with torch.no_grad():
            outputs = run_function(*args)
        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), please use .backward() if possible")
        inputs = ctx.saved_tensors
        # Stash the surrounding rng state, and mimic the state that was
        # present at this time during forward.  Restore the surrouding state
        # when we're done.
        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_cuda_in_fwd:
            rng_devices = ctx.fwd_gpu_devices
        bwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()
        with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_cuda_in_fwd:
                    set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)
                get_cuda_rng_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)
            detached_inputs = detach_variable(inputs)
            with torch.enable_grad():
                outputs = ctx.run_function(*detached_inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)
        torch.autograd.backward(outputs, args)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp
                      for inp in detached_inputs)
        return (None, None) + grads


def checkpoint(function, *args):
    """Checkpoint a model or part of the model.
    This has been directly copied from torch.utils.checkpoint."""
    return CheckpointFunction.apply(function, True, *args)
