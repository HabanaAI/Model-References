# Copyright The PyTorch Lightning team.
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
import functools
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union
from collections.abc import Mapping, Sequence
from collections import namedtuple
from copy import deepcopy
from distutils.version import LooseVersion

import os
import torch
from torch import nn

from pytorch_lightning.utilities.apply_func import apply_to_collection
# from pytorch_lightning.utilities.distributed import gather_all_tensors_if_available
# from pytorch_lightning.metrics.utils import _flatten, dim_zero_cat, dim_zero_mean, dim_zero_sum


class Metric(nn.Module, ABC):
    """
    Base class for all metrics present in the Metrics API.

    Implements ``add_state()``, ``forward()``, ``reset()`` and a few other things to
    handle distributed synchronization and per-step metric computation.

    Override ``update()`` and ``compute()`` functions to implement your own metric. Use
    ``add_state()`` to register metric state variables which keep track of state on each
    call of ``update()`` and are synchronized across processes when ``compute()`` is called.

    Note:
        Metric state variables can either be ``torch.Tensors`` or an empty list which can we used
        to store `torch.Tensors``.

    Note:
        Different metrics only override ``update()`` and not ``forward()``. A call to ``update()``
        is valid, but it won't return the metric value at the current step. A call to ``forward()``
        automatically calls ``update()`` and also returns the metric value at the current step.

    Args:
        compute_on_step:
            Forward only calls ``update()`` and returns None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
    """
    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__()

        self.dist_sync_on_step = dist_sync_on_step
        self.compute_on_step = compute_on_step
        self.process_group = process_group
        self._to_sync = True

        self.update = self._wrap_update(self.update)
        self.compute = self._wrap_compute(self.compute)
        self._computed = None
        self._forward_cache = None

        # initialize state
        self._reductions = {}
        self._defaults = {}

    def add_state(
        self, name: str, default, dist_reduce_fx: Optional[Union[str, Callable]] = None, persistent: bool = True
    ):
        """
        Adds metric state variable. Only used by subclasses.

        Args:
            name: The name of the state variable. The variable will then be accessible at ``self.name``.
            default: Default value of the state; can either be a ``torch.Tensor`` or an empty list. The state will be
                reset to this value when ``self.reset()`` is called.
            dist_reduce_fx (Optional): Function to reduce state accross mutliple processes in distributed mode.
                If value is ``"sum"``, ``"mean"``, or ``"cat"``, we will use ``torch.sum``, ``torch.mean``,
                and ``torch.cat`` respectively, each with argument ``dim=0``. The user can also pass a custom
                function in this parameter.
            persistent (Optional): whether the state will be saved as part of the modules ``state_dict``.

        Note:
            Setting ``dist_reduce_fx`` to None will return the metric state synchronized across different processes.
            However, there won't be any reduction function applied to the synchronized metric state.

            The metric states would be synced as follows

            - If the metric state is ``torch.Tensor``, the synced value will be a stacked ``torch.Tensor`` across
              the process dimension if the metric state was a ``torch.Tensor``. The original ``torch.Tensor`` metric
              state retains dimension and hence the synchronized output will be of shape ``(num_process, ...)``.

            - If the metric state is a ``list``, the synced value will be a ``list`` containing the
              combined elements from all processes.

        Note:
            When passing a custom function to ``dist_reduce_fx``, expect the synchronized metric state to follow
            the format discussed in the above note.

        """
        if (
            not isinstance(default, torch.Tensor)
            and not isinstance(default, list)                     # noqa: W503
            or (isinstance(default, list) and len(default) != 0)  # noqa: W503
        ):
            raise ValueError(
                "state variable must be a tensor or any empty list (where you can append tensors)"
            )

        if dist_reduce_fx == "sum":
            dist_reduce_fx = dim_zero_sum
        elif dist_reduce_fx == "mean":
            dist_reduce_fx = dim_zero_mean
        elif dist_reduce_fx == "cat":
            dist_reduce_fx = dim_zero_cat
        elif dist_reduce_fx is not None and not isinstance(dist_reduce_fx, Callable):
            raise ValueError(
                "`dist_reduce_fx` must be callable or one of ['mean', 'sum', 'cat', None]"
            )

        if isinstance(default, torch.Tensor):
            if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
                # persistent keyword is only supported in torch >= 1.6.0
                self.register_buffer(name, default, persistent=persistent)
            else:
                self.register_buffer(name, default)
        else:
            setattr(self, name, default)

        self._defaults[name] = deepcopy(default)
        self._reductions[name] = dist_reduce_fx

    def forward(self, *args, **kwargs):
        """
        Automatically calls ``update()``. Returns the metric value over inputs if ``compute_on_step`` is True.
        """
        # add current step
        self.update(*args, **kwargs)
        self._forward_cache = None

        if self.compute_on_step:
            self._to_sync = self.dist_sync_on_step

            # save context before switch
            self._cache = {attr: getattr(self, attr) for attr in self._defaults.keys()}

            # call reset, update, compute, on single batch
            self.reset()
            self.update(*args, **kwargs)
            self._forward_cache = self.compute()

            # restore context
            for attr, val in self._cache.items():
                setattr(self, attr, val)
            self._to_sync = True
            self._computed = None

            return self._forward_cache

    def _sync_dist(self):
        input_dict = {attr: getattr(self, attr) for attr in self._reductions.keys()}
        output_dict = apply_to_collection(
            input_dict,
            torch.Tensor,
            gather_all_tensors_if_available,
            group=self.process_group,
        )

        for attr, reduction_fn in self._reductions.items():
            # pre-processing ops (stack or flatten for inputs)
            if isinstance(output_dict[attr][0], torch.Tensor):
                output_dict[attr] = torch.stack(output_dict[attr])
            elif isinstance(output_dict[attr][0], list):
                output_dict[attr] = _flatten(output_dict[attr])

            assert isinstance(reduction_fn, (Callable)) or reduction_fn is None
            reduced = reduction_fn(output_dict[attr]) if reduction_fn is not None else output_dict[attr]
            setattr(self, attr, reduced)

    def _wrap_update(self, update):
        @functools.wraps(update)
        def wrapped_func(*args, **kwargs):
            self._computed = None
            return update(*args, **kwargs)
        return wrapped_func

    def _wrap_compute(self, compute):
        @functools.wraps(compute)
        def wrapped_func(*args, **kwargs):
            # return cached value
            if self._computed is not None:
                return self._computed

            if (
                self._to_sync
                and torch.distributed.is_available()  # noqa: W503
                and torch.distributed.is_initialized()  # noqa: W503
            ):
                self._sync_dist()

            self._computed = compute(*args, **kwargs)
            self.reset()

            return self._computed

        return wrapped_func

    @abstractmethod
    def update(self) -> None:  # pylint: disable=E0202
        """
        Override this method to update the state variables of your metric class.
        """
        pass

    @abstractmethod
    def compute(self):  # pylint: disable=E0202
        """
        Override this method to compute the final metric value from state variables
        synchronized across the distributed backend.
        """
        pass

    def reset(self):
        """
        This method automatically resets the metric state variables to their default value.
        """
        for attr, default in self._defaults.items():
            current_val = getattr(self, attr)
            if isinstance(current_val, torch.Tensor):
                setattr(self, attr, deepcopy(default).to(current_val.device))
            else:
                setattr(self, attr, deepcopy(default))

    def __getstate__(self):
        # ignore update and compute functions for pickling
        return {k: v for k, v in self.__dict__.items() if k not in ["update", "compute"]}

    def __setstate__(self, state):
        # manually restore update and compute functions for pickling
        self.__dict__.update(state)
        self.update = self._wrap_update(self.update)
        self.compute = self._wrap_compute(self.compute)




def gather_all_tensors_if_available(result: Union[torch.Tensor], group: Optional[Any] = None):
    """
    Function to gather all tensors from several ddp processes onto a list that
    is broadcasted to all processes
    Args:
        result: the value to sync
        group: the process group to gather results from. Defaults to all processes (world)
    Return:
        gathered_result: list with size equal to the process group where
            gathered_result[i] corresponds to result tensor from process i
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if group is None:
            group = torch.distributed.group.WORLD

        world_size = torch.distributed.get_world_size(group)

        gathered_result = [torch.zeros_like(result) for _ in range(world_size)]

        # sync and broadcast all
        torch.distributed.barrier(group=group)
        torch.distributed.all_gather(gathered_result, result, group)

        result = gathered_result
    return result


def dim_zero_cat(x):
    return torch.cat(x, dim=0)


def dim_zero_sum(x):
    return torch.sum(x, dim=0)


def dim_zero_mean(x):
    return torch.mean(x, dim=0)


def _flatten(x):
    return [item for sublist in x for item in sublist]
