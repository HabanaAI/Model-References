# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
from itertools import chain
import torch
from torch.nn.parallel import DataParallel

from .scatter_gather import scatter_kwargs
from mmcv.utils import is_hpu_enabled, groundtruth_processing_on_cpu, move_to_hpu, move_to_device, move_model_to_hpu_if_needed
import torch.nn as nn

class MMDataParallel(DataParallel):
    """The DataParallel module that supports DataContainer.

    MMDataParallel has two main differences with PyTorch DataParallel:

    - It supports a custom type :class:`DataContainer` which allows more
      flexible control of input data during both GPU and CPU inference.
    - It implement two more APIs ``train_step()`` and ``val_step()``.

    .. warning::
        MMDataParallel only supports single GPU training, if you need to
        train with multiple GPUs, please use MMDistributedDataParallel
        instead. If you have multiple GPUs and you just want to use
        MMDataParallel, you can set the environment variable
        ``CUDA_VISIBLE_DEVICES=0`` or instantiate ``MMDataParallel`` with
        ``device_ids=[0]``.

    Args:
        module (:class:`nn.Module`): Module to be encapsulated.
        device_ids (list[int]): Device IDS of modules to be scattered to.
            Defaults to None when GPU is not available.
        output_device (str | int): Device ID for output. Defaults to None.
        dim (int): Dimension used to scatter the data. Defaults to 0.
    """

    def __init__(self, *args, dim=0, **kwargs):
        if is_hpu_enabled():
            # because MMDataParallel is a subclass of DataParallel which is not supported
            # so the patched in __init__ function avoid parts that would have failed
            nn.Module.__init__(self)
            torch._C._log_api_usage_once("torch.nn.parallel.DataParallel")
            module = args[0]
            self.module = module
            self.device_ids = []
            move_model_to_hpu_if_needed(self.module)
        else:
            super(MMDataParallel, self).__init__(*args, dim=dim, **kwargs)
            # DataParallel will not transfer to HPU, so moving model here
            cpu_mode = not (is_hpu_enabled() or torch.cuda.is_available())
            if cpu_mode:
                self.module.to('cpu')
                self.device_ids = []
            self.dim = dim

    def forward(self, *inputs, **kwargs):
        """Override the original forward function.

        The main difference lies in the CPU inference where the data in
        :class:`DataContainers` will still be gathered.
        """
        if not self.device_ids:
            # We add the following line thus the module could gather and
            # convert data containers as those in GPU inference
            if is_hpu_enabled():
                assert len(inputs) == 0 #have not encountered case with len(inputs)>0, so have this assert here
                kwargs = (move_to_hpu(kwargs),)
                inputs = (inputs,)
            else:
                inputs, kwargs = self.scatter(inputs, kwargs, [-1])
            return self.module(*inputs[0], **kwargs[0])
        else:
            return super().forward(*inputs, **kwargs)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def train_step(self, *inputs, **kwargs):
        if not self.device_ids:
            # We add the following line thus the module could gather and
            # convert data containers as those in GPU inference
            # CPU and HPU follows this path
            if is_hpu_enabled():
                assert len(kwargs) == 0 #have not encountered case with len(kwargs)>0, so have this assert here
                if groundtruth_processing_on_cpu():
                    inputs = move_to_device(inputs, 'cpu')
                    inputs[0]['img'] = move_to_hpu(inputs[0]['img'])
                    inputs = (inputs,)
                else:
                    inputs = (move_to_hpu(inputs),)
                kwargs = (kwargs,)
            else:
                inputs, kwargs = self.scatter(inputs, kwargs, [-1])
            return self.module.train_step(*inputs[0], **kwargs[0])

        assert not is_hpu_enabled()
        assert torch.cuda.is_available()

        assert len(self.device_ids) == 1, \
            ('MMDataParallel only supports single GPU training, if you need to'
             ' train with multiple GPUs, please use MMDistributedDataParallel'
             ' instead.')

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(
                    'module must have its parameters and buffers '
                    f'on device {self.src_device_obj} (device_ids[0]) but '
                    f'found one of them on device: {t.device}')

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        return self.module.train_step(*inputs[0], **kwargs[0])

    def val_step(self, *inputs, **kwargs):
        if is_hpu_enabled():
            assert False, 'val step hasnt been enabled for HPU yet'

        if not self.device_ids:
            # We add the following line thus the module could gather and
            # convert data containers as those in GPU inference
            inputs, kwargs = self.scatter(inputs, kwargs, [-1])
            return self.module.val_step(*inputs[0], **kwargs[0])

        assert len(self.device_ids) == 1, \
            ('MMDataParallel only supports single GPU training, if you need to'
             ' train with multiple GPUs, please use MMDistributedDataParallel'
             ' instead.')

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(
                    'module must have its parameters and buffers '
                    f'on device {self.src_device_obj} (device_ids[0]) but '
                    f'found one of them on device: {t.device}')

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        return self.module.val_step(*inputs[0], **kwargs[0])


