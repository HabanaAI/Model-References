# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
from collections import OrderedDict
import numpy as np
import torch
import torch.distributed as dist
class LogBuffer:
    def __init__(self):
        self.val_history = OrderedDict()
        self.n_history = OrderedDict()
        self.running_sum = OrderedDict()
        self.running_count = OrderedDict()
        self.output = OrderedDict()
        self.ready = False
    def clear(self):
        self.val_history.clear()
        self.n_history.clear()
        self.running_sum.clear()
        self.running_count.clear()
        self.clear_output()
    def clear_output(self):
        self.output.clear()
        self.ready = False
    def update(self, vars, count=1):
        assert isinstance(vars, dict)
        for key, var in vars.items():
            if type(var) == type(torch.tensor([])):
                self.running_sum[key] = self.running_sum.get(key, 0) + count * var
                self.running_count[key] = self.running_count.get(key, 0) + count
            else:
                if key not in self.val_history:
                    self.val_history[key] = []
                    self.n_history[key] = []
                self.val_history[key].append(var)
                self.n_history[key].append(count)
    def average(self, n=0):
        """Average latest n values or all values."""
        assert n >= 0
        for key in self.val_history:
            values = np.array(self.val_history[key][-n:])
            nums = np.array(self.n_history[key][-n:])
            avg = np.sum(values * nums) / np.sum(nums)
            self.output[key] = avg
        for key in self.running_sum:
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(self.running_sum[key].div_(dist.get_world_size()))
            avg = (self.running_sum[key]/self.running_count[key]).item()
            self.running_sum[key] = 0
            self.running_count[key] = 0
            self.output[key] = avg

        self.ready = True