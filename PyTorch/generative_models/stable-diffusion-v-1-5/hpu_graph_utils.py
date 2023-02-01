#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
###############################################################################
import torch


class CachedParams:
    def __init__(self, graph_inputs, graph_outputs, graph = None):
        self.graph_inputs = graph_inputs
        self.graph_outputs = graph_outputs
        self.graph = graph


def input_hash(obj):
    if isinstance(obj, dict):
        return input_hash(tuple(obj.items()))
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return hash(tuple(input_hash(el) for el in obj))
    elif torch.is_tensor(obj):
        return hash(obj.shape)
    else:
        return hash(obj)


def copy_to(dst, src):
    assert type(dst) == type(src)
    if isinstance(dst, dict):
        for (dk, dv), (sk, sv) in zip(dst.items(), src.items()):
            assert dk == sk
            copy_to(dv, sv)
    elif isinstance(dst, list) or isinstance(dst, tuple):
        for d, s in zip(dst, src):
            copy_to(d, s)
    elif torch.is_tensor(dst):
        dst.copy_(src)
