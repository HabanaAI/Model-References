#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
###############################################################################
import habana_frameworks.torch as ht
import torch


class CachedParams:
    def __init__(self, graph_inputs, graph_outputs):
        self.graph_inputs = graph_inputs
        self.graph_outputs = graph_outputs
        self.graph = None


def input_hash(obj):
    if isinstance(obj, dict):
        return input_hash(tuple(obj.items()))
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return hash(tuple(input_hash(el) for el in obj))
    elif torch.is_tensor(obj):
        return hash((obj.shape, obj.dtype, obj.device))
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


def wrap_in_hpu_graph(module):
    stream = ht.hpu.Stream()
    cache = {}
    orig_fwd = module.forward

    def forward(*args, **kwargs):
        inputs = (args, kwargs)
        h = input_hash(inputs)
        cached = cache.get(h)
        if cached is None:
            outputs = orig_fwd(*args, **kwargs)
            graph_inputs = inputs
            graph_outputs = outputs
            cache[h] = CachedParams(graph_inputs, graph_outputs)
            return outputs

        with ht.hpu.stream(stream):
            copy_to(cached.graph_inputs, inputs)
            stream.synchronize()
            if cached.graph is None:
                cached.graph = ht.hpu.HPUGraph()
                cached.graph.capture_begin()
                outputs = orig_fwd(*cached.graph_inputs[0], **cached.graph_inputs[1])
                copy_to(cached.graph_outputs, outputs)
                cached.graph.capture_end()
            else:
                cached.graph.replay()
        return cached.graph_outputs
    module.forward = forward
    return module
