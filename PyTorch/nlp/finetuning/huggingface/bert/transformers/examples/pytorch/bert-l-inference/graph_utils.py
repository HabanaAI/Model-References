#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
###############################################################################

import torch

class CachedParams:
    def __init__(self, graph_inputs, graph_outputs, graph):
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
        dst.copy_(src, non_blocking=True)


def wrap_in_hpu_graph_func(func):
    import habana_frameworks.torch as ht
    stream = ht.hpu.Stream()
    cache = {}
    orig_fwd = func
    def forward(*args, **kwargs):
        inputs = (args, kwargs)
        h = input_hash(inputs)
        cached = cache.get(h)
        if cached is None:
            with ht.hpu.stream(stream):
                graph = ht.hpu.HPUGraph()
                graph.capture_begin()
                outputs = orig_fwd(*args, **kwargs)
                graph.capture_end()
                graph_inputs = inputs
                graph_outputs = outputs
                cache[h] = CachedParams(graph_inputs, graph_outputs, graph)
            return outputs
        copy_to(cached.graph_inputs, inputs)
        cached.graph.replay()
        return cached.graph_outputs
    return forward


def wrap_in_hpu_graph(module):
    import habana_frameworks.torch as ht
    stream = ht.hpu.Stream()
    cache = {}
    orig_fwd = module.forward
    def forward(*args, **kwargs):
        inputs = (args, kwargs)
        h = input_hash(inputs)
        cached = cache.get(h)
        if cached is None:
            with ht.hpu.stream(stream):
                graph = ht.hpu.HPUGraph()
                graph.capture_begin()
                outputs = orig_fwd(*args, **kwargs)
                graph.capture_end()
                graph_inputs = inputs
                graph_outputs = outputs
                cache[h] = CachedParams(graph_inputs, graph_outputs, graph)
            return outputs

        copy_to(cached.graph_inputs, inputs)
        cached.graph.replay()
        return cached.graph_outputs
    module.forward = forward
    return module
