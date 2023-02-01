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


def wrap_in_cuda_graph(module):
    stream = torch.cuda.Stream()
    cache = {}
    orig_fwd = module.forward

    def forward(*args, **kwargs):
        inputs = (args, kwargs)
        h = input_hash(inputs)
        cached = cache.get(h)
        if cached is None:
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                outputs = orig_fwd(*args, **kwargs)
                graph_inputs = inputs
                graph_outputs = outputs
                cache[h] = CachedParams(graph_inputs, graph_outputs)
            torch.cuda.current_stream().wait_stream(stream)
            return outputs
        if cached.graph is None:
            cached.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cached.graph):
                outputs = orig_fwd(*cached.graph_inputs[0], **cached.graph_inputs[1])
                copy_to(cached.graph_outputs, outputs)
        else:
            copy_to(cached.graph_inputs, inputs)
        cached.graph.replay()
        return cached.graph_outputs
    module.forward = forward
    return module


def wrap_in_graph(args, module):
    if args.device == 'hpu':
        import habana_frameworks.torch.hpu.graphs as htgraphs
        return htgraphs.wrap_in_hpu_graph(module)
    else:
        return wrap_in_cuda_graph(module)
