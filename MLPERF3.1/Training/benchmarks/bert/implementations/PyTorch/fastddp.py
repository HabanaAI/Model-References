# autopep8: off
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company

"""
    Fast and lightweight alternative of DistributeDataParallel for Intel® Gaudi® AI Accelerator
"""

import functools

import torch
import torch.nn

from habana_frameworks.torch.core import mark_step

_fusion_buffer = None
_fusion_views = None
_visited_gradient_elem_count = 0
_mark_step_on_gradients = None
_all_reduce_group_size = torch.distributed.group.WORLD.size()

_use_hpu_graph = None
_recorded_unfusion_graph = None

def FastDistributedDataParallel(model: torch.nn.Module, fusion_buffer_dtype: torch.dtype, mark_step_on_gradients=[], use_hpu_graph: bool=True):
    gradient_elem_count = 0
    for param in model.parameters():
        assert param.dtype == torch.float32, '...'
        gradient_elem_count += torch.numel(param)

    global _fusion_buffer
    global _fusion_views
    global _visited_gradient_elem_count
    global _mark_step_on_gradients
    _fusion_buffer = torch.zeros(size=(gradient_elem_count,), dtype=fusion_buffer_dtype, device='hpu:0')
    _fusion_views = {}
    _visited_gradient_elem_count = 0
    _mark_step_on_gradients = mark_step_on_gradients

    for param_index, param in enumerate(model.parameters()):
        param.register_hook(functools.partial(_on_gradient, param_index))

    model.all_reduce_gradients = functools.partial(_all_reduce_gradients, model)

    global _use_hpu_graph
    global _recorded_unfusion_graph
    _use_hpu_graph = use_hpu_graph
    _recorded_unfusion_graph = None

    if len(mark_step_on_gradients) > 0:
        assert not use_hpu_graph, 'HPU Graph in FastDDP cannot be used in conjuction with `mark_step_on_gradients`.'

    return model


def _on_gradient(param_index, grad):
    global _fusion_buffer
    global _fusion_views
    global _visited_gradient_elem_count
    global _mark_step_on_gradients

    view = _fusion_views.get(param_index, None)
    if view is None:
        grad_numel = torch.numel(grad)
        view = _fusion_buffer[_visited_gradient_elem_count:_visited_gradient_elem_count+grad_numel].reshape(grad.shape)
        _fusion_views[param_index] = view
        _visited_gradient_elem_count += grad_numel
        assert _visited_gradient_elem_count <= torch.numel(_fusion_buffer)

    if param_index in _mark_step_on_gradients:
        mark_step()

    view.copy_(grad if grad.dtype == _fusion_buffer.dtype else grad.to(_fusion_buffer.dtype), non_blocking=True)


def _all_reduce_gradients(model: torch.nn.Module, unfuse_gradients: bool = True):
    global _visited_gradient_elem_count
    global _fusion_buffer
    global _fusion_views
    global _all_reduce_group_size

    mark_step()

    grads = [param.grad for param in model.parameters() if param.grad is not None]

    clip_global_grad_norm = torch.ops.hpu.optimizer_lamb_fused_norm(grads, 1.0)
    _fusion_buffer.div_((clip_global_grad_norm * _all_reduce_group_size).to(_fusion_buffer.dtype))

    torch.distributed.all_reduce(_fusion_buffer, group=torch.distributed.group.WORLD, async_op=True)

    if unfuse_gradients:
        global _use_hpu_graph
        if _use_hpu_graph:
            global _recorded_unfusion_graph
            if _recorded_unfusion_graph is None:
                import habana_frameworks.torch as ht
                _recorded_unfusion_graph = ht.hpu.HPUGraph()
                s = ht.hpu.Stream()
                with ht.hpu.stream(s):
                    _recorded_unfusion_graph.capture_begin()
                    _unfuse_gradients(grads)
                    _recorded_unfusion_graph.capture_end()

            _recorded_unfusion_graph.replay()

        else:
            mark_step()
            _unfuse_gradients(grads)

def _unfuse_gradients(grads):
    global _fusion_views
    for grad_index, grad in enumerate(grads):
        view = _fusion_views[grad_index]
        grad.copy_(view if view.dtype == grad.dtype else view.to(grad.dtype), non_blocking=True)
