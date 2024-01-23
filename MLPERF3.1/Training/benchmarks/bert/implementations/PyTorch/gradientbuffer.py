# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company

"""
    Gradient Buffer for Intel® Gaudi® AI Accelerator

    A data structure consisting of a pair of fusion buffers storing gradients.
    The primary fusion buffer keeps accumulated gradients and is a subject for all-reduce operation.
    The secondary fusion buffer is an intermediate buffer where gradient clipping normalization takes place.
"""

import functools

import torch
import torch.nn

import habana_frameworks.torch.core as htcore

DEFAULT_CLIP_VALUE = 1.0


class GradientBuffer:
    def __init__(self,
                 model: torch.nn.Module,
                 hook_autograd: bool,
                 dtype: torch.dtype = torch.float32,
                 device: str = 'hpu:0',
                 allow_accumulation: bool = True,
                 max_grad_norm: float = DEFAULT_CLIP_VALUE,
                 grad_div_factor: float = 1.0):
        self.model = model
        self.dtype = dtype
        self.device = device
        self.allow_accumulation = allow_accumulation
        self.max_grad_norm = max_grad_norm
        self.grad_div_factor = torch.tensor(
            [grad_div_factor], device=device) if grad_div_factor != 1.0 else None

        assert self.dtype in [
            torch.float32, torch.bfloat16], f'GradientBuffer supports only float32 and bfloat16 dtypes, but dtype is: {self.dtype}'

        self.gradient_elem_count = 0
        for param in model.parameters():
            self.gradient_elem_count += torch.numel(param)

        self.visited_gradient_elem_count = 0

        self.is_empty = True

        self.fusion_buffer = torch.zeros(
            size=(self.gradient_elem_count,), dtype=self.dtype, device=self.device)
        self.gradient_views = {}

        if self.allow_accumulation:
            self.fusion_aux_buffer = torch.zeros(
                size=(self.gradient_elem_count,), dtype=self.dtype, device=self.device)
            self.gradient_aux_views = {}
        else:
            self.fusion_aux_buffer = None
            self.gradient_aux_views = None

        if hook_autograd:
            def on_gradient(param_index, grad):
                self._fuse_gradient(param_index, grad)
                if param_index == 0:
                    self._finish_fusing_gradients()

            for param_index, param in enumerate(model.parameters()):
                param.register_hook(
                    functools.partial(on_gradient, param_index))

    def mark_empty(self):
        self.is_empty = True

    def fuse_gradients(self):
        grads = [param.grad for param in self.model.parameters()]
        for grad_index, grad in enumerate(grads):
            self._fuse_gradient(grad_index, grad)
        self._finish_fusing_gradients()

    def _fuse_gradient(self,
                       grad_index: int,
                       grad: torch.Tensor):
        use_aux = not self.is_empty and self.max_grad_norm is not None

        if use_aux:
            assert self.allow_accumulation, 'GradientBuffer has been created with allow_accumulation=False, but tries to fuse the same gradient for the second time.'
            buffer = self.fusion_aux_buffer
            views = self.gradient_aux_views
        else:
            buffer = self.fusion_buffer
            views = self.gradient_views

        view = views.get(grad_index, None)
        if view is None:
            grad_numel = torch.numel(grad)

            view = self.fusion_buffer[self.visited_gradient_elem_count:self.visited_gradient_elem_count +
                                      grad_numel].reshape(grad.shape)
            self.gradient_views[grad_index] = view

            if self.allow_accumulation:
                aux_view = self.fusion_aux_buffer[self.visited_gradient_elem_count:self.visited_gradient_elem_count +
                                                  grad_numel].reshape(grad.shape)
                self.gradient_aux_views[grad_index] = aux_view

            self.visited_gradient_elem_count += grad_numel
            assert self.visited_gradient_elem_count <= torch.numel(
                buffer), 'Exceeding Fusion Buffer size.'

            if use_aux:
                view = aux_view

        view.copy_(grad if grad.dtype == self.dtype else grad.to(
            self.dtype), non_blocking=True)

    def _finish_fusing_gradients(self):
        use_aux = not self.is_empty and self.max_grad_norm is not None

        if use_aux:
            assert self.allow_accumulation, 'GradientBuffer has been created with allow_accumulation=False, but tries to fuse the same gradient for the second time.'
            buffer = self.fusion_aux_buffer
        else:
            buffer = self.fusion_buffer

        # Normalize the gradient buffer.
        grad_denom = None

        if self.max_grad_norm is not None:
            grad_denom = torch.ops.hpu.optimizer_lamb_fused_norm([buffer], self.max_grad_norm)

        if self.grad_div_factor is not None:
            if grad_denom is None:
                grad_denom = self.grad_div_factor
            else:
                grad_denom *= self.grad_div_factor

        if grad_denom is not None:
            buffer.div_(grad_denom.to(self.dtype))

        # If processed gradients are in auxiliary buffer, accumulate it to the primary buffer.
        if use_aux:
            self.fusion_buffer.add_(self.fusion_aux_buffer)

        self.is_empty = False

    def unfuse_gradients(self,
                         mark_empty: bool = True):
        assert not self.is_empty

        grads = [param.grad for param in self.model.parameters()]

        for grad_index, grad in enumerate(grads):
            view = self.gradient_views[grad_index]
            grad.copy_(view if view.dtype == grad.dtype else view.to(
                grad.dtype), non_blocking=True)

        if mark_empty:
            self.mark_empty()

    def all_reduce(self):
        torch.distributed.all_reduce(
            self.fusion_buffer, group=torch.distributed.group.WORLD, async_op=True)
