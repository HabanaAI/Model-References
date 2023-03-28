###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import torch
import collections
from functools import partial
from contextlib import contextmanager
from torch.functional import Tensor
from os import makedirs
from os.path import join


class TensorLogger:
    """ Records nn.Module's activations and gradients tensors

        Description:
        Records up to max_iterations (if 0, recording is disabled).
        If log_activations_enabled, nn.Module's activations are recorded during forward.
        If log_grads_enabled, nn.Module's gradients are recorded during back propagation.
        If log_inputs_enabled, model inputs are recorded.

        Usage:
            Integrated within the training loop:
                tensor_logger = TensorLogger(model, max_iterations=2)

                for i, samples in enumerate(data_loader) # training loop
                    with tensor_logger.log_iteration(i):
                        # run forward/backward iteration

                tensor_logger.save(filename)

            Another alternative:
                tensor_logger = TensorLogger(model, max_iterations=2)

                for i, samples in enumerate(data_loader) # training loop
                    with tensor_logger:
                        tensor_logger.set_iteration(i)
                        # run forward/backward iteration

                tensor_logger.save(filename)

        Implementation notes:
            forward/backward activations/gradients are collected using nn.Module hooks.
            However, model inputs are collected by overloading model.forward() method.
            Model inputs can't be collected using the hooks since the hooks only provide
            inputs and do not provide kwargs, if exist, of the forward method.
    """
    def __init__(self,
                 model,
                 max_iterations=0,
                 log_activations_enabled=False,
                 log_grads_enabled=False,
                 log_inputs_enabled=False,
                 prefix=None):

        # for now, no support for virtual pipeline (interleaved)
        if isinstance(model, list):
            assert len(model) == 1, 'No support for list of multiple models (len={})'.format(len(model))
            model = model[0]

        self.model = model
        self.max_iterations = max_iterations
        self.log_activations_enabled = log_activations_enabled
        self.log_grads_enabled = log_grads_enabled
        self.log_inputs_enabled = log_inputs_enabled
        self.prefix = 'model' if prefix is None else prefix

        # captured tensors are saved in the following hierarchy:
        #   {
        #        iteration:  {               # iteration number
        #            tensor_type:  {         # fwd_act/bwd_grad_in/bwd_grad_out
        #                name: [tensors]     # tensor name's tensors. list is required due to e.g. grad accumulation
        #            }
        #        }
        #    }
        class IterData(dict):
            def __init__(self):
                super(IterData, self).__init__()
                self['fwd_act'] = collections.defaultdict(list)
                self['bwd_grad_in'] = collections.defaultdict(list)
                self['bwd_grad_out'] = collections.defaultdict(list)
                self['model_inputs'] = collections.defaultdict(list)

        self.data = collections.defaultdict(IterData)
        self.active = False
        self.current_iteration = 0
        self.fwd_handles = []
        self.bwd_handles = []

    def _fqn(self, name):
        return '.'.join([self.prefix, name]) if name else self.prefix

    def set_iteration(self, iteration):
        self.current_iteration = iteration

    def get_num_recorded_iterations(self):
        return len(self.data)

    @contextmanager
    def log_iteration(self, iteration):
        self.current_iteration = iteration
        self._enable()
        yield self
        self._disable()

    def __enter__(self):
        self._enable()
        return self

    def __exit__(self):
        self._disable()

    def clear(self):
        self.data.clear()

    def save(self, filename, do_clear=True):
        def convert_for_pickle(obj):
            if isinstance(obj, dict):
                return {k: convert_for_pickle(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_pickle(e) for e in obj]
            elif isinstance(obj, tuple):
                return tuple([convert_for_pickle(e) for e in obj])
            else:
                if isinstance(obj, Tensor):
                    return obj.detach().cpu()
                else:
                    return obj


        data = convert_for_pickle(self.data)
        torch.save(data, filename)
        self.clear() if do_clear else None

    def _enable(self):
        if not self.active and self.get_num_recorded_iterations() < self.max_iterations:
            self.active = True
            self._enable_log_grads() if self.log_grads_enabled else None
            self._enable_log_activations() if self.log_activations_enabled else None
            self._enable_log_inputs() if self.log_inputs_enabled else None

    def _disable(self):
        if self.active:
            self.active = False
            self._disable_log_grads()
            self._disable_log_activations()
            self._disable_log_inputs()

    @staticmethod
    def _extract_tensors(t):
        if t is None:
            return None
        elif isinstance(t, int):
            return torch.tensor(t)
        elif isinstance(t, torch.Tensor):
            return t.detach().contiguous()
        elif isinstance(t, list):
            return [TensorLogger._extract_tensors(e) for e in t]
        elif isinstance(t, tuple):
            return tuple(TensorLogger._extract_tensors(e) for e in t)
        elif isinstance(t, dict):
            return {k: TensorLogger._extract_tensors(v) for k, v in t.items()}
        assert False, 'Unsupported type: {}'.format(type(t))

    def _save_fwd_activation(self, name, _mod, _inp, out):
        fwd_act = self._extract_tensors(out)
        self.data[self.current_iteration]['fwd_act'][name].append(fwd_act)

    def _save_bwd_grads(self, name, _mod, grad_input, grad_output):
        grad_in = self._extract_tensors(grad_input)
        grad_out = self._extract_tensors(grad_output)
        self.data[self.current_iteration]['bwd_grad_in'][name].append(grad_in)
        self.data[self.current_iteration]['bwd_grad_out'][name].append(grad_out)

    def _save_inputs(self, *inp, **kwargs):
        model_inputs = self._extract_tensors(inp)
        model_kwargs = self._extract_tensors(kwargs)
        self.data[self.current_iteration]['model_inputs']['inputs'].append(model_inputs)
        self.data[self.current_iteration]['model_inputs']['kwargs'].append(model_kwargs)

    def _enable_log_grads(self):
        #Revert after [SW-69765] is fixed
        full_bwd_hook_supported = False
        for name, m in self.model.named_modules():
            register_fn = m.register_full_backward_hook if full_bwd_hook_supported else m.register_backward_hook
            h = register_fn(partial(self._save_bwd_grads, self._fqn(name)))
            self.bwd_handles.append(h)

    def _enable_log_activations(self):
        for name, m in self.model.named_modules():
            h = m.register_forward_hook(partial(self._save_fwd_activation, self._fqn(name)))
            self.fwd_handles.append(h)

    def _enable_log_inputs(self):
        def wrapped_forward(*inputs, **kwargs):
            self._save_inputs(*inputs, **kwargs)
            return self.model.original_forward__(*inputs, **kwargs)

        self.model.original_forward__ = self.model.forward
        self.model.forward = wrapped_forward

    def _disable_log_grads(self):
        for h in self.bwd_handles:
            h.remove()
        self.bwd_handles = []

    def _disable_log_activations(self):
        for h in self.fwd_handles:
            h.remove()
        self.fwd_handles = []

    def _disable_log_inputs(self):
        if hasattr(self.model, 'original_forward__'):
            self.model.forward = self.model.original_forward__
            del self.model.original_forward__

def save_logged_tensors(tensor_logger: TensorLogger, tensor_logger_path, rank_no, iteration=None):
    if tensor_logger.get_num_recorded_iterations():
        makedirs(tensor_logger_path, exist_ok=True)
        filename = 'tensor_logger_rank_{}'.format(rank_no) + '.pt'
        if iteration is not None:
            filename = 'tensor_logger_rank_{}_iter_{}'.format(rank_no, iteration) + '.pt'
        fullname = join(tensor_logger_path, filename)
        tensor_logger.save(fullname)