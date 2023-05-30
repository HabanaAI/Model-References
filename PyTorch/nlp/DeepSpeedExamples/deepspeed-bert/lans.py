# Copyright (c) 2021, Habana Labs Ltd.  All rights reserved.
# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MIT License
#
# Copyright (c) 2019 cybertronai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from torch.optim import Optimizer


class LANS(Optimizer):
    """Implements a pure pytorch variant of LANS

    LANS was proposed in `Accelerated Large Batch Optimization of BERT Pretraining in 54 minutes`.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its norm. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        grad_averaging (bool, optional): whether apply (1-beta2) to grad when
            calculating running averages of gradient. (default: True)
        set_grad_none (bool, optional): whether set grad to None when zero_grad()
            method is called. (default: True)
        max_grad_norm (float, optional): value used to clip global grad norm
            (default: 1.0)
        adjust_step (boolean, optional): Decrement step for bias correction (needed for
            Zero01 optimization when using DeepSpeed).

        Accelerated Large Batch Optimization of BERT Pretraining in 54 minutes:
        https://arxiv.org/pdf/2006.13484.pdf
    """
    def __init__(self, params, lr=1e-3, bias_correction=True,
                 betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01,
                 grad_averaging=True, set_grad_none=True,
                 max_grad_norm=0.0, adjust_step=False):
        defaults = dict(lr=lr, bias_correction=bias_correction,
                        betas=betas, eps=eps, weight_decay=weight_decay,
                        grad_averaging=grad_averaging,
                        max_grad_norm=max_grad_norm)
        super().__init__(params, defaults)
        self.set_grad_none = set_grad_none
        self.adjust_step = adjust_step

    def zero_grad(self, set_to_none=False):
        super(LANS, self).zero_grad(set_to_none=(set_to_none or self.set_grad_none))

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        device = None

        loss = None
        if closure is not None:
            loss = closure()

        # if max_grad_enabled is enabled (> 0), calculate gradient clipping factor
        clip_global_grad_norm = 1.0
        max_grad_norm = self.defaults['max_grad_norm']
        if max_grad_norm > 0.:
            global_grad_norm = None
            for group in self.param_groups:
                for p in group['params']:
                    if device is None:
                       device = p.device
                       global_grad_norm = torch.zeros(1, device=device)
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError('LANS does not support sparse gradients')
                    global_grad_norm.add_(grad.pow(2).sum())

            assert device != None, "There are no params in param_groups"
            assert global_grad_norm != None, "global_grad_norm is None"
            global_grad_norm_ = torch.sqrt(global_grad_norm)
            clip_global_grad_norm = torch.maximum(global_grad_norm_.div(max_grad_norm), torch.ones(1, device=device))

        for group in self.param_groups:
            bias_correction = 1 if group['bias_correction'] else 0
            beta1, beta2 = group['betas']
            grad_averaging = 1 if group['grad_averaging'] else 0
            if grad_averaging:
                beta3 = 1 - beta1
            else:
                beta3 = 1.0

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            step_size = group['lr']

            if bias_correction:
                if self.adjust_step:
                    bias_correction_step = group['step'] - 1 if (group['step'] > 1) else group['step']
                else:
                    bias_correction_step = group['step']

                bias_correction1 = 1 - beta1 ** bias_correction_step
                bias_correction2 = 1 - beta2 ** bias_correction_step
            else:
                bias_correction1, bias_correction2 = 1.0, 1.0

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data.div_(clip_global_grad_norm)

                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg_, exp_avg_sq_ = state['exp_avg'], state['exp_avg_sq']

                # Scale the gradient by its norm
                grad_norm = grad.norm(2.0)
                scaled_grad = grad / (grad_norm+group['eps']) if (grad_norm != 0) else grad

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg_.mul_(beta1).add_(scaled_grad, alpha=beta3)
                # v_t
                exp_avg_sq_.mul_(beta2).addcmul_(scaled_grad, scaled_grad, value=(1 - beta2))

                # update the momentum values and create clones to avoid modifying runner stats
                exp_avg = exp_avg_.div(bias_correction1)
                exp_avg_sq = exp_avg_sq_.div(bias_correction2)

                # || w_t ||
                weight_norm = p.data.norm(2.0)

                # r_t
                exp_avg_sq_sqrt = torch.sqrt(exp_avg_sq)
                exp_avg_sq_sqrt.add_(group['eps'], alpha=1.0)
                adam_step = exp_avg.div_(exp_avg_sq_sqrt)

                # c_t
                c_step = scaled_grad.div_(exp_avg_sq_sqrt)

                # TODO: in case decay=0, evaluate using if to skip add_ operation
                #  vs. always performing add_ where we multiply by decay=0
                # if group['weight_decay'] != 0:
                adam_step.add_(p.data, alpha=group['weight_decay'])
                c_step.add_(p.data, alpha=group['weight_decay'])

                # || r_t || , || c_t ||
                adam_norm = adam_step.norm(2.0)
                c_norm = c_step.norm(2.0)

                # TODO: try avoiding conditional statement in the graph
                ratio_m = step_size * (weight_norm / adam_norm) if (adam_norm != 0 and weight_norm != 0) else step_size
                ratio_g = step_size * (weight_norm / c_norm) if (c_norm != 0 and weight_norm != 0) else step_size

                trust_ratio_m = adam_step * ratio_m * beta1
                trust_ratio_g = c_step * ratio_g * beta3

                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['c_norm'] = c_norm

                alpha = -(trust_ratio_m + trust_ratio_g)
                p.data.add_(alpha)

        return loss
