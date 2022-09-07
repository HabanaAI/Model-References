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
import numpy as np
from torch.optim import Optimizer


class NVLAMB_EXP(Optimizer):

    """Implements a pure pytorch variant of FuseLAMB (NVLAMB variant) optimizer from apex.optimizers.FusedLAMB
    reference: https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/Transformer-XL/pytorch/lamb.py

    LAMB was proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its norm. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            NOT SUPPORTED now! (default: False)
        adam_w_mode (boolean, optional): Apply L2 regularization or weight decay
            True for decoupled weight decay(also known as AdamW) (default: True)
        grad_averaging (bool, optional): whether apply (1-beta2) to grad when
            calculating running averages of gradient. (default: True)
        set_grad_none (bool, optional): whether set grad to None when zero_grad()
            method is called. (default: True)
        max_grad_norm (float, optional): value used to clip global grad norm
            (default: 1.0)
        use_nvlamb (boolean, optional): Apply adaptive learning rate to 0.0
            weight decay parameter (default: False)
        adjust_step (boolean, optional): Decrement step for bias correction (needed for
            Zero01 optimization when using DeepSpeed).
        max_trust (float, optional): maximum value for trust factor, 0 = no max
        min_trust (float, optional): minimum value for trust factor, 0 = no min
        use_ds_lamb_bias_correction (boolean, optional): Calculate bias correction factor once per group
            and apply it on alpha. NOTE that this mode modifies the behavior of weight decay.
            This behavior is replicated from DeepSpeed FusedLamb to enable to compare (default: False)

    .. _Large Batch Optimization for Deep Learning - Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """
    def __init__(self, params, lr=1e-3, bias_correction=True,
                 betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01,
                 amsgrad=False, adam_w_mode=True,
                 grad_averaging=True, set_grad_none=True,
                 max_grad_norm=1.0, use_nvlamb=False, fused=False, adjust_step=False,
                 max_trust=0., min_trust=0., use_ds_lamb_bias_correction=False):
        if amsgrad:
            raise RuntimeError('NVLAMB_EXP does not support the AMSGrad variant.')
        defaults = dict(lr=lr, bias_correction=bias_correction,
                        betas=betas, eps=eps, weight_decay=weight_decay,
                        grad_averaging=grad_averaging,
                        max_grad_norm=max_grad_norm,
                        max_trust=max_trust, min_trust=min_trust,
                        use_ds_lamb_bias_correction=use_ds_lamb_bias_correction)
        super().__init__(params, defaults)
        self.fused = fused
        self.adam_w_mode = 1 if adam_w_mode else 0  # dummy for now, always use adam_w mode (wd is excluded from EMA)
        self.set_grad_none = set_grad_none
        self.use_nvlamb = use_nvlamb
        self.adjust_step = adjust_step
        self.use_ds_lamb_bias_correction = use_ds_lamb_bias_correction
        assert bias_correction or not use_ds_lamb_bias_correction, \
            'use_ds_lamb_bias_correction=True requires bias_correction=True'

    def zero_grad(self):
        if self.set_grad_none:
            for group in self.param_groups:
                for p in group['params']:
                    p.grad = None
        else:
            super(NVLAMB_EXP, self).zero_grad()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        device = self.param_groups[0]["params"][0].device

        loss = None
        if closure is not None:
            loss = closure()

        # if max_grad_enabled is enabled (> 0), calculate gradient clipping factor
        clip_global_grad_norm = 1.0
        max_grad_norm = self.defaults['max_grad_norm']
        if max_grad_norm > 0.:
            global_grad_norm = torch.zeros(1, device=device)
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instead.')
                    global_grad_norm.add_(grad.pow(2).sum())

            global_grad_norm_ = torch.sqrt(global_grad_norm)

            if global_grad_norm_ > max_grad_norm:
                clip_global_grad_norm = global_grad_norm_ / max_grad_norm

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
                # TODO (SW-84908): (Assi) To verify if the below is correct for ZerO1 and Zero0.
                if self.adjust_step:
                    bias_correction_step = group['step'] - 1 if (group['step'] > 1) else group['step']
                else:
                    bias_correction_step = group['step']
                bias_correction1 = 1 - beta1 ** bias_correction_step
                bias_correction2 = 1 - beta2 ** bias_correction_step
            else:
                bias_correction1, bias_correction2 = 1.0, 1.0

            bias_factor = np.sqrt(bias_correction2) / bias_correction1 if self.use_ds_lamb_bias_correction else 1.0

            max_trust = group['max_trust']
            min_trust = group['min_trust']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.div_(clip_global_grad_norm)
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg_, exp_avg_sq_ = state['exp_avg'], state['exp_avg_sq']

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg_.mul_(beta1).add_(grad, alpha=beta3)
                # v_t
                exp_avg_sq_.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # create clones to avoid modifying runner stats
                exp_avg = exp_avg_.div(bias_correction1) \
                    if not self.use_ds_lamb_bias_correction else exp_avg_.clone()
                exp_avg_sq = exp_avg_sq_.div(bias_correction2) \
                    if not self.use_ds_lamb_bias_correction else exp_avg_sq_.clone()

                # || w_t ||
                weight_norm = p.data.norm(2.0)
                # u_t
                exp_avg_sq_sqrt = torch.sqrt(exp_avg_sq)
                adam_step = exp_avg.div_(exp_avg_sq_sqrt.add_(group['eps']))
                if group['weight_decay'] != 0:
                    adam_step.add_(p.data, alpha=group['weight_decay'])
                # || u_t ||
                adam_norm = adam_step.norm(2.0)
                if (group['weight_decay'] != 0 or self.use_nvlamb) and adam_norm > 0 and weight_norm > 0:
                    trust_ratio = weight_norm / adam_norm
                    trust_ratio = trust_ratio.item()
                else:
                    trust_ratio = 1

                trust_ratio = min(trust_ratio, max_trust) if max_trust != 0. else trust_ratio
                trust_ratio = max(trust_ratio, min_trust) if min_trust != 0. else trust_ratio

                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio

                alpha = -step_size * trust_ratio * bias_factor
                adam_step2 = adam_step * alpha
                p.data.add_(adam_step2)
