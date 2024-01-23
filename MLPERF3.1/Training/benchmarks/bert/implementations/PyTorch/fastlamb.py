# autopep8: off
#
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
#   __    __       __       _______       __      _____  ___        __           ___            __       ___      ___  _______          ______    _______  ___________  __     ___      ___   __   ________    _______   _______
#  /" |  | "\     /""\     |   _  "\     /""\    (\"   \|"  \      /""\         |"  |          /""\     |"  \    /"  ||   _  "\        /    " \  |   __ "\("     _   ")|" \   |"  \    /"  | |" \ ("      "\  /"     "| /"      \
# (:  (__)  :)   /    \    (. |_)  :)   /    \   |.\\   \    |    /    \        ||  |         /    \     \   \  //   |(. |_)  :)      // ____  \ (. |__) :))__/  \\__/ ||  |   \   \  //   | ||  | \___/   :)(: ______)|:        |
#  \/      \/   /' /\  \   |:     \/   /' /\  \  |: \.   \\  |   /' /\  \       |:  |        /' /\  \    /\\  \/.    ||:     \/      /  /    ) :)|:  ____/    \\_ /    |:  |   /\\  \/.    | |:  |   /  ___/  \/    |  |_____/   )
#  //  __  \\  //  __'  \  (|  _  \\  //  __'  \ |.  \    \. |  //  __'  \       \  |___    //  __'  \  |: \.        |(|  _  \\     (: (____/ // (|  /        |.  |    |.  |  |: \.        | |.  |  //  \__   // ___)_  //      /
# (:  (  )  :)/   /  \\  \ |: |_)  :)/   /  \\  \|    \    \ | /   /  \\  \     ( \_|:  \  /   /  \\  \ |.  \    /:  ||: |_)  :)     \        / /|__/ \       \:  |    /\  |\ |.  \    /:  | /\  |\(:   / "\ (:      "||:  __   \
#  \__|  |__/(___/    \___)(_______/(___/    \___)\___|\____\)(___/    \___)     \_______)(___/    \___)|___|\__/|___|(_______/       \"_____/ (_______)       \__|   (__\_|_)|___|\__/|___|(__\_|_)\_______) \_______)|__|  \___)
#
""" Highly-optimized sharded LAMB for PyTorch MLPerf BERT pre-training on Intel® Gaudi® AI Accelerator

    The paper: https://arxiv.org/abs/1904.00962
"""

import habana_frameworks.torch as ht
import torch
from torch.optim import Optimizer
from utils import get_rank, get_world_size


class FastLamb(Optimizer):
    def __init__(self,
                 params,
                 lr: float = 1e-3,
                 bias_correction: bool = True,
                 betas=(0.9, 0.999),
                 eps: float = 1e-6,
                 weight_decay: float = 0.01,
                 sharded: bool = False,
                 use_lowp: bool = True):

        super().__init__(params, dict(
            lr=lr,
            bias_correction=bias_correction,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            grad_averaging=True))

        self.graph1 = None
        self.graph2 = None
        self.step_count = 0
        self.sharded = sharded
        self.use_lowp = use_lowp

        self.external_param_to_grad_map = None  # may be manually set before first step() to instruct the logic to fetch gradients directly from external source to speed up the processing

    def step(self):
        """ Performs an optimization step, where model parameters are updated based on the gradient information and previous responses.
        """

        # If this is the first step, initialize all the auxiliary tensors.
        if self.step_count == 0:
            self._prepare_for_first_step()

        # Increment the step counter.
        self.step_count += 1
        for group in self.param_groups:
            group['lr_t'].copy_(group['lr'], non_blocking=True)  # set by the learning rate scheduler
            group['step'] = self.step_count  # read by the learning rate scheduler

        # Perform the first phase of the optimizer, where exponential moving averages are updated and Adam steps computed along additional information.
        if self.graph1 is None:
            self.graph1 = ht.hpu.HPUGraph()
            s = ht.hpu.Stream()
            with ht.hpu.stream(s), torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=False):
                self.graph1.capture_begin()

                beta1, beta2 = self.param_groups[0]['betas']
                self.beta1_decayed.mul_(beta1)
                self.beta2_decayed.mul_(beta2)

                adam_step_unbias_factor = torch.sqrt(1.0 - self.beta2_decayed) / (1.0 - self.beta1_decayed)

                if self.use_lowp:
                    adam_step_unbias_factor = adam_step_unbias_factor.to(torch.bfloat16)

                total_index = 0

                for group in self.param_groups:
                    assert group['betas'][0] == beta1
                    assert group['betas'][1] == beta2
                    eps = group['eps']
                    weight_decay = group['weight_decay']

                    for param, sharded, grad, exp_avg, exp_avg_sq, adam_step, adam_shard in zip(group['param_shards'], group['sharded'], group['grads'], group['exp_avg_list'], group['exp_avg_sq_list'], group['adam_step_list'], group['adam_shard_list']):
                        if self.use_lowp and grad.dtype != torch.bfloat16:
                            grad = grad.to(torch.bfloat16)
                        exp_avg.copy_((exp_avg * beta1) + grad * (1.0 - beta1), non_blocking=True)
                        exp_avg_sq.copy_((exp_avg_sq * beta2) + (grad * grad) * (1.0 - beta2), non_blocking=True)

                        param_data = param.data
                        if self.use_lowp and param.data.dtype != torch.bfloat16:
                            param_data = param_data.to(torch.bfloat16)

                        if weight_decay != 0.0:
                            adam_step.copy_(exp_avg * torch.rsqrt(exp_avg_sq + eps*eps) * adam_step_unbias_factor + param_data * weight_decay, non_blocking=True)
                        else:
                            adam_step.copy_(exp_avg * torch.rsqrt(exp_avg_sq + eps*eps) * adam_step_unbias_factor, non_blocking=True)

                        if self.sharded:
                            scale_factor = 1.0 if sharded else (1.0 / get_world_size())
                            self.w_norm_buffer[total_index] = torch.sum(param_data * param_data) * scale_factor
                            self.adam_norm_buffer[total_index] = torch.sum(adam_step * adam_step) * scale_factor
                        else:
                            self.w_norm_buffer[total_index] = param_data.norm(2.0)
                            self.adam_norm_buffer[total_index] = adam_step.norm(2.0)

                        if adam_shard is not None:
                            adam_shard.copy_(adam_step, non_blocking=True)

                        total_index += 1

                self.graph1.capture_end()

        self.graph1.replay()

        if self.sharded:
            torch.distributed.all_reduce(self.w_norm_buffer, async_op=True)
            torch.distributed.all_reduce(self.adam_norm_buffer, async_op=True)
            torch.distributed.all_gather_into_tensor(self.adam_full_buffer, self.adam_shard_buffer, async_op=True)

        # Perform the second phase of the optimizer, where model parameters are updated based on the previously computed information.
        if self.graph2 is None:
            self.graph2 = ht.hpu.HPUGraph()
            s = ht.hpu.Stream()
            with ht.hpu.stream(s), torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=False):
                self.graph2.capture_begin()

                lr = self.param_groups[0]['lr_t']
                if self.sharded:
                    step_factors = -lr * torch.where(torch.logical_and(self.w_norm_buffer > 0.0, self.adam_norm_buffer > 0.0), torch.sqrt(self.w_norm_buffer / self.adam_norm_buffer), 1.0)
                else:
                    step_factors = -lr * torch.where(torch.logical_and(self.w_norm_buffer > 0.0, self.adam_norm_buffer > 0.0), self.w_norm_buffer / self.adam_norm_buffer, 1.0)

                total_index = 0

                for group in self.param_groups:
                    assert lr == group['lr_t']
                    eps = group['eps']
                    weight_decay = group['weight_decay']

                    for param, param_sharded, adam_full in zip(group['params'], group['sharded'], group['adam_full_list']):
                        if param_sharded:
                            adam_full = torch.cat(adam_full, dim=0)
                        param.data.add_(adam_full * step_factors[total_index])

                        total_index += 1

                self.graph2.capture_end()

        self.graph2.replay()

    def _prepare_for_first_step(self):
        """ Called by the first step() to create all the auxiliary/intermediate tensors necessary for the logic.
        """

        assert self.step_count == 0
        device = torch.device('hpu')
        world_size = get_world_size()
        rank = get_rank()

        # Initialize one-element placeholder tensors for BETA1^STEP and BETA2^STEP.
        beta1, beta2 = self.param_groups[0]['betas']
        self.beta1_decayed = torch.full(size=(1,), fill_value=1.0, dtype=torch.float32, device=device)
        self.beta2_decayed = torch.full(size=(1,), fill_value=1.0, dtype=torch.float32, device=device)

        aux_dtype = torch.float32 if not self.use_lowp else torch.bfloat16

        # Initilize one-element placeholder tensors for weight norms and Adam norms min-max clamping.
        eps = self.param_groups[0]['eps']

        total_param_count = 0
        total_shard_numel = 0

        for group in self.param_groups:
            assert group['betas'][0] == beta1
            assert group['betas'][1] == beta2
            assert group['eps'] == eps

            param_count = len(group["params"])

            group['param_shards'] = []  # a list of model's (optionally) sharded parameters
            group['sharded'] = []  # a list of boolean indicators whether the param/grad is sharded
            group['grads'] = []  # a list of model's gradients
            group['exp_avg_list'] = []  # a list of exponential BETA1-moving averages of gradients
            group['exp_avg_sq_list'] = []  # a list of exponential BETA2-moving averages of squared gradients
            group['adam_step_list'] = []  # a list of Adam steps
            group['lr_t'] = torch.zeros(size=(1,), dtype=aux_dtype, device=device)  # a learning rate single-element placeholder tensor
            group['adam_shard_ranges'] = []  # a list of fusion linear ranges of Adam step shards

            for param in group['params']:
                param_data = param.data
                param_sharded = self.sharded and (param_data.shape[0] % world_size == 0 and param_data.shape[0] // world_size >= 8)
                group['sharded'].append(param_sharded)
                if param_sharded:
                    shard_width = param_data.shape[0] // world_size
                    param_data = param_data[rank*shard_width:(rank+1)*shard_width]
                    group['adam_shard_ranges'].append((total_shard_numel, total_shard_numel + torch.numel(param_data)))
                    shard_align = 256
                    total_shard_numel += (torch.numel(param_data)+shard_align-1)//shard_align*shard_align
                else:
                    group['adam_shard_ranges'].append(None)
                group['param_shards'].append(param_data)

                if self.external_param_to_grad_map is not None:
                    grad = self.external_param_to_grad_map[param]
                else:
                    grad = param.grad
                assert grad is not None

                if param_sharded:
                    grad = grad[rank*shard_width:(rank+1)*shard_width]

                assert grad.shape == param_data.shape

                group['grads'].append(grad)
                group['exp_avg_list'].append(torch.zeros(size=grad.shape, dtype=aux_dtype, device=device))
                group['exp_avg_sq_list'].append(torch.zeros(size=grad.shape, dtype=aux_dtype, device=device))
                group['adam_step_list'].append(torch.zeros(size=grad.shape, dtype=aux_dtype, device=device))

            total_param_count += param_count

        self.w_norm_buffer = torch.zeros(size=(total_param_count,), dtype=aux_dtype, device=device)  # fused norms of model parameters
        self.adam_norm_buffer = torch.zeros(size=(total_param_count,), dtype=aux_dtype, device=device)  # fused norms of Adam steps

        if self.sharded:
            assert total_shard_numel > 0, 'FastLamb works in sharded mode, but there are no shardeables.'
            self.adam_shard_buffer = torch.zeros(size=(total_shard_numel,), dtype=aux_dtype, device=device)  # ???
            self.adam_full_buffer = torch.zeros(size=(total_shard_numel * get_world_size(),), dtype=aux_dtype, device=device)  # ???

        for group in self.param_groups:
            group['adam_shard_list'] = []  # a list of views within adam_shard_buffer containing sharded Adam steps (only)
            group['adam_full_list'] = []  # a list of views within adam_full_buffer containing full Adam steps

            for param, sharded, param_shard, adam_shard_range, adam_step in zip(group['params'], group['sharded'], group['param_shards'], group['adam_shard_ranges'], group['adam_step_list']):
                if sharded:
                    adam_shard = self.adam_shard_buffer[adam_shard_range[0]:adam_shard_range[1]].reshape(param_shard.shape)
                else:
                    adam_shard = None
                group['adam_shard_list'].append(adam_shard)

                if sharded:
                    views = []
                    for rank in range(get_world_size()):
                        rank_base_index = rank * total_shard_numel
                        views.append(self.adam_full_buffer[rank_base_index+adam_shard_range[0]:rank_base_index+adam_shard_range[1]].reshape(adam_shard.shape))
                    group['adam_full_list'].append(views)
                else:
                    group['adam_full_list'].append(adam_step)
