# coding=utf-8
# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPT zero-shot evaluation."""

import math
import functools

import torch

from megatron import get_args
from megatron import print_rank_0, is_last_rank
from megatron import get_tokenizer
from megatron import mpu
from megatron.checkpointing import load_checkpoint
from megatron.model import GPTModel
from megatron.training import get_model
from megatron.utils import get_ltor_masks_and_position_ids, unwrap_model
from megatron.p2p_communication import recv_forward, send_forward
from tasks.finetune_utils import build_data_loader

from .datasets import build_dataset

# These are needed to unwrap the model, would be nice to put these in megatron.utils if possible?
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module

def get_model_provider(eval_metric):
    """Based on evaluation metric set the parallel-output flag and
    return the model provider."""

    def model_provider(pre_process=True, post_process=True):
        """Build the model."""

        if eval_metric == 'loss':
            parallel_output = True
        elif eval_metric == 'accuracy':
            parallel_output = False
        else:
            raise NotImplementedError('output type for {} evaluation metric '
                                      'is not supported.'.format(eval_metric))

        print_rank_0('building GPT model ...')
        model = GPTModel(num_tokentypes=0, parallel_output=parallel_output,
                         pre_process=pre_process, post_process=post_process)

        return model

    return model_provider


def process_batch(batch, device):
    """Process batch and produce inputs for the model."""
    args = get_args()
    tokenizer = get_tokenizer()

    loss_mask = batch['pad_mask'].long().to(device=device).contiguous().byte()
    tokens_ = batch['text'].long().to(device=device).contiguous()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and position ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, attention_mask, position_ids, loss_mask


def calculate_metric(output, labels, loss_mask, eval_metric):
    # For loss, return the unreduced loss.
    if eval_metric == 'loss':
        losses = mpu.vocab_parallel_cross_entropy(
            output.contiguous().float(), labels.contiguous())
        loss = torch.sum(
            losses.view(-1) * loss_mask.contiguous().view(-1).float())
        return loss

    # For accuracy, return the number of correctly predicted samples.
    if eval_metric == 'accuracy':
        outputs = torch.argmax(output, -1)
        correct = (outputs == labels).float()
        correct[(1 - loss_mask).bool()] = 1
        correct = correct.prod(-1)
        return correct.sum()

    raise NotImplementedError('Evaluation metric {} is not implemented.'.format(eval_metric))


def forward_step(batch, model, eval_metric):
    """Forward step."""

    # Get the batch.
    tokens, labels, attention_mask, position_ids, loss_mask = process_batch(
        batch, device=torch.device(f'cuda:{torch.cuda.current_device()}'))

    # Tell the model what our actual batch size will be
    args = get_args()
    assert args.micro_batch_size == args.eval_micro_batch_size, \
        "forward_step (zeroshot_gpt) - Unsupported for split micro batch size"
    args.micro_batch_size = len(labels)
    # Next line should be considered once eval_micro_batch_size is supported here
    args.eval_micro_batch_size = args.micro_batch_size

    input_tensor = recv_forward()

    # Forward pass through the model.
    unwrapped_model = unwrap_model(
        model, (torchDDP, LocalDDP, Float16Module))
    unwrapped_model.set_input_tensor(input_tensor)
    output = model(tokens, position_ids, attention_mask)

    send_forward(output)

    if mpu.is_pipeline_last_stage():
        return calculate_metric(output, labels, loss_mask, eval_metric)

    return None


def evaluate(data_loader, model, eval_metric):
    """Evaluation."""
    args = get_args()

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_output = 0.0
    with torch.no_grad():
        # For all the batches in the dataset.
        for iteration, batch in enumerate(data_loader):
            if iteration % args.log_interval == 0:
                print_rank_0('> working on iteration: {}'.format(iteration))
            # Forward evaluation.
            output = forward_step(batch, model, eval_metric)

            # Reduce across processes.
            if mpu.is_pipeline_last_stage():
                torch.distributed.all_reduce(output,
                                             group=mpu.get_data_parallel_group())

                total_output += output

    return total_output


class PeekableIterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.next_item = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_item is not None:
            item = self.next_item
            self.next_item = None
            return item
        else:
            return next(self.iterator)

    def peek(self):
        if self.next_item is None:
            self.next_item = next(self.iterator)
        return self.next_item


def evaluate_3d(loader, model, eval_metric, do_print):
    args = get_args()

    total_iters = len(loader)
    peekable_loader = PeekableIterator(iter(loader))

    dp_world_size = mpu.get_data_parallel_world_size()

    total_output, total_tokens = 0.0, 0
    last_batch_size = loader.batch_size
    for i in range(total_iters):
        batch = peekable_loader.peek()

        # We create the data_loader with drop_last=False
        # This can cause the last batch to be smaller than loader.batch_size
        # However, Megatron caches the size of the batch
        # Therefore, we detect that the current batch size has changed and reset the cache
        # In addition, Pipeline model engine calculates total_loss aggregated over micro batches.
        # However, total_loss has no meaning for eval, yet being calculated.
        # Reset total_loss to avoid above similar batch size issue
        batch_size = batch['text'].shape[0]
        if batch_size != last_batch_size:
            model.reset_activation_shape()
            mpu.data.reset_cached_broadcast_sizes()
            model.total_loss = None
            last_batch_size = batch_size

        output = model.eval_batch(peekable_loader, compute_loss=False, reduce_output=None)

        # output logits are available only on last stage pipeline workers
        if mpu.is_pipeline_last_stage():
            output = torch.cat(output)

            _, labels, _, _, loss_mask = process_batch(batch, device=output.device)

            res = calculate_metric(output, labels, loss_mask, eval_metric)
            total_output += res
            total_tokens += loss_mask.view(-1).eq(1).sum()

            # Average loss across DP
            # HCCL does not support torch.distributed.ReduceOp.AVG, so use SUM
            torch.distributed.all_reduce(total_output,
                                         group=mpu.get_data_parallel_group(),
                                         op=torch.distributed.ReduceOp.SUM)
            total_output = total_output / dp_world_size

            if do_print and (i+1) % args.log_interval == 0:
                avg_metric = total_output / total_tokens
                print(f'Iteration: {i+1}: avg_{eval_metric}={avg_metric}')

    loss = total_output * dp_world_size
    return loss


def evaluate_and_print_results(task, data_loader, model, eval_metric, using_3d):
    """Evaluate and print results on screen."""

    # Evaluate and get results.
    if using_3d:
        # only a single last stage worker will print
        do_print = mpu.is_pipeline_last_stage() \
                   and (mpu.get_data_parallel_rank() == 0) \
                   and (mpu.get_tensor_model_parallel_rank() == 0)
        output = evaluate_3d(data_loader, model, eval_metric, do_print)
    else:
        do_print = is_last_rank()
        output = evaluate(data_loader, model, eval_metric)

    if do_print:
        string = ' validation results on {} | '.format(task)
        if eval_metric == 'loss':
            num_tokenized_tokens = data_loader.dataset.num_tokenized_tokens
            num_original_tokens = data_loader.dataset.num_original_tokens
            val_loss = output / (num_tokenized_tokens - 1)
            ppl = math.exp(min(20, val_loss))
            token_ratio = (num_tokenized_tokens - 1) / (num_original_tokens - 1)
            adjusted_ppl = math.exp(min(20, val_loss * token_ratio))
            string += 'avg loss: {:.4E} | '.format(val_loss)
            string += 'ppl: {:.4E} | '.format(ppl)
            string += 'adjusted ppl: {:.4E} | '.format(adjusted_ppl)
            string += 'token ratio: {} |'.format(token_ratio)

        elif eval_metric == 'accuracy':
            num_examples = len(data_loader.dataset)
            acc = output / num_examples
            string += 'number correct: {:.4E} | '.format(output)
            string += 'total examples: {:.4E} | '.format(num_examples)
            string += 'avg accuracy: {:.4E}'.format(acc)

        else:
            raise NotImplementedError('evaluation method for {} metric is not '
                                      'implemented yet.'.format(eval_metric))

        length = len(string) + 1
        print('-' * length)
        print(string)
        print('-' * length)


def main():
    """Main program."""
    args = get_args()

    if args.num_layers_per_virtual_pipeline_stage is not None:
        raise NotImplementedError("Interleaved pipeline schedule is not yet supported for downstream tasks.")

    if args.task == 'LAMBADA':
        eval_metric = 'accuracy'
    elif args.task == 'WIKITEXT103':
        eval_metric = 'loss'
    else:
        raise NotImplementedError('{} task is not implemented.'.format(args.task))

    # Set up model and load checkpoint.
    if args.deepspeed:
        parallel_output = (eval_metric == 'loss')
        model = load_ds_checkpoint_and_setup_megatron(parallel_output)
    else:
        model = get_model(get_model_provider(eval_metric))
        assert len(model) == 1, "Above condition should have caught this"
        model = model[0]

    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    # Data stuff.
    dataset = build_dataset(args.task)
    assert args.micro_batch_size == args.eval_micro_batch_size, \
        "main (zeroshot_gpt) - Unsupported for split micro batch size"
    dataloader = build_data_loader(dataset, args.micro_batch_size,
                                   args.num_workers, drop_last=False)

    # Run evaluation.
    evaluate_and_print_results(args.task, dataloader, model, eval_metric, using_3d=args.deepspeed)

    print_rank_0('done :-)')


# Below function is adapted from tasks/eval_harness/evaluate.py
# Basically, it attempts to load a 3D model checkpoint onto a different 3D configuration
# This works for inference only!
def load_ds_checkpoint_and_setup_megatron(parallel_output):
    args = get_args()

    # Loading pipelined models in deepspeed with different TP than it was originally trained on fails
    # due to a sanity check, that makes sure that all state_dicts that we merge contains attention layers.
    # This, however, is not true for pipelining when we will merge the state_dict for the embeddings which
    # does not contain these attention-specific keys.
    #
    # Deepspeed does however manage to load the model if we just turn off this sanity check.
    import deepspeed
    from megatron.training import setup_model_and_optimizer
    if args.model_type == 'GPT':
        from pretrain_gpt import model_provider
    elif args.model_type == 'LLAMA':
        from pretrain_llama import model_provider
    else:
        raise Exception(f'unrecognized model type {args.model_type}')

    deepspeed.runtime.state_dict_factory.MegatronSDLoader.sanity_check = lambda self, ckpt_file_name: None

    cp_path = args.load
    args.load = None
    model_provider_fn = functools.partial(model_provider, parallel_output=parallel_output)
    model, _, _ = setup_model_and_optimizer(model_provider_fn)
    model = model[0]
    zero_enabled = model._config.zero_enabled
    model._config.zero_enabled = False
    _, _ = model.load_checkpoint(cp_path, tag=None, load_optimizer_states=False, load_lr_scheduler_states=False,
                                 load_module_only=True)
    model._config.zero_enabled = zero_enabled

    torch.distributed.barrier()
    return model
