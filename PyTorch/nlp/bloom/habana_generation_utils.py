#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################
import torch
import torch.nn.functional as F
from enum import Enum


MIN_INF = float('-inf')


class GenerationMode(Enum):
    VANILLA = 'vanilla'
    COMPATIBILITY = 'compatibility'
    OPTIMIZED = 'optimized'

    def __str__(self):
        return self.value


def get_device(model):
    if hasattr(model, 'device'):
        return model.device
    if hasattr(model, 'module'):
        return model.module.device
    assert False, 'Cannot extract device!'
    return None


def is_on_hpu(obj):
    return str(get_device(obj)).startswith('hpu')


@torch.no_grad()
def generate(model,
             max_length,
             num_beams=1,
             static_shapes=False,
             ignore_eos=True,
             max_iterations=None,
             **model_args):

    initial_ids = model_args['input_ids']
    model_args = prepare_input(model, max_length, static_shapes, **model_args)
    if num_beams == 1:
        return greedy_search(model, max_length, ignore_eos=ignore_eos, max_iterations=max_iterations, **model_args)
    if num_beams > 1:
        beam_trace = beam_search(model, max_length, num_beams, max_iterations=max_iterations, **model_args)
        return finalize_beams(initial_ids, move(beam_trace, 'cpu'), model.config)
    assert False, 'Unsupported combination of generation parameters!'


def prepare_input(model, max_length, static_shapes=False, **model_args):
    device = get_device(model)
    if static_shapes:
        input_ids = model_args['input_ids']
        attention_mask = model_args['attention_mask']
        cur_length = input_ids.shape[-1]
        padding_length = max_length - cur_length
        model_args['token_idx'] = torch.tensor(cur_length)
        model_args['input_ids'] = F.pad(input_ids, (0, padding_length), value=model.config.pad_token_id)
        model_args['attention_mask'] = F.pad(attention_mask, (0, padding_length), value=0)
    if 'token_idx' in model_args:
        model_args['token_idx'] = model_args['token_idx'].to(device)
    model_args['input_ids'] = model_args['input_ids'].to(device)
    model_args['attention_mask'] = model_args['attention_mask'].to(device)
    model_args['use_cache'] = model_args.get('use_cache')
    return model_args


def calc_iterations(cur_length, max_length, max_iterations):
    iterations = max_length - cur_length
    if max_iterations is not None:
        iterations = min(iterations, max_iterations)
    return range(max(iterations, 0))


def index_copy_or_cat(index, destination, source):
    if index is None:
        return torch.cat([destination, source], dim=-1)
    else:
        return destination.index_copy_(-1, index, source)


@torch.no_grad()
def greedy_search(model,
                  max_length,
                  max_iterations=None,
                  ignore_eos=True,
                  **model_input):

    input_ids = model_input['input_ids']
    attention_mask = model_input['attention_mask']

    token_idx = model_input.get('token_idx', None)

    if token_idx is None:
        cur_length = input_ids.shape[-1]
    else:
        cur_length = token_idx.item()

    eos_generated = torch.zeros((input_ids.shape[-2],), dtype=torch.bool, device=input_ids.device)

    if is_on_hpu(input_ids):
        import habana_frameworks.torch.core as htcore
        htcore.mark_step()
    for i in calc_iterations(cur_length, max_length, max_iterations):
        model_output = model(**model_input)

        logits = model_output['logits']
        if token_idx is None or logits.shape[-2] == 1:
            next_token_logits = logits[:, -1, :].unsqueeze(-2)
        else:
            next_token_logits = logits.index_select(-2, token_idx - 1)
        next_tokens = torch.argmax(next_token_logits, dim=-1).squeeze(-1)
        next_tokens = torch.logical_not(eos_generated) * next_tokens + eos_generated * model.config.pad_token_id
        eos_generated.logical_or_(next_tokens.eq(model.config.eos_token_id))
        next_tokens = next_tokens.unsqueeze(-1)

        if token_idx is None:
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            attention_mask = F.pad(attention_mask, (0, 1), value=1)
        else:
            input_ids.index_copy_(1, token_idx, next_tokens)
            attention_mask.index_fill_(1, token_idx, 1)
            token_idx.add_(1)

        if model_input['use_cache']:
            model_input['past_key_values'] = model_output['past_key_values']
            model_input['input_ids'] = next_tokens
        else:
            model_input['input_ids'] = input_ids
        model_input['attention_mask'] = attention_mask
        if not ignore_eos:
            if eos_generated.min() == 1:
                break
        if is_on_hpu(input_ids):
            import habana_frameworks.torch.core as htcore
            htcore.mark_step()
    return input_ids


@torch.no_grad()
def beam_search(model,
                max_length,
                num_beams,
                max_iterations=None,
                **model_input):

    input_ids = model_input['input_ids']
    attention_mask = model_input['attention_mask']
    bs = input_ids.shape[0]

    token_idx = model_input.get('token_idx', None)

    if token_idx is None:
        cur_length = input_ids.shape[-1]
    else:
        cur_length = token_idx.item()

    beam_scores = torch.zeros((bs,), device=input_ids.device, dtype=torch.float32)

    beam_trace_scores = torch.zeros((max_length, 2 * bs * num_beams), device=input_ids.device, dtype=torch.float32)
    beam_trace_indices = torch.zeros((max_length, 2 * bs * num_beams), device=input_ids.device, dtype=torch.int64)
    beam_trace_tokens = torch.zeros((max_length, 2 * bs * num_beams), device=input_ids.device, dtype=torch.int64)
    beam_trace_idx = torch.tensor(0, device=input_ids.device)

    beam_attention_mask = torch.ones((bs * num_beams), device=input_ids.device, dtype=attention_mask.dtype)

    if is_on_hpu(input_ids):
        import habana_frameworks.torch.core as htcore
        htcore.mark_step()
    for i in calc_iterations(cur_length, max_length, max_iterations):
        model_output = model(**model_input)

        logits = model_output['logits']

        if token_idx is None or logits.shape[-2] == 1:
            next_token_logits = logits[:, -1, :].unsqueeze(-2)
        else:
            next_token_logits = logits.index_select(-2, token_idx - 1)

        next_token_logits = next_token_logits.squeeze(-2)
        vocab_size = next_token_logits.shape[-1]

        next_token_logits = F.log_softmax(next_token_logits, dim=-1, dtype=torch.float32) + beam_scores.unsqueeze(-1)
        next_token_logits = next_token_logits.view(bs, -1).squeeze(0)
        next_token_values, next_token_indices = torch.topk(next_token_logits, k=(2 * num_beams), dim=-1, largest=True, sorted=True)

        beam_scores = next_token_values.flatten()
        beam_indices = next_token_indices.div(vocab_size, rounding_mode='floor').flatten()
        beam_tokens = next_token_indices.remainder(vocab_size).flatten()

        beam_trace_scores.index_copy_(0, beam_trace_idx, beam_scores.unsqueeze(0))
        beam_trace_indices.index_copy_(0, beam_trace_idx, beam_indices.unsqueeze(0))
        beam_trace_tokens.index_copy_(0, beam_trace_idx, beam_tokens.unsqueeze(0))
        beam_trace_idx.add_(1)

        beam_scores.add_(torch.where(beam_tokens.eq(model.config.eos_token_id), MIN_INF, 0.0))
        beam_scores = beam_scores.view(bs, -1).unsqueeze(0)
        _, selected = torch.topk(beam_scores, k=num_beams, dim=-1, largest=True, sorted=True)
        offset = torch.arange(0, torch.numel(beam_scores), beam_scores.shape[-1]).unsqueeze(-1)
        selected = (selected + offset).flatten()
        beam_scores = beam_scores.flatten().index_select(0, selected)
        beam_tokens = beam_tokens.index_select(0, selected)
        beam_indices = beam_indices.index_select(0, selected)

        prev_beams = logits.shape[0] // bs
        beam_offsets = torch.arange(0, logits.shape[0], prev_beams, dtype=torch.int32, device=logits.device)
        beam_indices = (beam_indices.view(bs, -1) + beam_offsets.unsqueeze(-1)).flatten()

        attention_mask = attention_mask.index_select(0, beam_indices)
        model_input['attention_mask'] = index_copy_or_cat(token_idx, attention_mask, beam_attention_mask.unsqueeze(-1))

        if model_input['use_cache']:
            model_input['past_key_values'] = tuple(tuple(kv.index_select(0, beam_indices) for kv in layer) for layer in model_output['past_key_values'])
            model_input['input_ids'] = beam_tokens.unsqueeze(-1)
        else:
            input_ids = input_ids.index_select(0, beam_indices)
            model_input['input_ids'] = index_copy_or_cat(token_idx, input_ids, beam_tokens.unsqueeze(-1))

        if token_idx is not None:
            token_idx.add_(1)

        if is_on_hpu(input_ids):
            import habana_frameworks.torch.core as htcore
            htcore.mark_step()

    return (beam_trace_idx, beam_trace_scores, beam_trace_indices, beam_trace_tokens)


def finalize_beams(initial_ids, beam_trace, model_config):
    beam_trace_idx, beam_trace_scores, beam_trace_indices, beam_trace_tokens = beam_trace

    bs = initial_ids.shape[0]
    num_beams = beam_trace_scores.shape[1] // (2 * bs)

    beam_trace_idx = beam_trace_idx.item()
    beam_trace_scores = beam_trace_scores[:beam_trace_idx, :]
    beam_trace_indices = beam_trace_indices[:beam_trace_idx, :]
    beam_trace_tokens = beam_trace_tokens[:beam_trace_idx, :]

    # (score, parent_beam, token_id, is_finished)
    root = (MIN_INF, None, None, False)

    def resolve_beam(beam):
        if beam == root:
            return []
        score, prev, tok, is_finished = beam
        rest = resolve_beam(prev)
        rest.append(tok)
        return rest

    prev_beams = [[root]] * bs
    best = [root] * bs

    def beam_score(beam):
        return (beam[3], beam[0])

    for step, (scores, indices, tokens) in enumerate(zip(beam_trace_scores, beam_trace_indices, beam_trace_tokens)):
        cur_beams = [[] for _ in range(bs)]
        for idx, (s, i, t) in enumerate(zip(scores, indices, tokens)):
            batch = idx // (num_beams * 2)
            idx = idx % (num_beams * 2)
            b_len = 1 + step + initial_ids.shape[0]
            b_score = s.item() / b_len
            b_tok = t.item()
            is_finished = b_tok == model_config.eos_token_id
            if len(cur_beams[batch]) >= num_beams:
                continue
            beam = (b_score, prev_beams[batch][i], b_tok, is_finished)
            if not is_finished:
                cur_beams[batch].append(beam)
            if is_finished or (step + 1 == beam_trace_idx):
                if beam_score(best[batch]) < beam_score(beam):
                    best[batch] = beam
        prev_beams = cur_beams

    result = [torch.cat([initial_ids[i], torch.tensor(resolve_beam(b), dtype=initial_ids.dtype, device=initial_ids.device)]) for i, b in enumerate(best)]
    max_length = max([t.shape[-1] for t in result])
    result = [expand_if_needed(res, max_length, model_config.pad_token_id) for res in result]
    input_ids = torch.stack(result)
    return input_ids


def move(obj, device):
    constructor = type(obj)
    if isinstance(obj, tuple):
        return constructor(move(v, device) for v in obj)
    if isinstance(obj, list):
        return constructor([move(v, device) for v in obj])
    if isinstance(obj, dict):
        return constructor({k: move(v, device) for k, v in obj.items()})
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    return obj


def expand_if_needed(tensor, new_size, value):
    orig_len = tensor.shape[-1]
    if orig_len > 1:
        padding_len = new_size - orig_len
        return F.pad(tensor, (0, padding_len), value=value)
    return tensor


def expand_and_update_if_needed(args, key, new_size, value):
    args[key] = expand_if_needed(args[key], new_size, value)


def enable_compatibility_mode(model, max_length, static_shapes):
    device = get_device(model)
    old_fwd = model.forward

    def fwd(*args, **kwargs):
        cur_idx = kwargs['attention_mask'].shape[-1]
        if static_shapes:
            kwargs['token_idx'] = torch.tensor(cur_idx)
            expand_and_update_if_needed(kwargs, 'input_ids', max_length, model.config.pad_token_id)
            expand_and_update_if_needed(kwargs, 'attention_mask', max_length, model.config.pad_token_id)
        output = old_fwd(*move(args, device), **move(kwargs, device))
        output = output.copy()
        output['logits'] = output['logits'].cpu()
        output['logits'] = output['logits'][:, :cur_idx, :]

        if is_on_hpu(model):
            import habana_frameworks.torch.core as htcore
            htcore.mark_step()

        return output

    model.forward = fwd
    model.old_fwd = old_fwd


def disable_compatibility_mode(model):
    model.forward = model.old_fwd
    del model.old_fwd


def create_pipeline(model, tokenizer, generation_mode):
    def pipeline(prompts, max_length, max_iterations=None, static_shapes=False, ignore_eos=True, **generation_args):
        model_args = tokenizer(prompts, return_tensors="pt", max_length=max_length, padding=True, truncation=True)
        if generation_mode == GenerationMode.VANILLA:
            model_args = model_args.to(model.device)
            output = model.generate(max_length=max_length, **generation_args, **model_args)
        elif generation_mode == GenerationMode.COMPATIBILITY:
            enable_compatibility_mode(model, max_length, static_shapes)
            output = model.generate(max_length=max_length, **generation_args, **model_args)
            disable_compatibility_mode(model)
        elif generation_mode == GenerationMode.OPTIMIZED:
            output = generate(model, max_length=max_length, max_iterations=max_iterations, static_shapes=static_shapes, ignore_eos=ignore_eos, **generation_args, **model_args)
        else:
            assert False, f'Unsupported generation mode: {generation_mode}'
        return tokenizer.batch_decode(output.cpu(), skip_special_tokens=True)

    return pipeline
