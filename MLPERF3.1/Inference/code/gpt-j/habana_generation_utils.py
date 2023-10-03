#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################
import time
import os
import glob
import torch
import torch.nn.functional as F
from enum import Enum
import habana_frameworks.torch.core as htcore

from collections import UserDict


def boolean(string):
    char = string.lower()[0]
    assert char == 't' or char == 'f', f"Invalid value: {string} - it should start with either 't' or 'f'"
    return char == 't'


def flip(dictionary):
    return {v: k for k, v in dictionary.items()}


def unwrap_ds(model):
    if hasattr(model, 'module'):
        return model.module
    return model


def defined(v):
    return v is not None


class Option:
    def __init__(self, opt_type, default=None, help=None, is_custom=False):
        self.opt_type = opt_type
        self.default = default
        self.is_custom = is_custom
        self.help = help

    def describe(self, name):
        type_str = FLIPPED_SUPPORTED_TYPES[self.opt_type]
        default_str = f'={self.default}' if defined(self.default) else ''
        custom_str = ' [custom]' if self.is_custom else ''
        help_str = f'\n\t{self.help}' if self.help else ''
        return f'{name}:{type_str}{default_str}{custom_str}{help_str}'


class CustomOption(Option):
    def __init__(self, opt_type, **kwargs):
        super().__init__(opt_type, **kwargs, is_custom=True)


SUPPORTED_TYPES = {
    'int': int,
    'bool': boolean,
    'float': float,
}
FLIPPED_SUPPORTED_TYPES = flip(SUPPORTED_TYPES)

OPTIONS = {
    # HF options
    'max_length': Option(int, default=128, help='Maximum input + output length. Overriden by max_new_tokens.'),
    'max_new_tokens': Option(int, help='Maximum number of tokens to generate.'),
    'min_length': Option(int, help='Minimum input + output length. Overriden by min_new_tokens.'),
    'min_new_tokens': Option(int, help='Minimum number of tokens to generate.'),

    'num_beams': Option(int, default=1, help='Number of beams. When num_beams=1 greedy_search is used, otherwise beam_search.'),
    'early_stopping': Option(boolean, default=False, help='Exit beam-search when N hypothesis are found'),
    'early_stopping_delay': Option(int, default=1, help='Determines how many iterations to schedule before checking for early exit condition'),
    'do_sample': Option(boolean, default=False, help='Enable sampling. Affects both greedy_search and beam_search.'),
    'temperature': Option(float, help='Value > 1.0 increase sampling randomness. Value < 1.0 makes tokens with best score more likely to be selected.'),
    'top_k': Option(int, help='Limit sampling to top_k best tokens at each step.'),
    'top_p': Option(float, help='Limit sampling to a minimal set of tokens S such as P(S) >= top_p.'),
    'repetition_penalty': Option(float, help='Penalize repeating tokens. Value > 1 makes tokens that have already appeared less likely.'),
    'no_repeat_ngram_size': Option(int, help='Forbid ngrams that have already appeared from reappearing.'),
    'length_penalty': Option(float, default=1.0, help='Applied as exponent to beam length. Value > 1.0 encourages longer sequences (because of log used in scoring). Value < 0.0 encourages shorter sequences. Beam-search only.'),
    'use_cache': Option(boolean, default=True, help='Run with KV-cache enabled.'),

    # Generic HPU options
    'use_graphs': CustomOption(boolean, default=True, help='Use HPU graphs if possible.'),
    'ignore_eos': CustomOption(boolean, default=True, help='Run greedy_search for full max_length to avoid device<>CPU synchronization.'),
    'max_iterations': CustomOption(int, help='Limit number of iterations. Useful for profiling and debugging.'),

    # Model specific HPU options
    'static_shapes': CustomOption(boolean, help='Run with static shapes to avoid graph recompilations.'),
    'bucket_width': CustomOption(int, help='Pad shapes to a multiple of bucket width when static_shapes are used.'),
    'max_input_length': CustomOption(int, help='Maximum length of input when static_shapes are used.'),
    'trim_logits': CustomOption(boolean, help='Calculate logits only for the last token in the initial run of the model.'),
    'limit_graphs': CustomOption(boolean, help='Use hpu graphs only for iterations > 0.'),
    'reuse_cache': CustomOption(boolean, help='Reuse kv-cache memory between prompts.'),
    'kv_cache_fp8': CustomOption(boolean, default=False, help='store kv-cache in float8 when kv-cache is used'),

    'use_position_ids': CustomOption(boolean, default=True, help='Use position ids in GPT-J'),
    'kv_cache_margin': CustomOption(int, help='Update only last K entries in KV-cache. Requires reuse_cache.'),
}

MIN_INF = float('-inf')


def custom_options():
    return [k for k, v in OPTIONS.items() if v.is_custom]


def generate_option_help():
    result = 'Options need to be specified in the form of KV1,KV2,[...] where each KV is either KEY_N=VALUE_N or KEY_N:TYPE_N=VALUE_N. '
    result += '\nKnown options:'
    for name, op in OPTIONS.items():
        result = result + '\n    ' + op.describe(name)
    result += '\nOptions that are not listed above but are supported by HF API can be passed by explicitly specifing their type. For example: penalty_alpha:float=0.5 . Note: this is only supported in "vanilla" and "compatibility" generation modes.'
    result += '\nOptions marked as "custom" are only used when running in "optimized" generation mode.'
    return result


def parse_key_type_value(ktv):
    if '=' in ktv:
        # Full key/type/value
        # key[:type]=value
        kt, value = ktv.split('=')
        kt = kt.split(':')
        name = kt[0]
        if len(kt) > 1:
            opt_type = kt[1]
            assert opt_type in SUPPORTED_TYPES, f'Unsupported type: {opt_type}. Supported types: {list(SUPPORTED_TYPES.keys())}'
            opt_type = SUPPORTED_TYPES[opt_type]
        else:
            assert name in OPTIONS, f'Cannot deduce type! Unknown option:{name}! Please specify type or use one of the following options: {list(OPTIONS.keys())}'
            opt_type = OPTIONS[name].opt_type
        return (name, opt_type(value))
    else:
        # Boolean shorthand
        # [!]key
        if ktv.startswith('!'):
            return (ktv[1:], False)
        else:
            return (ktv, True)


def parse_options(string, default_values={}):
    if string is None:
        return GenerationOptions(default_values)
    kvs = [parse_key_type_value(ktv) for ktv in string.split(',')]
    return GenerationOptions(default_values=default_values, **dict(kvs))


class GenerationOptions(dict):
    def __init__(self, default_values={}, **args):
        super().__init__(self, **args)
        self.set_defaults(default_values)

    def filter(self, *keywords):
        result = GenerationOptions(**self)
        for k in keywords:
            result.pop(k, None)
        return result

    def set_defaults(self, default_values):
        for k, v in default_values.items():
            if k not in self:
                self[k] = v
        for k, v in OPTIONS.items():
            if defined(v.default) and k not in self:
                self[k] = v.default

    def __getattr__(self, key):
        if key in self.keys():
            return self[key]
        return None

    def set(self, key, value):
        self[key] = value

    def print(self):
        print("Generation options:")
        for k, v in sorted(self.items()):
            print('  ', f'{k}={v}')


def fast_topk(tensor, k, dim):
    min_inf = torch.tensor(MIN_INF, dtype=tensor.dtype, device=tensor.device)
    best = []
    for i in range(k):
        value, index = torch.max(tensor, dim=dim)
        best.append((value.unsqueeze(-1), index.unsqueeze(-1)))
        if (i + 1 < k):
            tensor.scatter_(dim, index.unsqueeze(-1), min_inf.unsqueeze(0).expand(tensor.size(0), 1))
    best_value, best_index = zip(*best)
    best_value = torch.cat([b for b in best_value], dim=-1)
    best_index = torch.cat([b for b in best_index], dim=-1)
    return best_value, best_index


if os.environ.get('TOPK_ALGORITHM', 'FAST') == 'NATIVE':
    TOPK_IMPL = torch.topk
else:
    TOPK_IMPL = fast_topk


class SelectionBeam():
    def __init__(self, batch_size, beam_size):
        self.batch_size = batch_size
        self.beam_size = beam_size

    def __call__(self, logits, eos_token_id):
        eos_logits = logits[:, eos_token_id].clone()
        logits[:, eos_token_id] = MIN_INF
        logits = logits.view(self.batch_size, -1)
        topk = TOPK_IMPL(logits, k=self.beam_size, dim=-1)
        return (*topk, eos_logits)


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
def generate_on_prepared_input(model,
                               options,
                               model_inputs,
                               max_length,
                               input_length):
    if options.use_cache and options.reuse_cache:
        model_inputs['reuse_cache'] = True
        bs, _ = model_inputs['input_ids'].shape
        unwrap_ds(model).allocate_kv_cache(bs * options.num_beams, max_length, options.kv_cache_fp8)

    device = get_device(model)
    model_inputs = move(model_inputs, device)

    initial_ids = model_inputs['input_ids']
    bs = initial_ids.shape[0]
    selection_algorithm = SelectionBeam(bs, options.num_beams)
    beam_trace = beam_search(model, options, selection_algorithm, max_length, input_length, model_inputs)
    return initial_ids.cpu(), move(beam_trace, 'cpu')


def calculate_input_padding(input_length, options):
    if not options.static_shapes:
        return 0
    if defined(options.bucket_width):
        return round_up(input_length, options.bucket_width) - input_length
    if defined(options.max_input_length):
        return options.max_input_length - input_length
    assert False, "Running with static_shapes requires setting either 'bucket_width' or 'max_input_length'"


def calculate_max_length(input_length, options):
    if defined(options.max_new_tokens) and defined(options.bucket_width):
        return round_up(input_length + options.max_new_tokens, options.bucket_width)
    if defined(options.max_new_tokens) and defined(options.max_input_length):
        return options.max_input_length + options.max_new_tokens
    if defined(options.max_input_length):
        assert options.max_length >= options.max_input_length, \
            f"max_input_length={options.max_input_length} is bigger then max_length={options.max_length}! Either increase max_length or specify max_new_tokens."
    return options.max_length


def prepare_decoder_only_input_without_moving(pad_token_id, options, model_args):
    input_ids = model_args['input_ids']
    attention_mask = model_args['attention_mask']

    input_ids = input_ids.to(torch.int32)
    attention_mask = attention_mask.to(torch.bfloat16)

    input_length = input_ids.shape[-1]
    input_padding = calculate_input_padding(input_length, options)
    max_length = calculate_max_length(input_length, options)

    if options.static_shapes:
        model_args['token_idx'] = torch.tensor(input_length)
        if input_padding > 0:
            input_ids = F.pad(input_ids, (0, input_padding), value=pad_token_id)
            attention_mask = F.pad(attention_mask, (0, input_padding), value=0)

    position_ids = attention_mask.int().cumsum(-1) - 1
    start_end = torch.full((input_ids.shape[0], 2), input_length, dtype=torch.int32)
    start_end[:, 0] -= position_ids[:, -1].to(torch.int32)

    attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min
    attention_mask = attention_mask.unsqueeze(1)

    model_args['input_ids'] = input_ids
    model_args['attention_mask'] = attention_mask
    model_args['position_ids'] = position_ids
    model_args['start_end'] = start_end
    model_args['use_cache'] = options.use_cache
    if options.trim_logits:
        model_args['trim_logits'] = True

    return model_args, max_length, input_length


def round_up(n, multiple):
    return (n + multiple - 1) // multiple * multiple


def calc_iterations(input_length, max_length, options):
    if defined(options.max_new_tokens):
        iterations = options.max_new_tokens
    else:
        iterations = max_length - input_length
    if defined(options.max_iterations):
        iterations = min(iterations, options.max_iterations)
    return range(max(iterations, 0))


@torch.no_grad()
def beam_search(model,
                options,
                selection_algorithm,
                max_length,
                input_length,
                model_input):

    if model.config.is_encoder_decoder:
        input_ids_key = 'decoder_input_ids'
        attention_mask_key = 'decoder_attention_mask'
    else:
        input_ids_key = 'input_ids'
        attention_mask_key = 'attention_mask'
    past_key = 'past_key_values'

    input_ids = model_input[input_ids_key]
    attention_mask = model_input[attention_mask_key]

    token_idx = model_input.get('token_idx', None)
    position_ids = model_input.pop('position_ids')

    MIN_LENGTH = 30
    MAX_LENGTH = 128
    bs = input_ids.shape[0]
    beam_scores = torch.zeros((bs,), device=input_ids.device, dtype=torch.float32)
    beam_trace_scores = torch.zeros((MAX_LENGTH, bs * options.num_beams), device=input_ids.device, dtype=torch.float32)
    beam_trace_indices = torch.zeros((MAX_LENGTH, bs * options.num_beams), device=input_ids.device, dtype=torch.int32)
    beam_trace_tokens = torch.zeros((MAX_LENGTH, bs * options.num_beams), device=input_ids.device, dtype=torch.int32)
    beam_trace_eos = torch.zeros((MAX_LENGTH, bs * options.num_beams), device=input_ids.device, dtype=torch.float32)
    beam_trace_idx = torch.tensor(0, device=input_ids.device)

    total_eos_tokens = torch.zeros((1), device=input_ids.device, dtype=torch.int32).repeat(bs)
    max_eos_tokens = torch.tensor(options.num_beams, device=input_ids.device, dtype=torch.int32).repeat(bs)

    model_input['kv_cache_shape'] = (bs * options.num_beams, input_ids.shape[-1])

    if options.early_stopping:
        checks = [None] * options.early_stopping_delay

    start = torch.full([bs], input_length, dtype=torch.int32, device=input_ids.device)
    end = torch.full([bs], input_length, dtype=torch.int32, device=input_ids.device)
    mul = torch.tensor([[64, 16, 4, 1]], dtype=torch.int32, device=input_ids.device)

    htcore.mark_step()

    for i in calc_iterations(input_length, max_length, options):
        first_step = (i == 0)

        embed_positions = model.transformer.embed_positions.repeat(position_ids.shape[0], 1, 1)
        repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        sincos = torch.gather(embed_positions, 1, repeated_position_ids)
        sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        output_size = 2 * sin.shape[2]

        model_input['sin'] = torch.repeat_interleave(sin, 2, dim=2, output_size=output_size).unsqueeze(2)
        model_input['cos'] = torch.repeat_interleave(cos, 2, dim=2, output_size=output_size).unsqueeze(2)

        model_output = model(**model_input)

        logits = model_output['logits']
        if token_idx is None or logits.shape[-2] == 1:
            next_token_logits = logits[:, -1, :].unsqueeze(-2)
        else:
            next_token_logits = logits.index_select(-2, token_idx - 1)

        next_token_logits = next_token_logits.squeeze(-2)
        vocab_size = next_token_logits.shape[-1]

        if i < MIN_LENGTH:
            next_token_logits[:, model.config.eos_token_id] = MIN_INF

        next_token_logits = F.log_softmax(next_token_logits, dim=-1, dtype=torch.float32) + beam_scores.unsqueeze(-1)
        next_token_values, next_token_indices, eos_scores = selection_algorithm(next_token_logits, model.config.eos_token_id)
        beam_scores = next_token_values.flatten()
        beam_indices = next_token_indices.div(vocab_size, rounding_mode='floor').flatten().to(torch.int32)
        beam_tokens = next_token_indices.remainder(vocab_size).flatten().to(torch.int32)

        if first_step:
            model_input[past_key] = unwrap_ds(model).reorder_kv_cache_first_token(model_input['kv_cache_shape'])
        else:
            indices = beam_indices.view(bs, options.num_beams)
            indices = torch.sum(indices * mul, axis=-1).to(torch.uint8)
            end.add_(1)
            model_input[past_key] = unwrap_ds(model).reorder_kv_cache_next_token(start, end, indices, model_input['kv_cache_shape'])

        if options.early_stopping and i >= MIN_LENGTH:
            bs_beam_scores = beam_scores.reshape((bs, -1))
            bs_eos_scores = eos_scores.reshape((bs, -1))
            scores = torch.cat([bs_beam_scores, bs_eos_scores], dim=-1)
            best_indices = torch.topk(scores, options.num_beams)[1]
            eos_tokens = (best_indices >= options.num_beams).sum(dim=-1, dtype=torch.int32)
            total_eos_tokens.add_(eos_tokens)
            is_finished = (total_eos_tokens >= max_eos_tokens)
            end = torch.logical_not(is_finished).to(torch.int32) * end
            cur_check_idx = i % options.early_stopping_delay
            checks[cur_check_idx] = is_finished.all()

        if first_step:
            eos_scores = eos_scores.repeat_interleave(options.num_beams, dim=0, output_size=options.num_beams * bs)
        beam_trace_scores.index_copy_(0, beam_trace_idx, beam_scores.unsqueeze(0))
        beam_trace_indices.index_copy_(0, beam_trace_idx, beam_indices.unsqueeze(0))
        beam_trace_tokens.index_copy_(0, beam_trace_idx, beam_tokens.unsqueeze(0))
        beam_trace_eos.index_copy_(0, beam_trace_idx, eos_scores.unsqueeze(0))
        beam_trace_idx.add_(1)

        if first_step:
            attention_mask = torch.repeat_interleave(
                attention_mask, options.num_beams, dim=0, output_size=options.num_beams * bs
            )
        attention_mask.index_fill_(2, token_idx, 0)

        next_tokens = beam_tokens.unsqueeze(-1)

        token_idx.add_(1)

        model_input[input_ids_key] = next_tokens
        model_input[attention_mask_key] = attention_mask

        if first_step:
            model_input["start_end"] = None

        if first_step:
            position_ids = position_ids[:, -1].unsqueeze(-1)
            position_ids = position_ids.repeat_interleave(options.num_beams, dim=0, output_size=options.num_beams * bs)
        else:
            position_ids.add_(1)

        if options.early_stopping and i >= MIN_LENGTH:
            next_check_idx = (i + 1) % options.early_stopping_delay
            all_done = checks[next_check_idx]
            if all_done is not None and all_done.cpu().item():
                break

    return (beam_trace_idx, beam_trace_scores, beam_trace_indices, beam_trace_tokens, beam_trace_eos)


def finalize_beams(initial_ids, beam_trace, model_config, length_penalty):
    beam_trace_idx, beam_trace_scores, beam_trace_indices, beam_trace_tokens, beam_trace_eos = beam_trace

    bs = initial_ids.shape[0]
    num_beams = beam_trace_scores.shape[1] // bs

    beam_trace_idx = beam_trace_idx.item()
    beam_trace_scores = beam_trace_scores[:beam_trace_idx, :].reshape(beam_trace_idx, bs, -1)
    beam_trace_indices = beam_trace_indices[:beam_trace_idx, :].reshape(beam_trace_idx, bs, -1)
    beam_trace_tokens = beam_trace_tokens[:beam_trace_idx, :].reshape(beam_trace_idx, bs, -1)
    beam_trace_eos = beam_trace_eos[:beam_trace_idx, :].reshape(beam_trace_idx, bs, -1)

    input_lengths = torch.tensor(initial_ids.size(-1)) - torch.eq(initial_ids, model_config.eos_token_id).sum(-1)

    results = []
    for batch in range(bs):
        best_score = (False, MIN_INF)
        best_beam = 0
        best_step = 0
        total_finished = 0
        for step in range(beam_trace_idx):
            #b_len = initial_ids.shape[-1] + step
            b_len = input_lengths[batch] + step
            p_scores = torch.cat([beam_trace_scores[step, batch], beam_trace_eos[step, batch]])
            scores = p_scores / (b_len ** length_penalty)
            top_scores, top_beams = torch.sort(scores, dim=-1, descending=True, stable=True)
            # print(batch, step, top_scores.numpy().tolist(), top_beams.numpy().tolist())
            for beam in top_beams[:num_beams]:
                beam = beam.item()
                finished = beam >= num_beams
                score = (finished, scores[beam])
                total_finished += finished
                # print("\t", beam, score)
                if score > best_score or (not best_score[0] and beam == 0):
                    best_beam = beam
                    best_score = score
                    best_step = step
                    # print('new best', score, 'vs', best_score)
            if total_finished >= num_beams:
                break

        idx = best_beam
        tokens = []
        for step in range(best_step, -1, -1):
            if idx >= num_beams:
                tokens.append(model_config.eos_token_id)
                idx = idx - num_beams
            else:
                tokens.append(beam_trace_tokens[step, batch, idx].item())
                idx = beam_trace_indices[step, batch, idx].item()
        tokens.reverse()
        results.append(tokens)

    max_length = max(len(r) for r in results)
    results = [torch.tensor(r) for r in results]
    results = torch.cat([expand_if_needed(r, max_length, model_config.pad_token_id).unsqueeze(0) for r in results], dim=0)
    results = torch.cat([initial_ids, results], dim=-1)

    return results


def map_tensors(obj, fn):
    constructor = type(obj)
    if isinstance(obj, tuple):
        return constructor(map_tensors(v, fn) for v in obj)
    if isinstance(obj, list):
        return constructor([map_tensors(v, fn) for v in obj])
    if isinstance(obj, dict) or isinstance(obj, UserDict):
        return constructor({k: map_tensors(v, fn) for k, v in obj.items()})
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    return obj


def move(obj, device):
    return map_tensors(obj, lambda t: t.to(device))


def expand_if_needed(tensor, new_size, value, dim=-1):
    orig_len = tensor.shape[dim]
    padding_len = new_size - orig_len
    if padding_len > 0:
        if dim == -1:
            return F.pad(tensor, (0, padding_len), value=value)
        elif dim == -2:
            return F.pad(tensor, (0, 0, 0, padding_len), value=value)
        else:
            assert False, f'Unsupported dim value: {dim}'
    return tensor
