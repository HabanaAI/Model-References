#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################
import time
import glob
import torch
import torch.nn.functional as F
from enum import Enum

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


class GenerationMode(Enum):
    VANILLA = 'vanilla'
    OPTIMIZED = 'optimized'

    def __str__(self):
        return self.value


class LogitsModifierTemperature:
    def __init__(self, temperature):
        self.temperature = temperature

    def __call__(self, logits):
        return logits / self.temperature


class LogitsModifierTopK:
    def __init__(self, k):
        self.k = k

    def __call__(self, logits):
        topk = torch.topk(logits, self.k)[0][:, -1].unsqueeze(-1)
        mask = logits < topk
        return logits.masked_fill(mask, MIN_INF)


class LogitsModifierTopP:
    def __init__(self, p):
        self.p = p

    def __call__(self, logits):
        sorted, indices = torch.sort(logits, descending=False)
        cum_probs = sorted.softmax(dim=-1, dtype=torch.float32).cumsum(dim=-1)
        sorted_mask = cum_probs <= (1 - self.p)
        mask = sorted_mask.scatter(1, indices, sorted_mask)
        return logits.masked_fill(mask, MIN_INF)


class LogitsModifierMinOutputLength:
    def __init__(self, min_len, cur_len, eos_token_id):
        self.min_len = min_len
        self.cur_len = cur_len
        self.eos_token_id = eos_token_id

    def __call__(self, logits):
        if self.cur_len < self.min_len:
            logits[:, self.eos_token_id] = MIN_INF
            self.cur_len = self.cur_len + 1
        return logits


class TokensModifierRepetitionPenalty:
    def __init__(self, penalty):
        self.penalty = penalty

    def __call__(self, logits, input_ids, *args):
        score = torch.gather(logits, 1, input_ids)
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)
        return logits.scatter_(1, input_ids, score)


class TokensModifierNoRepeatNgram():
    """
    Prevents generation of ngrams of a given size.
    E.g. if n==3, every sequence of 3 (and more) tokens in the generated output is unique.

    """

    def __init__(self, ngram_size: int):
        self.N = ngram_size

    def generate_forbidden_tokens(self, token_ids: list, current_sequence_length: int) -> list:
        """
        Uses a sliding window of length n-1 to go through all the already generated tokens and
        compares this window against a sequence_to_match (n-1 most recently generated tokens).
        If a given window matches this sequence, we add the token that succeeds the window
        to a list of forbidden tokens.
        The function works with 1 beam at a time.

        Params:
        :token_ids: already generated tokens (padded)
        :current_sequence_length: number of already generated tokens (discarding padding)

        Returns:
        :forbidden: list of forbidden tokens

        """

        start_of_sequence_to_match = current_sequence_length - (self.N - 1)  # index, where the sequence_to_match begins
        sequence_to_match = token_ids[start_of_sequence_to_match : current_sequence_length]

        forbidden = []
        for i in range(start_of_sequence_to_match):
            window = token_ids[i:i + self.N - 1]
            if window == sequence_to_match:
                forbidden.append(token_ids[i + self.N - 1])
        return forbidden

    def __call__(self, logits: torch.Tensor, input_ids: torch.Tensor, current_sequence_length: int) -> torch.tensor:
        """
        Adds MIN_INF forbidden tokens logits so that these tokens are not chosen.

        Params:
        :logits: logits for the next token in each beam
        :input_ids: already generated token ids for each beam (padded)
        :current_sequence_length: number of already generated tokens (discarding padding)

        Returns:
        modified logits

        """
        bs, _ = logits.shape
        token_mask = torch.zeros_like(logits, device='cpu')
        input_ids = input_ids.to('cpu').tolist()
        forbidden = [self.generate_forbidden_tokens(ids, current_sequence_length) for ids in input_ids]
        for bs, tid in enumerate(forbidden):
            token_mask[bs][tid] = MIN_INF
        token_mask = token_mask.to(logits.device)
        return logits + token_mask


class SelectionGreedy():
    def __call__(self, logits):
        return torch.argmax(logits, dim=-1).squeeze(-1)


class SelectionGreedySampling():
    def __call__(self, logits):
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        return torch.multinomial(probs, num_samples=1).squeeze(1)


class SelectionBeam():
    def __init__(self, batch_size, beam_size):
        self.batch_size = batch_size
        self.beam_size = beam_size

    def __call__(self, logits):
        logits = logits.view(self.batch_size, -1)
        return torch.topk(logits, k=2 * self.beam_size, dim=-1, largest=True)


class SelectionBeamSampling(SelectionBeam):
    def __call__(self, logits):
        logits = logits.view(self.batch_size, -1)
        scores = F.softmax(logits, dim=-1, dtype=torch.float32)
        next_token_indices = torch.multinomial(scores, num_samples=2 * self.beam_size)
        next_token_scores = torch.gather(logits, -1, next_token_indices)
        next_token_scores, sorted_indices = torch.sort(next_token_scores, descending=True, dim=-1)
        next_token_indices = torch.gather(next_token_indices, -1, sorted_indices)
        return next_token_scores, next_token_indices


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
             options,
             model_inputs):

    if model.config.is_encoder_decoder:
        encoder_args = prepare_encoder_input(model.encoder, options, model_inputs)
        model_args = prepare_decoder_input(model, options, encoder_args)
        initial_ids = model_args['decoder_input_ids']
        max_length = options.max_length
    else:
        model_args, max_length = prepare_decoder_only_input(model, options, model_inputs)
        initial_ids = model_args['input_ids']

    token_modifiers = []
    if defined(options.repetition_penalty):
        token_modifiers.append(TokensModifierRepetitionPenalty(options.repetition_penalty))
    if defined(options.no_repeat_ngram_size):
        token_modifiers.append(TokensModifierNoRepeatNgram(options.no_repeat_ngram_size))

    logit_modifiers = []
    if defined(options.min_new_tokens):
        logit_modifiers.append(LogitsModifierMinOutputLength(options.min_new_tokens, 0, model.config.eos_token_id))
    elif defined(options.min_length):
        logit_modifiers.append(LogitsModifierMinOutputLength(options.min_length, initial_ids.shape[-1], model.config.eos_token_id))
    if defined(options.top_p):
        logit_modifiers.append(LogitsModifierTopP(options.top_p))
    if defined(options.top_k):
        logit_modifiers.append(LogitsModifierTopK(options.top_k))
    if defined(options.temperature):
        logit_modifiers.append(LogitsModifierTemperature(options.temperature))

    if options.num_beams == 1:
        selection_algorithm = SelectionGreedySampling() if options.do_sample else SelectionGreedy()
        return greedy_search(model, options, selection_algorithm, token_modifiers, logit_modifiers, max_length, model_args)
    if options.num_beams > 1:
        bs = initial_ids.shape[0]
        selection_algorithm = SelectionBeamSampling(bs, options.num_beams) if options.do_sample else SelectionBeam(bs, options.num_beams)
        beam_trace = beam_search(model, options, selection_algorithm, token_modifiers, logit_modifiers, max_length, model_args)
        return finalize_beams(initial_ids.cpu(), move(beam_trace, 'cpu'), model.config, options.length_penalty)
    assert False, 'Unsupported combination of generation options!'


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


def prepare_decoder_only_input(model, options, model_args):
    input_ids = model_args['input_ids']
    attention_mask = model_args['attention_mask']

    input_length = input_ids.shape[-1]
    input_padding = calculate_input_padding(input_length, options)
    max_length = calculate_max_length(input_length, options)
    device = get_device(model)

    if options.static_shapes:
        model_args['token_idx'] = torch.tensor(input_length)
        if input_padding > 0:
            model_args['input_ids'] = F.pad(input_ids, (0, input_padding), value=model.config.pad_token_id)
            model_args['attention_mask'] = F.pad(attention_mask, (0, input_padding), value=0)

    model_args['use_cache'] = options.use_cache

    if options.trim_logits:
        model_args['trim_logits'] = True

    if options.use_cache and options.reuse_cache:
        model_args['reuse_cache'] = True
        bs, _ = model_args['input_ids'].shape
        unwrap_ds(model).allocate_kv_cache(bs * options.num_beams, max_length, options.kv_cache_fp8)

    unwrap_ds(model).prepare_for_new_input(input_length, options.num_beams, bs, device)

    return move(model_args, device), max_length


def round_up(n, multiple):
    return (n + multiple - 1) // multiple * multiple


def prepare_encoder_input(model, options, model_args):
    device = get_device(model)
    if options.static_shapes:
        cur_len = model_args['input_ids'].shape[-1]
        if defined(options.bucket_width):
            max_length = round_up(cur_len, options.bucket_width)
        else:
            max_length = cur_len
        expand_and_update_if_needed(model_args, 'input_ids', max_length, model.config.pad_token_id)
        expand_and_update_if_needed(model_args, 'attention_mask', max_length, 0)
    result = move(model_args, device)
    return result


def prepare_decoder_input(model, options, encoder_args):
    device = get_device(model)
    encoder_input_ids = encoder_args['input_ids']
    batch_size = encoder_input_ids.shape[0]

    decoder_args = {}
    decoder_args['encoder_outputs'] = model.encoder(**encoder_args)
    if options.static_shapes:
        decoder_args['token_idx'] = torch.tensor(1)
        decoder_args['max_output_length'] = options.max_length
    decoder_args['decoder_attention_mask'] = torch.ones((batch_size, 1,))
    decoder_args['decoder_input_ids'] = torch.full((batch_size, 1), model.config.pad_token_id, dtype=encoder_input_ids.dtype)
    decoder_args['attention_mask'] = encoder_args['attention_mask']
    decoder_args['use_cache'] = options.use_cache
    return move(decoder_args, device)


def calc_iterations(cur_length, max_length, options):
    if defined(options.max_new_tokens):
        iterations = options.max_new_tokens
    else:
        iterations = max_length - cur_length
    if defined(options.max_iterations):
        iterations = min(iterations, options.max_iterations)
    return range(max(iterations, 0))


def apply_modifiers(fns, logits, *args):
    for f in fns:
        logits = f(logits, *args)
    return logits


@torch.no_grad()
def greedy_search(model,
                  options,
                  selection_algorithm,
                  token_modifiers,
                  logit_modifiers,
                  max_length,
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

    if token_idx is None:
        cur_length = input_ids.shape[-1]
        result = input_ids
    else:
        cur_length = token_idx.item()
        result = expand_if_needed(input_ids, max_length, model.config.pad_token_id)

    eos_generated = torch.zeros((input_ids.shape[-2],), dtype=torch.bool, device=input_ids.device)

    if is_on_hpu(input_ids):
        import habana_frameworks.torch.core as htcore
        htcore.mark_step()
    for i in calc_iterations(cur_length, max_length, options):
        first_step = (i == 0)
        if options.use_graphs and options.limit_graphs and first_step:
            model_output = model(**model_input, bypass_hpu_graphs=True)
        else:
            model_output = model(**model_input)

        logits = model_output['logits']
        if token_idx is None or logits.shape[-2] == 1:
            next_token_logits = logits[:, -1, :]
        else:
            next_token_logits = logits.index_select(-2, token_idx - 1).squeeze(-2)

        next_token_logits = apply_modifiers(token_modifiers, next_token_logits, result, cur_length + i)
        next_token_logits = apply_modifiers(logit_modifiers, next_token_logits)
        next_tokens = selection_algorithm(next_token_logits)

        next_tokens = torch.logical_not(eos_generated) * next_tokens + eos_generated * model.config.pad_token_id
        eos_generated.logical_or_(next_tokens.eq(model.config.eos_token_id))
        next_tokens = next_tokens.unsqueeze(-1)

        if token_idx is None:
            result = torch.cat([result, next_tokens], dim=-1)
            attention_mask = F.pad(attention_mask, (0, 1), value=1)
        else:
            result.index_copy_(1, token_idx, next_tokens)
            attention_mask = expand_if_needed(attention_mask, max_length, 0)
            attention_mask.index_fill_(1, token_idx, 1)
            token_idx.add_(1)

        if model_input['use_cache']:
            model_input[input_ids_key] = next_tokens
            model_input[past_key] = model_output[past_key]
            if first_step and defined(token_idx) and not options.reuse_cache:
                model_input[past_key] = expand_cache(model_input[past_key], max_length, 0)
        else:
            model_input[input_ids_key] = result
        model_input[attention_mask_key] = attention_mask
        if not options.ignore_eos:
            if eos_generated.min() == 1:
                break
        if is_on_hpu(input_ids):
            import habana_frameworks.torch.core as htcore
            htcore.mark_step()
    return result


@torch.no_grad()
def beam_search(model,
                options,
                selection_algorithm,
                token_modifiers,
                logit_modifiers,
                max_length,
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

    if token_idx is None:
        cur_length = input_ids.shape[-1]
        result = input_ids
    else:
        cur_length = token_idx.item()
        result = expand_if_needed(input_ids, max_length, model.config.pad_token_id)

    bs = input_ids.shape[0]
    beam_scores = torch.zeros((bs,), device=input_ids.device, dtype=torch.float32)
    beam_trace_scores = torch.zeros((max_length, 2 * bs * options.num_beams), device=input_ids.device, dtype=torch.float32)
    beam_trace_indices = torch.zeros((max_length, 2 * bs * options.num_beams), device=input_ids.device, dtype=torch.int64)
    beam_trace_tokens = torch.zeros((max_length, 2 * bs * options.num_beams), device=input_ids.device, dtype=torch.int64)
    beam_trace_idx = torch.tensor(0, device=input_ids.device)

    if is_on_hpu(input_ids):
        import habana_frameworks.torch.core as htcore
        htcore.mark_step()
    for i in calc_iterations(cur_length, max_length, options):
        first_step = (i == 0)
        if options.use_graphs and options.limit_graphs and first_step:
            model_output = model(**model_input, bypass_hpu_graphs=True)
        else:
            model_output = model(**model_input)

        logits = model_output['logits']
        if token_idx is None or logits.shape[-2] == 1:
            next_token_logits = logits[:, -1, :].unsqueeze(-2)
        else:
            next_token_logits = logits.index_select(-2, token_idx - 1)

        next_token_logits = next_token_logits.squeeze(-2)
        vocab_size = next_token_logits.shape[-1]

        next_token_logits = apply_modifiers(token_modifiers, next_token_logits, result, cur_length + i)
        next_token_logits = apply_modifiers(logit_modifiers, next_token_logits)

        next_token_logits = F.log_softmax(next_token_logits, dim=-1, dtype=torch.float32) + beam_scores.unsqueeze(-1)
        next_token_values, next_token_indices = selection_algorithm(next_token_logits)

        beam_scores = next_token_values.flatten()
        beam_indices = next_token_indices.div(vocab_size, rounding_mode='floor').flatten()
        beam_tokens = next_token_indices.remainder(vocab_size).flatten()

        beam_trace_scores.index_copy_(0, beam_trace_idx, beam_scores.unsqueeze(0))
        beam_trace_indices.index_copy_(0, beam_trace_idx, beam_indices.unsqueeze(0))
        beam_trace_tokens.index_copy_(0, beam_trace_idx, beam_tokens.unsqueeze(0))
        beam_trace_idx.add_(1)

        beam_scores.add_(torch.where(beam_tokens.eq(model.config.eos_token_id), MIN_INF, 0.0))
        beam_scores = beam_scores.view(bs, -1).unsqueeze(0)
        _, selected = torch.topk(beam_scores, k=options.num_beams, dim=-1, largest=True, sorted=True)
        offset = torch.arange(0, torch.numel(beam_scores), beam_scores.shape[-1]).unsqueeze(-1)
        selected = (selected + offset).flatten()
        beam_scores = beam_scores.flatten().index_select(0, selected)
        beam_tokens = beam_tokens.index_select(0, selected)
        beam_indices = beam_indices.index_select(0, selected)

        prev_beams = logits.shape[0] // bs
        beam_offsets = torch.arange(0, logits.shape[0], prev_beams, dtype=torch.int32, device=logits.device)
        beam_indices_offset = (beam_indices.view(bs, -1) + beam_offsets.unsqueeze(-1)).flatten()

        result = result.index_select(0, beam_indices_offset)
        attention_mask = attention_mask.index_select(0, beam_indices_offset)
        if 'encoder_outputs' in model_input:
            model_input['encoder_outputs']['last_hidden_state'] = model_input['encoder_outputs']['last_hidden_state'].index_select(0, beam_indices_offset)
            model_input['attention_mask'] = model_input['attention_mask'].index_select(0, beam_indices_offset)

        next_tokens = beam_tokens.unsqueeze(-1)
        if token_idx is None:
            result = torch.cat([result, next_tokens], dim=-1)
            attention_mask = F.pad(attention_mask, (0, 1), value=1)
        else:
            result.index_copy_(1, token_idx, next_tokens)
            attention_mask = expand_if_needed(attention_mask, max_length, 0)
            attention_mask.index_fill_(1, token_idx, 1)
            token_idx.add_(1)

        if model_input['use_cache']:
            model_input[input_ids_key] = next_tokens
            if options.reuse_cache:
                if first_step:
                    model_input[past_key] = unwrap_ds(model).reorder_kv_cache_first_token(beam_indices)
                else:
                    model_input[past_key] = unwrap_ds(model).reorder_kv_cache_next_token(beam_indices)
            else:
                model_input[past_key] = unwrap_ds(model)._reorder_cache(model_output[past_key], beam_indices_offset)
            if first_step and defined(token_idx) and not options.reuse_cache:
                model_input[past_key] = expand_cache(model_input[past_key], max_length, 0)
        else:
            model_input[input_ids_key] = result
        model_input[attention_mask_key] = attention_mask

        if is_on_hpu(input_ids):
            import habana_frameworks.torch.core as htcore
            htcore.mark_step()

    return (beam_trace_idx, beam_trace_scores, beam_trace_indices, beam_trace_tokens)


def finalize_beams(initial_ids, beam_trace, model_config, length_penalty):
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
        result = []
        while beam != root:
            score, prev, tok, is_finished = beam
            result = [tok] + result
            beam = prev
        return result

    prev_beams = [[root]] * bs
    best = [root] * bs

    def beam_score(beam):
        return (beam[3], beam[0])

    for step, (scores, indices, tokens) in enumerate(zip(beam_trace_scores, beam_trace_indices, beam_trace_tokens)):
        cur_beams = [[] for _ in range(bs)]
        for idx, (s, i, t) in enumerate(zip(scores, indices, tokens)):
            batch = idx // (num_beams * 2)
            idx = idx % (num_beams * 2)
            b_len = 1 + step
            b_score = s.item() / (b_len ** length_penalty)
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


def expand_layer_cache(past, new_size, value):
    new_k = expand_if_needed(past[0], new_size, value, dim=-2)
    new_v = expand_if_needed(past[1], new_size, value, dim=-2)
    return (new_k, new_v)


def expand_cache(cache, new_size, value):
    return tuple(expand_layer_cache(layer_past, new_size, value) for layer_past in cache)


def reorder_cache(cache, indices):
    return map_tensors(cache, lambda t: t.index_select(0, indices))


def expand_and_update_if_needed(args, key, new_size, value):
    if key in args:
        args[key] = expand_if_needed(args[key], new_size, value)


def trim_and_update_if_needed(args, key, idx):
    if key in args:
        args[key] = args[key].cpu()[:, :idx, :]


def enable_statistics(model):
    if hasattr(model, 'iterations'):
        return
    old_fwd = model.forward
    model.iterations = 0

    def fwd(*args, **kwargs):
        model.iterations = model.iterations + 1
        return old_fwd(*args, **kwargs)
    model.forward = fwd


def fmt_float(x, suffix=''):
    return f'{x:.3f}{suffix}'


def count_hpu_graphs():
    return len(glob.glob('.graph_dumps/*PreGraph*'))


def create_pipeline(model, tokenizer, mode, calc_stats=False):
    if calc_stats:
        enable_statistics(model)

    def pipeline(inputs, options):
        model_args = tokenizer(inputs, return_tensors="pt", padding=True)

        if calc_stats:
            input_tokens = torch.numel(model_args['input_ids'])
            model.iterations = 0
            generate_start = time.perf_counter()
        if mode == GenerationMode.VANILLA:
            model_args = model_args.to(get_device(model))
            output = model.generate(**options.filter(*custom_options()), **model_args)
        elif mode == GenerationMode.OPTIMIZED:
            output = generate(model, options, model_args)
        else:
            assert False, f'Unsupported generation mode: {mode}'
        output = output.cpu()
        if calc_stats:
            generate_end = time.perf_counter()
        tokens = tokenizer.batch_decode(output.cpu(), skip_special_tokens=True)
        if not calc_stats:
            return tokens

        bs = output.shape[0]
        iterations = model.iterations
        generate_time = generate_end - generate_start
        out_tok = torch.numel(output)
        out_latency = generate_time / out_tok
        stats = [
            ('duration', generate_time, 's'),
            ('iterations', model.iterations, ''),
            ('in_tok', input_tokens, ''),
            ('out_tok', out_tok, ''),
            ('out_tps', (out_tok / generate_time), ''),
            ('iter_tps', (iterations * bs / generate_time), ''),
            ('out_latency', (out_latency), 's'),
        ]
        if is_on_hpu(model):
            stats.append(('graphs', count_hpu_graphs()))
        return tokens, stats

    return pipeline


if __name__ == '__main__':
    print(generate_option_help())
