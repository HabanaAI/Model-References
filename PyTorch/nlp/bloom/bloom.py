#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
import sys
import torch
import torch.nn.functional as F
import time
import argparse
import shutil
import glob
import inspect
import json
import tempfile
from pathlib import Path


on_start = []
on_step_begin = []
on_token_begin = []
on_token_end = []
on_step_end = []
on_stop = []


def trigger(phase):
    [f() for f in phase]


def override_print(enable):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if force or enable:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def setup_distributed(args):
    args.local_rank = int(os.getenv('LOCAL_RANK', '0'))
    args.world_size = int(os.getenv('WORLD_SIZE', '0'))
    args.global_rank = int(os.getenv('RANK', '0'))
    override_print(args.global_rank == 0 or args.verbose_workers)


def setup_device(args):
    if args.device == 'hpu':
        import habana_frameworks.torch.core as htcore
        on_token_begin.append(htcore.mark_step)
    return torch.device(args.device)


def setup_code(args):
    import transformers.models.bloom.modeling_bloom as modeling_bloom
    import transformers.generation_utils as generation_utils
    if not args.vanilla:
        import modeling_bloom as hpu_modeling_bloom
        import generation_utils as hpu_generation_utils
        modeling_bloom.BloomForCausalLM = hpu_modeling_bloom.BloomForCausalLM
        modeling_bloom.BloomForCausalLM.generate = hpu_generation_utils.GenerationMixin.generate
        modeling_bloom.BloomForCausalLM.greedy_search = hpu_generation_utils.GenerationMixin.greedy_search
        modeling_bloom.BloomForCausalLM.beam_search = hpu_generation_utils.GenerationMixin.beam_search
        modeling_bloom.BloomForCausalLM._update_model_kwargs_for_generation = hpu_generation_utils.GenerationMixin._update_model_kwargs_for_generation
        modeling_bloom.BloomForCausalLM._get_stopping_criteria = hpu_generation_utils.GenerationMixin._get_stopping_criteria
        modeling_bloom.BloomBlock = hpu_modeling_bloom.BloomBlock
    return modeling_bloom


def get_model_name(args):
    return 'bigscience/' + args.model


def prepare_weights(args):
    from huggingface_hub import snapshot_download
    try:
        name = get_model_name(args)
        weights = snapshot_download(repo_id=name, local_files_only=True, cache_dir=args.weights)
    except FileNotFoundError:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        print(f"ERROR! Unable to find weights. Please download them using {script_dir}/utils/fetch_weights.py --model {name} --weights {args.weights}")
        sys.exit(1)
    return weights


def setup_pipeline(args, model, weights, device):
    print(f'Using model from: {inspect.getfile(model.__class__)}')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(weights, local_files_only=True)
    generated_tokens = 0
    def count_tokens():
        nonlocal generated_tokens
        generated_tokens = generated_tokens + 1
    on_token_begin.append(count_tokens)

    def pipe_fn(query, **kwargs):
        nonlocal generated_tokens
        generated_tokens = 0
        tokens = tokenizer(query, return_tensors="pt", max_length=args.max_length, padding=True, truncation=True)
        if args.static_shapes:
            input_token_len = tokens.input_ids.shape[-1]
            padding_len = args.max_length - input_token_len
            kwargs['token_idx'] = torch.tensor(input_token_len, device=device)
            kwargs['max_steps'] = padding_len
            tokens['input_ids'] = F.pad(tokens.input_ids, (0, padding_len), value=model.config.pad_token_id)
            tokens['attention_mask'] = F.pad(tokens.attention_mask, (0, padding_len), value=0)
        tokens = tokens.to(device)
        out = model.generate(**tokens,
                             **kwargs,
                             ignore_eos=args.ignore_eos,
                             pre_token_hook=on_token_begin,
                             post_token_hook=on_token_end,
                             search_on_cpu=args.search_on_cpu,
                             use_cache=args.use_kv_cache
                             ).cpu()
        return tokenizer.batch_decode(out, skip_special_tokens=True), generated_tokens
    return pipe_fn


def setup_model(args, weights, device):
    from transformers import AutoModelForCausalLM
    dtype = get_dtype(args)
    model = AutoModelForCausalLM.from_pretrained(weights, local_files_only=True, torch_dtype=dtype).to(device)
    if args.use_graphs:
        import graph_utils
        model.transformer = graph_utils.wrap_in_graph(args, model.transformer)
    return model


def update_checkpoints_json(f, weights):
    bin_files = [str(entry) for entry in Path(weights).rglob('*.bin') if entry.is_file()]
    data = {
        "type": "BLOOM",
        "checkpoints": bin_files,
        "version": 1.0
    }
    json.dump(data, f)
    f.flush()


def get_dtype(args):
    if args.dtype == 'bf16':
        return torch.bfloat16
    if args.dtype == 'fp16':
        return torch.float16
    if args.dtype == 'fp32':
        return torch.float32
    assert False, f'Uknown dtype: {args.dtype}'


def setup_distributed_model(args, code, weights, device):
    import deepspeed
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(weights, local_files_only=True)

    dtype = get_dtype(args)
    with deepspeed.OnDevice(dtype=dtype, device='meta'):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)

    #change training to false in all modules.
    model = model.eval()

    f = tempfile.NamedTemporaryFile(suffix=".json", mode="+w")
    update_checkpoints_json(f, weights)
    model = deepspeed.init_inference(model,
                                     mp_size=args.world_size,
                                     dtype=dtype,
                                     injection_policy={code.BloomBlock: ('mlp.dense_4h_to_h', 'self_attention.dense')},
                                     args=args,
                                     enable_cuda_graph=args.use_graphs,
                                     checkpoint=f.name)
    if args.model == "bloom" and args.device == 'hpu':
        #TODO: remove once MME issue is solved
        model.module.split_lm_head()
    return model


def setup_profiler(args, steps):
    if args.profile is None or args.global_rank != 0:
        return

    start_token, end_token = map(int, args.profile_tokens.split(','))
    active_steps = end_token - start_token + 1
    warmup_steps = start_token
    cur_step = 0
    cur_token = 0

    def on_step_begin_fn():
        nonlocal cur_step, cur_token
        cur_step = cur_step + 1
        cur_token = 0
    on_step_begin.append(on_step_begin_fn)

    def on_token_begin_fn():
        nonlocal cur_token
        cur_token = cur_token + 1
    on_token_begin.append(on_token_begin_fn)

    def when(cond, clbk):
        def fn():
            if cond():
                clbk()
        return fn

    def is_cur_step():
        return cur_step == steps

    def is_start_token():
        return is_cur_step() and cur_token == start_token

    def is_end_token():
        return is_cur_step() and cur_token == end_token

    if args.profile.startswith('tb'):
        schedule = torch.profiler.schedule(wait=0, warmup=warmup_steps, active=active_steps, repeat=1)
        activities = [torch.profiler.ProfilerActivity.CPU]
        activities.extend([torch.profiler.ProfilerActivity.HPU] if args.device == 'hpu' else [])
        activities.extend([torch.profiler.ProfilerActivity.CUDA] if args.device == 'cuda' else [])
        full = args.profile == 'tb-full'

        profiler = torch.profiler.profile(
            schedule=schedule,
            activities=activities,
            on_trace_ready=torch.profiler.tensorboard_trace_handler('.', use_gzip=True),
            record_shapes=full,
            with_stack=full)

        on_step_begin.append(when(is_cur_step, profiler.start))
        on_token_begin.append(when(is_cur_step, profiler.step))
        on_step_end.append(when(is_cur_step, profiler.stop))

    elif args.profile == 'hltv':
        sys.path.append(os.environ['PYTORCH_MODULES_ROOT_PATH'])
        from topologies.tools import SynapseProfilerApi, TraceType
        api = SynapseProfilerApi()

        on_token_begin.append(when(is_start_token, lambda: api.profiler_start(TraceType.TraceAll, 0)))
        on_token_end.append(when(is_end_token, lambda: api.profiler_stop(TraceType.TraceAll, 0)))
        on_token_end.append(when(is_end_token, lambda: api.profiler_get_trace_json(TraceType.TraceAll, 0)))


def setup_env(args):
    if args.debug:
        os.environ['ENABLE_CONSOLE'] = 'true'
        os.environ['LOG_LEVEL_ALL'] = '3'
    if args.global_rank == 0:
        os.environ['GRAPH_VISUALIZATION'] = 'true'
        if args.profile is None:
            os.environ['HABANA_PROFILE'] = '0'
        elif args.profile == 'default':
            os.environ['HABANA_PROFILE'] = '1'
        elif args.profile == 'hltv':
            os.environ['HABANA_PROFILE'] = 'profile_api_with_nics'
        else:
            os.environ['HABANA_PROFILE'] = 'profile_api_light'
        shutil.rmtree('.graph_dumps', ignore_errors=True)
    else:
        os.environ['HABANA_PROFILE'] = '0'
    if args.world_size > 0:
        os.environ.setdefault('ID', str(args.local_rank))
        os.environ.setdefault('WA_BETA_ALIBI', '1')
        os.environ.setdefault('PT_HPU_LAZY_ACC_PAR_MODE', '0')
        os.environ.setdefault('PT_HPU_ENABLE_LAZY_COLLECTIVES', 'true')


def set_default(args, device, param, value):
    v = vars(args)
    prev = v[param]
    if prev is None and args.device == device:
        print(f"Using default value: '{value}' for '--{param}'")
        v[param] = value


def setup_defaults(args):
    set_default(args, 'hpu', 'static_shapes', True)
    set_default(args, 'hpu', 'use_graphs', True)
    if args.beams > 1:
        set_default(args, 'hpu', 'search_on_cpu', True)
    print(f'Runtime params: {vars(args)}')


def count_hpu_graphs():
    return len(glob.glob('.graph_dumps/*PreGraph*'))


def read_file(filename, default):
    if filename is not None:
        with open(filename) as f:
            return json.load(f)
    return default


def read_input_file(args):
    return read_file(args.input_file, [])


def read_reference_file(args):
    return read_file(args.reference_file, {})


def write_output_file(args, output):
    if args.output_file is not None:
        with open(args.output_file, 'w') as f:
            json.dump(output, f, indent=4)


def flag(v):
    char = v.lower()[0]
    assert char == 't' or char == 'f', f"Invalid value: {v} - it should start with either 't' or 'f'"
    return char == 't'


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Bloom POC for HPU')
parser.add_argument('--device', '-d', type=str, choices=['cpu', 'cuda', 'hpu'], help='Device to run', default='hpu')
parser.add_argument('--model', '-m', type=str, choices=['bloom-560m', 'bloom-1b7', 'bloom-3b', 'bloom-7b1', 'bloom'], help='Model', default='bloom-7b1')
parser.add_argument('--weights', type=str, help="Weight dir for all pretrained models", required=True)
parser.add_argument('--dtype', '-dt', type=str, choices=['fp32', 'fp16', 'bf16'], help='Precision to use', default='fp32')
parser.add_argument('--debug', action='store_true', help="Enable additional logs")
parser.add_argument('--vanilla', action='store_true', help="Use default BloomModel impl from transformers lib")
parser.add_argument('--profile', choices=['tb', 'tb-full', 'hltv', 'default'], help="Enable profiling")
parser.add_argument('--profile_tokens', '-pt', type=str, default='1,10', help="Range of tokens to profile in last step")
parser.add_argument('--repeat', '-r', type=int, help="Number of times each query should be repeated", default=1)
parser.add_argument('--max_length', '-ml', type=int, help="Number of maximum output tokens", default=20)
parser.add_argument('--batch_size', '-bs', type=int, help="Number of queries per batch", default=1)
parser.add_argument('--local_rank', type=int, help="Local rank used by DeepSpeed", default=0)
parser.add_argument('--verbose_workers', action='store_true', help="Enable output from non-master workers")
parser.add_argument('--use_graphs', type=flag, help="Enable using HPU graphs")
parser.add_argument('--limit', '-l', type=int, help="Maximum number of queries to process")
parser.add_argument('--input_file', '-if', type=str, help="Read queries from a file")
parser.add_argument('--output_file', '-of', type=str, help="Save output to a file")
parser.add_argument('--reference_file', '-rf', type=str, help="Compare output with references read from a file")
parser.add_argument('--beams', '-b', type=int, help="Number of decoding beams", default=1)
parser.add_argument('--sample', '-s', action='store_true', help="Enable multinomial sampling", default=False)
parser.add_argument('--static_shapes', type=flag, help="Enable static shapes. Default=True on HPU")
parser.add_argument('--ignore_eos', type=flag, help="Ignore eos token in greedy_search", default=True)
parser.add_argument('--search_on_cpu', type=flag, help="Move searching logic to CPU. Currently only beam-search is affected. Default=True on HPU")
parser.add_argument('--use_kv_cache', type=flag, help="Use KV caching", default=True)
parser.add_argument('--async_collectives', type=flag, help="flag passed to deepspeed for activate asynchronous collective", default=True)
parser.add_argument('queries', type=str, nargs='*', help="Input queries", default=[])
args = parser.parse_args()

init_start = time.perf_counter()
setup_distributed(args)
setup_defaults(args)
setup_env(args)
device = setup_device(args)
code = setup_code(args)
weights = prepare_weights(args)
model = setup_model(args, weights, device) if args.world_size == 0 else setup_distributed_model(args, code, weights, device)
pipe = setup_pipeline(args, model, weights, device)
init_end = time.perf_counter()

print(f"Init took {(init_end - init_start):.3f}s")
print("Starting inference...")
bs = args.batch_size
references = read_reference_file(args)
queries = read_input_file(args) + args.queries
if args.limit is not None:
    queries = queries[:args.limit]
queries = queries * args.repeat
steps = (len(queries) + bs - 1) // bs
queries = [queries[i % len(queries)] for i in range(steps * bs)]
queries = [queries[(i * bs):(i + 1) * bs] for i in range(steps)]
setup_profiler(args, steps)

output = {}

kwargs = {
    'max_length': args.max_length,
    'num_beams': args.beams,
    'do_sample': args.sample,
}
errors = 0
trigger(on_start)
for batch_idx, batch in enumerate(queries):
    trigger(on_step_begin)
    ts = time.perf_counter()
    answers, generated_tokens = pipe(batch, **kwargs)
    te = time.perf_counter()
    trigger(on_step_end)

    duration = te - ts
    stats = f'step:{batch_idx} time:{duration:.3f}s tokens:{generated_tokens} tps:{(generated_tokens / duration):.3f}'
    if args.device == 'hpu':
        stats = stats + f' hpu_graphs:{count_hpu_graphs()}'
    separator = '-' * len(stats)

    print(separator)
    print(stats)
    print(separator)

    for i, (q, a) in enumerate(zip(batch, answers)):
        def print_with_label(prefix, value):
            if value is not None:
                print(f'{prefix}{batch_idx}.{i}: {value}')
        print_with_label('Q', q)
        print_with_label('A', a)
        ref = references.get(q, None)
        print_with_label('R', ref)
        if ref is not None and a != ref:
            print_with_label('E', 'Output doesn\'t match reference!')
            errors = errors + 1
        output[q] = a
trigger(on_stop)

write_output_file(args, output)
sys.exit(errors > 0)
