#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
import sys
import torch
import time
import argparse
import glob
import json
import tempfile
import shutil
from pathlib import Path
import numpy as np
import random
import habana_generation_utils as hgu


def flag(v):
    char = v.lower()[0]
    assert char == 't' or char == 'f', f"Invalid value: {v} - it should start with either 't' or 'f'"
    return char == 't'


extra_arg_types = {
    'early_stopping': flag,
    'min_length': int,
    'do_sample': flag,
    'num_beam_groups': int,
    'temperature': float,
    'top_p': float,
    'diversity_penalty': float,
    'repetition_penalty': float,
    'length_penalty': float,
    'no_repeat_ngram_size': int,
    'renormalize_logits': flag,
}


def override_print(enable):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if force or enable:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def setup_seed(args):
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)


def setup_distributed(args):
    args.local_rank = int(os.getenv('LOCAL_RANK', '0'))
    args.world_size = int(os.getenv('WORLD_SIZE', '0'))
    args.global_rank = int(os.getenv('RANK', '0'))
    override_print(args.global_rank == 0 or args.verbose_workers)


def setup_device(args):
    if args.device == 'hpu':
        import habana_frameworks.torch.core as htcore
    return torch.device(args.device)


def setup_code(args):
    if args.vanilla_model:
        import transformers.models.bloom.modeling_bloom as code
        return code
    else:
        import modeling_bloom as code
        return code


def prepare_weights(args):
    from huggingface_hub import snapshot_download
    try:
        name = 'bigscience/' + args.model
        weights = snapshot_download(repo_id=name, local_files_only=True, cache_dir=args.weights)
    except FileNotFoundError:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        print(f"ERROR! Unable to find weights. Please download them using {script_dir}/utils/fetch_weights.py --model {name} --weights {args.weights}")
        sys.exit(1)
    return weights


def setup_model(args, code, weights):
    dtype = get_dtype(args)
    model = code.BloomForCausalLM.from_pretrained(weights, local_files_only=True, torch_dtype=dtype).to(args.device)
    model = model.eval()
    if args.use_graphs:
        import habana_frameworks.torch.hpu.graphs as htgraphs
        model = htgraphs.wrap_in_hpu_graph(model)
    return model


def setup_tokenizer(weights):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(weights, local_files_only=True)
    return tokenizer


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


def setup_distributed_model(args, code, weights):
    import deepspeed
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(weights, local_files_only=True)

    dtype = get_dtype(args)
    with deepspeed.OnDevice(dtype=dtype, device='meta'):
        model = code.BloomForCausalLM(config)

    #change training to false in all modules.
    model = model.eval()

    f = tempfile.NamedTemporaryFile(suffix=".json", mode="+w")
    update_checkpoints_json(f, weights)
    kwargs = dict(dtype=dtype, checkpoint=f.name)
    kwargs["tensor_parallel"] = {"tp_size": args.world_size}
    if args.no_ds_fork:
        kwargs['replace_with_kernel_inject'] =True
        kwargs['base_dir'] = weights
    else:
        kwargs['injection_policy'] = {code.BloomBlock: ("self_attention.dense", "mlp.dense_4h_to_h")}
        kwargs['enable_cuda_graph'] = args.use_graphs

    model = deepspeed.init_inference(model,
                                     **kwargs)
    if args.model == "bloom" and args.device == 'hpu':
        #TODO: remove once MME issue is solved
        model.module.split_lm_head()
    return model


def setup_env(args):
    if args.debug:
        os.environ.setdefault('ENABLE_CONSOLE', 'true')
        os.environ.setdefault('LOG_LEVEL_ALL', '3')

    profile_flag = '0'
    if args.global_rank == 0:
        os.environ.setdefault('GRAPH_VISUALIZATION', 'true')
        if args.profile_tokens is not None and args.profile_tokens > 0:
            if args.world_size > 0:
                profile_flag = 'profile_api_with_nics'
            else:
                profile_flag = 'profile_api_light'
        shutil.rmtree('.graph_dumps', ignore_errors=True)
    os.environ.setdefault('HABANA_PROFILE', profile_flag)

    if args.world_size > 0:
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


def setup_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Bloom POC for HPU')
    parser.add_argument('--device', '-d', type=str, choices=['cpu', 'cuda', 'hpu'], help='Device to run', default='hpu')
    parser.add_argument('--debug', action='store_true', help="Enable additional logs")

    parser.add_argument('--model', '-m', type=str, choices=['bloom-560m', 'bloom-1b7', 'bloom-3b', 'bloom-7b1', 'bloom'], help='Model', default='bloom-7b1')
    parser.add_argument('--weights', type=str, help="Weight dir for all pretrained models", required=True)
    parser.add_argument('--dtype', '-dt', type=str, choices=['fp32', 'fp16', 'bf16'], help='Precision to use', default='fp32')

    parser.add_argument('--vanilla_model', action='store_true', help="Use default BloomModel impl from transformers lib")
    parser.add_argument('--generation_mode', type=hgu.GenerationMode, choices=list(hgu.GenerationMode), default=hgu.GenerationMode.OPTIMIZED, help="Selected generation mode")
    parser.add_argument('--extra_generation_args', type=str, help="Experimental. Extra arguments passed to generate()"
                        " in the form of KEY1:VALUE1,KEY2:VALUE2,[...] . Requires running in compatibility_mode."
                        " Supported keys:" + ', '.join(extra_arg_types.keys()), default='')

    parser.add_argument('--profile_tokens', '-pt', type=int, help="Enable profiling and capture K tokens", default=None)
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
    parser.add_argument('--no_ds_fork', action='store_true', help="Whether using original deepspeed or deepspeed-fork")
    parser.add_argument('--static_shapes', type=flag, help="Enable static shapes. Default=True on HPU")
    parser.add_argument('--ignore_eos', type=flag, help="Ignore eos token in greedy_search", default=True)
    parser.add_argument('--use_kv_cache', type=flag, help="Use KV caching", default=True)
    parser.add_argument('--seed', type=int, help="random seed to use")
    parser.add_argument('--iters_to_ignore', type=int, help="number of iterations to ignore when measuring avg duration", default=3)
    parser.add_argument('queries', type=str, nargs='*', help="Input queries", default=[])
    return parser


def initialize_model(args):
    init_start = time.perf_counter()
    setup_seed(args)
    if args.no_ds_fork:
        assert args.vanilla_model, "Can't use regular DeepSpeed without vanilla BloomModel implementation"
        assert args.device != "hpu", "Can't use hpu device with regular DeepSpeed implementation"
    setup_distributed(args)
    setup_defaults(args)
    setup_env(args)
    setup_device(args)
    code = setup_code(args)
    print(f'Using model code from {code.__file__}')
    weights = prepare_weights(args)
    model = setup_model(args, code, weights) if args.world_size == 0 else setup_distributed_model(args, code, weights)
    tokenizer = setup_tokenizer(weights)
    init_end = time.perf_counter()
    print(f"Model initialization took {(init_end - init_start):.3f}s")
    return model, tokenizer


def setup_profiler(args, steps):
    if args.profile_tokens is None or args.profile_tokens <= 0:
        def noop():
            pass
        return (noop, noop, noop)

    active = 1 if steps > 0 else 0
    warmup = 1 if steps > 1 else 0
    wait = steps - warmup - active

    schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1)
    activities = [torch.profiler.ProfilerActivity.CPU]
    activities.extend([torch.profiler.ProfilerActivity.HPU] if args.device == 'hpu' else [])
    activities.extend([torch.profiler.ProfilerActivity.CUDA] if args.device == 'cuda' else [])

    profiler = torch.profiler.profile(
        schedule=schedule,
        activities=activities,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('.', use_gzip=True),
        record_shapes=True,
        with_stack=True)
    return (profiler.start, profiler.step, profiler.stop)


def setup_generation_args(args):
    kwargs = {
        'max_length': args.max_length,
        'num_beams': args.beams,
        'use_cache': args.use_kv_cache,
        'static_shapes': args.static_shapes,
        'ignore_eos': args.ignore_eos,
    }

    if args.generation_mode != hgu.GenerationMode.OPTIMIZED:
        extra = [kv.split(':') for kv in args.extra_generation_args.split(',') if len(kv) > 0]
        for k, v in extra:
            assert k in extra_arg_types, f'Unsupported generation argument: {k}'
            kwargs[k] = extra_arg_types[k](v)
    else:
        assert not args.extra_generation_args, "Extra generation args are not supported in 'optimized' generation mode"
    return kwargs


def count_fwd_passes(model):
    old_fwd = model.forward
    model.runs = 0

    def fwd(*args, **kwargs):
        model.runs = model.runs + 1
        return old_fwd(*args, **kwargs)
    model.forward = fwd
    return model


def main():
    parser = setup_parser()
    args = parser.parse_args()
    model, tokenizer = initialize_model(args)
    model = count_fwd_passes(model)
    pipeline = hgu.create_pipeline(model, tokenizer, generation_mode=args.generation_mode)

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
    batches = len(queries)

    on_start, on_step, on_stop = setup_profiler(args, batches)

    output = {}

    kwargs = setup_generation_args(args)

    errors = 0
    total_time = 0.
    separator = ''

    on_start()
    for batch_idx, batch in enumerate(queries):
        profiling_enabled = args.profile_tokens is not None and batch_idx == batches - 1
        max_iterations = args.profile_tokens if profiling_enabled else None

        model.runs = 0
        ts = time.perf_counter()
        answers = pipeline(batch, max_iterations=max_iterations, **kwargs)
        te = time.perf_counter()
        generated_tokens = model.runs * args.batch_size

        duration = te - ts
        stats = f'step:{batch_idx} time:{duration:.3f}s tokens:{generated_tokens} tps:{(generated_tokens / duration):.3f}'
        if args.device == 'hpu':
            stats = stats + f' hpu_graphs:{count_hpu_graphs()}'
        separator = '-' * len(stats)

        print(separator)
        print(stats)
        if batch_idx >= args.iters_to_ignore:
            total_time += duration
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
        on_step()
    on_stop()

    total_queries = len(queries) - args.iters_to_ignore
    total_time = total_time / total_queries if total_queries > 0 else 0
    print(separator)
    print(separator)
    print(f"Average query time is {(total_time):.3f}s")

    write_output_file(args, output)
    sys.exit(errors > 0)


if __name__ == '__main__':
    main()
