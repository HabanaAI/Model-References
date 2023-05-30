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
    assert args.weights is not None, "Please specify pretrained weight location using '--weights'"
    from huggingface_hub import snapshot_download
    try:
        name = 'bigscience/' + args.model
        weights = snapshot_download(repo_id=name, local_files_only=True, cache_dir=args.weights)
    except FileNotFoundError:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        print(f"ERROR! Unable to find weights. Please download them using {script_dir}/utils/fetch_weights.py --model {name} --weights {args.weights}")
        sys.exit(1)
    return weights


def setup_model(args, code, weights, options):
    dtype = get_dtype(args)
    model = code.BloomForCausalLM.from_pretrained(weights, local_files_only=True, torch_dtype=dtype).to(args.device)
    model = model.eval()
    if options.use_graphs and args.device == 'hpu':
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


def setup_distributed_model(args, code, weights, options):
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
        kwargs['replace_with_kernel_inject'] = True
        kwargs['base_dir'] = weights
    else:
        kwargs['injection_policy'] = {code.BloomBlock: ("self_attention.dense", "mlp.dense_4h_to_h")}
        kwargs['enable_cuda_graph'] = options.use_graphs

    model = deepspeed.init_inference(model,
                                     **kwargs)
    return model


def setup_env(args):
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
        os.environ.setdefault('HLS_MODULE_ID', str(args.local_rank))
        os.environ.setdefault('ID', str(args.global_rank))
        os.environ.setdefault('HCL_USE_IN_ORDER_COLLECTIVE_GAUDI2', '1')


def set_default(args, device, param, value):
    v = vars(args)
    prev = v[param]
    if prev is None and args.device == device:
        print(f"Using default value: '{value}' for '--{param}'")
        v[param] = value


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

    parser.add_argument('--model', '-m', type=str, choices=['bloom-560m', 'bloom-1b7', 'bloom-3b', 'bloom-7b1', 'bloom'], help='Model', default='bloom-7b1')
    parser.add_argument('--weights', type=str, help="Weight dir for all pretrained models")
    parser.add_argument('--dtype', '-dt', type=str, choices=['fp32', 'fp16', 'bf16'], help='Precision to use', default='fp32')

    parser.add_argument('--vanilla_model', action='store_true', help="Use default BloomModel impl from transformers lib")
    parser.add_argument('--no_ds_fork', action='store_true', help="Whether using original deepspeed or deepspeed-fork")
    parser.add_argument('--local_rank', type=int, help="Local rank used by DeepSpeed", default=0)
    parser.add_argument('--verbose_workers', action='store_true', help="Enable output from non-master workers")

    parser.add_argument('--mode', type=hgu.GenerationMode, choices=list(hgu.GenerationMode), default=hgu.GenerationMode.OPTIMIZED, help="Selected generation mode")
    parser.add_argument('--options', type=str, help="Coma-seperated list of options used in generation. For more details run with --help_options")
    parser.add_argument('--help_options', action='store_true', help='Show detailed option help')
    parser.add_argument('--profile_tokens', '-pt', type=int, help="Enable profiling and capture K tokens")

    parser.add_argument('--repeat', '-r', type=int, help="Number of times each query should be repeated", default=1)
    parser.add_argument('--batch_size', '-bs', type=int, help="Number of queries per batch", default=1)
    parser.add_argument('--limit', '-l', type=int, help="Maximum number of queries to process")

    parser.add_argument('--input_file', '-if', type=str, help="Read queries from a file")
    parser.add_argument('--output_file', '-of', type=str, help="Save output to a file")
    parser.add_argument('--reference_file', '-rf', type=str, help="Compare output with references read from a file")
    parser.add_argument('--seed', type=int, help="random seed to use")

    parser.add_argument('--max_length', '-ml', type=int, help="[DEPRECATED] Number of maximum output tokens")
    parser.add_argument('--use_kv_cache', type=flag, help="[DEPRECATED] Use KV caching")
    parser.add_argument('--ignore_eos', type=flag, help="[DEPRECATED] Ignore eos token in greedy_search")
    parser.add_argument('--static_shapes', type=flag, help="[DEPRECATED] Enable static shapes")
    parser.add_argument('--min_length', type=int, help="[DEPRECATED] Min length")
    parser.add_argument('--beams', '-b', type=int, help="[DEPRECATED] Number of decoding beams")
    parser.add_argument('--use_graphs', type=flag, help="[DEPRECATED] Enable using HPU graphs")

    parser.add_argument('queries', type=str, nargs='*', help="Input queries", default=[])
    return parser


def setup_options(args):
    print(f'Runtime params: {vars(args)}\n')
    options = hgu.parse_options(args.options)
    kv = vars(args)

    def handle_deprecated(old_name, new_name=None):
        if new_name is None:
            new_name = old_name
        old_value = kv[old_name]
        if old_value is not None:
            print(f'*** Warning! Using --{old_name}={old_value} is deprecated! Append {new_name}={old_value} to generation options instead!\n')
            options[new_name] = old_value

    handle_deprecated('max_length')
    handle_deprecated('use_kv_cache', 'use_cache')
    handle_deprecated('ignore_eos')
    handle_deprecated('static_shapes')
    handle_deprecated('beams', 'num_beams')
    handle_deprecated('use_graphs', 'use_graphs')
    print(f'Generation options: {options}\n')
    return options


def initialize_model(args):
    init_start = time.perf_counter()
    setup_seed(args)
    if args.no_ds_fork:
        assert args.vanilla_model, "Can't use regular DeepSpeed without vanilla BloomModel implementation"
        assert args.device != "hpu", "Can't use hpu device with regular DeepSpeed implementation"
    setup_distributed(args)
    options = setup_options(args)
    setup_env(args)
    setup_device(args)
    code = setup_code(args)
    print(f'Using model code from {code.__file__}')
    weights = prepare_weights(args)
    model = setup_model(args, code, weights, options) if args.world_size == 0 else setup_distributed_model(args, code, weights, options)
    tokenizer = setup_tokenizer(weights)
    init_end = time.perf_counter()
    print(f"Model initialization took {(init_end - init_start):.3f}s")
    return model, tokenizer, options


def setup_profiler(args, steps):
    if args.global_rank > 0 or args.profile_tokens is None or args.profile_tokens <= 0:
        return None

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
    return profiler


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

    if args.help_options:
        print(hgu.generate_option_help())
        sys.exit(0)

    model, tokenizer, options = initialize_model(args)
    inner_model = count_fwd_passes(hgu.unwrap_ds(model))
    pipeline = hgu.create_pipeline(model, tokenizer, mode=args.mode)

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

    profiler = setup_profiler(args, batches)

    output = {}

    errors = 0
    separator = ''

    if profiler:
        profiler.start()
    for batch_idx, batch in enumerate(queries):
        if profiler is not None and batch_idx == batches - 1:
            options.max_iterations = args.profile_tokens
        else:
            options.max_iterations = None

        inner_model.runs = 0
        ts = time.perf_counter()
        answers = pipeline(batch,
                           options)
        te = time.perf_counter()
        generated_tokens = inner_model.runs * args.batch_size

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
        if profiler:
            profiler.step()
    if profiler:
        profiler.stop()

    print(separator)

    write_output_file(args, output)
    sys.exit(errors > 0)


if __name__ == '__main__':
    main()
