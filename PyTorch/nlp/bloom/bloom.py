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
import itertools
import statistics
from pathlib import Path
import numpy as np
import random
import habana_generation_utils as hgu

DEFAULT_NUM_PROFILE_TOKENS = 5


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
    if hasattr(args,'seed') and args.seed is not None:
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
        if args.quantization_file:
            htcore.hpu_set_env()
    return torch.device(args.device)


def get_default_options(model_type, is_vanilla, world_size):
    if model_type == 'bloom' and not is_vanilla:
        return {'static_shapes': True,
                'trim_logits': True,
                'reuse_cache': True,
                'limit_graphs': world_size > 0,
                'kv_cache_fp8': False}
    elif not is_vanilla:
        return {'static_shapes':True,
                'limit_graphs': world_size > 0,
                'kv_cache_fp8': False}
    else:
        return {}


def setup_code(args, config):
    class ModelCode:
        def __init__(self, from_config, from_pretrained, injection_policy, default_options):
            self.from_config = from_config
            self.from_pretrained = from_pretrained
            self.injection_policy = injection_policy
            self.default_options = default_options

    model_type = config.model_type
    default_options = get_default_options(model_type, args.vanilla_model if hasattr(args, 'vanilla_model') else False, args.world_size)

    if model_type == 'bloom':
        if hasattr(args, 'vanilla_model') and args.vanilla_model:
            import transformers.models.bloom.modeling_bloom as code
        else:
            import modeling_bloom as code
        print(f'Using model-specific code from {code.__file__}')
        return ModelCode(code.BloomForCausalLM,
                         code.BloomForCausalLM.from_pretrained,
                         {code.BloomBlock: ("self_attention.dense", "mlp.dense_4h_to_h")},
                         default_options)
    else:
        if not args.vanilla_model:
            import optimum.habana.transformers.modeling_utils as utils
            print(f'Enabling optimizations from optimum-habana')
            utils.adapt_transformers_to_gaudi()

        import transformers.models.auto.modeling_auto as code
        print(f'Using auto-model code from {code.__file__}')
        return ModelCode(code.AutoModelForCausalLM.from_config,
                         code.AutoModelForCausalLM.from_pretrained,
                         {},
                         default_options)


def get_model_name(name):
    # Prepend for backward compatibility
    if name.startswith('bloom'):
        return 'bigscience/' + name
    return name


def find_weights(args):
    if not hasattr(args, 'weights') or args.weights is None:
        return None
    assert args.model is not None, '--model is required when using --weights'
    model_name = get_model_name(args.model)
    from huggingface_hub import snapshot_download
    try:
        weights = snapshot_download(repo_id=model_name, local_files_only=True, cache_dir=args.weights)
    except FileNotFoundError:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        print(f"ERROR! Unable to find weights. Please download them using {script_dir}/utils/fetch_weights.py --model {model_name} --weights {args.weights}")
        sys.exit(1)
    return weights


def setup_config(args, weights):
    assert weights is not None or args.config is not None, 'Cannot find default config! Use either --model and --weights or --config'
    if not hasattr(args, 'config') or args.config is None:
        cfg = weights + '/config.json'
    else:
        cfg = args.config
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(cfg, local_files_only=True)
    if config.pad_token_id is None:
        config.pad_token_id = config.eos_token_id
    return config


def setup_model(args, code, weights, config, options):
    dtype = get_dtype(args)
    if hasattr(args, 'config') and args.config is not None:
        with torch.device("meta"):
            model = code.from_config(config).to(dtype)
        model = model.to_empty(device=args.device)
    else:
        model = code.from_pretrained(weights, local_files_only=True, torch_dtype=dtype)
        model = model.to(args.device)
    model = model.eval()
    if options.use_graphs and args.device == 'hpu':
        import habana_frameworks.torch.hpu.graphs as htgraphs
        model = htgraphs.wrap_in_hpu_graph(model)
    return model


def setup_tokenizer(weights, config):
    from transformers import AutoTokenizer, BatchEncoding
    if weights is not None:
        tokenizer = AutoTokenizer.from_pretrained(weights, local_files_only=True)
    else:
        class FakeTokenizer:
            def __init__(self, config):
                self.pad_token = config.pad_token_id
                self.eos_token = config.eos_token_id

            def __call__(self, batch, *args, **kwargs):
                max_words = max([len(s.split()) for s in batch])
                input_ids = torch.full((len(batch), max_words), self.eos_token)
                attention_mask = torch.full((len(batch), max_words), 1)
                return BatchEncoding(data={'input_ids': input_ids, 'attention_mask': attention_mask})

            def batch_decode(self, batch, *args, **kwargs):
                return ['<FakeOutput>'] * batch.size(0)

        tokenizer = FakeTokenizer(config)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def setup_quantization(model, quantization_config_file):
    from quantization import quantization as quant
    quant_config = quant.parse_configuration(quantization_config_file)
    if quant_config.quantization_enabled:
        print("Initializing inference with quantization")
        quant.apply_quantization(model, quant_config)
        import habana_frameworks.torch.core as htcore
        htcore.hpu_initialize(model)
    return model

def list_bin_files(weights):
    return [str(entry) for entry in Path(weights).rglob('*.bin') if entry.is_file()]


def update_checkpoints_json(f, bin_files):
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


def split(tensor, dim, world_size, global_rank):
    assert tensor.size(dim) % world_size == 0
    split_size = tensor.size(dim) // world_size
    return torch.split(tensor, split_size, dim)[global_rank].contiguous()


def join(partial, world_size):
    all = [torch.empty_like(partial) for i in range(world_size)]
    torch.distributed.all_gather(all, partial)
    return torch.cat(all, dim=-1)


class SplitEmbedding(torch.nn.Module):
    def __init__(self, weight, world_size):
        super().__init__()
        self.weight = torch.nn.parameter.Parameter(data=weight, requires_grad=False)
        self.world_size = world_size

    def forward(self, input):
        result = torch.nn.functional.embedding(input, self.weight)
        return join(result, self.world_size)


class SplitLinear(torch.nn.Module):
    def __init__(self, weight, world_size):
        super().__init__()
        self.weight = torch.nn.parameter.Parameter(data=weight.t().contiguous(), requires_grad=False)
        self.world_size = world_size

    def forward(self, input):
        result = torch.matmul(input, self.weight)
        return join(result, self.world_size)


def setup_distributed_model(args, model_code, weights, config, options):
    import deepspeed

    dtype = get_dtype(args)
    with deepspeed.OnDevice(dtype=dtype, device='meta'):
        model = model_code.from_config(config)
    # change training to false in all modules.
    model = model.eval()

    kwargs = dict(dtype=dtype)
    use_dummy_weights = hasattr(args, 'config') and args.config is not None
    if not use_dummy_weights:
        f = tempfile.NamedTemporaryFile(suffix=".json", mode="+w")
        bin_files = list_bin_files(weights)
        update_checkpoints_json(f, bin_files)
        kwargs["checkpoint"] = f.name
    kwargs["tensor_parallel"] = {"tp_size": args.world_size}
    kwargs['enable_cuda_graph'] = options.use_graphs
    if hasattr(args, 'kernel_inject') and args.kernel_inject:
        kwargs['replace_with_kernel_inject'] = True
        kwargs['base_dir'] = weights
    else:
        kwargs['injection_policy'] = model_code.injection_policy

    model = deepspeed.init_inference(model,
                                     **kwargs)

    if not args.no_split_emb and config.model_type == 'bloom':
        new_emb = split(model.module.transformer.word_embeddings.weight, 1, args.world_size, args.global_rank)
        model.module.transformer.word_embeddings = SplitEmbedding(new_emb, args.world_size)
        if args.device == 'hpu':
            import habana_frameworks.torch.core as htcore
            htcore.mark_step()

    if use_dummy_weights:
        model.to_empty(device=args.device)

    return model


def setup_env(args):
    os.environ.setdefault('PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES', '0')
    os.environ.setdefault('EXPERIMENTAL_WEIGHT_SHARING', 'FALSE')

    profile_flag = '0'
    if args.global_rank == 0:
        os.environ.setdefault('GRAPH_VISUALIZATION', 'true')
        if hasattr(args, 'profile') and args.profile:
            if args.world_size > 0:
                profile_flag = 'profile_api_with_nics'
            elif args.profile_type == 'hltv':
                profile_flag = 'profile_api'
            else:
                profile_flag = 'profile_api_light'
        shutil.rmtree('.graph_dumps', ignore_errors=True)

    # DeepSpeed loads htcore which sets HABANA_PROFILE=profile_api_light by default.
    # Need to override that value
    os.environ['HABANA_PROFILE'] = profile_flag

    if args.world_size > 0:
        os.environ.setdefault('PT_HPU_ENABLE_LAZY_COLLECTIVES', 'true')
        os.environ.setdefault('HLS_MODULE_ID', str(args.local_rank))
        os.environ.setdefault('ID', str(args.global_rank))
        if args.world_size < 9:
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


def list_models(prefix, models, sizes):
    return [prefix + m[0] + m[1] for m in itertools.product(models, sizes)]


def setup_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='LLM inference script for HPU')
    parser.add_argument('--device', '-d', type=str, choices=['cpu', 'cuda', 'hpu'], help='Device to run', default='hpu')

    all_models = list_models('bigscience/', ['bloom', 'bloomz'], ['-560m', '-1b7', '-3b', '-7b1', '']) + \
        list_models('facebook/', ['opt'], ['-13b', '-30b', '-66b'])

    model_or_config = parser.add_mutually_exclusive_group(required=True)
    model_or_config.add_argument('--model', '-m', type=str, help='Model name. Example values:' + ', '.join(all_models))
    model_or_config.add_argument('--config', type=str, help="Path to model config file. Implies running with uninitialized weights")

    parser.add_argument('--dtype', '-dt', type=str, choices=['fp32', 'fp16', 'bf16'], help='Precision to use', default='fp32')
    parser.add_argument('--weights', type=str, help="Weight dir for all pretrained models and tokenizers")

    parser.add_argument('--vanilla_model', action='store_true', help="Use default model implementation from transformers lib")
    parser.add_argument('--kernel_inject', action='store_true', help="Enable replace_with_kernel_inject mode in DeepSpeed")
    parser.add_argument('--local_rank', type=int, help="Local rank used by DeepSpeed", default=0)
    parser.add_argument('--verbose_workers', action='store_true', help="Enable output from non-master workers")

    parser.add_argument('--mode', type=hgu.GenerationMode, choices=list(hgu.GenerationMode), default=hgu.GenerationMode.OPTIMIZED, help="Selected generation mode")
    parser.add_argument('--options', type=str, help="Coma-seperated list of options used in generation. See habana_generation_utils for more details")
    parser.add_argument('--profile', action='store_true', help="Enable profiling in last step")
    parser.add_argument('--profile_type', type=str, choices=['tb', 'hltv'], default='tb')

    parser.add_argument('--repeat', '-r', type=int, help="Number of times each query should be repeated", default=1)
    parser.add_argument('--batch_size', '-bs', type=int, help="Number of queries per batch", default=1)
    parser.add_argument('--limit', '-l', type=int, help="Maximum number of queries to process")

    parser.add_argument('--input_file', '-if', type=str, help="Read queries from a file")
    parser.add_argument('--output_file', '-of', type=str, help="Save output to a file")
    parser.add_argument('--reference_file', '-rf', type=str, help="Compare output with references read from a file")
    parser.add_argument('--seed', type=int, help="random seed to use")
    parser.add_argument('--quantization_file', '-qf', type=str, help="Read quantization configuration from a file")
    parser.add_argument('--const_serialization_path', '-csp', type=str, help="Path to serialize const params")

    parser.add_argument('--max_length', '-ml', type=int, help="[DEPRECATED] Number of maximum output tokens")
    parser.add_argument('--use_kv_cache', type=flag, help="[DEPRECATED] Use KV caching")
    parser.add_argument('--ignore_eos', type=flag, help="[DEPRECATED] Ignore eos token in greedy_search")
    parser.add_argument('--static_shapes', type=flag, help="[DEPRECATED] Enable static shapes")
    parser.add_argument('--min_length', type=int, help="[DEPRECATED] Min length")
    parser.add_argument('--beams', '-b', type=int, help="[DEPRECATED] Number of decoding beams")
    parser.add_argument('--use_graphs', type=flag, help="[DEPRECATED] Enable using HPU graphs")
    parser.add_argument('--no_split_emb', action='store_true', help="Don't split Embedding when run under DeepSpeed [Bloom only]")

    parser.add_argument('queries', type=str, nargs='*', help="Input queries", default=[])
    return parser


def setup_options(args, default_values):
    print(f'Runtime params: {vars(args)}\n')
    options = hgu.parse_options(args.options if hasattr(args, 'options') else None, default_values)
    kv = vars(args)

    def handle_deprecated(old_name, new_name=None):
        if new_name is None:
            new_name = old_name
        old_value = kv.get(old_name, None)
        if old_value is not None:
            print(f'*** Warning! Using --{old_name}={old_value} is deprecated! Append {new_name}={old_value} to generation options instead!\n')
            options[new_name] = old_value

    handle_deprecated('max_length')
    handle_deprecated('use_kv_cache', 'use_cache')
    handle_deprecated('ignore_eos')
    handle_deprecated('static_shapes')
    handle_deprecated('beams', 'num_beams')
    handle_deprecated('use_graphs', 'use_graphs')

    return options


def initialize_model(args):
    init_start = time.perf_counter()
    setup_seed(args)
    if hasattr(args, 'kernel_inject') and args.kernel_inject:
        assert hasattr(args, 'vanilla_model') and args.vanilla_model, "Can't use regular DeepSpeed without vanilla model implementation"
    setup_distributed(args)
    setup_env(args)
    setup_device(args)
    weights = find_weights(args)
    config = setup_config(args, weights)
    model_code = setup_code(args, config)
    options = setup_options(args, model_code.default_options)
    tokenizer = setup_tokenizer(weights, config)
    if args.const_serialization_path:
        import uuid
        args.const_serialization_path = os.path.join(args.const_serialization_path  + uuid.uuid4().hex)
        os.makedirs(args.const_serialization_path)
        from habana_frameworks.torch.hpu import enable_const_section_serialization
        print("Serializing const params to {}".format(args.const_serialization_path))
        enable_const_section_serialization(args.const_serialization_path, False)

    model = setup_model(args, model_code, weights, config, options) if args.world_size == 0 else setup_distributed_model(args, model_code, weights, config, options)
    if args.quantization_file or os.environ.get('MARK_CONSTS', '0') == '1':
        from habana_frameworks.torch.core.quantization import _mark_params_as_const, _check_params_as_const
        _mark_params_as_const(model)
        _check_params_as_const(model)
    if args.quantization_file:
        model = setup_quantization(model, args.quantization_file)
    init_end = time.perf_counter()
    print(f"Model initialization took {(init_end - init_start):.3f}s")
    return model, tokenizer, options


def setup_pt_profiler(schedule, device):
    activities = [torch.profiler.ProfilerActivity.CPU]
    activities.extend([torch.profiler.ProfilerActivity.HPU] if device == 'hpu' else [])
    activities.extend([torch.profiler.ProfilerActivity.CUDA] if device == 'cuda' else [])

    profiler = torch.profiler.profile(
        schedule=schedule,
        activities=activities,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('.', use_gzip=True),
        record_shapes=True,
        with_stack=True)
    return profiler


def setup_hltv_profiler(schedule):
    sys.path.append(os.environ['PYTORCH_MODULES_ROOT_PATH'])
    from topologies.tools import SynapseProfilerApi, TraceType
    api = SynapseProfilerApi()

    class SynapseProfiler:
        def check(self):
            if schedule(self.cur_step) == torch.profiler.ProfilerAction.RECORD_AND_SAVE:
                api.profiler_start(TraceType.TraceAll, 0)

        def start(self):
            self.cur_step = 0
            self.check()

        def step(self):
            self.cur_step = self.cur_step + 1
            self.check()

        def stop(self):
            api.profiler_stop(TraceType.TraceAll, 0)
            api.profiler_get_trace_json(TraceType.TraceAll, 0)

    return SynapseProfiler()


def setup_profiler(args, step):
    if args.global_rank > 0 or not args.profile:
        return None

    active = 1
    warmup = 1 if step > 0 else 0
    wait = max(step - warmup, 0)

    schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1)

    if args.profile_type == 'tb':
        return setup_pt_profiler(schedule, args.device)
    else:
        return setup_hltv_profiler(schedule)


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

    model, tokenizer, options = initialize_model(args)
    inner_model = count_fwd_passes(hgu.unwrap_ds(model))
    pipeline = hgu.create_pipeline(model, tokenizer, mode=args.mode)

    print("Preparing inputs...")
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

    def query_length(q):
        return tokenizer(q, return_tensors="pt", padding=True)['input_ids'].size(-1)

    qlens = [query_length(q) for q in queries]
    max_input_length = max(qlens)
    input_stats = [
        f"count={len(qlens)}",
        f"min={min(qlens)}",
        f"max={max_input_length}",
        f"avg={statistics.mean(qlens)}",
        f"median={statistics.median(qlens)}",
    ]
    print("Input length statistics [tokens]:", ', '.join(input_stats))
    if 'max_input_length' not in options:
        options.set('max_input_length', max_input_length)

    profiler = setup_profiler(args, batches - 1)

    output = {}

    errors = 0
    separator = ''
    orig_max_iterations = options.max_iterations

    options.print()
    print("Starting inference...")
    if profiler:
        profiler.start()
    for batch_idx, batch in enumerate(queries):
        if profiler is not None and batch_idx == batches - 1:
            options.max_iterations = DEFAULT_NUM_PROFILE_TOKENS
        else:
            options.max_iterations = orig_max_iterations

        inner_model.runs = 0
        ts = time.perf_counter()
        answers = pipeline(batch, options)
        te = time.perf_counter()
        generated_tokens = inner_model.runs * args.batch_size

        duration = te - ts
        stats = f'step:{batch_idx} time:{duration:.3f}s tokens:{generated_tokens} tps:{(generated_tokens / duration):.3f}'
        if args.device == 'hpu':
            stats = stats + f' hpu_graphs:{count_hpu_graphs()}'

            import habana_frameworks.torch as ht
            mem_stats = ht.hpu.memory.memory_stats()
            max_used = hgu.fmt_float(mem_stats['MaxInUse'] / 1024.0 / 1024.0 / 1024.0, 'G')
            perc_used = hgu.fmt_float(100 * mem_stats['MaxInUse'] / mem_stats['Limit'], '%')
            stats = stats + f' max_hpu_mem:{max_used} ({perc_used})'
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
    if args.const_serialization_path and os.path.isdir(args.const_serialization_path):
        shutil.rmtree(args.const_serialization_path)
    write_output_file(args, output)
    sys.exit(errors > 0)


if __name__ == '__main__':
    main()
