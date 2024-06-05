###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
import pandas as pd
import torch
import time
import json
import argparse
import struct
import contextlib
from utils import initialize_model, setup_parser, logger, print_stats


def measure_perf(txt="", tps=False, reset=False):
    if hasattr(measure_perf, "prev") and not reset and txt:
        duration = time.perf_counter()-measure_perf.prev
        if tps:
            tps_str = f"Throughput: {(args.batch_size * args.max_new_tokens)/duration:.0f} TPS"
        else:
            tps_str = ""
        logger.info(
            f"{txt} took {duration:.3f} sec. {tps_str}")
    else:
        duration = 0
    measure_perf.prev = time.perf_counter()
    return duration


def get_ds(args):
    ds = pd.read_pickle(args.dataset)

    if args.n_iterations:
        ds = ds.head(args.n_iterations * args.batch_size)

    return ds


def get_input(ds, batch_size):
    queries = []
    tok_input = ds["tok_input"].tolist()
    for start in range(0, len(ds), batch_size):
        end = start + batch_size
        batch = tok_input[start:end]
        input_ids = []
        attention_mask = []
        for query in batch:
            input_ids.append(
                [0] * (args.max_input_tokens - len(query)) + query)
            attention_mask.append(
                [0] * (args.max_input_tokens - len(query)) + [1] * len(query))
        queries.append({
            'input_ids': torch.tensor(input_ids, dtype=torch.int32),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.int32)
        })
    return queries


def setup_profiler(args):
    if args.profiling_scope == "batch" and args.profiling_steps != 0:
        import habana_frameworks.torch.core as htcore
        profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=0, warmup=args.profiling_warmup_steps, active=args.profiling_steps, repeat=1),
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.HPU],
            on_trace_ready=torch.profiler.tensorboard_trace_handler('hpu_profile'))
        return 0, 0, profiler, profiler.step
    else:
        def step():
            pass
        return args.profiling_warmup_steps, args.profiling_steps, contextlib.nullcontext(), step


def main(args, ds):
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    print_logs = (local_rank == 0)
    print(f"Dataset has {len(ds)} samples.")
    os.makedirs(args.output_dir, exist_ok=True)

    batches = get_input(ds, args.batch_size)

    model, _, generation_config = initialize_model(args, logger)
    _, _, profiler, _ = setup_profiler(args)

    def generate(input_queries, warmup=False):
        for t in input_queries:
            if torch.is_tensor(input_queries[t]):
                input_queries[t] = input_queries[t].to(args.device)
        with torch.autograd.profiler.record_function("generate:"):
            outputs = model.generate(
                **input_queries,
                generation_config=generation_config,
                lazy_mode=True,
                hpu_graphs=args.use_hpu_graphs,
                profiling_steps=args.profiling_steps if not warmup else 0,
                profiling_warmup_steps=args.profiling_warmup_steps,
            ).cpu()
        outputs = outputs.tolist()
        for i in range(len(outputs)):
            outputs[i] = outputs[i][args.max_input_tokens:]
        return outputs

    results = []
    N = len(batches)
    i = 1
    durations = []
    measure_perf("Start")
    generate(batches[0], warmup=True)
    measure_perf("Warmup")
    t_start = time.perf_counter()
    with profiler:
        for batch in batches:
            result = generate(batch)
            results.extend(result)
            durations.append(measure_perf(
                f"Generating batch {i} / {N}", tps=True))
            i += 1
    duration = time.perf_counter() - t_start

    if print_logs:
        print("Inference took {:.1f} secs".format(duration))
        print("Saving mlperf-accuracy-file...")
        acc_file = []
        num_token = 0
        for i, idx in enumerate(ds.index):
            pred = results[i]
            eos_token_id = 2
            try:
                ind_eos = pred.index(eos_token_id)+1
            except:
                ind_eos = len(pred)
            pred = pred[:ind_eos]
            num_token += len(pred)
            acc_file.append({
                "seq_id": idx,
                "qsl_idx": idx,
                "data": bytes(struct.pack('L' * len(pred), *pred)).hex().upper()
            })
        os.makedirs(args.log_path, exist_ok=True)
        path = args.log_path + "/mlperf_log_accuracy.json"
        with open(path, "w") as outfile:
            outfile.write(json.dumps(acc_file))
        estimated_performance = num_token/duration
        print("Estimated performance for accuracy run is {:.1f} tokens per second".format(
            estimated_performance))
        print("Saved to {}".format(path))

        print_stats(durations, args, len(batches))

    if args.quant_config:
        import habana_quantization_toolkit
        habana_quantization_toolkit.finish_measurements(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = setup_parser(parser)

    ds = get_ds(args)
    main(args, ds)
