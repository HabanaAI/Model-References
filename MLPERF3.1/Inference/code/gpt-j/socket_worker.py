#!/usr/bin/env python3

###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################

import argparse
import os
import torch
import time
import random
import threading
import queue

import habana_generation_utils as hgu
import modeling_gptj as hpu_modeling_gptj
import quantization.quantize as quantize
from hgu_options import get_options_dict

import socket_utils
from dataset import Dataset


MIN_NEW_TOKENS = 30
MAX_NEW_TOKENS = 128


def fatal(e):
    import traceback
    traceback.print_exc()
    print("EXCEPTION:", e, flush=True)
    os._exit(1)


def get_fake_delay(dtype: str) -> dict:
    class FakeDelayDict(dict):
        def __getitem__(self, length: int) -> int:
            key = min([key for key in self.keys() if key >= length - MAX_NEW_TOKENS - 1])
            return dict.__getitem__(self, key)

    # dict {
    #   input_length: average processing time on real device [us]
    # }
    if dtype == 'float8':
        return FakeDelayDict({
            1919: 207946,
            1663: 177573,
            1407: 162134,
            1151: 141677,
            1023: 144127,
            895: 105898,
            767: 94835,
            639: 79685,
            511: 63538
        })
    else:
        return FakeDelayDict({
            1919: 418798,
            1663: 367299,
            1407: 337564,
            1151: 292790,
            1023: 289867,
            895: 234328,
            767: 211056,
            639: 156582,
            511: 143436
        })


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=str, required=True, help="Unix socket to connect to")
    parser.add_argument("--quantization_file", "-qf", type=str,
                        help="Read quantization configuration from a file")
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint")
    parser.add_argument("--dtype", choices=["bfloat16", "float32", "float8"], required=True,
                        help="data type of the model, choose from bfloat16, float32 and float8")
    parser.add_argument("--dataset-path", required=True, help="")
    parser.add_argument("--max_examples", type=int, required=True, help="Maximum number of examples to consider (not limited by default)")
    parser.add_argument("--options", type=str, required=True,
                        help="Coma-seperated list of options used in generation")
    parser.add_argument("--fake_device", action='store_true', help="Enable dummy device with estimated delay")
    parser.add_argument("--fake_dataset", action='store_true', help="Enable dummy dataset")
    args = parser.parse_args()
    return args


def handle(sock, prepare_input_func, pipeline_func, finalize_beams_func, options):
    pipeline_queue = queue.Queue()
    thread = threading.Thread(target=run_pipeline, args=(pipeline_queue, pipeline_func, finalize_beams_func))
    thread.start()

    while True:
        try:
            data = socket_utils.receive(sock)
            if data is None:
                break
            pipeline_input = prepare_input_func(data, options)
            pipeline_queue.put(pipeline_input)
        except Exception as e:
            fatal(e)

    pipeline_queue.put(None)
    thread.join()


def prepare_input(data, options):
    batch, new_options, batch_size = data
    options.update(new_options)

    req_ids = [b[0][0] for b in batch]
    sample_ids = [b[0][1] for b in batch]
    while len(sample_ids) < batch_size:
        sample_ids.append(sample_ids[0])

    def getter(src):
        def get(idx):
            if idx != -1:
                return src[idx]
            else:
                return torch.ones((1, 1), dtype=src[0].dtype)
        return get

    src_input_ids = getter(dataset.source_encoded_input_ids)
    src_attn_masks = getter(dataset.source_encoded_attn_masks)
    input_ids = [src_input_ids(id) for id in sample_ids]
    attention_mask = [src_attn_masks(id) for id in sample_ids]
    batch, max_input_length = align_batch(input_ids, attention_mask, dataset.tokenizer.pad_token_id, options.max_input_length)

    options.set('max_input_length', max_input_length + MAX_NEW_TOKENS + 1)
    options.set('max_length', max_input_length + MAX_NEW_TOKENS + 1)
    options.set('min_length', max_input_length + MIN_NEW_TOKENS)

    batch, max_length, input_length = hgu.prepare_decoder_only_input_without_moving(dataset.tokenizer.pad_token_id, options, batch)
    return (batch, options, max_length, input_length, req_ids)


def run_pipeline(pipeline_queue, pipeline_func, finalize_beams_func):
    try:
        with torch.inference_mode():
            thread = None
            while True:
                items = pipeline_queue.get()
                if items is None:
                    break

                batch, options, max_length, input_length, req_ids = items
                initial_ids, beam_trace = pipeline_func(batch, options, max_length, input_length)

                thread = threading.Thread(target=finalize_beams_func, args=(initial_ids, beam_trace, max_length, req_ids))
                thread.start()
            thread.join()
    except Exception as e:
        fatal(e)


def finalize_beams(initial_ids, beam_trace, max_input_length, req_ids):
    try:
        output = hgu.finalize_beams(initial_ids, beam_trace, model.config, options.length_penalty)

        response = []
        for req_id, output in zip(req_ids, output):
            response.append((req_id, output[max_input_length:].numpy().tobytes()))
        socket_utils.send(sock, response)
    except Exception as e:
        fatal(e)

def left_pad(tensor, max_len, value):
    return torch.nn.functional.pad(tensor, (max_len - tensor.size(-1), 0), value=value)


def align_batch(input_ids, attention_mask, pad_token_id, max_length=None):
    input_lengths = [t.size(-1) for t in input_ids]
    if max_length is None:
        max_length = max(input_lengths)
    input_ids = [left_pad(t, max_length, pad_token_id) for t in input_ids]
    attention_mask = [left_pad(t, max_length, 0) for t in attention_mask]
    return {"input_ids": torch.cat(input_ids), "attention_mask": torch.cat(attention_mask)}, max_length


if __name__ == "__main__":
    args = get_args()

    dataset = Dataset(args.model_path, args.dataset_path, total_count_override=args.max_examples, add_padding=False, fake_data=args.fake_dataset)
    options = get_options_dict(args.options)
    options = hgu.GenerationOptions(**options)
    hgu_pipeline = None
    device = torch.device("cpu")

    if not args.fake_device:
        if int(os.environ.get('OMPI_COMM_WORLD_SIZE', 1)) > 1:
            local_rank = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', "0")
            os.environ["HLS_MODULE_ID"] = local_rank

        import habana_frameworks.torch.core as htcore
        import habana_frameworks.torch.hpu.graphs as htgraphs
        device = torch.device('hpu')

        print("Loading PyTorch model...")
        model_path = args.model_path

        model = hpu_modeling_gptj.GPTJForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16
        )

        if model.config.pad_token_id is None:
            model.config.pad_token_id = model.config.eos_token_id

        model.to(torch.bfloat16)
        model.to(device)

        model = htgraphs.wrap_in_hpu_graph(model)

        if args.quantization_file:
            model = quantize.setup_quantization(model, args.quantization_file)

        def pipeline(batch, options, max_length, input_length):
            return hgu.generate_on_prepared_input(model, options, batch, max_length, input_length)

        prepare_input_func = prepare_input
        pipeline_func = pipeline
        finalize_beams_func = finalize_beams
    else:
        fake_delay_dict = get_fake_delay(args.dtype)

        def fake_pipeline(batch, *args):
            batch_size, length = batch['input_ids'].shape
            fake_delay = fake_delay_dict[length] * random.uniform(0.9, 1.1)
            total_fake_delay = batch_size * fake_delay / 1e6
            time.sleep(total_fake_delay / 10)
            return batch['input_ids'], None

        def fake_finalize_beams(initial_ids, _, max_input_length, req_ids):
            try:
                output = initial_ids.repeat(1, 2)
                response = []
                for req_id, output in zip(req_ids, output):
                    response.append((req_id, output[max_input_length:].numpy().tobytes()))
                socket_utils.send(sock, response)
            except Exception as e:
                fatal(e)

        prepare_input_func = prepare_input
        pipeline_func = fake_pipeline
        finalize_beams_func = fake_finalize_beams

    if args.dtype == "float8":
        options.kv_cache_fp8 = True

    sock = socket_utils.connect(args.socket)
    handle(sock, prepare_input_func, pipeline_func, finalize_beams_func, options)
