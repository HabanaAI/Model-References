#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
###############################################################################

import argparse
import os
import sys
import torch
import time
from collections.abc import Mapping

from transformers import BertTokenizer, BertModel
import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore

CPU = torch.device('cpu')
HPU = torch.device('hpu')

def recopy_to(data, device, non_blocking=True):
    """
    copies `data` be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: recopy_to(v, device) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(recopy_to(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=non_blocking)
    return data

def recopy(static_inputs, data, non_blocking=True):
    """
    Used for hpu graph replaying
    :return: static_inputs with data recursively copied to it
    """

    if isinstance(data, Mapping):
        for k, v in data.items():
            static_inputs[k] = recopy(static_inputs.get(k, {}), v, non_blocking)
    elif isinstance(data, (tuple, list)):
        for i, _ in enumerate(data):
            static_inputs[i] = recopy(static_inputs[i], data[i], non_blocking)
    elif isinstance(data, torch.Tensor):
        static_inputs = static_inputs.copy_(data, non_blocking=non_blocking)
    return static_inputs

def get_dtype(dtype):
    if dtype == 'fp32':
        return torch.float

    if dtype == 'bf16':
        return torch.bfloat16

    return None

def get_model(use_graphs, name, checkpoint, dtype):
    model = BertModel.from_pretrained(name)

    if checkpoint is not None:
        cp = torch.load(checkpoint)
        model.load_state_dict(checkpoint)

    model = model.eval()
    if use_graphs:
        model = ht.hpu.wrap_in_hpu_graph(model)

    model = model.to(HPU)

    return model


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Resnet50 inference example on HPU')
    parser.add_argument('--use_graphs', action='store_true', help="Enable using HPU graphs")
    parser.add_argument('--checkpoint', '-cp', type=str, help="Optional path to a pre-trained checkpoint")
    parser.add_argument('--batch_size', '-bs', type=int, help="Batch size to use. Each input will be duplicated <batch size> times", default=1)
    parser.add_argument('--model', '-m', type=str, choices=['bert-large-uncased'], help="Model to use", default='bert-large-uncased')
    parser.add_argument('--perf', '-p', action='store_true', help="Measure throughput")
    parser.add_argument('--dtype', type=str, choices=['fp32', 'bf16'], help="Model data type", default='fp32')
    parser.add_argument('--iterations', '-i', type=int, help="Number of iterations to run on the input", default=1)
    parser.add_argument('--max_length', '-ml', type=int, help="Maximum input length", default=128)
    parser.add_argument('text', type=str, nargs='*', help="Text to run the model on", default=["Hello world!"])
    args = parser.parse_args()

    dtype = get_dtype(args.dtype)

    tokenizer = BertTokenizer.from_pretrained(args.model)
    model = get_model(args.use_graphs, args.model, args.checkpoint, dtype)

    max_length = 0
    total_length = 0
    for t in args.text:
        encoded_input = tokenizer(t, return_tensors='pt', padding='max_length', truncation=True, max_length=args.max_length)
        l = torch.numel(encoded_input['input_ids'])
        total_length = total_length + l
        if l > max_length:
            max_length = l

    if args.max_length > 0:
        max_length = args.max_length

    if total_length == 0 or args.batch_size < 1:
        print("No text to process.")
        exit(0)

    if args.perf:
        duration = 0

    if args.perf:
        #warm-up iteration
        encoded_input = tokenizer([args.text[0]] * args.batch_size, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
        encoded_input = recopy_to(data=encoded_input, device=HPU)
        with torch.inference_mode():
            with torch.autocast(dtype=dtype, device_type='hpu', enabled=(args.dtype != torch.float)):
                output = model(**encoded_input)

    for i in range(0, args.iterations):
        for t in args.text:
            encoded_input = tokenizer([t] * args.batch_size, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
            if args.perf:
                perf_start = time.perf_counter()
            encoded_input = recopy_to(data=encoded_input, device=HPU)
            with torch.inference_mode():
                with torch.autocast(dtype=dtype, device_type='hpu', enabled=(args.dtype != torch.float)):
                    output = model(**encoded_input)
                if i == args.iterations - 1:
                    print(recopy_to(data=output, device=CPU))
            if args.perf:
                duration = duration + (time.perf_counter() - perf_start)

    if args.perf:
        print("Did", total_length*args.batch_size*args.iterations, "tokens in", duration*1000., "ms", total_length*args.batch_size*args.iterations/duration, "tokens/s")
        print("SPS:", args.batch_size*args.iterations*len(args.text)/duration)

if __name__ == "__main__":
    main()

