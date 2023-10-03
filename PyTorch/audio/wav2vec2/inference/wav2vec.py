#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
###############################################################################

import time
import os
import argparse
import numpy
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import torch
import csv
import multiprocessing
from threading import Lock

# Environment variables
# Note these need to be set before loading habana_framworks package
# Please do not move these from here
os.environ['PT_HPU_ENABLE_GENERIC_STREAM'] = '1'
os.environ['PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES'] = '0'
import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu.graphs as htgraphs

hpu = torch.device('hpu')
cpu = torch.device('cpu')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Wav2Vec on HPU')
parser.add_argument('--data_path', type=str, help='Local path for Librispeech test clean dataset. If the folder is empty, download only test clean', default='./dataset')
parser.add_argument('--dtype', '-dt', type=str, choices=['fp32', 'bf16'], help='Precision to use', default='fp32')
parser.add_argument('--debug', action='store_true', help="Enable additional logs")
parser.add_argument('--profile', action='store_true', help="Enable profiling")
parser.add_argument('--limit', '-l', type=int, help="Number of Queries to process", default=0)
parser.add_argument('--large', action='store_true', help="Run the large flavor of the model and not base")
parser.add_argument('--dev_clean_ds', action='store_true', help="Run the model with dev-clean(73 samples) dataset")
parser.add_argument('--repeat', '-r', type=int, help="Number of times each query should be repeated", default=1)
parser.add_argument('--use_graphs', action='store_true', help="Enable using HPU graphs")
parser.add_argument('--buckets', '-b', type=int, help="Number of buckets to use", default=0)
parser.add_argument('--accuracy', '-a', action='store_true', help="Enable accuracy measurement")
parser.add_argument('--perf', '-p', action='store_true', help="Measure performance")
parser.add_argument('--multi_thread', '-m', type=int, choices=[0,1], help="Generic Stream", default=1)
parser.add_argument('--graph_async', '-g', type=int, choices=[0,1], help="Enable HPU Graph thread", default=1)

args = parser.parse_args()
if args.use_graphs == False:
    args.multi_thread = 0
    args.graph_async = 0

def get_output(model, input_values, dtype):
    if dtype != torch.float32:
        with torch.autocast(device_type="hpu", dtype=dtype):
            logits = model(input_values).logits
    else:
        logits = model(input_values).logits
    return logits

if args.multi_thread:
    lock = Lock()

    def sync(stream, processor, logits, args, decodes, e2e, predicted, ground_truth, ds, r, i, tokens, perfs, perf_start):
        with ht.hpu.stream(stream):
          flag = False
          while flag == False:
            flag = stream.query() # check if all work on stream is done
            time.sleep(0.002) # sleep for 2 ms to unblock main thread

        with lock:
          if args.perf:
            ts = time.perf_counter()
            perfs.append(ts - perf_start)
            perf_start = ts
          # take argmax and decode
          predicted_ids = torch.argmax(logits, dim=-1)
          transcription = processor.batch_decode(predicted_ids)
          if args.perf:
              decodes.append(time.perf_counter() - perf_start)
              e2e.append(tokens[-1] + perfs[-1] + decodes[-1])
          if args.accuracy and r == 0:
              predicted.append(transcription[0])
              ground_truth.append(ds[i]['text'])

prof = args.profile

if not prof:
    os.environ['HABANA_PROFILE'] = '0'

if args.debug:
    os.environ['ENABLE_CONSOLE'] = 'true'
    os.environ['LOG_LEVEL_ALL'] = '3'
    os.environ['GRAPH_VISUALIZATION'] = 'true'

dtype = None
if args.dtype == 'fp32':
    dtype = torch.float32
elif args.dtype == 'bf16':
    dtype = torch.bfloat16
else:
    print("Invalid data type specified:", args.dtype)
    exit(1)
# load model and tokenizer
if not args.large:
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
else:
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

model = model.eval()
if args.use_graphs:
    model = htgraphs.wrap_in_hpu_graph(model, asynchronous=args.graph_async)
model = model.to(hpu)

# load dataset and read soundfiles
if args.dev_clean_ds:
    ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
else:
    ds = load_dataset('./librispeech_asr_test_clean.py', "clean", split="test", cache_dir=args.data_path, verification_mode='no_checks')

sampling_rate = ds.features['audio'].sampling_rate

limit = len(ds)
if args.limit > 0 and args.limit < limit:
    limit = args.limit

bucket_sizes = []
if args.buckets > 0:
    print("Bucketing input data")
    lengths = [len(ds[i]['audio']['array']) for i in range(0, limit)]
    lengths.sort()
    bucket_step = limit // args.buckets
    bucket_start = limit - ((args.buckets - 1 ) * bucket_step) - 1
    for i in range(bucket_start, limit, bucket_step):
        bucket_sizes.append(lengths[i])

if (len(bucket_sizes) > 0):
#warm-up
    for bucket in bucket_sizes:
        print("Compiling model for length", bucket)
        input_values = processor(ds[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt", padding="max_length", truncation=True, max_length=bucket).input_values.to(dtype=dtype)
        input_values = input_values.to(hpu, non_blocking=True)
        if args.dtype != "fp32":
            with torch.autocast(device_type="hpu", dtype=dtype):
                model(input_values).logits.to(cpu, non_blocking=False)
        else:
            model(input_values).logits.to(cpu, non_blocking=False)

if prof:
    schedule = torch.profiler.schedule(wait=0, warmup=1, active=limit*args.repeat, repeat=1)
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU]

    profiler = torch.profiler.profile(
    schedule=schedule,
    activities=activities,
    on_trace_ready=torch.profiler.tensorboard_trace_handler('.', use_gzip=True),
    record_shapes=True,
    with_stack=True)

    profiler.start()

errors = 0
perf_start = 0
perf_metric = []
perfs = []
tokens = []
decodes = []
e2e = []
lengths = []
ground_truth = []
predicted = []
stream_pool = []

if args.perf:
  e2e_start = time.perf_counter()

for r in range(0, args.repeat):
    if args.multi_thread:
        pool = multiprocessing.pool.ThreadPool(4) # (2 * pipeline depth) threads spawned
    for i in range(0, limit):
        if args.perf:
            perf_start = time.perf_counter()
# tokenize
        if len(bucket_sizes) > 0:
            seq_len = len(ds[i]['audio']['array'])
            b = 0
            while bucket_sizes[b] < seq_len:
                b = b + 1
            input_values = processor(ds[i]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt", padding="max_length", max_length=bucket_sizes[b]).input_values.to(dtype=dtype)  # Batch size 1
        else:
            input_values = processor(ds[i]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt", padding="longest").input_values.to(dtype=dtype)  # Batch size 1
        if args.perf:
            ts = time.perf_counter()
            tokens.append(ts - perf_start)
            perf_start = ts
        lengths.append(seq_len)
        input_values = input_values.to(hpu, non_blocking=True)
# retrieve logits
        logits = get_output(model, input_values, dtype)

        if args.multi_thread:
            logits_pinned = torch.empty_like(logits, device='cpu').pin_memory(device='hpu')
            if r == 0: #create new stream for each sample
                stream_obj = ht.hpu.Stream()
                stream_pool.append(stream_obj)
            else: #reuse streams on repeat to avoid running out of resources
                stream_obj = stream_pool[i]
            with ht.hpu.stream(stream_obj):
                logits_pinned.copy_(logits, non_blocking=True)
        else:
            logits = logits.to(cpu, non_blocking=False)

        if prof:
            profiler.step()

        if args.multi_thread:
            pool.apply_async(sync, args=(stream_obj, processor, logits_pinned, args, decodes, e2e, predicted, ground_truth, ds, r, i, tokens, perfs, perf_start))
        else:
            if args.perf:
               ts = time.perf_counter()
               perfs.append(ts - perf_start)
               perf_start = ts
# take argmax and decode
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)
            if args.perf:
                decodes.append(time.perf_counter() - perf_start)
                e2e.append(tokens[-1] + perfs[-1] + decodes[-1])
            if args.accuracy and r == 0:
                predicted.append(transcription[0])
                ground_truth.append(ds[i]['text'])

    if args.multi_thread:
       # close all threads at end of full repetition to ensure resources (e.g. pinned memory) are released
       pool.close()
       pool.join()

if args.perf:
    total_e2e_sec = time.perf_counter() - e2e_start
    with open('./perf_metric.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["Sample Length(tokens)","Tokenize(ms)","Compute(ms)","Decode(ms)","End-to-End(ms)","ThroughPut(ms/10k_tokens)"])
        total_tokens = 0
        for i in range(0, len(lengths)):
            total_tokens = total_tokens + lengths[i]
            perf_metric.append(((tokens[i] + perfs[i] + decodes[i])*1000.*10000.)/lengths[i])
            writer.writerow([lengths[i],tokens[i]*1000.,perfs[i]*1000.,decodes[i]*1000.,e2e[i]*1000.,(e2e[i]*1000.*10000.)/lengths[i]])

if prof:
    profiler.stop()

if args.accuracy:
    from jiwer import wer
    wer_val = wer(ground_truth, predicted)
    print("WER:", wer_val)

if args.perf:
    if len(perf_metric) > 1:
        print("Ran", args.repeat, "times on", limit, "samples")
        print("Average latency:", ((total_e2e_sec*1000)/total_tokens)*10000, "ms/10k_tokens")
        print("Throughput(1/(avg latency)):", total_tokens/(total_e2e_sec*1000), "tokens/ms")
        print("Avg end-to-end time per sample:", (total_e2e_sec*1000)/len(lengths), "ms/sample")
        print("Total end-to-end time for the dataset:", total_e2e_sec, "sec")
        if not args.dev_clean_ds: # for test-clean dataset
            print("Real-time factor:", (5.4*60*60*args.repeat)/total_e2e_sec) # 5.4h/(e2e time for test clean dataset)
