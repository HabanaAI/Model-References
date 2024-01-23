###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################

import time
import math
import array
import statistics
import torch
from contextlib import contextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer
import mlperf_loadgen as lg

from dataset import Dataset
import habana_generation_utils as hgu
import modeling_gptj as hpu_modeling_gptj
import quantization.quantize as quantize
from torch.utils.tensorboard import SummaryWriter


gen_kwargs = {
    "max_new_tokens": 128,
    "min_new_tokens": 30,
}


def setup_pt_profiler(schedule):
    activities = [torch.profiler.ProfilerActivity.CPU]
    activities.extend([torch.profiler.ProfilerActivity.HPU])

    profiler = torch.profiler.profile(
        schedule=schedule,
        activities=activities,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('.', use_gzip=True),
        record_shapes=True,
        with_stack=True)
    return profiler


def setup_hltv_profiler(schedule):
    import sys
    import os
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


def setup_profiler(step, profile_type):
    active = 1
    warmup = 1 if step > 0 else 0
    wait = max(step - warmup, 0)

    schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1)

    if profile_type == 'tb':
        return setup_pt_profiler(schedule)
    else:
        return setup_hltv_profiler(schedule)


class SUT_base():
    def __init__(self, args, options):
        print("Loading PyTorch model...")
        self.dataset_path = args.dataset_path
        self.model_path = args.model_path
        self.batch_size = args.batch_size
        self.input_length = 1919
        self.max_length = self.input_length + gen_kwargs['max_new_tokens'] + 1
        self.profile = args.profile
        self.profile_type = args.profile_type
        self.inference_times = []
        self.tb_writer = SummaryWriter() if args.enable_tensorboard_logging else None
        self.is_eager = args.eager

        gen_kwargs["num_beams"] = options["num_beams"]
        gen_kwargs["early_stopping"] = options["early_stopping"]

        if args.device == "cuda":
            assert torch.cuda.is_available(), "CUDA device is not available!"
        elif args.device == "hpu":
            import habana_frameworks.torch.core
            assert torch.hpu.is_available(), "HPU device is not available!"
        self.device = torch.device(args.device)

        self.model = self.setup_model(args)

        self.hgu_opts = hgu.GenerationOptions(
            max_length=self.max_length,
            min_length=self.input_length+gen_kwargs['min_new_tokens'],
            max_input_length=self.max_length,
             **options,
        )
        if self.profile:
            self.hgu_opts.max_iterations = args.profile_tokens
        if args.dtype == "float8":
            self.hgu_opts.kv_cache_fp8 = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            model_max_length=self.max_length,
            padding_side="left",
            use_fast=True,)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.data_object = Dataset(
            self.model_path, self.dataset_path, total_count_override=args.max_examples)
        self.qsl = lg.ConstructQSL(self.data_object.count, self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam, self.data_object.UnloadSamplesFromRam)

    def setup_model(self, args):
        if self.device.type == "hpu":
            model = hpu_modeling_gptj.GPTJForCausalLM.from_pretrained(
                self.model_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16
            )
        else:
            is_gpu = self.device.type == "cuda"
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto" if not is_gpu else None,
                low_cpu_mem_usage=True if not is_gpu else False,
                torch_dtype=torch.bfloat16
            )

        if model.config.pad_token_id is None:
            model.config.pad_token_id = model.config.eos_token_id
        model.to(torch.bfloat16)
        model.to(self.device)

        if self.device.type == "hpu":
            if not self.is_eager:
                import habana_frameworks.torch.hpu.graphs as htgraphs
                model = htgraphs.wrap_in_hpu_graph(model)
            if args.quantization_file:
                model = quantize.setup_quantization(model, args.quantization_file)
        return model

    def warmup(self):
        print("Warming up...")
        dummy_tensor = torch.ones([self.batch_size, self.input_length], dtype=torch.int64)
        input_batch = {
            "input_ids": dummy_tensor, "attention_mask": dummy_tensor.detach().clone()
        }
        input_batch, _, _ = hgu.prepare_decoder_only_input_without_moving(
            self.tokenizer.pad_token_id, self.hgu_opts, input_batch)

        t_start = time.time()
        _ = self.inference_call(input_batch).cpu().numpy()
        t_end = time.time()
        print("Warmup took {:.2f} ms".format((t_end-t_start)*1000))

    def issue_queries(self, query_samples):
        num_samples = len(query_samples)
        batches = math.ceil(num_samples / self.batch_size)
        print("Number of Samples in query_samples : ", num_samples)

        profiler = None
        if self.profile:
            profiler = setup_profiler(batches - 1, self.profile_type)
            profiler.start()
        for batch_id in range(batches):
            start_index = batch_id * self.batch_size
            batch_size = min(num_samples - start_index, self.batch_size)

            input_batch = self.prepare_input_batch(query_samples, start_index, batch_size)
            input_batch, _, _ = hgu.prepare_decoder_only_input_without_moving(
                self.tokenizer.pad_token_id, self.hgu_opts, input_batch)

            with self.measure_and_save_time(batch_id):
                output_batch = self.inference_call(input_batch).cpu().numpy()
            if profiler:
                profiler.step()

            self.send_responses(query_samples, start_index, batch_size, output_batch)
        if profiler:
            profiler.stop()

    def prepare_input_batch(self, query_samples, start_index, batch_size):
        indices = [
            query_samples[start_index + j].index for j in range(batch_size)
        ]
        while len(indices) < self.batch_size:
            indices.append(indices[0])

        input_ids = [
            self.data_object.source_encoded_input_ids[index] for index in indices
        ]
        attention_masks = [
            self.data_object.source_encoded_attn_masks[index] for index in indices
        ]
        return {
            "input_ids": torch.cat(input_ids), "attention_mask": torch.cat(attention_masks)
        }

    @contextmanager
    def measure_and_save_time(self, batch_id):
        t_start = time.time()
        yield
        t_end = time.time()
        time_taken = t_end - t_start
        if self.tb_writer:
            self.tb_writer.add_scalar('batch_time [seconds]', time_taken, batch_id)
        print("Batch {} : {:.2f} ms".format(batch_id, (time_taken)*1000))
        self.inference_times.append(time_taken)

    def inference_call(self, input_batch):
        with torch.inference_mode():
            input_batch_lengths = [x.shape[0] for x in input_batch["input_ids"]]

            if self.device.type == "hpu":
                initial_ids, beam_trace = hgu.generate_on_prepared_input(
                    self.model, self.hgu_opts, input_batch, self.max_length, self.input_length)
                output_batch = hgu.finalize_beams(
                    initial_ids, beam_trace, self.model.config, self.hgu_opts.length_penalty)
            else:
                output_batch = self.model.generate(
                    **input_batch, **gen_kwargs, pad_token_id=self.tokenizer.eos_token_id)

            output_batch_truncated = []
            for data, source_len in zip(output_batch, input_batch_lengths):
                output_batch_truncated.append(data[source_len:])
            output_batch_truncated = torch.stack(output_batch_truncated)
        return output_batch_truncated

    def send_responses(self, query_samples, start_index, batch_size, output_batch):
        responses_array = [
            array.array("B", output_batch[i].tobytes()) for i in range(batch_size)
        ]
        bi = [
            response_array.buffer_info() for response_array in responses_array
        ]
        lg.QuerySamplesComplete([
            lg.QuerySampleResponse(
                query_samples[start_index + j].id, bi[j][0], bi[j][1]
            ) for j in range(batch_size)
        ])

    def flush_queries(self):
        pass

    def close_log_file(self):
        pass

    def __del__(self):
        if self.inference_times:
            mean = statistics.fmean(self.inference_times)
            print(f"Average performance: {self.batch_size / mean:.3f} samples/s")

        if self.device.type == "hpu":
            from habana_frameworks.torch.hpu.memory import memory_stats
            GB = 1024**3
            memory_stats_dict = memory_stats(self.device)
            max_in_use = memory_stats_dict['MaxInUse'] / GB
            limit = memory_stats_dict['Limit'] / GB
            print(
                "HPU memory usage: {:.1f} GB / {:.1f} GB ({:.0f}%)".format(
                    max_in_use, limit, max_in_use / limit * 100.0
                )
            )
        print("Finished destroying SUT.")


class SUT_Offline(SUT_base):
    def __init__(self, args, options):
        SUT_base.__init__(self, args, options)
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        self.warmup()
    '''IssueQuery and inference methods implemented in Base class'''


class SUT_Server(SUT_base):
    def __init__(self, args, options):
        SUT_base.__init__(self, args, options)
        self.batch_size = 1 # batching is not supported currently in Server mode
        self.total_samples_done = 0
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        self.warmup()

    def issue_queries(self, query_samples):
        input_batch = self.prepare_input_batch(query_samples, start_index=0, batch_size=1)
        input_batch, _, _ = hgu.prepare_decoder_only_input_without_moving(
            self.tokenizer.pad_token_id, self.hgu_opts, input_batch)

        t_start = time.time()
        output_batch = self.inference_call(input_batch).cpu().numpy()
        t_end = time.time()
        print("Sample time : {:.2f} ms".format((t_end-t_start)*1000))

        self.send_responses(
            query_samples, start_index=0, batch_size=1, output_batch=output_batch)

        self.total_samples_done += 1
        if self.total_samples_done % 5 == 0:
            print("Completed : ", self.total_samples_done)
