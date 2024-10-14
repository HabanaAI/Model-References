###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################
# Changes:
# - Added hpu support
# - Added hpu graph support for UNET
# - --count value not reflecting issue resolved
# - Random error in ConstructQSL resolved
# - Multi-card inference support added
# - Higher batch size inference support added
# - Added quantization and measurement support
# - Updated warmup implementation

"""
mlperf inference benchmarking tool
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import array
import collections
import json
import logging
import os
import sys
import threading
import time
from queue import Queue
import subprocess

import mlperf_loadgen as lg
import numpy as np
import torch

import dataset
import coco

import habana_frameworks.torch.hpu as torch_hpu
import habana_frameworks.torch.core as htcore

import requests
import signal
import shutil
import pandas as pd

import tools.generate_fp32_weights as gw

global time_measurements
time_measurements = []
arrival_time=0

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

NANO_SEC = 1e9
MILLI_SEC = 1000

PORT = [3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007]

SUPPORTED_DATASETS = {
    "coco-1024": (
        coco.Coco,
        dataset.preprocess,
        coco.PostProcessCoco(device="cuda"),
        {"image_size": [3, 1024, 1024]},
    )
}


SUPPORTED_PROFILES = {
    "defaults": {
        "dataset": "coco-1024",
        "backend": "pytorch",
        "model-name": "stable-diffusion-xl",
    },
    "debug": {
        "dataset": "coco-1024",
        "backend": "debug",
        "model-name": "stable-diffusion-xl",
    },
    "stable-diffusion-xl-pytorch": {
        "dataset": "coco-1024",
        "backend": "pytorch",
        "model-name": "stable-diffusion-xl",
    },
    "stable-diffusion-xl-pytorch-dist": {
        "dataset": "coco-1024",
        "backend": "pytorch-dist",
        "model-name": "stable-diffusion-xl",
    },
}

SCENARIO_MAP = {
    "SingleStream": lg.TestScenario.SingleStream,
    "MultiStream": lg.TestScenario.MultiStream,
    "Server": lg.TestScenario.Server,
    "Offline": lg.TestScenario.Offline,
}


def get_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS.keys(), help="dataset")
    parser.add_argument("--dataset-path", required=True, help="path to the dataset")
    parser.add_argument(
        "--profile", choices=SUPPORTED_PROFILES.keys(), help="standard profiles"
    )
    parser.add_argument(
        "--scenario",
        default="SingleStream",
        help="mlperf benchmark scenario, one of " + str(list(SCENARIO_MAP.keys())),
    )
    parser.add_argument(
        "--max-batchsize",
        type=int,
        default=1,
        help="max batch size in a single inference",
    )
    parser.add_argument("--threads", default=1, type=int, help="threads")
    parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    parser.add_argument(
        "--find-peak-performance",
        action="store_true",
        help="enable finding peak performance pass",
    )
    parser.add_argument("--backend", help="Name of the backend")
    parser.add_argument("--model-name", help="Name of the model")
    parser.add_argument("--output", default="output", help="test results")
    parser.add_argument("--qps", type=int, help="target qps")
    parser.add_argument("--model-path", help="Path to model weights")

    parser.add_argument(
        "--dtype",
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="dtype of the model",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="device to run the benchmark",
    )
    parser.add_argument(
        "--latent-framework",
        default="torch",
        choices=["torch", "numpy"],
        help="framework to load the latents",
    )

    # file to use mlperf rules compliant parameters
    parser.add_argument(
        "--mlperf_conf", default="mlperf.conf", help="mlperf rules config"
    )
    # file for user LoadGen settings such as target QPS
    parser.add_argument(
        "--user_conf",
        default="user.conf",
        help="user config for user LoadGen settings such as target QPS",
    )
    # file for LoadGen audit settings
    parser.add_argument(
        "--audit_conf", default="audit.config", help="config for LoadGen audit settings"
    )
    # arguments to save images
    # pass this argument for official submission
    # parser.add_argument("--output-images", action="store_true", help="Store a subset of the generated images")
    # do not modify this argument for official submission
    parser.add_argument("--ids-path", help="Path to caption ids", default="tools/sample_ids.txt")

    # below will override mlperf rules compliant settings - don't use for official submission
    parser.add_argument("--time", type=int, help="time to scan in seconds")
    parser.add_argument("--count", type=int, help="dataset items to use")
    parser.add_argument("--debug", action="store_true", help="debug")
    parser.add_argument(
        "--performance-sample-count", type=int, help="performance sample count", default=5000
    )
    parser.add_argument(
        "--max-latency", type=float, help="mlperf max latency in pct tile"
    )
    parser.add_argument(
        "--samples-per-query",
        default=8,
        type=int,
        help="mlperf multi-stream samples per query",
    )
    # hpu specific arguments
    parser.add_argument('--hpu-graph', const=True, default=True, type=str2bool, nargs="?")
    parser.add_argument(
        "--hpus",
        type=int,
        default=8,
        help="number of hpu devices to run",
    )
    parser.add_argument("--measurements-dump-path",
                        type=str,
                        default="/tmp/time_measurements.csv",
                        help="Path to csv file where time measurements will be dumped")
    parser.add_argument("--quantize",
                        action="store_true",
                        help="enable quantization")
    parser.add_argument("--measure",
                        action="store_true",
                        help="measure to gather statistics for quantize")


    args = parser.parse_args()
    # don't use defaults in argparser. Instead we default to a dict, override that with a profile
    # and take this as default unless command line give
    defaults = SUPPORTED_PROFILES["defaults"]

    if args.profile:
        profile = SUPPORTED_PROFILES[args.profile]
        defaults.update(profile)
    for k, v in defaults.items():
        kc = k.replace("-", "_")
        if getattr(args, kc) is None:
            setattr(args, kc, v)

    if args.scenario not in SCENARIO_MAP:
        parser.error("valid scanarios:" + str(list(SCENARIO_MAP.keys())))
    return args


def get_backend(backend, **kwargs):
    if backend == "pytorch":
        from backend_pytorch import BackendPytorch
        backend = BackendPytorch(**kwargs)

    elif backend == "debug":
        from backend_debug import BackendDebug

        backend = BackendDebug()
    else:
        raise ValueError("unknown backend: " + backend)
    return backend

def start_sut_servers(args):
    cmd = []

    if args.hpus > 1:
        cmd += ['torchrun',
                    '--nnodes=1',
                    f'--nproc-per-node={args.hpus}', 'hpu_multicard.py',
                    f'--dataset={args.dataset}',
                    f'--dataset-path={args.dataset_path}',
                    f'--device={args.device}',
                    f'--dtype={args.dtype}',
                    f'--backend={args.backend}',
                    f'--max-batchsize={args.max_batchsize}',
                    f'--hpu-graph={args.hpu_graph}',
                    f'--latent-framework={args.latent_framework}']
        if args.model_path is not None:
            cmd += [f'--model-path={args.model_path}']
        if args.quantize:
            cmd +=[f'--quantize={args.quantize}']
        log.info('START: ' + " ".join(cmd))
        return subprocess.Popen(" ".join(cmd), shell=True)

def stop_sut_servers():
    for line in os.popen("ps ax | grep hpu_multicard  | grep -v grep"):
        fields = line.split()
        pid = fields[0]
        os.kill(int(pid), signal.SIGKILL)

class Item:
    """An item that we queue for processing by the thread pool."""

    def __init__(self, query_id, content_id, inputs, img=None):
        self.query_id = query_id
        self.content_id = content_id
        self.img = img
        self.inputs = inputs
        self.start = time.time()


class RunnerBase:
    def __init__(self, model, ds, threads, post_proc=None, max_batchsize=128, hpus=8):
        self.take_accuracy = False
        self.ds = ds
        self.model = model
        self.post_process = post_proc
        self.threads = threads
        self.take_accuracy = False
        self.max_batchsize = max_batchsize
        self.result_timing = []
        self.hpus = hpus

    def handle_tasks(self, tasks_queue):
        pass

    def start_run(self, result_dict, take_accuracy):
        self.result_dict = result_dict
        self.result_timing = []
        self.take_accuracy = take_accuracy
        self.post_process.start()

    def run_one_item(self, qitem: Item, task_id=0):
        # run the prediction
        processed_results = []
        try:
            if self.hpus == 1:
                results = self.model.predict(qitem.inputs)
                processed_results = self.post_process(
                    results, qitem.content_id, qitem.inputs, self.result_dict
                )
                post_process_time= time.time()
                if self.take_accuracy:
                    self.post_process.add_results(processed_results)
            else:
                url = 'http://localhost:'+ str(PORT[task_id]) + '/predict/'
                response = requests.post(url, json={'id': qitem.content_id})
                output_shape = json.loads(response.headers['output_shape'])
                processed_results = np.frombuffer(response.content, np.uint8).reshape(output_shape)
                if self.take_accuracy:
                    self.post_process.add_results(processed_results, qitem.content_id)
            self.result_timing.append(time.time() - qitem.start)
        except Exception as ex:  # pylint: disable=broad-except
            src = [self.ds.get_item_loc(i) for i in qitem.content_id]
            log.error("thread: failed on contentid=%s, %s", src, ex)
            # since post_process will not run, fake empty responses
            processed_results = [[]] * len(qitem.query_id)
        finally:
            response_array_refs = []
            response = []
            results = []
            for idx, query_id in enumerate(qitem.query_id):
                results = processed_results[idx]
                response_array = array.array(
                    "B", np.array(results, np.uint8).tobytes()
                )
                response_array_refs.append(response_array)
                bi = response_array.buffer_info()
                response.append(lg.QuerySampleResponse(query_id, bi[0], bi[1]))
            lg.QuerySamplesComplete(response)
            if self.hpus == 1:
                response_time = time.time()
                time_measurements.append({
                    "sample_id": qitem.query_id,
                    "input":  len(qitem.query_id),
                    "output": len(response),
                    "arrival_time": arrival_time,
                    "post_process_time": post_process_time,
                    "response_time": response_time,
                    "Latency": response_time-arrival_time,
                    "Execution_time":post_process_time-arrival_time,
                })

    def enqueue(self, query_samples):
        idx = [q.index for q in query_samples]
        query_id = [q.id for q in query_samples]
        if len(query_samples) < self.max_batchsize:
            data, label = self.ds.get_samples(idx)
            self.run_one_item(Item(query_id, idx, data, label))
        else:
            bs = self.max_batchsize
            for i in range(0, len(idx), bs):
                data, label = self.ds.get_samples(idx[i : i + bs])
                self.run_one_item(
                    Item(query_id[i : i + bs], idx[i : i + bs], data, label)
                )

    def finish(self):
        pass


class QueueRunner(RunnerBase):
    def __init__(self, model, ds, threads, post_proc=None, max_batchsize=128, hpus=8):
        super().__init__(model, ds, threads, post_proc, max_batchsize, hpus)
        self.tasks = [Queue(maxsize=hpus * 700) for i in range(hpus)]
        self.workers = []
        self.result_dict = {}
        self.accum_indx = []
        self.accum_id = []
        self.task_counter = 0

        for i in range(self.hpus):
            worker = threading.Thread(target=self.handle_tasks, args=(self.tasks[i], i,))
            worker.daemon = True
            self.workers.append(worker)
            worker.start()

    def handle_tasks(self, tasks_queue, task_id):
        """Worker thread."""
        while True:
            qitem = tasks_queue.get()
            if qitem is None:
                # None in the queue indicates the parent want us to exit
                tasks_queue.task_done()
                break
            self.run_one_item(qitem, task_id)
            tasks_queue.task_done()

    def enqueue(self, query_samples, flush=False):
        idx = [q.index for q in query_samples]
        query_id = [q.id for q in query_samples]

        if flush:
            if len(self.accum_indx) > 0:
                task_id = self.task_counter % self.hpus
                data, label = self.ds.get_samples(self.accum_indx)
                self.tasks[task_id].put(Item(self.accum_id, self.accum_indx, data, label))
        elif len(query_samples) < self.max_batchsize:
            if len(self.accum_indx) + len(query_samples) < self.max_batchsize:
                self.accum_indx.extend(idx)
                self.accum_id.extend(query_id)
            else:
                self.accum_indx.extend(idx)
                self.accum_id.extend(query_id)
                bs = self.max_batchsize
                for i in range(0, len(self.accum_indx), bs):
                    ie = i + bs
                    task_id = self.task_counter % self.hpus
                    data, label = self.ds.get_samples(self.accum_indx[i:ie])
                    self.tasks[task_id].put(Item(self.accum_id[i:ie], self.accum_indx[i:ie], data, label))
                    self.task_counter += 1
                self.accum_indx = []
                self.accum_id = []
        else:
            bs = self.max_batchsize
            for i in range(0, len(idx), bs):
                ie = i + bs
                task_id = self.task_counter % self.hpus
                data, label = self.ds.get_samples(idx[i:ie])
                self.tasks[task_id].put(Item(query_id[i:ie], idx[i:ie], data, label))
                self.task_counter += 1

    def finish(self):
        # exit all threads
        for i, _ in enumerate(self.workers):
            self.tasks[i].put(None)
        for worker in self.workers:
            worker.join()


def main():
    args = get_args()

    log.info(args)
    start_sut_servers(args)

    device = args.device
    if args.hpus > 1:
        device = "cpu"

    # find backend
    backend = get_backend(
        args.backend,
        precision=args.dtype,
        device=device,
        model_path=args.model_path,
        batch_size=args.max_batchsize
     )

    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    def dump_time_measurements():
        pd_measurements = pd.DataFrame(time_measurements)
        pd_measurements.to_csv(args.measurements_dump_path)
    # --count applies to accuracy mode only and can be used to limit the number of images
    # for testing.
    count_override = False
    count = args.count
    if count:
        count_override = True

    htcore.hpu_set_env()

    # load model to backend
    model = backend.load()

    if args.hpus == 1:
        setattr(model.pipe, 'quantized', args.quantize)
        if args.quantize:
            # additional unet for last 2 steps
            import copy
            unet_bf16 = copy.deepcopy(backend.pipe.unet).to(args.device)

            if args.hpu_graph and torch_hpu.is_available():
                unet_bf16 = torch_hpu.wrap_in_hpu_graph(unet_bf16)
            setattr(backend.pipe, 'unet_bf16', unet_bf16)

            # replace bf16 weights to be quantized with fp32 weights
            temp_dict = gw.get_unet_weights(args.model_path)
            for name, module in backend.pipe.unet.named_modules():
                if name in temp_dict.keys():
                    del module.weight
                    setattr(module,'weight',torch.nn.Parameter(temp_dict[name].clone().to(args.device)))
            temp_dict.clear()

        if args.quantize or args.measure:
            try:
                from neural_compressor.torch.quantization import FP8Config, convert, prepare
            except ImportError:
                raise ImportError(
                    "Module neural_compressor is missing. Please use a newer Synapse version to use quantization")
            quant_config_full_fp8 = os.getenv('QUANT_CONFIG')
            config_fp8 = FP8Config.from_json_file(quant_config_full_fp8)
            if args.measure:
                backend.pipe.unet = prepare(backend.pipe.unet, config_fp8)
            elif args.quantize:
                quant_config_partial_fp8 = os.getenv('QUANT_CONFIG_2')
                config_fp8_2 = FP8Config.from_json_file(quant_config_partial_fp8)
                backend.pipe.unet = convert(backend.pipe.unet, config_fp8)
                backend.pipe.unet_bf16 = convert(backend.pipe.unet_bf16, config_fp8_2)
                htcore.hpu_initialize(backend.pipe.unet_bf16, mark_only_scales_as_const=True)
            htcore.hpu_initialize(backend.pipe.unet, mark_only_scales_as_const=True)

        if args.hpu_graph and torch_hpu.is_available():
            backend.pipe.unet = torch_hpu.wrap_in_hpu_graph(backend.pipe.unet)


    # dataset to use
    dataset_class, pre_proc, post_proc, kwargs = SUPPORTED_DATASETS[args.dataset]
    ds = dataset_class(
        data_path=args.dataset_path,
        name=args.dataset,
        pre_process=pre_proc,
        pipe_tokenizer=model.pipe.tokenizer,
        pipe_tokenizer_2=model.pipe.tokenizer_2,
        latent_dtype=dtype,
        latent_device=device,
        latent_framework=args.latent_framework,
        **kwargs,
    )
    final_results = {
        "runtime": model.name(),
        "version": model.version(),
        "time": int(time.time()),
        "args": vars(args),
        "cmdline": str(args),
    }

    mlperf_conf = os.path.abspath(args.mlperf_conf)
    if not os.path.exists(mlperf_conf):
        log.error("{} not found".format(mlperf_conf))
        sys.exit(1)

    user_conf = os.path.abspath(args.user_conf)
    if not os.path.exists(user_conf):
        log.error("{} not found".format(user_conf))
        sys.exit(1)

    audit_config = os.path.abspath(args.audit_conf)

    if args.accuracy:
        ids_path = os.path.abspath(args.ids_path)
        with open(ids_path) as f:
            saved_images_ids = [int(_) for _ in f.readlines()]


    if args.output:
        output_dir = os.path.abspath(args.output)
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(output_dir)

    #
    # make one pass over the dataset to validate accuracy
    #
    if count_override is False:
        count = ds.get_item_count()

    # warmup
    if args.hpus== 1:
        syntetic_str = "Lorem ipsum dolor sit amet, consectetur adipiscing elit"
        latents_pt= torch.rand(ds.latents.shape, dtype=dtype).to(args.device)
        warmup_samples = [
            {
                "input_tokens": ds.preprocess(syntetic_str, model.pipe.tokenizer),
                "input_tokens_2": ds.preprocess(syntetic_str, model.pipe.tokenizer_2),
                "latents":latents_pt,
            }
            for _ in range(args.max_batchsize)
        ]
        for i in range(5):
            _ = backend.predict(warmup_samples)
    else:
        def send_query(port, warmup_step=5):
            step_count = 1
            while True:
                try:
                    url = 'http://localhost:'+ str(port) + '/warmup/'
                    response = requests.post(url)
                    if step_count == warmup_step:
                        break
                    step_count += 1
                except Exception as e:
                    # wait for flask server to be ready
                    time.sleep(10)


        workers = []
        for idx, server_port in enumerate(PORT):
            worker = threading.Thread(target=send_query, args=(server_port,))
            worker.daemon = True
            workers.append(worker)
            worker.start()
        for worker in workers:
            worker.join()

    scenario = SCENARIO_MAP[args.scenario]
    runner_map = {
        lg.TestScenario.SingleStream: RunnerBase,
        lg.TestScenario.MultiStream: QueueRunner,
        lg.TestScenario.Server: QueueRunner,
        lg.TestScenario.Offline: QueueRunner,
    }
    runner = runner_map[scenario](
        model, ds, args.threads, post_proc=post_proc, max_batchsize=args.max_batchsize, hpus=args.hpus
    )

    def issue_queries(query_samples):
        if args.hpus == 1:
            global arrival_time
            arrival_time = time.time()
        runner.enqueue(query_samples)

    def flush_queries():
        runner.enqueue(query_samples=[], flush=True)

    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = output_dir
    log_output_settings.copy_summary_to_stdout = False
    log_settings = lg.LogSettings()
    log_settings.enable_trace = args.debug
    log_settings.log_output = log_output_settings

    settings = lg.TestSettings()
    settings.FromConfig(mlperf_conf, args.model_name, args.scenario)
    settings.FromConfig(user_conf, args.model_name, args.scenario)
    settings.scenario = scenario
    settings.mode = lg.TestMode.PerformanceOnly
    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    if args.find_peak_performance:
        settings.mode = lg.TestMode.FindPeakPerformance

    if args.time:
        # override the time we want to run
        settings.min_duration_ms = args.time * MILLI_SEC
        settings.max_duration_ms = args.time * MILLI_SEC

    if args.qps:
        qps = float(args.qps)
        settings.server_target_qps = qps
        settings.offline_expected_qps = qps

    if count_override:
        settings.min_query_count = count
        settings.max_query_count = count

    if args.samples_per_query:
        settings.multi_stream_samples_per_query = args.samples_per_query
    if args.max_latency:
        settings.server_target_latency_ns = int(args.max_latency * NANO_SEC)
        settings.multi_stream_expected_latency_ns = int(args.max_latency * NANO_SEC)

    performance_sample_count = (
        args.performance_sample_count
        if args.performance_sample_count
        else min(count, 500)
    )
    sut = lg.ConstructSUT(issue_queries, flush_queries)
    qsl = lg.ConstructQSL(
        count, performance_sample_count, ds.load_query_samples, ds.unload_query_samples
    )

    log.info("starting {}".format(scenario))
    result_dict = {"scenario": str(scenario)}
    runner.start_run(result_dict, args.accuracy)

    lg.StartTestWithLogSettings(sut, qsl, settings, log_settings, audit_config)
    if args.measure:
        try:
            from neural_compressor.torch.quantization import finalize_calibration
        except ImportError:
            raise ImportError(
                "Module neural_compressor is missing. Please use a newer Synapse version to use quantization")
        finalize_calibration(backend.pipe.unet)

    runner.finish()
    stop_sut_servers()

    # wait for all servers to shutdown
    time.sleep(5)

    if args.accuracy:
        post_proc.finalize(result_dict, ds, output_dir=args.output)
        final_results["accuracy_results"] = result_dict
        post_proc.save_images(saved_images_ids, ds)

    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)
    dump_time_measurements()

    #
    # write final results
    #
    if args.output:
        with open("results.json", "w") as f:
            json.dump(final_results, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    main()
