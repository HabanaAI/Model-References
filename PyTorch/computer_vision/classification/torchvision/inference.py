###############################################################################
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
###############################################################################

import argparse
import json
import logging
import sys
from contextlib import contextmanager
import subprocess
import time
from multiprocessing import Process, Event, Queue
from statistics import mean

import torch
import os
import utils  # PyTorch/computer_vision/classification/torchvision/utils.py
import torchvision
from torch import nn
from torchvision import transforms
import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu.graphs as htgraphs
from torchvision.models.resnet import resnet50
#Import local copy of the model only for ResNext101_32x4d
#which is not part of standard torchvision package.
import model as resnet_models # PyTorch/computer_vision/classification/torchvision/model
from data_loaders import build_data_loader

HPU = torch.device("hpu")
data_type = {'bfloat16': torch.bfloat16, 'float32': torch.float32}

schedule = torch.profiler.schedule(wait=10, warmup=1, active=10, repeat=1)
activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU]

profiler = torch.profiler.profile(
            schedule=schedule,
            activities=activities,
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./runs/', use_gzip=True),
            record_shapes=True,
            with_stack=True)

def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def get_imagenet_dataset(dir, cache=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    cache_path = _get_cache_path(dir)
    if cache and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_test from {}".format(cache_path))
        dataset, _ = torch.load(cache_path)
    else:
        dataset = torchvision.datasets.ImageFolder(
            dir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
        if cache:
            print("Saving dataset_test to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, dir), cache_path)

    return dataset

class HPUModel:  # TODO add warm up iteration
    def __init__(self,
                 model_def: torch.nn.Module = None,
                 parameters_path: str = None,
                 example_input: torch.Tensor = None,
                 dtype: str = 'bfloat16',
                 model_path: str = None,
                 compile_mode = False
                 ):
        self.model = model_def
        print(f'Inference data type {dtype}')
        self.dtype = data_type[dtype]
        self.latency_counter = list()
        if model_path:
            print("Loading model : " + model_path)
            self.model = torch.load(model_path, map_location=torch.device("cpu"))
        elif parameters_path:
            checkpoint = torch.load(parameters_path, map_location=torch.device("cpu"))
            self.model.load_state_dict(checkpoint['model'])

        self.model.eval()
        self.model = htcore.hpu_set_env(self.model)

        if compile_mode:
            self.model = torch.compile(self.model, backend="hpu_backend")

        self.model.to(device=HPU)

        if not compile_mode:
            htcore.hpu_initialize(self.model)


    def __call__(self,
                 data: torch.Tensor, measurement='latency'):
        latency_timer = time.time()
        data = data.to(device=HPU, non_blocking=True)
        with torch.autocast(device_type="hpu", dtype=self.dtype, enabled=(self.dtype != torch.float32), cache_enabled=False):
            output = self.model(data)
            if measurement == 'latency':
                output = output.to('cpu')
                self.latency_counter.append(time.time()-latency_timer)
            else:
                ht.core.mark_step()
        return output

    def benchmark_runner(self, data_loader, run_with_profiler, measurement: str = 'latency'):
        with torch.no_grad():
            for data, target in data_loader:
                output = self(data, measurement)
                break
            start = time.perf_counter()
            for data, target in data_loader:
                if measurement == 'throughput' and run_with_profiler:
                    profiler.step()
                output = self(data, measurement)
            if measurement == 'throughput':
                if run_with_profiler:
                    profiler.stop()
                output.to('cpu')
            finish = time.perf_counter()
        return finish, start

    def benchmark(self, data_loader, run_with_profiler):
        finish_latency, start_latency = self.benchmark_runner(data_loader, run_with_profiler, 'latency')
        duration_latency = finish_latency - start_latency
        print(f'duration latency {duration_latency}')
        total_samples = None
        batch_size = None
        if isinstance(data_loader, torch.utils.data.dataloader.DataLoader):
            total_samples = len(data_loader.dataset)
            batch_size = data_loader.batch_size
        else:
            total_samples = len(data_loader.dataloader.dataset)
            batch_size = data_loader.dataloader.batch_size
        if run_with_profiler:
            profiler.start()
        if self.latency_counter:
            avg_latency = sum(self.latency_counter) / len(self.latency_counter)
        else:
            print(f"There is no latency measurements")
            avg_latency = 0
        finish_tp, start_tp = self.benchmark_runner(data_loader, run_with_profiler, 'throughput')
        duration_tp = finish_tp - start_tp
        performance = total_samples / duration_tp
        print(f'duration throughput {duration_tp}')
        print(f'total_samples {total_samples}')
        metrics = {
            'avg_latency (ms)': avg_latency * 1000,
            'performance (img/s)': performance
        }
        return metrics


class HPUJITModel(HPUModel):
    def __init__(self,
                 model_def: torch.nn.Module = None,
                 parameters_path: str = None,
                 traced_model_path: str = None,
                 example_input: torch.Tensor = None,
                 dtype: str = 'bfloat16',
                 model_path: str = None,
                 compile_mode=False
                 ):
        self.dtype = data_type[dtype]
        print(f'Inference data type {dtype}')
        if traced_model_path:
            model = torch.jit.load(traced_model_path, map_location=torch.device('cpu'))
            model.to(device=HPU)
        else:
            super().__init__(model_def, parameters_path, model_path=model_path)
            self._trace(example_input)

    def _trace(self, example_input):
        with torch.no_grad():
            with torch.autocast(device_type="hpu", dtype=self.dtype, enabled=(self.dtype != torch.float32), cache_enabled=False):
                example_input = example_input.to(device=HPU)
                self.model = torch.jit.trace(self.model, example_input, check_trace=False, strict=False)


class HPUGraphModel(HPUModel):
    def __init__(self,
                 model_def: nn.Module = None,
                 parameters_path: str = None,
                 example_input=None,
                 dtype: str = 'bfloat16',
                 model_path: str = None,
                 compile_mode = False
                 ):
        super().__init__(model_def, parameters_path, example_input=example_input, dtype=dtype, model_path=model_path)
        self.dtype = data_type[dtype]
        self.model = htgraphs.wrap_in_hpu_graph(self.model)
        print(f'Inference data type {dtype}')

def resnet_accuracy(hpu_model: HPUModel,
                    data_loader):
    acc1_sum = 0
    acc5_sum = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader, start=1):
            output = hpu_model(data, measurement='latency')  # latency measurement is with copy output
            if output.size()[0] != target.size()[0]:
                output = output[0:target.size()[0]]
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            acc1_sum += acc1
            acc5_sum += acc5
            print(f'Top 1 accuracy: {acc1_sum / i}', end="\r", flush=True)
    metrics = {
        'top_1': float(acc1_sum) / i,
        'top_5': float(acc5_sum) / i
    }
    return metrics


def get_model_defs(model_def: str):
    if model_def == 'resnext101_32x4d':
        return resnet_models.__dict__[model_def]
    else:
        model_defs = {resnet50}
        model_defs = {func.__name__: func for func in model_defs}
        return model_defs[model_def]

def main(model_type: type,
         model_def: callable,
         model_dtype: str,
         batch_size: int,
         data_dir: str,
         ckpt_pth: str,
         run_accuracy=False,
         run_with_profiler=False,
         run_benchmarks=False,
         use_pt_dataloader=False,
         model_path: str=None,
         use_compile_mode=False):
    val_dir = os.path.join(data_dir, 'val')
    dataset = get_imagenet_dataset(val_dir)
    sampler=torch.utils.data.SequentialSampler(dataset)
    if use_pt_dataloader:
        data_loader = build_data_loader(is_training=False, dl_worker_type="MP", seed=123,
                                        dataset=dataset, batch_size=batch_size, sampler=sampler,
                                        num_workers=8)
    else:
        data_loader = build_data_loader(is_training=False, dl_worker_type="HABANA", seed=123,
                                        dataset=dataset, batch_size=batch_size, sampler=sampler,
                                        num_workers=8, pin_memory=True, pin_memory_device='hpu')

    with torch.no_grad():
        example_input = torch.ones((batch_size, 3, 224, 224), device="cpu")

    pretrained=True
    if os.path.isfile(ckpt_pth) or os.path.isfile(model_path):
        pretrained=False
    if use_compile_mode:
        if os.environ.get('PT_HPU_LAZY_MODE') is None:
            sys.exit("Please use PT_HPU_LAZY_MODE=0 in the command line for torch.compile")
        elif not os.environ['PT_HPU_LAZY_MODE'] == '0':
            sys.exit("Please use PT_HPU_LAZY_MODE=0 in the command line for torch.compile")
        if not model_type is HPUModel:
            sys.exit("Please use HPUModel as the modeltype in the command line for torch.compile")

    model = model_type(model_def(pretrained=pretrained), parameters_path=ckpt_pth,
        example_input=example_input, dtype=model_dtype, model_path=model_path, compile_mode=use_compile_mode)
    if run_benchmarks:
        benchmarks = model.benchmark(data_loader, run_with_profiler)
        print(benchmarks)

    if run_accuracy:
        accuracy = resnet_accuracy(model, data_loader)
        print(accuracy)


model_def_strs = {'resnet50', 'resnext101_32x4d'}
modes = {HPUJITModel, HPUModel, HPUGraphModel}
modes = {mode.__name__: mode for mode in modes}

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-t', '--model_type',
                            choices=(modes.keys()),
                            help='inference model type',
                            required=True)
    arg_parser.add_argument('-m', '--model', choices=model_def_strs,
                            help='model name',
                            required=True)
    arg_parser.add_argument('-b', '--batch_size', type=int,
                            required=True)
    arg_parser.add_argument('--benchmark', action='store_true')
    arg_parser.add_argument('--accuracy', action='store_true')
    arg_parser.add_argument('--profile', action='store_true')
    arg_parser.add_argument('-dt', '--dtype',
                            choices=(data_type.keys()),
                            nargs='?',
                            const='bfloat16',
                            default='bfloat16',
                            help='inference model dtype')
    arg_parser.add_argument('-data', '--dataset_path',
                            default='/data/pytorch/imagenet/ILSVRC2012/',
                            required=False,
                            help='path to Imagenet dataset')
    arg_parser.add_argument('-ckpt', '--checkpoint_path',
                            default='./pretrained_checkpoint/pretrained_checkpoint.pt',
                            required=False,
                            help='path to pre-trained checkpoint')
    arg_parser.add_argument('--pt_dataloader', action='store_true')
    arg_parser.add_argument('-mp', '--model_path',
                            default=None,
                            required=False,
                            help='path to model')
    arg_parser.add_argument('--compile', action='store_true',
                            help='enable and run with torch.compile')

    args = arg_parser.parse_args()

    model_type = modes[args.model_type]
    model_def = get_model_defs(model_def=args.model)
    main(model_type=model_type,
         model_def=model_def,
         model_dtype=args.dtype,
         batch_size=args.batch_size,
         data_dir=args.dataset_path,
         ckpt_pth=args.checkpoint_path,
         run_benchmarks=args.benchmark,
         run_accuracy=args.accuracy,
         run_with_profiler=args.profile,
         use_pt_dataloader=args.pt_dataloader,
         model_path=args.model_path,
         use_compile_mode=args.compile)
