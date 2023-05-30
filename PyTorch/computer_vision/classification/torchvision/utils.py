# Copyright (c) 2021-2022, Habana Labs Ltd.  All rights reserved.

import datetime
import time
import errno
import os
from collections import defaultdict, deque

import torch
import torch.distributed as dist

import habana_frameworks.torch.utils.experimental as htexp

mpi_comm = None

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self,device):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        if device.type == 'hpu':
            t = torch.tensor([self.count, self.total], dtype=torch.float32).to('hpu')
        else:
            t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
            dist.barrier()

        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t",device=torch.device('cuda')):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.device = device

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes(self.device)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if (i + 1) % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i+1, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i+1, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {}'.format(header, total_time_str))

# Modified version of accuracy. target and pred tensors are pytorch Long
# which is not supported by habana kernels yet. So fall back to CPU for
# ops involving these(and remain on CPU since this is the last oprton of
# iteration and we need the accuracy values to be printed out on host)
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()
        pred_cpu = torch.tensor(pred, device='cpu')
        target_cpu = torch.tensor(target, device='cpu')

        correct = pred_cpu.eq(target_cpu[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def barrier():
    if mpi_comm is not None:
        mpi_comm.Barrier()

def init_distributed_mode(args):
    from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu
    args.world_size, args.rank, args.local_rank = initialize_distributed_hpu()
    if args.world_size == 1:
        args.distributed = False
        return

    args.distributed = True
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)

    if args.device == 'hpu':
        args.dist_backend = 'hccl'
        # To improve resnet dist performance, decrease number of all_reduce calls to 1 by increasing bucket size to 230
        dist._DEFAULT_FIRST_BUCKET_BYTES = 230*1024*1024
        dist.init_process_group(args.dist_backend, rank=args.rank, world_size=args.world_size)
    else:
        torch.cuda.set_device(args.gpu)
        args.dist_backend = 'nccl'
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                             world_size=args.world_size, rank=args.rank)

    setup_for_distributed(args.rank == 0)

def get_device_type():
    return htexp._get_device_type()

def is_gaudi():
    return get_device_type() == htexp.synDeviceType.synDeviceGaudi

def is_gaudi2():
    return get_device_type() == htexp.synDeviceType.synDeviceGaudi2

def get_device_string():
    if is_gaudi():
        return "gaudi"
    elif is_gaudi2():
        return "gaudi2"
    else:
        raise ValueError("Unsupported device")
