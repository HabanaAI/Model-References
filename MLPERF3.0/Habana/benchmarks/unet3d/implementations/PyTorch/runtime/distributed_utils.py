###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
import random

import torch
import torch.distributed as dist
import numpy as np


def get_device(device_name, local_rank):
    if device_name == "hpu":
        device = torch.device(device_name)
    elif device_name == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
        device = torch.device(device_name)
    else:
        device = torch.device("cpu")
    return device


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():  # just in case
        torch.cuda.manual_seed_all(seed)


def generate_seeds(rng, size):
    """
    Generate list of random seeds

    :param rng: random number generator
    :param size: length of the returned list
    """
    seeds = [rng.randint(0, 2**31 - 1) for _ in range(size)]
    return seeds


def broadcast_seeds(seeds, device):
    """
    Broadcasts random seeds to all distributed workers.
    Returns list of random seeds (broadcasted from workers with rank 0).

    :param seeds: list of seeds (integers)
    :param device: torch.device
    """
    if dist.is_available() and dist.is_initialized():
        seeds_tensor = torch.IntTensor(seeds).to(device)
        torch.distributed.broadcast(seeds_tensor, 0)
        seeds = seeds_tensor.tolist()
    return seeds


def get_seed(master_seed, device):
    """
    Generates per-worker seed from one master_seed, which is later used
    to initialize random number generators (mostly for dropouts).
    Seeds are generated on worker with rank 0 and broadcasted to all other
    workers.

    :param master_seed: master RNG seed used to initialize other generators
    :param device: torch.device (used for distributed.broadcast)
    """
    local_rank = get_rank()
    if master_seed == -1:
        # random master seed, random.SystemRandom() uses /dev/urandom on Unix
        master_seed = random.SystemRandom().randint(0, 2**31 - 1)
        if local_rank == 0:
            # master seed is reported only from rank=0 worker, it's to avoid
            # confusion, seeds from rank=0 are later broadcasted to other
            # workers
            print(f'Using random master seed: {master_seed}')
    else:
        # master seed was specified from command line
        print(f'Using master seed from command line: {master_seed}')

    # initialize seeding RNG
    seeding_rng = random.Random(master_seed)

    # generate worker seeds, one seed for every distributed worker
    worker_seeds = generate_seeds(seeding_rng, get_world_size())

    # broadcast seeds from rank=0 to other workers
    worker_seeds = broadcast_seeds(worker_seeds, device)
    return worker_seeds[local_rank]


def reduce_tensor(tensor, num_gpus):
    if num_gpus > 1:
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.reduce_op.SUM)
        if rt.is_floating_point():
            rt = rt / num_gpus
        else:
            rt = rt // num_gpus
        return rt
    return tensor


def init_distributed(device):
    if device == 'hpu':
        from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu

        world_size, rank, _ = initialize_distributed_hpu()
        distributed = world_size > 1
        if distributed:
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '12345'
            dist.init_process_group(backend='hccl',
                                    init_method='env://',
                                    rank=rank,
                                    world_size=world_size)
            assert dist.is_initialized()

    else:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        distributed = world_size > 1
        if distributed:
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            dist.init_process_group(backend=backend,
                                    init_method='env://')
            assert dist.is_initialized()

    if get_rank() == 0:
        print("Distributed initialized. World size:", world_size)

    return distributed


def get_world_size():
    """
    Gets distributed world size or returns one if distributed is not initialized.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank():
    """
    Gets distributed rank or returns zero if distributed is not initialized.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def is_main_process():
    return get_rank() == 0


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
