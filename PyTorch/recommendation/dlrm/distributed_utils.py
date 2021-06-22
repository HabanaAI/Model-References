# Copyright (c) 2021, Habana Labs Ltd.  All rights reserved.

from __future__ import print_function
from collections import defaultdict, deque
import datetime
import time
import torch
import torch.distributed as dist
import sys

import errno

import os
import numpy as np

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

def dlrm_get_emb_table_map(ln_emb, rank, world_size):
    if len(ln_emb) <= 1:
        print("multinode doesn't support single embedding table yet !")
        sys.exit(0)
    if ((world_size & (world_size-1)) != 0):
        print("unsupported world size !")
        sys.exit(0)
    #Assign the embedding tables so that larger tables are distributed evently
    sorted_size = np.argsort(ln_emb)
    selected_tables = np.sort(sorted_size[rank : :world_size])
    # Needed only for convergence comparison
    all_reduce_reorder = []
    for i in range(world_size):
        all_reduce_reorder.append(np.sort(sorted_size[i::world_size]))
    all_reduce_reorder = np.concatenate(all_reduce_reorder)
    all_reduce_reorder = np.argsort(all_reduce_reorder)
    return selected_tables, all_reduce_reorder

def init_distributed_mode(args):
    if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ and 'OMPI_COMM_WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    print('Start initilaizing distributed backend')
    args.distributed = True
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.world_size), flush=True)

    use_hpu = not args.no_habana
    if use_hpu == True:
        import habana_torch_hcl
    if use_hpu == True and 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ and 'OMPI_COMM_WORLD_SIZE' in os.environ:
        args.dist_backend = 'hcl'
        os.environ["ID"] = str(args.rank)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        torch.distributed.init_process_group(args.dist_backend, rank=args.rank, world_size=args.world_size)
    elif use_hpu == True and 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.dist_backend = 'hcl'
        os.environ["ID"] = str(args.rank)
        torch.distributed.init_process_group(args.dist_backend, rank=args.rank, world_size=args.world_size)
    elif args.use_gpu == False:
        args.dist_backend = 'gloo'
        torch.distributed.init_process_group(args.dist_backend, rank=args.rank, world_size=args.world_size)
        print("Started process group {}, with rank {} and world_size {}".format(args.dist_backend, torch.distributed.get_rank(), torch.distributed.get_world_size()))
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
    else:
        torch.cuda.set_device(args.gpu)
        args.dist_backend = 'nccl'
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                             world_size=args.world_size, rank=args.rank)
    print('Backend = {}'.format(args.dist_backend))
    #set to True to enable printing for all cards

    # Change the condition for distributed setup
    # If print_all_ranks is enabled, print the loss and
    # other metrics from each node for all iterations
    # else print the loss and other metrics for only one node
    setup_for_distributed(True if args.print_all_ranks else (args.rank==0))

