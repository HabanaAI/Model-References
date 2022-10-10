# Copyright (c) 2021, Habana Labs Ltd.  All rights reserved.


import torch
import torch.distributed as dist
import os
mpi_comm = None

def init_distributed_mode(args):
    if args.use_hpu:
        from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu

        args.world_size, args.rank, args.local_rank = initialize_distributed_hpu()
        args.distributed = args.world_size > 1

        if args.distributed:
            # necessary pytorch imports
            import torch.utils.data.distributed
            import torch.distributed as dist
            args.dist_backend = 'hccl'
            dist._DEFAULT_FIRST_BUCKET_BYTES = 100*1024*1024
            dist.init_process_group(args.dist_backend, init_method='env://',rank=args.rank, world_size=args.world_size)
            print("world_size = {}".format(args.world_size))
            print("distributed={}".format(args.distributed))
        else:
            args.local_rank = 0
            print('Not using distributed mode') 
