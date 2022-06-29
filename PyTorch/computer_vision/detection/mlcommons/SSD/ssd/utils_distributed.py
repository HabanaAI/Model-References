# Copyright (c) 2021, Habana Labs Ltd.  All rights reserved.


import torch
import torch.distributed as dist
import os
mpi_comm = None

def init_distributed_mode(args):
    if args.use_hpu:
        if 'WORLD_SIZE' in os.environ:
            args.distributed = int(os.environ['WORLD_SIZE']) > 1
            args.world_size = int(os.environ['WORLD_SIZE'])
        else :
            try:
               from mpi4py import MPI
               global mpi_comm
               mpi_comm = MPI.COMM_WORLD
               size = mpi_comm.Get_size() # new: gives number of ranks in comm
               rank = mpi_comm.Get_rank()
               if size > 1:
                  os.environ['MASTER_ADDR'] = 'localhost'
                  os.environ['MASTER_PORT'] = '12355'
                  os.environ['RANK'] = str(rank)
                  os.environ['WORLD_SIZE'] = str(size)
                  args.world_size = int(os.environ['WORLD_SIZE'])
                  if 'LOCAL_RANK' not in os.environ:
                      args.local_rank = rank
                  args.distributed = True
               else:
                  print('Not using distributed mode')
                  args.distributed = False
                  return
            except Exception as e:
               args.distributed = False
               print(e)
               print("**mpi4py is not available, using mpirun will not run distributed mode")
               return
        # necessary pytorch imports
        import torch.utils.data.distributed
        import torch.distributed as dist
        args.dist_backend = 'hccl'
        import habana_frameworks.torch.core.hccl
        dist._DEFAULT_FIRST_BUCKET_BYTES = 100*1024*1024
        dist.init_process_group(args.dist_backend, init_method='env://')
        print("world_size = {}".format(args.world_size))
        print("distributed={}".format(args.distributed))
