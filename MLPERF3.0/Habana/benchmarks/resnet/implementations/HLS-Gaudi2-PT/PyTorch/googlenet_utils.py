# Copyright (c) 2022, Habana Labs Ltd.  All rights reserved.

from __future__ import print_function
import torch
import torch.distributed as dist
import os
mpi_comm = None


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


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


def barrier():
    if mpi_comm is not None:
        mpi_comm.Barrier()


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        msg = 'Not using distributed mode'
        try:
            from mpi4py import MPI
            global mpi_comm
            mpi_comm = MPI.COMM_WORLD
            size = mpi_comm.Get_size()  # new: gives number of ranks in comm
            rank = mpi_comm.Get_rank()
            if size > 1:
                args.rank = rank
                args.world_size = size
                if os.getenv('MASTER_ADDR') is None:
                    os.environ['MASTER_ADDR'] = 'localhost'
                if os.getenv('MASTER_PORT') is None:
                    os.environ['MASTER_PORT'] = '12355'
                if args.device == 'gpu':
                    args.gpu = args.rank % torch.cuda.device_count()
            else:
                print(msg)
                args.distributed = False
                return
        except Exception as e:
            print(e)
            print("**mpi4py is not available, using mpirun will not run distributed mode")
            args.distributed = False
            return

    args.distributed = True
    if args.device == 'gpu':
        print('| distributed init (rank {}, gpu {}): {}'.format(
            args.rank, args.gpu, args.dist_url), flush=True)
    else:
        print('| distributed init (rank {}): {}'.format(
            args.rank, args.dist_url), flush=True)

    if args.device == 'hpu' and args.world_size > 1:
        args.dist_backend = 'hccl'
        import habana_frameworks.torch.distributed.hccl
        dist._DEFAULT_FIRST_BUCKET_BYTES = 200*1024*1024  # 200MB
        dist.init_process_group(
            args.dist_backend, rank=args.rank, world_size=args.world_size)
    elif args.device == 'gpu':
        torch.cuda.set_device(args.gpu)
        args.dist_backend = 'nccl'
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                             world_size=args.world_size, rank=args.rank)

    setup_for_distributed(args.rank == 0)
