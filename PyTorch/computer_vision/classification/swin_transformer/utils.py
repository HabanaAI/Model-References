# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from __future__ import print_function
import os
import torch
import torch.distributed as dist
from collections import defaultdict, deque
import datetime
import time
import errno
mpi_comm = None


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if config.MODEL.DEVICE == 'cuda':
            try:
                # noinspection PyUnresolvedReferences
                from apex import amp
            except ImportError:
                amp = None
            if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
                amp.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    if config.MODEL.DEVICE == 'cuda':
        try:
            # noinspection PyUnresolvedReferences
            from apex import amp
        except ImportError:
            amp = None
        if config.AMP_OPT_LEVEL != "O0":
            save_state['amp'] = amp.state_dict()

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def habana_mark_step():
    import habana_frameworks.torch.core as htcore
    htcore.mark_step()


#permute the params from filters first (KCRS) to filters last(RSCK) or vice versa.
#and permute from RSCK to KCRS is used for checkpoint saving
def permute_params(model, to_filters_last, lazy_mode):
    import habana_frameworks.torch.core as htcore
    if htcore.is_enabled_weight_permute_pass() is True:
        return
    with torch.no_grad():
        for name, param in model.named_parameters():
            if(param.ndim == 4):
                if to_filters_last:
                    param.data = param.data.permute((2, 3, 1, 0))
                else:
                    param.data = param.data.permute((3, 2, 0, 1))  # permute RSCK to KCRS

    habana_mark_step()


# permute the momentum from filters first (KCRS) to filters last(RSCK) or vice versa.
# and permute from RSCK to KCRS is used for checkpoint saving
# Used for Habana device only
def permute_momentum(optimizer, to_filters_last, lazy_mode):
    import habana_frameworks.torch.core as htcore
    if htcore.is_enabled_weight_permute_pass() is True:
        return
    for group in optimizer.param_groups:
        for p in group['params']:
            param_state = optimizer.state[p]
            # for SGD
            if 'momentum_buffer' in param_state:
                buf = param_state['momentum_buffer']
                if(buf.ndim == 4):
                    if to_filters_last:
                        buf = buf.permute((2,3,1,0))
                    else:
                        buf = buf.permute((3,2,0,1))
                    param_state['momentum_buffer'] = buf
            # for AdamW
            if 'exp_avg' in param_state:
                buf = param_state['exp_avg']
                if(buf.ndim == 4):
                    if to_filters_last:
                        buf = buf.permute((2, 3, 1, 0))
                    else:
                        buf = buf.permute((3, 2, 0, 1))
                    param_state['exp_avg'] = buf
            if 'exp_avg_sq' in param_state:
                buf = param_state['exp_avg_sq']
                if(buf.ndim == 4):
                    if to_filters_last:
                        buf = buf.permute((2, 3, 1, 0))
                    else:
                        buf = buf.permute((3, 2, 0, 1))
                    param_state['exp_avg_sq'] = buf

    habana_mark_step()


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

def init_distributed_mode(config):
    local_rank = config.LOCAL_RANK
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and 'LOCAL_RANK' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = rank % torch.cuda.device_count()
    else:
        msg = 'Not using distributed mode'
        try:
            from mpi4py import MPI
            global mpi_comm
            mpi_comm = MPI.COMM_WORLD
            world_size = mpi_comm.Get_size() # new: gives number of ranks in comm
            rank = mpi_comm.Get_rank()
            if os.getenv('MASTER_ADDR') is None:
                os.environ['MASTER_ADDR'] = 'localhost'
            if os.getenv('MASTER_PORT') is None:
                os.environ['MASTER_PORT'] = '12355'
        except Exception as e:
            print(e)
            print("**mpi4py is not available, using mpirun will not run distributed mode")
            return False

    print('| distributed init (rank {}): {}'.format(
        rank, config.DIST_URL), flush=True)

    if config.MODEL.DEVICE == 'hpu':
        if world_size  > 1:
            os.environ["ID"] = str(rank % config.PROCESS_PER_MODE )
            #not used currently
            os.environ["LOCAL_RANK"] = str(rank % config.PROCESS_PER_MODE )
            import habana_frameworks.torch.core.hccl
            dist.init_process_group('hccl', rank=rank, world_size=world_size)
        else:
            return False
    else:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method=config.DIST_URL,
                                             world_size=world_size, rank=rank)

    setup_for_distributed(rank == 0)

    return True
