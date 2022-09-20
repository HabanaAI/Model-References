# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor, init_distributed_mode, habana_mark_step, barrier

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--device', type=str, default='hpu', help='The device for training. Default: hpu')
    parser.add_argument('--mode', type=str, default='eager', choices=[
                      'eager', 'lazy'], help='Different modes avaialble. Possible values: eager, lazy. Default: eager')

    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--distributed', action='store_true', help="distributed status")

    # distributed training
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')

    # use Mixed precision on HPU
    parser.add_argument('--hmp', dest='is_hmp', action='store_true', help='enable hmp mode')
    parser.add_argument('--hmp-bf16', default='', help='path to bf16 ops list in hmp O1 mode')
    parser.add_argument('--hmp-fp32', default='', help='path to fp32 ops list in hmp O1 mode')
    parser.add_argument('--hmp-opt-level', default='O1', help='choose optimization level for hmp')
    parser.add_argument('--hmp-verbose', action='store_true', help='enable verbose mode for hmp')

    parser.add_argument("--epochs", default=None, help='training epochs')
    parser.add_argument("--train-steps", default=None, help='training steps')
    parser.add_argument("--test-steps", default=None, help='test steps')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):

    device = torch.device(config.MODEL.DEVICE)

    if config.TRAIN.MODE == 'lazy':
        os.environ["PT_HPU_LAZY_MODE"] = '1'

    if config.MODEL.DEVICE == 'hpu' and config.IS_HMP:
        from habana_frameworks.torch.hpex import hmp
        hmp.convert(opt_level=config.HMP_OPT_LEVEL, bf16_file_path=config.HMP_BF16,
                    fp32_file_path=config.HMP_FP32, isVerbose=config.HMP_VERBOSE)

    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.to(device)
    logger.info(str(model))

    optimizer = build_optimizer(config, model)

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    # Permute convolution weight tensors & 'momentum' for better performance on HPU
    if config.MODEL.DEVICE == 'hpu' and not config.MODEL.RESUME:
        run_lazy_mode = True if config.TRAIN.MODE == 'lazy' else False

    if config.MODEL.DEVICE == 'cuda':
        try:
            # noinspection PyUnresolvedReferences
            from apex import amp
        except ImportError:
            amp = None
        if config.AMP_OPT_LEVEL != "O0":
            model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model, optimizer, lr_scheduler, logger)
        # Permute convolution weight tensors & 'momentum' for better performance on HPU
        if config.MODEL.DEVICE == 'hpu':
            run_lazy_mode = True if config.TRAIN.MODE == 'lazy' else False
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    if config.MODEL.DEVICE == 'hpu':
        model_without_ddp = model
        if config.DISTRIBUTED:
            # increase the size of the bucket to combine all the all-reduce calls to a single call for better performance on HPU
            bucket_size_mb = 100
            model = torch.nn.parallel.DistributedDataParallel(model, bucket_cap_mb=bucket_size_mb, broadcast_buffers=False,
                    gradient_as_bucket_view=False)
            model_without_ddp = model.module
    else:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
        model_without_ddp = model.module

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        if config.DISTRIBUTED:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, device)
        if not config.DISTRIBUTED or (config.DISTRIBUTED and dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1))):
            # Permute back the convolution weight tensors & 'momentum' for checkpoint saving
            if config.MODEL.DEVICE == 'hpu':
                model_copy = build_model(config)
                state_dict = model_without_ddp.state_dict()

                model_copy.load_state_dict(state_dict)

                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to('cpu')

                save_checkpoint(config, epoch, model_copy, max_accuracy, optimizer, lr_scheduler, logger)
            else:
                save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)

            if config.MODEL.DEVICE == 'hpu':
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to('hpu')

        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, device):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        if config.MODEL.DEVICE == 'cuda':
            samples,targets = samples.to(device, non_blocking=False), targets.to(device, non_blocking=False)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if config.MODEL.DEVICE == 'hpu':
            samples,targets = samples.to(device, non_blocking=False), targets.to(device, non_blocking=False)
            if config.TRAIN.MODE == 'lazy':
                habana_mark_step()

        outputs = model(samples)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = criterion(outputs, targets)
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            if config.MODEL.DEVICE == 'cuda' and config.AMP_OPT_LEVEL != "O0":
                try:
                    # noinspection PyUnresolvedReferences
                    from apex import amp
                except ImportError:
                    amp = None
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.MODEL.DEVICE == 'hpu' and config.TRAIN.MODE == 'lazy':
                    habana_mark_step()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                if config.MODEL.DEVICE == 'hpu' and config.IS_HMP:
                    from habana_frameworks.torch.hpex import hmp
                    with hmp.disable_casts():
                        optimizer.step()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            if config.MODEL.DEVICE == 'cuda' and config.AMP_OPT_LEVEL != "O0":
                try:
                    # noinspection PyUnresolvedReferences
                    from apex import amp
                except ImportError:
                    amp = None
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.MODEL.DEVICE == 'hpu' and config.TRAIN.MODE == 'lazy':
                    habana_mark_step()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())

            if config.MODEL.DEVICE == 'hpu' and config.IS_HMP:
                from habana_frameworks.torch.hpex import hmp
                with hmp.disable_casts():
                    optimizer.step()
            else:
                optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        if config.MODEL.DEVICE == 'hpu' and config.TRAIN.MODE == 'lazy':
            habana_mark_step()
        elif config.MODEL.DEVICE == 'cuda':
            torch.cuda.synchronize()

        if idx % config.PRINT_FREQ == 0:
            # the loss value is fetched to CPU only when needed for better performance on HPU
            loss_meter.update(loss.item(), targets.size(0))
            norm_meter.update(grad_norm)
            if idx == 0:
                batch_time.update(time.time() - end)
            else:
                batch_time.update((time.time() - end) / config.PRINT_FREQ)
            end = time.time()

            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

        if config.TRAIN.STEPS and (idx + 1) == config.TRAIN.STEPS:
            break

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.to(config.MODEL.DEVICE, non_blocking=False)
        target = target.to(config.MODEL.DEVICE, non_blocking=False)
        if config.MODEL.DEVICE == 'hpu':
            if config.TRAIN.MODE == 'lazy':
                habana_mark_step()

        # compute output
        output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)

        if config.MODEL.DEVICE == 'cuda':
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            acc1 = reduce_tensor(acc1)
            acc5 = reduce_tensor(acc5)
            loss = reduce_tensor(loss)
        elif config.MODEL.DEVICE == 'hpu':
            output_cpu = output.detach().to('cpu')
            target_cpu=torch.tensor(target, device='cpu')
            acc1, acc5 = accuracy(output_cpu, target_cpu, topk=(1, 5))

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')

        if config.TEST.STEPS and (idx + 1) == config.TEST.STEPS:
            break

    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    args, config = parse_option()

    if config.MODEL.DEVICE == 'cuda':
        try:
            # noinspection PyUnresolvedReferences
            from apex import amp
        except ImportError:
            amp = None
        if config.AMP_OPT_LEVEL != "O0":
            assert amp is not None, "amp not installed!"

    if init_distributed_mode(config) :
        torch.distributed.barrier()
        args.distributed = True
        dist_world_size = dist.get_world_size()
        dist_rank = dist.get_rank()
    else:
        args.distributed = False
        dist_world_size = 1
        dist_rank = 0
    config = get_config(args)

    seed = config.SEED + dist_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist_world_size / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist_world_size / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist_world_size / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist_rank, name=f"{config.MODEL.NAME}")

    if dist_rank == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
