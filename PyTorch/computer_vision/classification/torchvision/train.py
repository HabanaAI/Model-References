# Copyright (c) 2021-2024, Habana Labs Ltd.  All rights reserved.


from __future__ import print_function

import datetime
import os
import time
import sys
import random
import utils
from math import ceil

import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms

#Import local copy of the model only for ResNext101_32x4d
#which is not part of standard torchvision package.
import model as resnet_models
import habana_frameworks.torch.core as htcore
from habana_frameworks.torch.dynamo.compile_backend.experimental import enable_compiled_autograd
from data_loaders import build_data_loader

try:
    from apex import amp
except ImportError:
    amp = None


def train_one_epoch(lr_scheduler, model, criterion, optimizer, data_loader, device, epoch, print_freq, apex=False,
                    tb_writer=None, steps_per_epoch=0, is_autocast=False, lazy_mode=True, prof=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ",device=device)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)
    step_count = 0
    total_no_of_steps = steps_per_epoch * epoch
    batch_size_used = 0
    if args.dl_time_exclude:
        dl_ex_time = 0.0
        dl_ex_start_time = 0.0

    last_print_time_tensorboard= time.time()
    last_print_time_metric_logger = time.time()

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)
        batch_size_used = batch_size_used + image.size()[0]

        dl_ex_start_time=time.time()

        if args.channels_last:
            image = image.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=is_autocast):
            output = model(image)
            loss = criterion(output, target)
            loss = loss / image.shape[0]
        optimizer.zero_grad(set_to_none=True)

        if apex:
           with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
           loss.backward()

        if lazy_mode:
            htcore.mark_step()

        optimizer.step()

        if lazy_mode:
            htcore.mark_step()

        if (step_count + 1) % print_freq == 0:
            loss = loss.item()
            images_processed = batch_size_used
            output_cpu = output.detach().to('cpu')
            acc1, acc5 = utils.accuracy(output_cpu, target, topk=(1, 5))
            metric_logger.update(loss=loss, lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters['acc1'].update(acc1.item(), n=images_processed)
            metric_logger.meters['acc5'].update(acc5.item(), n=images_processed)
            current_time = time.time()
            if args.dl_time_exclude:
                dl_ex_time += time.time() - dl_ex_start_time
                dl_ex_start_time = current_time
            last_print_time_metric_logger = dl_ex_time if args.dl_time_exclude else (current_time - last_print_time_metric_logger)
            metric_logger.meters['img/s'].update(images_processed / last_print_time_metric_logger)

            if tb_writer is not None:
                tb_writer.add_scalar('Loss', loss, total_no_of_steps)
                current_time = time.time()
                if args.dl_time_exclude:
                    dl_ex_time += time.time() - dl_ex_start_time
                    dl_ex_start_time = current_time
                last_print_time_tensorboard = dl_ex_time if args.dl_time_exclude else (current_time - last_print_time_tensorboard)
                tb_writer.add_scalar('img/s', (images_processed / last_print_time_tensorboard), total_no_of_steps)

                last_print_time_tensorboard = time.time()

            if args.dl_time_exclude:
                dl_ex_time = 0.0
                dl_ex_start_time=time.time()
            batch_size_used = 0
            last_print_time_metric_logger= time.time()

        step_count = step_count + 1
        total_no_of_steps = total_no_of_steps + 1
        if step_count >= args.num_train_steps:
            break

        if args.optimizer == "lars":
            lr_scheduler.step()

        if args.dl_time_exclude:
            dl_ex_time += time.time() - dl_ex_start_time

        if prof:
            prof.step()

    if lr_scheduler is not None and args.optimizer == "sgd":
        lr_scheduler.step()

def evaluate(model, criterion, data_loader, device, print_freq=100, tb_writer=None, epoch=0, is_autocast=False):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", device=device)
    header = 'Test:'
    step_count = 0
    with torch.no_grad(), torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=is_autocast):
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)

            if args.channels_last:
                image = image.contiguous(memory_format=torch.channels_last)

            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)
            loss = loss / image.shape[0]

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            loss_cpu = loss.to('cpu').detach()
            metric_logger.update(loss=loss_cpu.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
            step_count = step_count + 1
            if step_count >= args.num_eval_steps:
                break
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    # Return from here if evaluation phase does not go through any iterations.(eg, The data set is so small that
    # there is only one eval batch, but that was skipped in data loader due to drop_last=True)
    if len(metric_logger.meters) == 0:
        return

    if tb_writer is not None:
        tb_writer.add_scalar('Evaluation accuracy', metric_logger.acc1.global_avg, epoch)

    print(' * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5))
    return metric_logger.acc1.global_avg


def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha256(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, cache_dataset, distributed):
    # Data loading code
    print("Loading data")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_train from {}".format(cache_path))
        dataset, _ = torch.load(cache_path)
    else:
        # Note that transforms are used only by native python data loader: torch.utils.data.DataLoader
        # and Aeon data loader. In case of MediaAPI, transforms are implemented independently using
        # API calls (see resnet_media_pipe.py code)
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        if cache_dataset:
            print("Saving dataset_train to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_test from {}".format(cache_path))
        dataset_test, _ = torch.load(cache_path)
    else:
        # Transforms are not used by MediaAPI data loader. See comment above for 'dataset' transforms.
        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
        if cache_dataset:
            print("Saving dataset_test to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating samplers")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def lr_vec_fcn(values, milestones):
    lr_vec = []
    for n in range(len(milestones)-1):
        lr_vec += [values[n]]*(milestones[n+1]-milestones[n])
    return lr_vec


def adjust_learning_rate(optimizer, epoch, lr_vec):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr_vec[epoch]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def create_rank_dir(dir):
    worker_output_dir = f"{dir}/worker_{utils.get_rank()}"
    utils.mkdir(worker_output_dir)
    return worker_output_dir


def main(args):
    if args.eval_offset_epochs < 0:
        assert False, "Eval offset has to be 0 or bigger"

    if args.epochs_between_evals < 1:
        assert False, "Epochs between evaluations has to be 1 or bigger"

    if args.dl_worker_type == "MP":
        try:
            # Default 'fork' doesn't work with synapse. Use 'forkserver' or 'spawn'
            torch.multiprocessing.set_start_method('spawn')
        except RuntimeError:
            pass
    elif args.dl_worker_type == "HABANA":
        try:
            import habana_dataloader
        except ImportError:
            assert False, "Could Not import habana dataloader package"

    # Enable hpu dynamic shape
    lazy_mode = os.getenv('PT_HPU_LAZY_MODE', '1') == '1'
    if args.device == 'hpu' and not utils.is_gaudi() and lazy_mode:
        try:
            import habana_frameworks.torch.hpu as hthpu
            hthpu.enable_dynamic_shape()
        except ImportError:
            logger.info("habana_frameworks could not be loaded")

    if not args.no_compiled_autograd and args.use_torch_compile:
        enable_compiled_autograd()

    if args.apex:
        if sys.version_info < (3, 0):
            raise RuntimeError("Apex currently only supports Python 3. Aborting.")
        if amp is None:
            raise RuntimeError("Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
                               "to enable mixed-precision training.")

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    if args.enable_tensorboard_logging:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer_dir = create_rank_dir(args.output_dir)
        tb_writer = SummaryWriter(log_dir=tb_writer_dir)
        tb_writer.add_scalar('_hparams_/session_start_info', time.time(), 0)
    else:
        tb_writer = None

    if args.deterministic and args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.device == 'cuda':
            torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True

    if args.device == 'hpu' and args.world_size > 0:
        # patch torch cuda functions that are being unconditionally invoked
        # in the multiprocessing data loader
        torch.cuda.current_device = lambda: None
        torch.cuda.set_device = lambda x: None

    pin_memory_device = None
    pin_memory = False
    if args.device == 'cuda' or args.device == 'hpu':
        pin_memory_device = args.device
        pin_memory = True

    train_dir = os.path.join(args.data_path, 'train')
    val_dir = os.path.join(args.data_path, 'val')
    num_images_train = 0
    num_images_validation = 0
    for _, _, files in os.walk(train_dir, followlinks=True):
        num_images_train += len(files)
    for _, _, files in os.walk(val_dir, followlinks=True):
        num_images_validation += len(files)
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir,
                                                                   args.cache_dataset, args.distributed)
    data_loader_seed = args.seed if args.deterministic else None

    data_loader = build_data_loader(is_training=True, dl_worker_type=args.dl_worker_type, seed=data_loader_seed,
                                    dataset=dataset, batch_size=args.batch_size, sampler=train_sampler,
                                    num_workers=args.workers, pin_memory=pin_memory,
                                    pin_memory_device=pin_memory_device)

    data_loader_test = build_data_loader(is_training=False, dl_worker_type=args.dl_worker_type, seed=data_loader_seed,
                                         dataset=dataset_test, batch_size=args.batch_size, sampler=test_sampler,
                                         num_workers=args.workers, pin_memory=pin_memory,
                                         pin_memory_device=pin_memory_device)

    print("Creating model")
    #Import only resnext101_32x4d from a local copy since torchvision
    # package doesn't support resnext101_32x4d variant
    if 'resnext101_32x4d' in args.model:
        model = resnet_models.__dict__[args.model](pretrained=args.pretrained)
    else:
        model = torchvision.models.__dict__[
            args.model](pretrained=args.pretrained)
    model.to(device)
    if args.device == 'hpu' and lazy_mode and args.hpu_graphs and not utils.is_gaudi():
        import habana_frameworks.torch.hpu.graphs as htgraphs
        htgraphs.ModuleCacher()(model, have_grad_accumulation=True)
    if args.channels_last:
        if device == torch.device('cuda'):
            print('Converting model to channels_last format on CUDA')
            model.to(memory_format=torch.channels_last)
        elif args.device == 'hpu':
            print('Converting model params to channels_last format on Habana')
            # TODO:
            # model.to(device).to(memory_format=torch.channels_last)
            # The above model conversion doesn't change the model params
            # to channels_last for many components - e.g. convolution.
            # So we are forced to rearrange such tensors ourselves.

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, reduction='sum')

    if args.optimizer == "lars":
        from habana_frameworks.torch.hpex.optimizers import FusedResourceApplyMomentum
        optimizer = FusedResourceApplyMomentum(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.lars_weight_decay)

        skip_list = ['batch_normalization', 'bias', 'bn', 'downsample.1']
        skip_mask = []
        for n_param, param in zip(model.named_parameters(), model.parameters()):
            assert n_param[1].shape == param.shape
            skip_mask.append(not any(v in n_param[0] for v in skip_list))

        from habana_frameworks.torch.hpex.optimizers import FusedLars
        optimizer = FusedLars(optimizer, skip_mask, eps=0.0)
    elif args.optimizer == "sgd":
        if args.device == 'hpu' and not args.force_native_sgd:
            from habana_frameworks.torch.hpex.optimizers import FusedSGD
            sgd_optimizer = FusedSGD
        else:
            sgd_optimizer = torch.optim.SGD
        optimizer = sgd_optimizer(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.apex:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.apex_opt_level
                                          )

    steps_per_epoch = ceil(num_images_train / args.world_size / args.batch_size)
    steps_per_eval = ceil(num_images_validation / args.world_size / args.batch_size)
    train_print_freq = min(args.print_freq, steps_per_epoch - 1)
    eval_print_freq = min(args.print_freq, steps_per_eval - 1)
    if args.optimizer == "lars":
        train_steps = steps_per_epoch * args.epochs
        print("************* PolynomialDecayWithWarmup  ************")
        from model.optimizer import PolynomialDecayWithWarmup
        lr_scheduler = PolynomialDecayWithWarmup(optimizer,
                                                batch_size=args.batch_size,
                                                steps_per_epoch=steps_per_epoch,
                                                train_steps=train_steps,
                                                initial_learning_rate=args.lars_base_learning_rate,
                                                warmup_epochs=args.lars_warmup_epochs,
                                                end_learning_rate=args.lars_end_learning_rate,
                                                power=2.0,
                                                lars_decay_epochs=args.lars_decay_epochs,
                                                opt_name='lars')
    elif args.optimizer == "sgd":
        if args.custom_lr_values is not None:
            lr_vec = lr_vec_fcn([args.lr]+args.custom_lr_values, [0]+args.custom_lr_milestones+[args.epochs])
            lr_scheduler = None
        else:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    model_for_eval = model

    # TBD: pass the right module for ddp
    model_without_ddp = model

    if args.distributed:
        if args.device == 'hpu':
            # W/A for First bucket should be smaller/equal to other buckets issue
            model = torch.nn.parallel.DistributedDataParallel(model, bucket_cap_mb=256, broadcast_buffers=False,
                gradient_as_bucket_view=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    model_for_train = model

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        evaluate(model_for_eval, criterion, data_loader_test, device=device,
                print_freq=eval_print_freq, is_autocast=args.is_autocast)
        return

    next_eval_epoch = args.eval_offset_epochs - 1
    while next_eval_epoch < args.start_epoch:
        next_eval_epoch += args.epochs_between_evals

    if args.use_torch_compile:
        model_for_train = torch.compile(model_for_train, backend="hpu_backend")
        model_for_eval = torch.compile(model_for_eval, backend="hpu_backend")

    print("Start training")

    prof = None
    if args.profile:
        assert args.profile_steps is not None, "please provide profile_steps argument"
        step_words = args.profile_steps.split(":")
        assert step_words[0] != '', "please provide valid profile_steps argument"
        warmup_steps = int(step_words[0]) - 1 if int(step_words[0]) > 0 else 0
        active_steps = 1
        if len(step_words) == 2:
            active_steps = int(step_words[1]) - warmup_steps
        profiler_dir = os.path.join(args.output_dir, "traces")
        if not os.path.exists(profiler_dir) and utils.get_rank() == 0:
            os.makedirs(profiler_dir)

        prof = torch.profiler.profile(
            activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU),
            schedule=torch.profiler.schedule(wait=0, warmup=warmup_steps, active=active_steps, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_dir),
            record_shapes=True,
            with_stack=True)
        prof.start()

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # Setting epoch is done by Habana dataloader internally
        if args.distributed and args.dl_worker_type != "HABANA":
            train_sampler.set_epoch(epoch)

        if lr_scheduler is None:
            adjust_learning_rate(optimizer, epoch, lr_vec)

        train_one_epoch(lr_scheduler, model_for_train, criterion, optimizer, data_loader,
                        device, epoch, print_freq=train_print_freq, apex=args.apex,
                        tb_writer=tb_writer, steps_per_epoch=steps_per_epoch,
                        is_autocast=args.is_autocast, lazy_mode=lazy_mode, prof=prof)
        if epoch == next_eval_epoch:
            evaluate(model_for_eval, criterion, data_loader_test, device=device,
                     print_freq=eval_print_freq, tb_writer=tb_writer, epoch=epoch, is_autocast=args.is_autocast)
            next_eval_epoch += args.epochs_between_evals

        if args.output_dir and args.save_checkpoint:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': None if lr_scheduler is None else lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args}
            # W/A: In resnext without lazy new_zipfile_serialization doesn't work correctly.
            # There is a need to use lagacy method.
            _use_new_zipfile_serialization = not ('resnext101' in args.model and not lazy_mode)
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)), _use_new_zipfile_serialization=_use_new_zipfile_serialization)
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'checkpoint.pth'), _use_new_zipfile_serialization=_use_new_zipfile_serialization)

            if args.save_model:
                model_save_name = f"{args.model}.model"
                utils.save_on_master(model_without_ddp, os.path.join(args.output_dir, model_save_name))

    if prof:
        prof.stop()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    print('Training time {}'.format(total_time_str))


def set_env_params():
    os.environ["MAX_WAIT_ATTEMPTS"] = "50"
    os.environ['HCL_CPU_AFFINITY'] = '1'


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--data-path', default='/data/pytorch/imagenet/ILSVRC2012/', help='dataset')
    parser.add_argument('--dl-time-exclude', default='True', type=lambda x: x.lower() == 'true', help='Set to False to include data load time')
    parser.add_argument('--model', default='resnet18',
                        help='select Resnet models from resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2')
    parser.add_argument('--device', default='hpu', help='device')
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-ebe', '--epochs_between_evals', default=1, type=int, metavar='N',
                        help='number of epochs to be completed before evaluation (default: 1)')
    parser.add_argument('-eoe', '--eval_offset_epochs', default=0, type=int, metavar='N',
                        help='offsets the epoch on which the evaluation starts (default: 0)')
    parser.add_argument('--dl-worker-type', default='HABANA', type=lambda x: x.upper(),
                        choices = ["MP", "HABANA"], help='select multiprocessing or habana accelerated')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers (default: 10)')
    parser.add_argument('--process-per-node', default=8, type=int, metavar='N',
                        help='Number of process per node')
    parser.add_argument('--hls_type', default='HLS1', help='Node type')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--lars_base_learning_rate', default='9', type=float, help='base learning rate for lars')
    parser.add_argument('--lars_end_learning_rate', default='0.0001', type=float, help='end learning rate for lars')
    parser.add_argument('--lars_warmup_epochs', default='3', type=int, help='number of warmup epochs for lars')
    parser.add_argument('--lars_decay_epochs', default='36', type=int, help='number of decay epochs for lars')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lwd', '--lars-weight-decay', default=5e-5, type=float,
                        metavar='W', help='lars weight decay (default: 5e-5)',
                        dest='lars_weight_decay')
    parser.add_argument('--lr-step-size', default=30, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--custom-lr-values', default=None, metavar='N', type=float, nargs='+', help='custom lr values list')
    parser.add_argument('--custom-lr-milestones', default=None, metavar='N', type=int, nargs='+',
                        help='custom lr milestones list')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--label-smoothing', default=0.0, type=float,
                        help='Apply label smoothing to the loss. This applies to'
                             'CrossEntropyLoss, when label_smoothing is greater than 0.')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')

    parser.add_argument('--channels-last', default='False', type=lambda x: x.lower() == 'true',
                        help='Whether input is in channels last format.'
                        'Any value other than True(case insensitive) disables channels-last')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    parser.add_argument(
        "--use_torch_compile",
        dest="use_torch_compile",
        help="Use torch.compile feature to run the model",
        action="store_true",
    )
    parser.add_argument(
        '--no-compiled-autograd',
        action="store_true",
        help='Disable compiled_autograd for torch.compile'
    )
    parser.add_argument(
        "--hpu_graphs",
        dest="hpu_graphs",
        help="Use HPU graphs feature to run the model by default",
        default='True', type=lambda x: x.lower() == 'true',
    )

    parser.add_argument('--optimizer', default='sgd', type=lambda x: x.lower(), choices = ["lars", "sgd"],
                        help='Select an optimizer from `lars` or `sgd`')
    parser.add_argument('--enable-tensorboard-logging', action='store_true',
                        help='enable logging using tensorboard things such as accuracy, loss or performance (img/s)')
    parser.add_argument('--force_native_sgd', action='store_true',
                        help='forces to use native SGD optimizer - to be used only together with --optimizer=sgd')

    # Mixed precision training parameters
    parser.add_argument('--apex', action='store_true',
                        help='Use apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O1', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
                        )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--num-train-steps', type=int, default=sys.maxsize, metavar='T',
                        help='number of steps a.k.a iterations to run in training phase')
    parser.add_argument('--num-eval-steps', type=int, default=sys.maxsize, metavar='E',
                        help='number of steps a.k.a iterations to run in evaluation phase')
    parser.add_argument('--save-checkpoint', action="store_true",
                        help='Whether or not to save checkpoint; True: to save, False to avoid saving. '
                             'Checkpoint will be stored in {output_dir}/checkpoint.pth, '
                             'where {output_dir} is the --output_dir argument value.'
                       )
    parser.add_argument('--save-model', action="store_true",
                        help='Whether or not to save model, True: to save, False to avoid saving. '
                             'Model will be stored in {output_dir}/{model}.model, '
                             'where {output_dir} and {model} are --output_dir and --model argument values.'
                        )
    parser.add_argument('--run-lazy-mode', default='True', type=lambda x: x.lower() == 'true',
                        help='[DEPRECATED] Do not use, it has no effect anymore. Instead, set env variable PT_HPU_LAZY_MODE to 1')
    parser.add_argument('--deterministic', action="store_true",
                        help='Whether or not to make data loading deterministic;This does not make execution deterministic')
    parser.add_argument("--profile", action="store_true",
                        help='enable/disable pytorch profiler')
    parser.add_argument("--profile_steps",
                        default='0',
                        help='warmup and active steps when to take profiler. Syntax is x:y where x is warmup steps and y is number of steps for which the profiler will be active')

    mixed_precision_group = parser.add_mutually_exclusive_group()
    mixed_precision_group.add_argument('--autocast', dest='is_autocast', action='store_true', help='enable autocast mode on Gaudi')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    set_env_params()
    args = parse_args()
    main(args)

