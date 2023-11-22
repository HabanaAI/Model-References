# Copyright (c) 2021-2023, Habana Labs Ltd.  All rights reserved.


from __future__ import print_function
import copy
from math import ceil
import shutil
import uuid

# Import local copy of the model only for ResNext101_32x4d
# which is not part of standard torchvision package.
import model as resnet_models
import datetime
import os
import time
import sys
import json

import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
import random
import utils
import habana_frameworks.torch.core as htcore
import habana_dataloader
from mlp_log import get_mllog_mlloger

try:
    from apex import amp
except ImportError:
    amp = None

DEFAULT_IMAGE_SIZE = 224
NUM_CHANNELS = 3

def get_mlperf_variable_map():
    try:
        script_path = os.path.realpath(__file__)
        head_tail = os.path.split(script_path)
        mlperf_map_file = head_tail[0] + '/mlperf_variable_map.json'
        with open(mlperf_map_file, mode='r') as file_handle:
            json_content = file_handle.read()
            mlperf_map = json.loads(json_content)
    except IOError:
      raise IOError(f"MLPerf variable map file: {mlperf_map_file} not accesible")
    return mlperf_map


def train_one_epoch(lr_scheduler, model, criterion, optimizer, data_loader, device, epoch,
                    print_freq, args, apex=False, warmup=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ", device=device)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = f'Warmup epoch: [{epoch}]' if warmup else f'Epoch: [{epoch}]'
    step_count = 0
    last_print_time = time.time()

    profiler = None
    if args.profile_steps is not None and not warmup:
        profile_steps = [int(i) for i in args.profile_steps.split(',')]
        profiling_duration = profile_steps[1] - profile_steps[0]
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU],
            schedule=torch.profiler.schedule(wait=0, warmup=profile_steps[0], active=profiling_duration, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(args.output_dir,
                                                                    worker_name=f"worker_{utils.get_rank()}",
                                                                    use_gzip=True),
            record_shapes=True,
            with_stack=True)
        profiler.start()

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)
        dl_ex_start_time = time.time()
        if args.channels_last:
            image = image.contiguous(memory_format=torch.channels_last)

        with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=args.use_autocast):
            output = model(image)
            loss = criterion(output, target)
            loss = loss / image.shape[0]
        optimizer.zero_grad(set_to_none=True)

        if apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if args.run_lazy_mode and not args.use_torch_compile:
            htcore.mark_step()

        optimizer.step()

        if args.run_lazy_mode and not args.use_torch_compile:
            htcore.mark_step()

        if step_count % print_freq == 0:
            output_cpu = output.detach().to('cpu')
            acc1, acc5 = utils.accuracy(output_cpu, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size * print_freq)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size * print_freq)
            current_time = time.time()
            last_print_time = dl_ex_start_time if args.dl_time_exclude else last_print_time
            images_processed = batch_size * print_freq if step_count != 0 else batch_size
            metric_logger.meters['img/s'].update(images_processed / (current_time - last_print_time))
            last_print_time = time.time()

        step_count = step_count + 1
        if profiler is not None:
            profiler.step()

        if step_count >= args.num_train_steps:
            break

        if lr_scheduler is not None:
            lr_scheduler.step()

    if profiler is not None:
        profiler.stop()


def evaluate(model, criterion, data_loader, device, print_freq=100, warmup=False):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", device=device)
    header = 'Warmup test:' if warmup else 'Test:'
    step_count = 0
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)

            if args.channels_last:
                image = image.contiguous(memory_format=torch.channels_last)

            target = target.to(device, non_blocking=True)
            with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=args.use_autocast):
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

    print(' * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5))
    return metric_logger.acc1.global_avg


def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path

def warmup(model_for_train, model_for_eval, device, criterion, optimizer, args, data_loader_type, pin_memory_device, pin_memory):
    state_backup = copy.deepcopy(optimizer.optim.state)
    dataset, dataset_test, train_sampler, test_sampler = load_data(f'{args.output_dir}/resnet_synth_data/train', f'{args.output_dir}/resnet_synth_data/val', args=args, synthetic=True)
    data_loader = data_loader_type(
        dataset, args.batch_size, sampler=train_sampler,
        num_workers=args.workers, pin_memory=pin_memory, pin_memory_device=pin_memory_device)
    data_loader_test = data_loader_type(
        dataset_test, args.batch_size, sampler=test_sampler,
        num_workers=args.workers, pin_memory=pin_memory, pin_memory_device=pin_memory_device)
    train_one_epoch(None, model_for_train, criterion, optimizer, data_loader, device, 0,
                    1, args, apex=args.apex, warmup=True)
    evaluate(model_for_eval, criterion, data_loader_test, device, print_freq=1, warmup=True)
    optimizer.zero_grad(True)
    optimizer.optim.state = state_backup
    optimizer.state = optimizer.optim.__getstate__()['state']

def load_data(traindir, valdir, args, manifest=None, synthetic=False):
    # Data loading code
    print("Loading data")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset_transforms = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    dataset_test_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
    dataset_loader_train = habana_dataloader.habana_dataset.ImageFolderWithManifest if args.dl_worker_type == "HABANA" and not synthetic else torchvision.datasets.ImageFolder
    dataset_loader_eval = torchvision.datasets.ImageFolder
    loader_params = {'root': traindir, 'transform': dataset_transforms}
    loader_test_params = {'root': valdir, 'transform': dataset_test_transforms}
    if args.dl_worker_type == "HABANA" and not synthetic:
        loader_params['manifest'] = manifest
    if (synthetic):
        steps = 4
        size = steps*args.batch_size
        img_shape = (NUM_CHANNELS, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
        all_images_shape = (size, NUM_CHANNELS, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
        chunks = torch.ones(all_images_shape, dtype=torch.uint8).chunk(size)
        images = [img.reshape(img_shape) for img in chunks]
        for i, image in enumerate(images):
            if (i % args.batch_size == 0):
                batch_class_name = uuid.uuid4()
                utils.mkdir(f'{traindir}/{batch_class_name}')
                utils.mkdir(f'{valdir}/{batch_class_name}')
            torchvision.io.write_jpeg(image, f'{traindir}/{batch_class_name}/{i}.JPEG')
            torchvision.io.write_jpeg(image, f'{valdir}/{batch_class_name}/{i}.JPEG')

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    cache_dataset = args.cache_dataset and not synthetic
    if cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_train from {}".format(cache_path))
        dataset, _ = torch.load(cache_path)
    else:
        dataset = dataset_loader_train(**loader_params)
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
        dataset_test = dataset_loader_eval(**loader_test_params)
        if cache_dataset:
            print("Saving dataset_test to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def lr_vec_fcn(values, milestones):
    lr_vec = []
    for n in range(len(milestones) - 1):
        lr_vec += [values[n]] * (milestones[n + 1] - milestones[n])
    return lr_vec


def adjust_learning_rate(optimizer, epoch, lr_vec):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr_vec[epoch]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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

    if args.device == 'hpu':
        if args.run_lazy_mode:
            assert os.getenv('PT_HPU_LAZY_MODE') == '1', f"run-lazy-mode == True, but PT_HPU_LAZY_MODE={os.getenv('PT_HPU_LAZY_MODE')}"
        else:
            assert os.getenv('PT_HPU_LAZY_MODE') == '0' or os.getenv('PT_HPU_LAZY_MODE')== '2', f"args.use_lazy_mode == False, but PT_HPU_LAZY_MODE={os.getenv('PT_HPU_LAZY_MODE')}"

        try:
            import habana_frameworks.torch.hpu as ht
            ht.disable_dynamic_shape()
        except ImportError:
            logger.info("habana_frameworks could not be loaded")

    if args.apex:
        if sys.version_info < (3, 0):
            raise RuntimeError(
                "Apex currently only supports Python 3. Aborting.")
        if amp is None:
            raise RuntimeError("Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
                               "to enable mixed-precision training.")

    utils.init_distributed_mode(args)
    print(args)

    synth_data_dir = (args.output_dir if args.output_dir else '/tmp') + '/resnet_synth_data'

    if utils.get_rank() == 0:
        if args.output_dir:
            utils.mkdir(args.output_dir)
        if args.log_dir:
            utils.mkdir(args.log_dir)

        try:
            shutil.rmtree(synth_data_dir)
        except:
            pass

        utils.mkdir(f'{synth_data_dir}')
        utils.mkdir(f'{synth_data_dir}/train')
        utils.mkdir(f'{synth_data_dir}/val')

    if utils.get_world_size() > 1:
        utils.barrier()

    mlperf_mlloger, mlperf_mllog = get_mllog_mlloger(args.log_dir if args.log_dir else args.output_dir)
    mlperf_mlloger.event(key=mlperf_mllog.constants.CACHE_CLEAR, value=True)
    mlperf_mlloger.start(key=mlperf_mllog.constants.INIT_START, value=None)
    mlperf_mlloger.event(key=mlperf_mllog.constants.SUBMISSION_BENCHMARK, value=mlperf_mllog.constants.RESNET)
    mlperf_mlloger.event(key=mlperf_mllog.constants.SUBMISSION_ORG, value='Habana')
    mlperf_mlloger.event(key=mlperf_mllog.constants.SUBMISSION_DIVISION, value='closed')
    mlperf_mlloger.event(key=mlperf_mllog.constants.SUBMISSION_PLATFORM, value='gaudi-{}'.format(args.num_gpus))
    mlperf_mlloger.event(key=mlperf_mllog.constants.SUBMISSION_STATUS, value='onprem')

    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True

    if args.device == 'hpu' and utils.get_world_size() > 0:
        # patch torch cuda functions that are being unconditionally invoked
        # in the multiprocessing data loader
        torch.cuda.current_device = lambda: None
        torch.cuda.set_device = lambda x: None

    if args.dl_worker_type == "MP":
        data_loader_type = torch.utils.data.DataLoader
    elif args.dl_worker_type == "HABANA":
        data_loader_type = habana_dataloader.HabanaDataLoader

    pin_memory_device = None
    pin_memory = False
    if args.device == 'cuda' or args.device == 'hpu':
        pin_memory_device = args.device
        pin_memory = True

    print("Creating model")
    # Import only resnext101_32x4d from a local copy since torchvision
    # package doesn't support resnext101_32x4d variant
    if 'resnext101_32x4d' in args.model:
        model = resnet_models.__dict__[args.model](pretrained=args.pretrained)
    else:
        model = torchvision.models.__dict__[
            args.model](pretrained=args.pretrained)
    model.to(device)
    if args.device=='hpu' and args.run_lazy_mode and args.hpu_graphs:
        import habana_frameworks.torch.hpu.graphs as htgraphs
        htgraphs.ModuleCacher()(model, have_grad_accumulation=True)
    if args.channels_last:
        if(device == torch.device('cuda')):
            print('Converting model to channels_last format on CUDA')
            model.to(memory_format=torch.channels_last)
        elif(args.device == 'hpu'):
            print('Converting model params to channels_last format on Habana')
            # TODO:
            # model.to(device).to(memory_format=torch.channels_last)
            # The above model conversion doesn't change the model params
            # to channels_last for many components - e.g. convolution.
            # So we are forced to rearrange such tensors ourselves.

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, reduction='sum')

    print("************* Running FusedResourceApplyMomentum optimizer ************")
    from habana_frameworks.torch.hpex.optimizers import FusedResourceApplyMomentum
    optimizer = FusedResourceApplyMomentum(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    mlperf_mlloger.event(key=mlperf_mllog.constants.LARS_OPT_WEIGHT_DECAY, value=args.weight_decay)
    mlperf_mlloger.event(key='lars_opt_momentum', value=args.momentum)

    skip_list = ['batch_normalization', 'bias', 'bn', 'downsample.1']
    skip_mask = []
    for n_param, param in zip(model.named_parameters(), model.parameters()):
        assert n_param[1].shape == param.shape
        skip_mask.append(not any(v in n_param[0] for v in skip_list))

    mlperf_variable_map = get_mlperf_variable_map()
    for model_weight, param in model.named_parameters():
        mlperf_mlloger.event(key=mlperf_mllog.constants.WEIGHTS_INITIALIZATION, metadata={'tensor': mlperf_variable_map[model_weight]})

    print("************* Running FusedLARS optimizer ************")
    from habana_frameworks.torch.hpex.optimizers import FusedLars
    optimizer = FusedLars(optimizer, skip_mask, eps=0.0)

    mlperf_mlloger.event(key=mlperf_mllog.constants.OPT_NAME, value='lars')
    mlperf_mlloger.event(key=mlperf_mllog.constants.LARS_EPSILON, value=0.0)

    if args.apex:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.apex_opt_level
                                          )
    NUM_IMAGES = {
        'train': 1281167,
        'validation': 50000,
    }
    steps_per_epoch = ceil(NUM_IMAGES['train'] / utils.get_world_size() / args.batch_size)
    steps_per_eval = ceil(NUM_IMAGES['validation'] / utils.get_world_size() / args.batch_size)
    train_print_freq = min(args.print_freq, steps_per_epoch - 1)
    eval_print_freq = min(args.print_freq, steps_per_eval - 1)

    print("************* PolynomialDecayWithWarmup  ************")
    from model.optimizer import PolynomialDecayWithWarmup
    train_steps = steps_per_epoch * args.epochs
    lr_scheduler = PolynomialDecayWithWarmup(optimizer,
                                             batch_size=args.batch_size,
                                             steps_per_epoch=steps_per_epoch,
                                             train_steps=train_steps,
                                             initial_learning_rate=args.base_learning_rate,
                                             warmup_epochs=args.warmup_epochs,
                                             end_learning_rate=args.end_learning_rate,
                                             power=2.0,
                                             lars_decay_epochs=args.lars_decay_epochs,
                                             mlperf_mllog=mlperf_mllog,
                                             mlperf_mlloger=mlperf_mlloger,
                                             opt_name='lars')

    model_for_eval = model

    # TBD: pass the right module for ddp
    model_without_ddp = model

    if args.distributed:
        if args.device == 'hpu':
            # To improve resnext101 dist performance,
            # decrease number of all_reduce calls to 1 by increasing bucket size to 200
            bucket_size_mb = 200
            is_grad_view = True
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              bucket_cap_mb=bucket_size_mb,
                                                              broadcast_buffers=False,
                                                              gradient_as_bucket_view=is_grad_view)
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
                 print_freq=eval_print_freq)
        return

    if (args.enable_warmup):
        warmup_start_time = time.time()
        print("Start warmup")
        warmup(model_for_train, model_for_eval, device, criterion, optimizer, args, data_loader_type, pin_memory_device, pin_memory)
        warmup_total_time = time.time() - warmup_start_time
        warmup_total_time_str = str(datetime.timedelta(seconds=int(warmup_total_time)))
        print(f'Warmup time {warmup_total_time_str}')

    mlperf_mlloger.event(key=mlperf_mllog.constants.GLOBAL_BATCH_SIZE, value=args.batch_size*utils.get_world_size())
    mlperf_mlloger.event(key=mlperf_mllog.constants.TRAIN_SAMPLES, value=NUM_IMAGES['train'])
    mlperf_mlloger.event(key=mlperf_mllog.constants.EVAL_SAMPLES, value=NUM_IMAGES['validation'])
    group_batch_norm = 1
    mlperf_mlloger.event(key=mlperf_mllog.constants.MODEL_BN_SPAN, value= args.batch_size * group_batch_norm)
    mlperf_mlloger.event(key=mlperf_mllog.constants.GRADIENT_ACCUMULATION_STEPS, value=args.num_acc_steps)

    next_eval_epoch = args.eval_offset_epochs - 1 + args.start_epoch
    if next_eval_epoch < 0:
        next_eval_epoch += args.epochs_between_evals

    train_dir = os.path.join(args.data_path, 'train')
    val_dir = os.path.join(args.data_path, 'val')

    if utils.get_rank() == 0:
        try:
            shutil.rmtree(synth_data_dir)
        except:
            pass

    dataset_manifest = prepare_dataset_manifest(args)

    print("Start training")
    top1_acc = 0
    start_time = time.time()
    mlperf_mlloger.end(key=mlperf_mllog.constants.INIT_STOP)

    if utils.get_world_size() > 1:
        utils.barrier()

    mlperf_mlloger.start(key=mlperf_mllog.constants.RUN_START)

    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args, dataset_manifest)
    data_loader = data_loader_type(
        dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.workers, pin_memory=pin_memory, pin_memory_device=pin_memory_device)

    data_loader_test = data_loader_type(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler,
        num_workers=args.workers, pin_memory=pin_memory, pin_memory_device=pin_memory_device)

    if args.use_torch_compile:
        model_for_train = torch.compile(model_for_train, backend="aot_hpu_training_backend")
        model_for_eval = torch.compile(model_for_eval, backend="aot_hpu_training_backend")

    mlperf_mlloger.start(
        key=mlperf_mllog.constants.BLOCK_START,
        value=None,
        metadata={
            'first_epoch_num': 1,
            'epoch_count':
                (args.eval_offset_epochs if args.eval_offset_epochs > 0
                 else args.epochs_between_evals)
        })
    for epoch in range(args.start_epoch, args.epochs):
        # Setting epoch is done by Habana dataloader internally
        if args.distributed and args.dl_worker_type != "HABANA":
            train_sampler.set_epoch(epoch)

        train_one_epoch(lr_scheduler, model_for_train, criterion, optimizer, data_loader,
                        device, epoch, print_freq=train_print_freq, args=args, apex=args.apex)

        if epoch == next_eval_epoch:
            mlperf_mlloger.start(
                key=mlperf_mllog.constants.EVAL_START, value=None, metadata={'epoch_num': epoch + 1})
            top1_acc = evaluate(model_for_eval, criterion, data_loader_test, device=device,
                 print_freq=eval_print_freq) / 100.
            mlperf_mlloger.end(
                key=mlperf_mllog.constants.EVAL_STOP, value=None, metadata={'epoch_num': epoch + 1})
            mlperf_mlloger.event(
                key=mlperf_mllog.constants.EVAL_ACCURACY, value=top1_acc, metadata={'epoch_num': epoch + 1})

            first_epoch_num = max(epoch - args.epochs_between_evals + 1, 0)
            epoch_count = args.epochs_between_evals
            if first_epoch_num == 0:
                epoch_count = args.eval_offset_epochs
                if epoch_count == 0:
                    epoch_count = args.epochs_between_evals
            mlperf_mlloger.end(
                key=mlperf_mllog.constants.BLOCK_STOP,
                value=None,
                metadata={
                    'first_epoch_num': first_epoch_num + 1,
                    'epoch_count': epoch_count
                })

            if top1_acc >= args.target_accuracy:
                break

            next_eval_epoch += args.epochs_between_evals

            if next_eval_epoch < args.epochs:
                mlperf_mlloger.start(
                    key=mlperf_mllog.constants.BLOCK_START,
                    value=None,
                    metadata={
                        'first_epoch_num': epoch + 2,
                        'epoch_count': args.epochs_between_evals
                    })


        if (args.output_dir and args.save_checkpoint):
            if args.device == 'hpu':
                checkpoint = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': None if lr_scheduler is None else lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args}

                utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
                utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, 'checkpoint.pth'))

            else:
                checkpoint = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': None if lr_scheduler is None else lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args}
                utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
                utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, 'checkpoint.pth'))

    if top1_acc >= args.target_accuracy:
        mlperf_mlloger.end(key=mlperf_mllog.constants.RUN_STOP, value=None, metadata={'status': 'success'})
    else:
        mlperf_mlloger.end(key=mlperf_mllog.constants.RUN_STOP, value=None, metadata={'status': 'fail'})

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def set_env_params():
    os.environ["MAX_WAIT_ATTEMPTS"] = "50"
    os.environ['HCL_CPU_AFFINITY'] = '1'


def prepare_dataset_manifest(args):
    import glob
    import pathlib

    if args.data_path is not None:
        # get files list
        dataset_dir = os.path.join(args.data_path, 'train')

        print(f"dataset dir: {dataset_dir}")
        manifest_data = {}
        manifest_data["file_list"] = sorted(
            glob.glob(dataset_dir + "/*/*.{}".format("JPEG")))

        # get class list
        data_dir = pathlib.Path(dataset_dir)
        manifest_data["class_list"] = sorted(
            [item.name for item in data_dir.glob('*') if item.is_dir() == True])

        file_sizes = {}

        for filename in manifest_data["file_list"]:
            file_sizes[filename] = os.stat(filename).st_size

        manifest_data['file_sizes'] = file_sizes

        return manifest_data


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description='PyTorch Classification Training')

    parser.add_argument('--data-path', default='/root/software/lfs/data/data/pytorch/imagenet/ILSVRC2012/',
                        help='dataset')
    parser.add_argument('--dl-time-exclude', default='True', type=lambda x: x.lower() == 'true',
                        help='Set to False to include data load time')
    parser.add_argument('--model', default='resnet18',
                        help='select Resnet models from resnet18, resnet34, resnet50, resnet101, resnet152,'
                             'resnext50_32x4d, resnext101_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2')
    parser.add_argument('--device', default='hpu', help='device')
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-ebe', '--epochs_between_evals', default=4, type=int, metavar='N',
                        help='number of epochs to be completed before evaluation (default: 4)')
    parser.add_argument('-eoe', '--eval_offset_epochs', default=0, type=int, metavar='N',
                        help='offsets the epoch on which the evaluation starts (default: 0)')
    parser.add_argument('--dl-worker-type', default='HABANA', type=lambda x: x.upper(),
                        choices=["MP", "HABANA"], help='select multiprocessing or habana accelerated')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers (default: 10)')
    parser.add_argument('--process-per-node', default=8, type=int, metavar='N',
                        help='Number of process per node')
    parser.add_argument('--hls_type', default='HLS2', help='Node type')
    parser.add_argument('--lars_decay_epochs', default='36', type=int, help='number of decay epochs')
    parser.add_argument('--warmup_epochs', default='3', type=int, help='number of warmup epochs')
    parser.add_argument('--base_learning_rate', default='9', type=float, help='base learning rate')
    parser.add_argument('--end_learning_rate', default='0.0001', type=float, help='end learning rate')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=5e-5, type=float,
                        metavar='W', help='weight decay (default: 5e-5)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=30, type=int,
                        help='decrease lr every step-size epochs')
    parser.add_argument('--custom-lr-values', default=None, metavar='N', type=float, nargs='+',
                        help='custom lr values list')
    parser.add_argument('--custom-lr-milestones', default=None, metavar='N', type=int, nargs='+',
                        help='custom lr milestones list')
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                        help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--label-smoothing', default=0.1, type=float,
                        help='Apply label smoothing to the loss. This applies to'
                             'CrossEntropyLoss, when label_smoothing is greater than 0.')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--log-dir', default='', help='destination path for mllogs')
    parser.add_argument('--profile-steps', default=None,
                        help='Profile steps range separated by comma (e.g. `--profile_steps 100,105`)')

    parser.add_argument('--channels-last', default='False', type=lambda x: x.lower() == 'true',
                        help='Whether input is in channels last format.'
                        'Any value other than True(case insensitive) disables channels-last')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_acc_steps', type=int, default=1, help='Number of gradient accumulation steps')
    parser.add_argument('--num_gpus', type=int, default=0, help='Number of used gpus to run the model')
    parser.add_argument('--target_accuracy', default=0.759, type=float, help='Quality target of training')
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--use_torch_compile",
        dest="use_torch_compile",
        help="Use torch.compile feature to run the model",
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
        "--hpu_graphs",
        dest="hpu_graphs",
        help="Use HPU graphs feature to run the model by default",
        default='True', type=lambda x: x.lower() == 'true',
    )
    parser.add_argument('--enable-warmup', default='True', type=lambda x: x.lower() == 'true',
                    help='Whether the warmup is enabled')

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
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--num-train-steps', type=int, default=sys.maxsize, metavar='T',
                        help='number of steps a.k.a iterations to run in training phase')
    parser.add_argument('--num-eval-steps', type=int, default=sys.maxsize, metavar='E',
                        help='number of steps a.k.a iterations to run in evaluation phase')
    parser.add_argument('--save-checkpoint', action="store_true",
                        help='Whether or not to save model/checkpont; True: to save, False to avoid saving')
    parser.add_argument('--run-lazy-mode', default='True', type=lambda x: x.lower() == 'true',
                        help='run model in lazy execution mode(enabled by default).'
                        'Any value other than True(case insensitive) disables lazy mode')
    parser.add_argument('--use_autocast', action='store_true', help='enable autocast')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    set_env_params()
    args = parse_args()
    main(args)
