# Copyright (c) 2021, Habana Labs Ltd.  All rights reserved.


from __future__ import print_function
import datetime
import os
import time
import sys

import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
import random
import utils

#Instead of importing resnet model from the standard torchvision package,
#import from a local copy. A local copy of resnet model file is used so that
#modifications can be done to the resnet model if necessary.
import model as resnet_models

try:
    from apex import amp
except ImportError:
    amp = None

def train_model(model, criterion, optimizer, image, target, apex):
    output = model(image)
    loss = criterion(output, target)
    optimizer.zero_grad()
    if apex:
       with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
       loss.backward()
    optimizer.step()

    return loss.item(),output.detach().to('cpu')

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, apex=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ",device=device)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)
    step_count = 0
    for image, target in metric_logger.log_every(data_loader, print_freq, header):

        if args.distributed:
            utils.barrier()

        start_time = time.time()

        image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)

        if args.channels_last:
            image = image.contiguous(memory_format=torch.channels_last)

        loss_cpu,output_cpu = train_model(model, criterion, optimizer, image, target, apex)

        acc1, acc5 = utils.accuracy(output_cpu, target, topk=(1, 5))
        batch_size = image.shape[0]
        #Bring the loss tensor back to CPU before printing. Certainly needed if running on Habana.
        metric_logger.update(loss=loss_cpu, lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))
        step_count = step_count + 1
        if step_count >= args.num_train_steps:
            break;


def evaluate(model, criterion, data_loader, device, print_freq=100):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ",device=device)
    header = 'Test:'
    step_count = 0
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            if args.channels_last:
                image = image.contiguous(memory_format=torch.channels_last)

            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            #Bring the loss tensor back to CPU before printing. Certainly needed if running on Habana.
            loss_cpu = loss.to('cpu').detach()
            metric_logger.update(loss=loss_cpu.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
            step_count = step_count + 1
            if step_count >= args.num_eval_steps:
                break;
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    #Return from here if evaluation phase does not go through any iterations.(eg, The data set is so small that
    #there is only one eval batch, but that was skipped in data loader due to drop_last=True)
    if len(metric_logger.meters) == 0 :
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

def enable_tracing(device):
    with torch.jit.optimized_execution(True):
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        if(device==torch.device('habana')):
            import hb_torch
            hb_torch.enable()
        sample_trace_tensor = torch.zeros(args.batch_size, 3, 224, 224).to(device)
        return sample_trace_tensor

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

    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler

#permute the params from filters first (KCRS) to filters last(RSCK) or vice versa.
#and permute from RSCK to KCRS is used for checkpoint saving
def permute_params(model, to_filters_last):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if(param.ndim == 4):
                if to_filters_last:
                    param.data = param.data.permute((2,3,1,0))
                else:
                    param.data = param.data.permute((3,2,0,1))

#permute the momentum from filters first (KCRS) to filters last(RSCK) or vice versa.
#and permute from RSCK to KCRS is used for checkpoint saving
def permute_momentum(optimizer, to_filters_last):
    #Permute the momentum buffer before using for checkpoint
    for group in optimizer.param_groups:
        for p in group['params']:
            param_state = optimizer.state[p]
            if 'momentum_buffer' in param_state:
                buf = param_state['momentum_buffer']
                if(buf.ndim == 4):
                    if to_filters_last:
                        buf = buf.permute((2,3,1,0))
                    else:
                        buf = buf.permute((3,2,0,1))
                    param_state['momentum_buffer'] = buf

#Data loader worker init function
def dl_worker_init_fn(seed):
    if seed is not None:
        random.seed(seed)

def main(args):
    if args.is_hmp:
        from hmp import hmp
        hmp.convert(opt_level=args.hmp_opt_level, bf16_file_path=args.hmp_bf16,
                    fp32_file_path=args.hmp_fp32, isVerbose=args.hmp_verbose)

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

    if args.device == 'habana':
        sys.path.append(os.path.realpath(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "../../../common")))
        from library_loader import load_habana_module
        load_habana_module()

    torch.manual_seed(args.seed)

    if args.deterministic:
        seed = args.seed
        if args.device == 'cuda':
            torch.cuda.manual_seed(seed)
    else:
        seed = None

    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True

    # Limit the test(eval) phase batch size to a lower value to reduce overall device memory pressure
    test_batch_size = args.batch_size
    if args.batch_size > 32 :
        test_batch_size = 32

    train_dir = os.path.join(args.data_path, 'train')
    val_dir = os.path.join(args.data_path, 'val')
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir,
                                                                   args.cache_dataset, args.distributed)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, worker_init_fn=dl_worker_init_fn(seed), pin_memory=True, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=test_batch_size,
        sampler=test_sampler, num_workers=args.workers, worker_init_fn=dl_worker_init_fn(seed), pin_memory=True, drop_last=True)

    print("Creating model")
    #model = torchvision.models.__dict__[args.model](pretrained=args.pretrained)
    #Instead of importing resnet model from the standard torchvision package,
    #import from a local copy. A local copy of resnet model file is used so that
    #modifications can be done to the resnet model if necessary.
    model = resnet_models.__dict__[args.model](pretrained=args.pretrained)

    model.to(device)
    if args.channels_last:
        if(device==torch.device('cuda')):
            print('Converting model to channels_last format on CUDA')
            model.to(memory_format=torch.channels_last)
        elif(args.device == 'habana'):
            print('Converting model params to channels_last format on Habana')
            #TODO:
            #model.to(device).to(memory_format=torch.channels_last)
            #The above model conversion doesn't change the model params
            #to channels_last for many components - e.g. convolution.
            #So we are forced to rearrange such tensors ourselves.

    if(args.device == 'habana'):
        permute_params(model, True)


    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.apex:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.apex_opt_level
                                          )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    model_for_eval = model
    if args.run_trace_mode:
        sample_trace_tensor = enable_tracing(device)

        if args.channels_last:
            sample_trace_tensor = sample_trace_tensor.contiguous(memory_format=torch.channels_last)
        # Create traced model for eval
        model.eval()
        model_for_eval = torch.jit.trace(model, sample_trace_tensor, check_trace=False)
        # Create traced model for train
        model.train()
        model = torch.jit.trace(model, sample_trace_tensor, check_trace=False)
        model_for_train = model

    # TBD: pass the right module for ddp
    model_without_ddp = model

    if args.distributed:
        if args.device == 'habana':
            model = torch.nn.parallel.DistributedDataParallel(model, bucket_cap_mb=100, broadcast_buffers=False)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    model_for_train = model

    if args.resume:
        if(args.device == 'habana'):
            permute_params(model_without_ddp, False)
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if(args.device == 'habana'):
            permute_momentum(optimizer, True)
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if(args.device == 'habana'):
            permute_params(model_without_ddp, True)

    if args.test_only:
        evaluate(model_for_eval, criterion, data_loader_test, device=device,
                print_freq=args.print_freq)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_one_epoch(model_for_train, criterion, optimizer, data_loader,
                device, epoch, print_freq=args.print_freq, apex=args.apex)
        lr_scheduler.step()
        evaluate(model_for_eval, criterion, data_loader_test, device=device,
                print_freq=args.print_freq)

        if (args.output_dir and args.save_checkpoint):
            if args.device == 'habana':
                permute_params(model_without_ddp, False)
                #Use this model only to copy the state_dict of the actual model
                copy_model = resnet_models.__dict__[args.model](pretrained=args.pretrained)

                copy_model.load_state_dict(model_without_ddp.state_dict())
                permute_momentum(optimizer, False)
                for state in optimizer.state.values():
                  for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to('cpu')

                checkpoint = {
                    'model': copy_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args}
                utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
                utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, 'checkpoint.pth'))

                for state in optimizer.state.values():
                  for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to('habana')
                permute_params(model_without_ddp, True)

            else:
                checkpoint = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args}
                utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
                utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, 'checkpoint.pth'))


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', help='dataset')
    parser.add_argument('--model', default='resnet18', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=30, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')

    parser.add_argument('--channels-last', default='True', type=lambda x:x.lower() == 'true',
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
    parser.add_argument('--save-checkpoint',  action="store_true",
                        help='Whether or not to save model/checkpont; True: to save, False to avoid saving')
    parser.add_argument('--run-trace-mode', action='store_true', default=False,
                        help='run JIT mode with fusion enabled')
    parser.add_argument('--deterministic',  action="store_true",
                        help='Whether or not to make data loading deterministic;This does not make execution deterministic')
    parser.add_argument('--hmp', dest='is_hmp', action='store_true',help='enable hmp mode')
    parser.add_argument('--hmp-bf16', default='', help='path to bf16 ops list in hmp O1 mode')
    parser.add_argument('--hmp-fp32', default='', help='path to fp32 ops list in hmp O1 mode')
    parser.add_argument('--hmp-opt-level', default='O1', help='choose optimization level for hmp')
    parser.add_argument('--hmp-verbose', action='store_true', help='enable verbose mode for hmp')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
