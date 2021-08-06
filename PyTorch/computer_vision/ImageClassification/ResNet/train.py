# Copyright (c) 2021, Habana Labs Ltd.  All rights reserved.


from __future__ import print_function

#Instead of importing resnet model from the standard torchvision package,
#import from a local copy. A local copy of resnet model file is used so that
#modifications can be done to the resnet model if necessary.
import model as resnet_models
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

try:
    from apex import amp
except ImportError:
    amp = None

def train_model(model, criterion, optimizer, image, target, apex, lazy_mode):
    output = model(image)
    loss = criterion(output, target)

    if apex:
       with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
       loss.backward()

    if lazy_mode:
        import habana_frameworks.torch.core as htcore
        htcore.mark_step()

    optimizer.step()

    for param in model.parameters():
        param.grad = None

    if lazy_mode:
        import habana_frameworks.torch.core as htcore
        htcore.mark_step()

    return loss, output

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, apex=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ",device=device)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)
    step_count = 0
    last_print_time= time.time()

    for image, target in metric_logger.log_every(data_loader, print_freq, header):

        image, target = image.to(device, non_blocking=False), target.to(device, non_blocking=False)

        if args.distributed:
            utils.barrier()

        dl_ex_start_time=time.time()

        if args.channels_last:
            image = image.contiguous(memory_format=torch.channels_last)

            if args.run_lazy_mode:
                # This mark_step is added so that the the lazy kernel can
                # create and evaluate the graph to infer the resulting tensor
                # as channels_last
                import habana_frameworks.torch.core as htcore
                htcore.mark_step()


        loss,output = train_model(model, criterion, optimizer, image, target, apex, args.run_lazy_mode)

        if step_count % print_freq == 0:
            loss_cpu = loss.item()
            output_cpu = output.detach().to('cpu')
            acc1, acc5 = utils.accuracy(output_cpu, target, topk=(1, 5))
            batch_size = image.shape[0]
            # Bring the loss tensor back to CPU before printing. Certainly needed if running on Habana.
            metric_logger.update(loss=loss_cpu, lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size*print_freq)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size*print_freq)
            current_time = time.time()
            last_print_time = dl_ex_start_time if args.dl_time_exclude else last_print_time
            metric_logger.meters['img/s'].update(batch_size*print_freq / (current_time - last_print_time))
            last_print_time = time.time()

        step_count = step_count + 1
        if step_count >= args.num_train_steps:
            break


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
                if args.run_lazy_mode:
                    # This mark_step is added so that the the lazy kernel can
                    # create and evaluate the graph to infer the resulting tensor
                    # as channels_last
                    import habana_frameworks.torch.core as htcore
                    htcore.mark_step()

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

#permute the params from filters first (KCRS) to filters last(RSCK) or vice versa.
#and permute from RSCK to KCRS is used for checkpoint saving
def permute_params(model, to_filters_last, lazy_mode):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if(param.ndim == 4):
                if to_filters_last:
                    param.data = param.data.permute((2, 3, 1, 0))
                else:
                    param.data = param.data.permute((3, 2, 0, 1))  # permute RSCK to KCRS

    if args.run_lazy_mode:
        import habana_frameworks.torch.core as htcore
        htcore.mark_step()


# permute the momentum from filters first (KCRS) to filters last(RSCK) or vice versa.
# and permute from RSCK to KCRS is used for checkpoint saving
# Used for Habana device only


def permute_momentum(optimizer, to_filters_last, lazy_mode):
    # Permute the momentum buffer before using for checkpoint
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

    if lazy_mode:
        import habana_frameworks.torch.core as htcore
        htcore.mark_step()

def main(args):

    if args.dl_worker_type == "MP":
        try:
            # Default 'fork' doesn't work with synapse. Use 'forkserver' or 'spawn'
            torch.multiprocessing.set_start_method('spawn')
        except RuntimeError:
            pass
    else:
        try:
            import habana_torch_dataloader
        except ImportError:
            assert False, "Could Not import habana_torch_dataloader"

    if args.run_lazy_mode:
        os.environ["PT_HPU_LAZY_MODE"] = "1"
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

    if args.device == 'hpu':
        from habana_frameworks.torch.utils.library_loader import load_habana_module
        load_habana_module()

    torch.manual_seed(args.seed)

    if args.deterministic:
        seed = args.seed
        random.seed(seed)
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
    if args.device == 'hpu' and args.workers > 0:
        # patch torch cuda functions that are being unconditionally invoked
        # in the multiprocessing data loader
        torch.cuda.current_device = lambda: None
        torch.cuda.set_device = lambda x: None

    if args.dl_worker_type == "MP":
        data_loader_type = torch.utils.data.DataLoader
    else:
        data_loader_type = habana_torch_dataloader.DataLoader

    data_loader = data_loader_type(
        dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    data_loader_test = data_loader_type(
        dataset_test, batch_size=test_batch_size, sampler=test_sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    print("Creating model")
    #model = torchvision.models.__dict__[args.model](pretrained=args.pretrained)
    #Instead of importing resnet model from the standard torchvision package,
    #import from a local copy. A local copy of resnet model file is used so that
    #modifications can be done to the resnet model if necessary.
    model = resnet_models.__dict__[args.model](pretrained=args.pretrained)

    model.to(device)
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

    criterion = nn.CrossEntropyLoss()

    if args.run_lazy_mode:
        from hb_custom import FusedSGD
        import habana_frameworks.torch.core as htcore
        htcore.enable_eliminate_common_subexpression(False)
        htcore.enable_constant_pooling(False)
        sgd_optimizer = FusedSGD
    else:
        sgd_optimizer = torch.optim.SGD
    optimizer = sgd_optimizer(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if(args.device == 'hpu'):
        permute_params(model, True, args.run_lazy_mode)
        permute_momentum(optimizer, True, args.run_lazy_mode)

    if args.apex:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.apex_opt_level
                                          )
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
            # To improve resnext101 dist performance, decrease number of all_reduce calls to 1 by increasing bucket size to 200
            bucket_size_mb = 200 if 'resnext101' in args.model else 100
            model = torch.nn.parallel.DistributedDataParallel(model, bucket_cap_mb=bucket_size_mb, broadcast_buffers=False,
                    first_bucket_cap_mb=bucket_size_mb)

        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    model_for_train = model

    if args.resume:
        if(args.device == 'hpu'):
            permute_params(model_without_ddp, False, args.run_lazy_mode)
            permute_momentum(optimizer, False, args.run_lazy_mode)

        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        #Permute the weight momentum buffer before using for checkpoint
        if(args.device == 'hpu'):
            permute_momentum(optimizer, True, args.run_lazy_mode)

        args.start_epoch = checkpoint['epoch'] + 1
        if(args.device == 'hpu'):
            permute_params(model_without_ddp, True, args.run_lazy_mode)

    if args.test_only:
        evaluate(model_for_eval, criterion, data_loader_test, device=device,
                print_freq=args.print_freq)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if lr_scheduler is None:
            adjust_learning_rate(optimizer, epoch, lr_vec)

        train_one_epoch(model_for_train, criterion, optimizer, data_loader,
                device, epoch, print_freq=args.print_freq, apex=args.apex)
        if lr_scheduler is not None:
            lr_scheduler.step()
        evaluate(model_for_eval, criterion, data_loader_test, device=device,
                print_freq=args.print_freq)

        if (args.output_dir and args.save_checkpoint):
            if args.device == 'hpu':
                permute_params(model_without_ddp, False, args.run_lazy_mode)
                # Use this model only to copy the state_dict of the actual model
                copy_model = resnet_models.__dict__[args.model](pretrained=args.pretrained)

                copy_model.load_state_dict(model_without_ddp.state_dict())
                # Permute the weight momentum buffer before saving in checkpoint
                permute_momentum(optimizer, False, args.run_lazy_mode)

                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to('cpu')

                checkpoint = {
                    'model': copy_model.state_dict(),
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

                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to('hpu')
                permute_params(model_without_ddp, True, args.run_lazy_mode)
                permute_momentum(optimizer, True, args.run_lazy_mode)

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

    if args.run_lazy_mode:
        os.environ.pop("PT_HPU_LAZY_MODE")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def set_env_params():
    os.environ["MAX_WAIT_ATTEMPTS"] = "50"
    os.environ["RUN_TPC_FUSER"] = "0"
    os.environ['HCL_CPU_AFFINITY'] = '1'
    os.environ['PT_HPU_ENABLE_SYNC_OUTPUT_HOST'] = 'false'

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--data-path', default='/software/data/pytorch/imagenet/ILSVRC2012/', help='dataset')
    parser.add_argument('--dl-time-exclude', default='True', type=lambda x: x.lower() == 'true', help='Set to False to include data load time')
    parser.add_argument('--model', default='resnet18',
                        help='select Resnet models from resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2')
    parser.add_argument('--device', default='hpu', help='device')
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--dl-worker-type', default='MT', type=lambda x: x.upper(),
                        choices = ["MT", "MP"], help='select multithreading or multiprocessing')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--process-per-node', default=8, type=int, metavar='N',
                        help='Number of process per node')
    parser.add_argument('--hls_type', default='HLS1', help='Node type')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=30, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--custom-lr-values', default=None, metavar='N', type=float, nargs='+', help='custom lr values list')
    parser.add_argument('--custom-lr-milestones', default=None, metavar='N', type=int, nargs='+',
                        help='custom lr milestones list')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')

    parser.add_argument('--channels-last', default='True', type=lambda x: x.lower() == 'true',
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
    parser.add_argument('--save-checkpoint', action="store_true",
                        help='Whether or not to save model/checkpont; True: to save, False to avoid saving')
    parser.add_argument('--run-lazy-mode', action='store_true',
                        help='run model in lazy execution mode')
    parser.add_argument('--deterministic', action="store_true",
                        help='Whether or not to make data loading deterministic;This does not make execution deterministic')
    parser.add_argument('--hmp', dest='is_hmp', action='store_true', help='enable hmp mode')
    parser.add_argument('--hmp-bf16', default='', help='path to bf16 ops list in hmp O1 mode')
    parser.add_argument('--hmp-fp32', default='', help='path to fp32 ops list in hmp O1 mode')
    parser.add_argument('--hmp-opt-level', default='O1', help='choose optimization level for hmp')
    parser.add_argument('--hmp-verbose', action='store_true', help='enable verbose mode for hmp')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    set_env_params()
    args = parse_args()
    main(args)

