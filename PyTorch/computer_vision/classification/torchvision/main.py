# Copyright (c) 2021, Habana Labs Ltd.  All rights reserved.

import argparse
import os
import random
import shutil
import datetime
import time
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.distributed as dist
import googlenet_utils as utils


parser = argparse.ArgumentParser(description='PyTorch GoogleNet Training')
parser.add_argument('--amp', dest='is_amp', action='store_true', help='enable GPU amp mode')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--data-path', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset-type', help='Imagenet',
                    type=str, choices=['Imagenet'],
                    default='Imagenet')
parser.add_argument('--device', help='cpu,hpu,gpu',
                    type=str, choices=['cpu', 'hpu', 'gpu'],
                    default='hpu')
parser.add_argument('--distributed', action='store_true', help='whether to enable distributed mode and run on multiple devices')
# distributed training parameters BEGIN
parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
parser.add_argument('--dl-worker-type', default='MP', type=lambda x: x.upper(),
                    choices = ["MP", "HABANA"], help='select multiprocessing or habana accelerated')
parser.add_argument('--process-per-node', default=8, type=int, metavar='N',
                        help='Number of process per node')
# distributed training parameters END
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--enable-lazy', action='store_true',
                    help='whether to enable Lazy mode, if it is not set, your code will run in Eager mode')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--gpu', default=0, type=int, metavar='N',
                    help='GPU device to target')
parser.add_argument('--hmp', dest='is_hmp', action='store_true', help='enable hmp mode')
parser.add_argument('--hmp-bf16', default='ops_bf16_googlenet.txt', help='path to bf16 ops list in hmp O1 mode')
parser.add_argument('--hmp-fp32', default='ops_fp32_googlenet.txt', help='path to fp32 ops list in hmp O1 mode')
parser.add_argument('--hmp-opt-level', default='O1', help='choose optimization level for hmp')
parser.add_argument('--hmp-verbose', action='store_true', help='enable verbose mode for hmp')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr-step-size', default=30, type=int, help='decrease lr every step-size epochs')
parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
parser.add_argument('--model', default='googlenet', help='model')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--no-aux-logits', action='store_true', help='disable aux logits in GoogleNet')
parser.add_argument('--num-train-steps', type=int, default=sys.maxsize,
                    help='Number of training steps to run.')
parser.add_argument('-p', '--print-interval', default=1, type=int,
                    metavar='N', help='print frequency (default: 1)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: None)')
parser.add_argument('--save-checkpoint', default=1, const=5, type=int, nargs='?', help='Save checkpoint after every <N> epochs')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers (default: 10)')

best_acc1 = 0

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

    if lazy_mode:
        import habana_frameworks.torch.core as htcore
        htcore.mark_step()


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


def get_model(model, pretrained, resume, no_aux_logits):
    modeldict = models.__dict__

    if(model=='googlenet' and no_aux_logits):
        model = modeldict[model](pretrained=pretrained, init_weights=False if resume else None, aux_logits=False)
    else:
        model = modeldict[model](pretrained=pretrained, init_weights=False if resume else None)

    return model


def main():

    if args.dl_worker_type == "MP":
        try:
            # Default 'fork' doesn't work with synapse. Use 'forkserver' or 'spawn'
            torch.multiprocessing.set_start_method('spawn')
            #work around for multi-process data loading for single card training. for multi card, use habana_torch_dataloader
        except RuntimeError:
            pass
    elif args.dl_worker_type == "HABANA":
        try:
            import habana_dataloader
        except ImportError:
            assert False, "Could Not import habana dataloader package"

    utils.init_distributed_mode(args)
    print(args)
    if args.enable_lazy:
        os.environ["PT_HPU_LAZY_MODE"]="1"
        import habana_frameworks.torch.core as htcore

    if args.is_hmp:
        from habana_frameworks.torch.hpex import hmp
        hmp.convert(opt_level=args.hmp_opt_level, bf16_file_path=args.hmp_bf16,
                    fp32_file_path=args.hmp_fp32, isVerbose=args.hmp_verbose)
    if args.device == 'hpu':
        from habana_frameworks.torch.utils.library_loader import load_habana_module
        load_habana_module()
        device=torch.device('hpu')
    elif args.device == 'gpu':
        if torch.cuda.is_available():
            device_name = "cuda:" + str(args.gpu)
            print(device_name)
            device = torch.device(device_name)
        else:
            assert False, "No GPU device"
    elif args.device == 'cpu':
        device=torch.device('cpu')
    else:
        assert False, "Need device type"
    print('Using', device)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
     # Data loading code
    traindir = os.path.join(args.data_path, 'train')
    valdir = os.path.join(args.data_path, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler=None
        val_sampler = None

    if args.device != 'gpu' and args.workers > 0:
        # patch torch cuda functions that are being unconditionally invoked
        # in the multiprocessing data loader
        torch.cuda.current_device = lambda: None
        torch.cuda.set_device = lambda x: None
    if args.dl_worker_type == "MP":
        data_loader_type = torch.utils.data.DataLoader
    elif args.dl_worker_type == "HABANA":
        data_loader_type = habana_dataloader.HabanaDataLoader
    train_loader = data_loader_type(
        train_dataset, batch_size=args.batch_size,
        shuffle=True if args.dl_worker_type == "MP" and args.distributed == False else False,
        num_workers=args.workers, pin_memory=True if args.device != 'cpu' else False, sampler=train_sampler)
    val_loader = data_loader_type(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True if args.device != 'cpu' else False, sampler=val_sampler)


    global best_acc1
    # create model
    print("Creating model ", args.model)
    model = get_model(args.model, args.pretrained, args.resume, args.no_aux_logits)
    model.to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    if args.enable_lazy:
        from habana_frameworks.torch.hpex.optimizers import FusedSGD
        sgd_optimizer = FusedSGD
    else:
        sgd_optimizer = torch.optim.SGD

    optimizer = sgd_optimizer(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    if args.device=='gpu' and args.is_amp:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))

            checkpoint = torch.load(args.resume, map_location='cpu')
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.device == 'hpu':
        permute_params(model, True, args.enable_lazy)
        permute_momentum(optimizer, True, args.enable_lazy)

    model_for_eval = model

    model_without_ddp = model

    if args.distributed:
        if args.device == 'hpu':
            bucket_size_mb = 200
            model = torch.nn.parallel.DistributedDataParallel(model, bucket_cap_mb=bucket_size_mb, broadcast_buffers=False,
                    gradient_as_bucket_view=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    model_for_train = model

    if args.evaluate:
        validate(val_loader, model_for_eval, criterion, device, args)
        return

    epoch_time = AverageMeter('EpochTime', ':6.3f')
    start_time = time.time()
    e_time = start_time
    for epoch in range(args.start_epoch, args.epochs):
        #adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        if args.distributed and args.dl_worker_type != "HABANA":
            train_sampler.set_epoch(epoch)

        print("training epoch ", epoch)
        train(train_loader, model_for_train, criterion, optimizer, epoch, device, args)
        lr_scheduler.step()

        # evaluate on validation set
        print("validating epoch ", epoch)
        acc1 = validate(val_loader, model_for_eval, criterion, device, args)

        # measure elapsed time
        epoch_time.update(time.time() - e_time)
        e_time = time.time()
        epoch_progress = ProgressMeter(
            len(range(args.start_epoch, args.epochs)),
            [epoch_time],
            prefix="END OF EPOCH [{}]:".format(epoch))
        epoch_progress.display(epoch - args.start_epoch + 1)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if args.save_checkpoint > 0 and (epoch+1)%args.save_checkpoint == 0:
            print("saving ckpt epoch ", epoch)
            if args.device == 'hpu':
                # Permute model parameters from RSCK to KCRS
                permute_params(model_without_ddp, False, args.enable_lazy)
                # Use this model only to copy the state_dict of the actual model
                copy_model = get_model(args.model, args.pretrained, args.resume, args.no_aux_logits)#models.__dict__[args.model](pretrained=args.pretrained)
                state_dict = model_without_ddp.state_dict()
                for k,v in state_dict.items():
                    if 'num_batches_tracked' in k and v.dim() == 1:
                        state_dict[k] = v.squeeze(0)

                copy_model.load_state_dict(state_dict)
                # Permute the weight momentum buffer before saving in checkpoint
                permute_momentum(optimizer, False, args.enable_lazy)

                # Bring all model parameters and optimizer parameters to CPU
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to('cpu')

            # Save model parameters in checkpoint
            filename = 'checkpoint_'+str(epoch)+'_'+args.device+'.pth.tar'
            save_checkpoint({
                'epoch': epoch,
                'arch': args.model,
                'state_dict': copy_model.state_dict() if args.device=='hpu' else model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
            }, is_best, filename)
            if args.device == 'hpu':
                #Take back model parameters and optimizer parameters to HPU
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to('hpu')
                # Permute back from KCRS to RSCK
                permute_params(model, True, args.enable_lazy)
                permute_momentum(optimizer, True, args.enable_lazy)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train(train_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter('BatchTime', ':6.3f')
    #Images per second with data loading time
    image_time_DL = AverageMeter('imgs/s(Inc. DL)', ':6.3f')
    image_time = AverageMeter('imgs/s(Exc. DL)', ':6.3f')
    data_time = AverageMeter('DL Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':0.2f')
    top5 = AverageMeter('Acc@5', ':0.2f')

    if  args.print_interval == 1:
        progress = ProgressMeter(
        len(train_loader),
        [batch_time, image_time, image_time_DL, data_time, losses, top1, top5],
        prefix='Epoch: [{}]'.format(epoch))
    else:
        progress = ProgressMeter(
        len(train_loader),
        [batch_time, image_time_DL, data_time, losses, top1, top5],
        prefix='Epoch: [{}]'.format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
        # measure data loading time
        data_end = time.time()
        data_loading_time = data_end - end

        images = images.contiguous(memory_format=torch.channels_last)
        if args.enable_lazy:
            import habana_frameworks.torch.core as htcore
            htcore.mark_step()

        # compute output
        output = model(images)

        if not args.no_aux_logits:
            aux_logits2 = output.aux_logits2
            aux_logits1 = output.aux_logits1
            output = output.logits
            # "Going Deeper with Convolutions" <http://arxiv.org/abs/1409.4842>, Page 6.
            loss = criterion(output, target) + 0.3*(criterion(aux_logits2, target) + criterion(aux_logits1, target))
        else:
            loss = criterion(output, target)

        optimizer.zero_grad(set_to_none = True)
        if args.device =='gpu' and args.is_amp:
            from apex import amp
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if args.enable_lazy:
            import habana_frameworks.torch.core as htcore
            htcore.mark_step()

        if args.is_hmp:
            from habana_frameworks.torch.hpex import hmp
            with hmp.disable_casts():
                optimizer.step()
        else:
            optimizer.step()

        if args.enable_lazy:
            import habana_frameworks.torch.core as htcore
            htcore.mark_step()

        if i % args.print_interval == 0:
            # measure accuracy and record loss
            acc1, acc5 = accuracy_classification(output, target, topk=(1, 5))
            batch_size = images.shape[0]
            losses.update(loss.item(), n=batch_size)
            top1.update(acc1[0].item(), n=batch_size*args.print_interval)
            top5.update(acc5[0].item(), n=batch_size*args.print_interval)

            batch_elapsed_time = time.time() - data_end
            total_elapsed_time = time.time() - end
            # measure elapsed time
            if epoch == 0:
                batch_time.update(batch_elapsed_time, n=args.print_interval, skip=2*args.print_interval, avoid_warmup=True)
                data_time.update(data_loading_time, n=args.print_interval, avoid_warmup=True)
                image_time_DL.update(batch_size*args.print_interval/total_elapsed_time, n=args.print_interval, skip=2*args.print_interval, avoid_warmup=True)
                if (args.print_interval == 1):
                    image_time.update(batch_size/batch_elapsed_time, n=args.print_interval, skip=2*args.print_interval, avoid_warmup=True)
            else:
                batch_time.update(batch_elapsed_time, n=args.print_interval)
                data_time.update(data_loading_time,n=args.print_interval)
                image_time_DL.update(batch_size*args.print_interval/total_elapsed_time, n=args.print_interval)
                if (args.print_interval == 1):
                    image_time.update(batch_size/batch_elapsed_time, n=args.print_interval, skip=2*args.print_interval)
            progress.display(i)
            end = time.time()

        if i == args.num_train_steps-1:
            break


def validate(val_loader, model, criterion, device, args):
    #Images per second with data loading time
    image_time_DL = AverageMeter('imgs/s(Inc. DL)', ':6.3f')
    batch_time = AverageMeter('BatchTime', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':0.2f')
    top5 = AverageMeter('Acc@5', ':0.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, image_time_DL, losses, top1, top5],
        prefix='Test: ')

    print("MODEL EVAL")
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        data_end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
            images = images.contiguous(memory_format=torch.channels_last)
            if args.enable_lazy:
                import habana_frameworks.torch.core as htcore
                htcore.mark_step()

            # compute output
            output = model(images)

            loss = criterion(output, target)

            if i % args.print_interval == 0:
                acc1, acc5 = accuracy_classification(output, target, topk=(1, 5))
                batch_size = images.shape[0]
                losses.update(loss.item())
                top1.update(acc1[0].item(), n=batch_size*args.print_interval)
                top5.update(acc5[0].item(), n=batch_size*args.print_interval)

                # measure elapsed time
                image_time_DL.update(batch_size*args.print_interval/(time.time() - data_end), n=args.print_interval)
                batch_time.update(time.time() - data_end, n=1, avoid_warmup=True)
                progress.display(i)
                data_end = time.time()

            if i == args.num_train_steps-1:
                break

    # gather the stats from all processes
    top1.synchronize_between_processes(device)
    top5.synchronize_between_processes(device)

    print(' * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'.format(top1=top1, top5=top5))

    return top1.global_avg


def save_checkpoint(state, is_best, filename):
    if utils.is_main_process():
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best'+str(state['epoch'])+'.pth.tar')


def set_env_params():
    os.environ["MAX_WAIT_ATTEMPTS"] = "50"
    os.environ['HCL_CPU_AFFINITY'] = '1'
    os.environ['PT_HPU_ENABLE_SYNC_OUTPUT_HOST'] = 'false'


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.total = 0.0
        self.warmup_val = 0

    def update(self, val, n=20, skip=20, avoid_warmup=False):
        self.val = val
        avg_count = self.count
        if self.count != 0 and (avoid_warmup and self.count >= skip or avoid_warmup == False):
           if avoid_warmup:
               avg_count = self.count - skip + n
           if avg_count > 0:
               self.sum += val * n
               self.avg = self.sum / avg_count
        self.count += n
        self.total += val * n

    def synchronize_between_processes(self,device):
        """
        Warning: does not synchronize the deque!
        """
        if not utils.is_dist_avail_and_initialized():
            return
        if device.type == 'hpu':
            t = torch.tensor([self.count, self.total], dtype=torch.float32).to('hpu')
        else:
            t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
            dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def global_avg(self):
        return self.total / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy_classification(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    set_env_params()
    args = parser.parse_args()
    main()

