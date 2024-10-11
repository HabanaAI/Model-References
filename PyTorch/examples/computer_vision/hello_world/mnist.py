# Copyright (C) 2021-2022 Habana Labs, Ltd. an Intel Company

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import os
import utils
import sys
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.utils.debug as htdebug


# todo: [SW-165872] revert below W/A when PR 113374 included in pytorch fork
torch._dynamo.config.optimize_ddp = False


class Net(nn.Module):
    def __init__(self, use_autocast=False):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(7744, 128)
        self.fc2 = nn.Linear(128, 10)
        self._use_autocast = use_autocast

    def forward(self, x):
        with torch.autocast("hpu", dtype=torch.bfloat16, enabled=self._use_autocast):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 3, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    if args.use_torch_compile:
        model = torch.compile(model,backend="hpu_backend")

    def train_function(data, target):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        return loss

    training_step = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        loss = train_function(data, target)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx *
                len(data), len(train_loader.dataset)/args.world_size,
                100. * batch_idx / len(train_loader), loss.item()))
            if batch_idx != 0 and args.dry_run:
                break

        if args.max_training_step != 0:
            training_step +=1
            if training_step == args.max_training_step:
                break

def test(args, model, device, test_loader):
    model.eval()
    if args.use_torch_compile:
       model = torch.compile(model,backend="hpu_backend")

    def test_function(data, target, test_loss):
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        acc, _ = utils.accuracy(output, target, topk=(1, 5))
        return test_loss, acc

    test_loss = 0
    correct = 0
    metric_logger = utils.MetricLogger(delimiter="  ", device=device)
    with torch.no_grad():
        for data, target in test_loader:
            batch_size = data.shape[0]
            data, target = data.to(device), target.to(device)
            test_loss, acc = test_function(
                data, target, test_loss)
            metric_logger.meters['acc'].update(acc.item(), n=batch_size)

    test_loss /= (len(test_loader.dataset)/args.world_size)
    metric_logger.meters['loss'].update(test_loss)
    metric_logger.synchronize_between_processes()

    print('\nTotal test set: {}, number of workers: {}'.format(
        len(test_loader.dataset), args.world_size))
    print('* Average Acc {top.global_avg:.3f} Average loss {value.global_avg:.3f}'.format(
        top=metric_logger.acc, value=metric_logger.loss))


test._fnc = None


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--use-torch-compile', action='store_true',
                        default=False, help='Use torch.compile and HPU dynamo backend')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--hpu', action='store_true', default=False,
                        help='Use hpu device')
    parser.add_argument('--data-path', type=str, default='../data', metavar='STR',
                        help='input data path for train and test')
    parser.add_argument('--dl-worker-type', default='MP', type=lambda x: x.upper(),
                        choices=["MT", "MP"], help='select multithreading or multiprocessing')
    parser.add_argument('--world_size', default=1, type=int, metavar='N',
                        help='number of total workers (default: 1)')
    parser.add_argument('--process-per-node', default=8, type=int, metavar='N',
                        help='Number of process per node')
    parser.add_argument('--start-epoch', type=int, default=0,
                        metavar='N', help='starting epoch number, default 0')
    parser.add_argument('--checkpoint', default='',
                        help='resume from checkpoint')
    parser.add_argument('--distributed', action='store_true',
                        help='whether to enable distributed mode and run on multiple devices')
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--autocast', action='store_true', default=False)
    parser.add_argument('--max-training-step', type=int, default=0, metavar='N',
                    help='Number of training steps to run. (default: 0)')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    torch.multiprocessing.set_start_method('spawn')
    if args.hpu:
        device = torch.device("hpu")
        # patch torch cuda functions that are being unconditionally invoked
        # in the multiprocessing data loader
        torch.cuda.current_device = lambda: None
        torch.cuda.set_device = lambda x: None

    if args.use_torch_compile:
        assert int(torch.__version__.split('.')[
                   0]) >= 2, "Graph mode is available only in PyTorch 2.x."

    utils.init_distributed_mode(args)

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if args.distributed and int(os.environ["LOCAL_RANK"]) != 0:
        # might be downloading mnist data, let local rank 0 download first
        torch.distributed.barrier()

    dataset1 = datasets.MNIST(args.data_path, train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST(args.data_path, train=False,
                              transform=transform)

    if args.distributed:

        if int(os.environ["LOCAL_RANK"]) == 0:
            # mnist data is downloaded, indicate other ranks can proceed
            torch.distributed.barrier()
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset1)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset2)

        train_loader = torch.utils.data.DataLoader(
            dataset1, batch_size=args.batch_size, sampler=train_sampler,
            num_workers=12, pin_memory=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(
            dataset2, batch_size=args.batch_size, sampler=test_sampler,
            num_workers=12, pin_memory=True, drop_last=True)
    else:
        train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net(use_autocast=args.autocast).to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    if args.distributed and args.hpu:
        model = torch.nn.parallel.DistributedDataParallel(model, bucket_cap_mb=100, broadcast_buffers=False,
                                                          gradient_as_bucket_view=True)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch']

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(args.start_epoch + 1, args.epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train(args, model, device, train_loader,
              optimizer, epoch)
        test(args, model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save({"model": model.state_dict(), 'optimizer': optimizer.state_dict(
        ), 'epoch': args.epochs}, "mnist_cnn.pt")


if __name__ == '__main__':
    main()
