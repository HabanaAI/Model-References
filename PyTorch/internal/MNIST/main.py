from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from torchvision import datasets, transforms
import time
from torch.utils import data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class TrainMetaData():
    def __init__(self):
        self.current_train_step = 0
        self.current_eval_step = 0
        self.num_train_steps = sys.maxsize
        self.num_eval_steps = sys.maxsize
        self.logging = True #Enable - default

    def increment_train_step(self):
        self.current_train_step += 1
        return self.current_train_step

    def increment_eval_step(self):
        self.current_eval_step += 1
        return self.current_eval_step

    def set_num_train_steps(self, x):
        self.num_train_steps = x

    def set_num_eval_steps(self, x):
        self.num_eval_steps = x

    def end_train(self):
        if (self.current_train_step == self.num_train_steps):
            return True
        else:
            return False

    def end_eval(self):
        if (self.current_eval_step == self.num_eval_steps):
            return True
        else:
            return False

    def end_train_n_eval(self):
        if (self.end_train() and self.end_eval()):
            return True
        else:
            return False

    def set_logging(self, x):
        self.logging = x

    def is_logging(self):
        return self.logging

    @staticmethod
    def accuracy(output, target, topk=(1,)):
        """
        Computes the accuracy over the k top
        predictions for the specified values of k
        """
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)

            pred = pred.t()
            pred_cpu = torch.tensor(pred, device='cpu')
            target_cpu = torch.tensor(target, device='cpu')

            correct = pred_cpu.eq(target_cpu[None])

            res = []
            for k in topk:
                correct_k = correct[:k].flatten().sum(dtype=torch.float32)
                res.append(correct_k * (100.0 / batch_size))
            return res



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(3 * 3 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = x.view(-1, 3 * 3 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train_model(model, data, target):
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    return loss.item(), output.detach().to('cpu')

def train(args, model, device, train_loader, optimizer, epoch, trainMetaData,rank):
    model.train()
    if(trainMetaData.is_logging() and rank==0):
        with open('mnistpy.log', 'w') as file:  # reset file
            file.write('')

    for batch_idx, (data, target) in enumerate(train_loader):
        iter_timer_start = time.time()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss_cpu, output_cpu = train_model(model, data, target)
        optimizer.step()
        iter_duration = time.time() - iter_timer_start
        # if batch_idx % args.log_interval == 0:
        acc1, acc5 = trainMetaData.accuracy(output_cpu, target, topk=(1, 5))
        log_msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} '\
                  'acc1: {:.6f} acc5: {:.6f} time: {:.6f}\n'.format(
                  epoch, batch_idx * len(data), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader),
                  loss_cpu, acc1, acc5,
                  iter_duration)

        if(trainMetaData.is_logging() and rank==0):
            with open('mnistpy.log', 'a') as file:
                file.write(log_msg)
        print(log_msg)
        trainMetaData.increment_train_step()
        if trainMetaData.end_train() is True:
            break

def test(args, model, device, test_loader, trainMetaData):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            output_cpu = output
            pred = output_cpu.to(torch.device('cpu')).argmax(dim=1, keepdim=True)
            target_cpu = target
            target_cpu = target_cpu.to(torch.device('cpu'))
            new_view = target_cpu.view_as(pred)
            correct += pred.eq(new_view).sum().item()
            trainMetaData.increment_eval_step()
            if trainMetaData.end_eval() is True:
                break


    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def setup_dist(rank, world_size,backend):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ["ID"] = str(rank)
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup_dist():
    dist.destroy_process_group()

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-habana', action='store_true', default=False,
                        help='disables habana training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--run-trace-mode', action='store_true', default=False,
                        help='run JIT mode with fusion enabled')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--num-train-steps', type=int, default=sys.maxsize, metavar='T',
                        help='number of steps for training')
    parser.add_argument('--num-eval-steps', type=int, default=sys.maxsize, metavar='E',
                        help='number of steps for evaluation')
    parser.add_argument('--no-log', action='store_true', default=False,
                        help='disable log')
    parser.add_argument('--hmp', dest='is_hmp', action='store_true', help='enable hmp mode')
    #Distributed parameters
    parser.add_argument('--backend',default='hcl', help='Device backend for distributed')
    args = parser.parse_args()
    return args

def permute_params_on_device(model):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if(param.ndim == 4):
                param.data = param.data.permute((2,3,1,0))

def main(args):

    rank = args.rank

    if args.is_hmp:
        from hmp import hmp
        hmp.convert()

    use_habana = not args.no_habana
    if use_habana:
        torch.ops.load_library(os.path.join(os.environ['BUILD_ROOT_LATEST'], "libhabana_pytorch_plugin.so"))
        sys.path.insert(0, os.path.join(os.environ['BUILD_ROOT_LATEST']))

    torch.manual_seed(args.seed)

    device = torch.device("habana" if use_habana else "cpu")

    model = Net().to(device)
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_habana else {}
    kwargs = {'pin_memory': True} if use_habana else {}
    if(device==torch.device('habana')):
        permute_params_on_device(model)

    if args.run_trace_mode:
        with torch.jit.optimized_execution(True):
            torch._C._jit_override_can_fuse_on_cpu(False)
            torch._C._jit_set_profiling_executor(False)
            torch._C._jit_set_profiling_mode(False)
            if(device==torch.device('habana')):
                import hb_torch
                hb_torch.enable()
            sample_trace_tensor = torch.FloatTensor(64, 1, 28, 28).to(device)
            model = torch.jit.trace(model, sample_trace_tensor, check_trace=False)

    if(args.distributed == True):
        sampler = data.DistributedSampler(args.train_dataset)
        train_loader = torch.utils.data.DataLoader(
            args.train_dataset, sampler = sampler,
            batch_size = args.batch_size, shuffle = (sampler is None), **kwargs)
        model = DDP(model)
    else:
        train_loader = torch.utils.data.DataLoader(args.train_dataset,batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    trainMetaData = TrainMetaData()
    trainMetaData.set_num_train_steps(args.num_train_steps)
    trainMetaData.set_num_eval_steps(args.num_eval_steps)
    log = not args.no_log
    trainMetaData.set_logging(True if log else False)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, trainMetaData, rank)
        test(args, model, device, test_loader, trainMetaData)

        if (trainMetaData.end_train_n_eval()):
            break
    if args.save_model and rank==0:
        model_cpu = model
        torch.save(model_cpu.to('cpu').state_dict(), "mnist_cnn.pt")

    if(args.distributed == True):
        cleanup_dist()


if __name__ == '__main__':
    args = parse_args()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        if os.getenv('HCL_CONFIG_PATH') is None:
            print("HCL_CONFIG_PATH is not set")
            exit(0)
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.distributed = True
    else:
        print('Not using distributed mode')
        args.rank=0
        args.distributed = False

    if args.rank == 0:
        #If in distributed mode download once. Assuming setup_dist will be a blocking call
        train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ]))

    if args.distributed == True:
        setup_dist(args.rank, args.world_size,args.backend)

    #If distributed mode data should be downloaded before the control reaches here.
    if args.rank != 0:
        train_dataset = datasets.MNIST('../data', train=True, download=False, transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ]))

    args.train_dataset = train_dataset
    main(args)
