# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This example is largely adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py.

Before you can run this example, you will need to download the ImageNet dataset manually from the
`official website <http://image-net.org/download>`_ and place it into a folder `path/to/imagenet`.

Train on ImageNet with default parameters:

.. code-block: bash

    python imagenet.py fit --model.data_path /path/to/imagenet

or show all options you can change:

.. code-block: bash

    python imagenet.py --help
    python imagenet.py fit --help
"""
import os
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed
import torchvision.models as models
from torch.utils.data import Dataset
from torchmetrics import MaxMetric
from typing import Any, Optional
from habana_frameworks.torch.hpex.optimizers import FusedSGD
import habana_frameworks.torch.core as htcore
from habana_frameworks.torch.dynamo.compile_backend.experimental import enable_compiled_autograd
import time
import argparse
import numpy as np

from lightning_utilities import module_available

if module_available("lightning"):
    from lightning.pytorch import LightningModule, Trainer, seed_everything
    from lightning.pytorch.callbacks import TQDMProgressBar, Callback, LearningRateMonitor, EarlyStopping
    from lightning.pytorch.plugins import MixedPrecisionPlugin
    from lightning.pytorch.utilities.rank_zero import rank_zero_info
elif module_available("pytorch_lightning"):
    from pytorch_lightning import LightningModule, Trainer, seed_everything
    from pytorch_lightning.callbacks import TQDMProgressBar, Callback, LearningRateMonitor, EarlyStopping
    from pytorch_lightning.plugins import MixedPrecisionPlugin
    from pytorch_lightning.utilities.rank_zero import rank_zero_info

from lightning_habana.pytorch.accelerator import HPUAccelerator
from lightning_habana.pytorch.strategies import HPUParallelStrategy, SingleHPUStrategy

from warnings import filterwarnings
from datamodules import get_data_module
filterwarnings("ignore")

# Modified version of accuracy. target and pred tensors are pytorch Long
# which is not supported by habana kernels yet. So fall back to CPU for
# ops involving these(and remain on CPU since this is the last oprton of
# iteration and we need the accuracy values to be printed out on host)
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    output = output.to('cpu')
    target = target.to('cpu')
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k.to('hpu') * (100.0 / batch_size))
        return res

class LitProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.set_description("Training ")
        return bar

class LoggingCallback(Callback):
    def __init__(self, global_batch_size, warmup=50, mode="train", perform_epoch=0):
        self.warmup_steps = warmup
        self.global_batch_size = global_batch_size
        self.steps = 0
        self.mode = mode
        self.epoch_start = None
        self.epoch_end = None
        self.perform_epoch = perform_epoch
        self.train_epoch_times = []
        self.best = 10000000

    def on_train_epoch_start(self, trainer, pl_module:Optional[Any]=None):
        rank_zero_info("Train Epoch start")
        self.steps = 0

    def on_train_batch_start(self, trainer, pl_module:Optional[Any]=None, batch:Optional[int]=0, batch_idx:Optional[int]=0):
        self.steps  += 1
        if self.steps == self.warmup_steps:
            self.epoch_start=time.time()

    def on_train_epoch_end(self, trainer, pl_module:Optional[Any]=None):
        self.epoch_end = time.time()
        diff = self.epoch_end - self.epoch_start
        self.train_epoch_times.append(diff)
        throughput = (self.steps - self.warmup_steps) * self.global_batch_size / (diff)
        rank_zero_info(f" throughput_{self.mode} = {throughput}")

    def on_train_end(self, trainer, pl_module:Optional[Any]=None):
        for i in self.train_epoch_times:
            if(i < self.best):
                self.best = i
        rank_zero_info(f" Best Epoch Time: %.2f" % self.best)
        avg = np.mean(self.train_epoch_times[1:])
        rank_zero_info(f" Avg Epoch Time: %.2f" % avg)

class CustomLR:
    def __init__(self,kwargs):
        self.lr_vec = []
        lr_values=kwargs.get('lr_values',[])
        lr_values.insert(0, kwargs.get('init_lr',0.1))
        lr_milestones=kwargs.get('lr_milestones',[])
        lr_milestones.insert(0,0)
        lr_milestones.append(kwargs.get('total_epochs',10) + 1)
        for n in range(len(lr_milestones) - 1):
           self.lr_vec +=[lr_values[n]] * (lr_milestones[n+1] - lr_milestones[n])
        rank_zero_info(f"The predetermined LR scheduler is: {self.lr_vec} for all epochs." )

    def __call__(self, current_epoch):
        return self.lr_vec[current_epoch]


class ImageNetLightningModel(LightningModule):
    """
    >>> ImageNetLightningModel(data_path='missing')  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ImageNetLightningModel(
      (model): ResNet(...)
    )
    """

    def __init__(
        self,
        data_path: str,
        arch: str = "resnet50",
        pretrained: bool = False,
        init_lr: float=0.1, #0.0069183,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        workers: int = 8,
        hpus: int = 8,
        print_freq: int = 1,
        benchmark: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.arch = arch
        self.pretrained = pretrained
        self.init_lr = init_lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.data_path = data_path
        self.batch_size = batch_size
        self.workers = workers
        self.hpus=hpus
        self.print_freq = print_freq
        self.model = models.__dict__[self.arch](pretrained=self.pretrained)
        self.train_dataset: Optional[Dataset] = None
        self.eval_dataset: Optional[Dataset] = None
        self.eval_best_acc = MaxMetric()
        if (hasattr(args, 'hpu_torch_compile') and args.hpu_torch_compile) or (hasattr(args, 'use_torch_compile') and args.use_torch_compile):
            self.model = torch.compile(self.model, backend="aot_hpu_training_backend")

        self.eval_accuracys = []
        #self.train_accuracys = []

        self.steps = 0
        self.benchmark = benchmark
        self.customlr = CustomLR(kwargs) if kwargs else None
        if self.customlr is None:
            rank_zero_info("No predetermined LR scheduler")

    def forward(self, x):
        return self.model(x)

    def on_after_backward(self):
        htcore.mark_step()

    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self.model(images)
        loss_train = F.cross_entropy(output, target)
        self.log("train_loss", loss_train)
        #update metrics
        if self.benchmark == False:
            acc1, acc5 = accuracy(output, target, topk=(1,5))
            self.log("train_acc1", acc1, prog_bar=True)
            self.log("train_acc5", acc5, prog_bar=True)
            #self.train_accuracys.append(acc1)
        return loss_train

    def eval_step(self, batch, batch_idx, prefix: str):
        images, targets = batch
        preds = self.model(images)
        loss_val = F.cross_entropy(preds, targets)
        self.log(f"{prefix}_loss", loss_val)
        # update metrics
        if self.benchmark == False:
            acc1, acc5 = accuracy(preds, targets, topk=(1,5))
            self.log(f"{prefix}_acc1", acc1, prog_bar=True, sync_dist=True if self.hpus > 1 else False)
            self.log(f"{prefix}_acc5", acc5, prog_bar=True)
            self.eval_accuracys.append(acc1)
        return  loss_val

    def on_validation_epoch_end(self) -> None:
        if self.benchmark == True:
            return None
        val_accuracy = torch.mean(torch.stack(self.eval_accuracys))
        self.log(f"epoch_eval_accuracy", val_accuracy, on_epoch=True, prog_bar=True)
        self.eval_best_acc.update(val_accuracy)
        self.log("top_eval_accuracy", self.eval_best_acc.compute(), on_epoch=True, prog_bar=True)
        self.eval_accuracys.clear()
        return super().on_validation_epoch_end()
    '''
    We currently use top_eval_accuracy to measure and report the accuracy,
    please uncomment the following function if we need to report the training accuracy

    def on_train_epoch_end(self) -> None:
        if self.benchmark == True:
            return None
        train_accuracy = torch.mean(torch.stack(self.train_accuracys))
        self.log(f"epoch train_accuracy", train_accuracy, on_epoch=True, prog_bar=True)
        self.train_accuracys.clear()
        return super().training_epoch_end()
    '''

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        # Here for the predetermined LR, we set lr=1 to allow the values directly passed to optimizer. Otherwise, all
        # predetermined LR values will be shrinked by lr.
        optimizer = FusedSGD(self.parameters(), lr=1 if self.customlr else self.init_lr,
                              momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=self.customlr) if self.customlr\
                    else lr_scheduler.StepLR(optimizer, step_size = args.lr_step_size, gamma=args.lr_gamma)
        return [optimizer], [scheduler]

def train_model(args):
    model = ImageNetLightningModel(
            data_path=args.data_path,
            batch_size=args.batch_size,
            workers=args.workers,
            hpus=args.hpus,
            benchmark=args.benchmark,
            init_lr=args.lr,
            **{ 'lr': args.lr,
                'lr_values': args.custom_lr_values,
                'lr_milestones': args.custom_lr_milestones,
                'total_epochs': args.epochs,
            } if args.custom_lr_values is not None else {}
        )

    class CustomLearningRateMonitor(LearningRateMonitor):
        def __init__(self,logging_interval):
            super().__init__(logging_interval)

        def on_train_epoch_start(self, trainer: Trainer, *args: Any, **kwargs: Any) -> None:
           if(len(self.lrs['lr-FusedSGD']) > 0):
             rank_zero_info(f"The learning rate on epoch:{trainer.current_epoch} is {self.lrs['lr-FusedSGD'][len(self.lrs['lr-FusedSGD']) - 1]}")
           return super().on_train_epoch_start(trainer, *args, **kwargs)

    start_time = time.time()
    parallel_hpus = [torch.device('hpu')] * args.hpus
    strategy=HPUParallelStrategy(parallel_devices=parallel_hpus,
                                    broadcast_buffers=False,
                                    gradient_as_bucket_view=True,
                                    bucket_cap_mb=256,
                                    static_graph=True) if args.hpus > 1 else SingleHPUStrategy() if args.hpus == 1 else None
    if isinstance(strategy, HPUParallelStrategy):
        # Improves distributed performance by limiting the number of all_reduce calls to 1
        # Tuning first bucket is supported through HPUParallelStrategy only
        torch.distributed._DEFAULT_FIRST_BUCKET_BYTES = 230*1024*1024

    callbacks = []
    if args.benchmark == False:
        callbacks.append(CustomLearningRateMonitor(logging_interval='step'))
        callbacks.append(LitProgressBar(refresh_rate = args.print_freq, process_position=0))
        callbacks.append(EarlyStopping(monitor="top_eval_accuracy", mode="max", min_delta=0.0, patience=args.patience))
    else:
        callbacks.append(LoggingCallback(global_batch_size=args.batch_size * args.hpus, warmup=args.warmup))

    if args.is_autocast:
        plugins = [MixedPrecisionPlugin(precision='bf16-mixed', device="hpu")]
    else:
        plugins = []

    trainer = Trainer(
                      max_epochs=args.epochs,
                      enable_progress_bar=False if args.benchmark else True,
                      enable_model_summary=False if args.benchmark else True,
                      enable_checkpointing=False if args.benchmark else True,
                      devices = args.hpus,
                      accelerator=HPUAccelerator(),
                      callbacks = callbacks,
                      check_val_every_n_epoch=args.check_val_every_n_epoch,
                      benchmark=args.benchmark,
                      limit_train_batches = None if args.max_train_batches == 0 else args.max_train_batches,
                      strategy=strategy,
                      plugins=plugins,
                      logger=False if args.benchmark else True,
                      use_distributed_sampler=False,
                      deterministic=False,
                      num_sanity_val_steps=0,
                      limit_val_batches=0.0 if args.benchmark else None,
                )
    data_module=get_data_module(args.data_path, args.dl_type, args.workers, args.batch_size, args.hpus)
    trainer.fit(model, datamodule=data_module)
    end_time = time.time()
    return end_time - start_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')
    parser.add_argument('--data_path', default='/data/pytorch/data/imagenet/ILSVRC2012/', help='dataset')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--lr_step_size',  default=30, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr_gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--hpus', default=1, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--warmup', default=50, type=int)
    parser.add_argument('--check_val_every_n_epoch',default=1, type=int)
    parser.add_argument('--print_freq', default=1, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--dl_type', default='HABANA', type=lambda x: x.upper(),
                        choices = ["MP", "HABANA"], help='select multiprocessing or habana accelerated')
    parser.add_argument('--max_train_batches', default=0, type=int)
    parser.add_argument('--autocast', dest='is_autocast', action='store_true', help='enable autocast mode on Gaudi')
    parser.add_argument('--benchmark', action='store_true', help='benchmark performance measurement')
    # hpu_torch_compile flag has been replaced with use_torch_compile and shall be deprecated
    parser.add_argument('--hpu_torch_compile', action='store_true', help='enable torch compile for hpu. This flag has been replaced with use_torch_compile and shall be deprecated')
    parser.add_argument('--use_torch_compile', action='store_true', help='enable torch compile for hpu')
    parser.add_argument('--compiled_autograd', action='store_true', help='[EXPERIMENTAL] Enable compiled_autograd for hpu')
    parser.add_argument('--custom_lr_values', default=None, metavar='N', type=float, nargs='+', help='custom lr values list')
    parser.add_argument('--custom_lr_milestones', default=None, metavar='N', type=int, nargs='+',
                        help='custom lr milestones list')
    parser.add_argument('--patience', default=30, type=int, help='Patience for early stopping')

    args = parser.parse_args()
    print(args)
    if  args.workers > 0:
        if args.dl_type == "MP":
            torch.multiprocessing.set_start_method('spawn')
        # patch torch cuda functions that are being unconditionally invoked
        # in the multiprocessing data loader
        torch.cuda.current_device = lambda: None
        torch.cuda.set_device = lambda x: None

    seed_everything(1234)

    if args.compiled_autograd:
        if not args.hpu_torch_compile:
            raise ValueError("--compiled_autograd requires --hpu_torch_compile")
        enable_compiled_autograd()

    time_interval=train_model(args)

    print("Total Training time %.2f" % time_interval)
