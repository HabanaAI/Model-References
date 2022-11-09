
import enum
import torch
from typing import Any, Callable, cast, Dict, Iterable, List, Optional, Type, Union
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from data_loading.data_module import DataModule
from types import SimpleNamespace
from habana_frameworks.torch.hpex import hmp
from utils.utils import get_device, mark_step, is_main_process
from models.nn_unet import NNUnet
import time
from tqdm import tqdm
import operator
import dllogger as logger
from dllogger import JSONStreamBackend, StdOutBackend, Verbosity

from .early_stopping_unet import EarlyStopping
import numpy as np
from  utils.logger import LoggingCallback
from models.unet import UNet
from dataclasses import asdict, dataclass, field
import torch.distributed as dist

def main_process():
    return dist.get_rank() == 0


def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    if size == 1:
        return
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size


class NativeLogging(object):
   def __init__(self, log_dir, global_batch_size, mode, warmup, dim, profile, perform_epoch=1):
      logger.init(backends=[JSONStreamBackend(Verbosity.VERBOSE, log_dir), StdOutBackend(Verbosity.VERBOSE)])
      self.global_batch_size = global_batch_size
      self.mode = mode
      self.dim = dim
      self.profile = profile
      self.perform_epoch = perform_epoch
      self.warmup = warmup
      self.start_times = []
      self.end_times =[]

   def batch_start(self):
       self.start_times.append(time.time())

   def batch_end(self):
       self.end_times.append(time.time())

   def process_performance_stats(self, deltas):
        def _round3(val):
            return round(val, 3)

        throughput_imgps = _round3(self.global_batch_size / np.mean(deltas))
        timestamps_ms = 1000 * deltas
        stats = {
            f"throughput_{self.mode}": throughput_imgps,
            f"latency_{self.mode}_mean": _round3(timestamps_ms.mean()),
        }
        for level in [90, 95, 99]:
            stats.update({f"latency_{self.mode}_{level}": _round3(np.percentile(timestamps_ms, level))})

        return stats

   def calculate_stats(self):
       diffs = list(map(operator.sub, self.end_times, self.start_times))
       deltas = np.array(diffs[self.warmup:])
       stats = self.process_performance_stats(deltas)
       self.start_times.clear()
       self.end_times.clear()
       return stats

   def epoch_end(self):
       stats = self.calculate_stats()
       logger.log(step=(), data=stats)
       logger.flush()
       return stats

   def info_log(self, stats):
       logger.log(step=(), data=stats)
       logger.flush()





class Trainer:
    def __init__(
        self,
        hparams: Optional['SimpleNamespace'],
    ) -> None:
        self.args = hparams.args
        self.current_epoch = 0
        self.val_check_interval: Union[int,float] = 1
        self.check_val_every_n_epoch: int = 1
        self.should_stop = False
        self.world_size = hparams.hpus
        self.global_rank = 0
        self.earlystopping = None
        self.val_dataloaders=list()
        self.train_dataloaders=list()
        self.optimizer = None
        self.scheduler = None
        pass

    def scheduler_step(self,val_loss):
        if self.scheduler is None:
            return
        scheduler_name = type(self.scheduler).__name__
        if scheduler_name == 'CosineAnnealingLR':
            self.scheduler.step()
        if scheduler_name == 'ReduceLROnPlateau':
            self.scheduler.step(val_loss)
        if scheduler_name == 'MultiStepLR':
            self.scheduler.step()
        pass

    def _implement_train(self, model:"NNUnet", hparams):

       model.train()
       train_losses = []
       self.optimizer = hparams.opt_dict['optimizer']
       torch.set_grad_enabled(True)

       with tqdm(self.train_dataloaders[0],  unit="it", leave=True, position=0) as tepoch:
            for i, batch in enumerate(tepoch):
                tepoch.set_description(f"Train Epoch {self.current_epoch}")
                self.optimizer.zero_grad(set_to_none=True)
                loss = model.training_step(batch, i)
                loss.backward()
                #gradient clipping improves training stability
                if model.args.gradient_clip == True:
                   torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), model.args.gradient_clip_norm)
                if i % hparams.args.progress_bar_refresh_rate == 0:
                   average_gradients(model)
                   tepoch.set_postfix(it=self.current_epoch)
                mark_step(model.args.run_lazy_mode)
                self.optimizer.step()
                mark_step(model.args.run_lazy_mode)
                train_losses.append(loss)
       output_loss = torch.mean(torch.stack(train_losses), dim=0)
       if main_process():
           print(f"Rank: {dist.get_rank()}, Batch: {i}, Train Loss is: {output_loss:.3f}", end="\r")
           print(flush=True)
       return output_loss

    def _implement_validate(self, model:"NNUnet", hparams:Optional[SimpleNamespace]=None):
        val_losses = []
        model.eval()
        with torch.set_grad_enabled(False):
            with tqdm(self.val_dataloaders[0], unit="it", leave=True, position=0) as tepoch:
                for i, batch in enumerate(tepoch):
                    tepoch.set_description(f"Validation Epoch {self.current_epoch}")
                    loss = model.validation_step(batch, i)
                    mark_step(model.args.run_lazy_mode)
                    val_losses.append(loss)
            output_loss = torch.mean(torch.stack([n['val_loss'] for n in val_losses]), dim = 0)
            tepoch.set_postfix(loss=output_loss.item())
        kwargs = model.validation_epoch_end(val_losses)
        if main_process():
            output_loss_print = output_loss.item()
            val_loss= kwargs['val_loss'].item()
            dice_sum = kwargs['dice_sum'].item()
            print(f"Rank: {dist.get_rank()}, Batch:{i}, Val Loss is: {output_loss_print:.3f}", end="\r")
            print(f" val_loss {val_loss:.2f}")
            print(f" dice_sum {dice_sum:.2f}")
            print(flush=True)
        return kwargs

    def fit(self, model:"NNUnet", hparams) -> None:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model = model.module
            model.trainer = self
            self.earlystopping =  EarlyStopping(monitor="dice_sum", patience=model.args.patience, verbose=True, mode="max")
            self.earlystopping.setup(self)
            self.scheduler = hparams.opt_dict['lr_scheduler'] if 'lr_scheduler' in  hparams.opt_dict else None

            self.train_dataloaders.append(hparams.data_module.train_dataloader())
            self.val_dataloaders.append(hparams.data_module.val_dataloader())
            for self.current_epoch in range(model.args.max_epochs):
                time_1 = time.time()
                train_output_loss = self._implement_train(model, hparams)
                val_output = self._implement_validate(model, hparams)
                self.scheduler_step(val_output['val_loss'])

                if main_process():
                    time_2 = time.time()
                    time_interval = time_2 - time_1
                    print(f"End epoch: {self.current_epoch} with time interval: {time_interval:.3f} secs")

                if self.current_epoch >= model.args.min_epochs:
                     self.earlystopping.on_validation_end(self, val_output['dice_sum'])
                     if self.earlystopping.stopped_epoch == self.current_epoch:
                        print(f"Training stopped with epoch: {self.current_epoch}")
                        break

    def _implement_benchmark_train(self, model:"NNUnet", hparams):
       train_loader = hparams.data_module.train_dataloader()
       log = hparams.log
       model.train()
       train_losses = []
       self.optimizer = hparams.opt_dict['optimizer']
       torch.set_grad_enabled(True)
       with tqdm(train_loader, unit="it", leave=True, position=0) as tepoch:
         for i, batch in enumerate(tepoch):
               log.batch_start()
               tepoch.set_description(f"Train Epoch {self.current_epoch}")
               self.optimizer.zero_grad(set_to_none=True)
               loss = model.training_step(batch, i)
               loss.backward()
               #gradient clipping improves training stability
               if model.args.gradient_clip == True:
                   torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), model.args.gradient_clip_norm)
               if i % hparams.args.progress_bar_refresh_rate == 0:
                  average_gradients(model)
                  tepoch.set_postfix(it=self.current_epoch)
               mark_step(model.args.run_lazy_mode)
               self.optimizer.step()
               mark_step(model.args.run_lazy_mode)

               train_losses.append(loss)
               log.batch_end()

       output_loss = torch.mean(torch.stack(train_losses), dim=0)
       return output_loss


    def benchmark_train(self, model:"NNUnet", hparams):
        train_loader = hparams.data_module.train_dataloader()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        log = hparams.log
        model.train()

        # For Native PyTorch, we found that the throughput stablized after 2 epochs
        # instead of 1, so we pick up the value from the last epoch
        model.args.max_epochs = model.args.max_epochs + 1

        for self.current_epoch in range(model.args.max_epochs):
            self._implement_benchmark_train(model, hparams)
            stats = log.calculate_stats()
            #To monitor the perf for the previous epoch
            print(stats)
        throughput = stats['throughput_train']
        if dist.get_world_size() > 1:
             throughput=torch.tensor(throughput,
                      dtype=torch.float64).to(torch.device('hpu'))
             dist.all_reduce(throughput, op=dist.ReduceOp.SUM)
             throughput = throughput.item()

        if main_process():
           stats['throughput_train'] = round(throughput,3)
           log.info_log(stats)
           print(stats)
           print(flush=True)










