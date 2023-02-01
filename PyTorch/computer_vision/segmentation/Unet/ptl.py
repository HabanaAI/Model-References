###############################################################################
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
import time
import torch
import itertools
import subprocess
import numpy as np
from statistics import mean
from typing import Optional, Any
from collections import namedtuple
from contextlib import contextmanager
from multiprocessing import Process, Event, Queue

from utils.utils import set_seed
from models.nn_unet import NNUnet
import habana_frameworks.torch as ht
from pytorch_lightning import Callback
import pytorch_lightning.utilities.seed
from utils.logger import LoggingCallback
from utils.utils import mark_step, layout_2d
import habana_frameworks.torch.core as htcore
from data_loading.data_module import DataModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.plugins import HPUPrecisionPlugin
from pytorch_lightning.strategies import HPUParallelStrategy
from data_loading.dali_loader import fetch_dali_loader, BenchmarkPipeline, LightningWrapper
from utils.utils import  is_main_process, log, make_empty_dir, set_cuda_devices, verify_ckpt_path, \
    get_config_file, get_path, get_split, get_test_fnames, load_data


def get_accuracy(dice):
    metrics = {}
    metrics.update({"Mean dice": round(dice.mean().item(), 2)})
    metrics.update({f"L{j+1}": round(m.item(), 2) for j, m in enumerate(dice)})
    return metrics


class LoggingCallback(Callback):
    def __init__(self, log_dir, global_batch_size, mode, warmup, dim, profile, perform_epoch=1):
        self.warmup_steps = warmup
        self.global_batch_size = global_batch_size
        self.step = 0
        self.dim = dim
        self.mode = mode
        self.profile = profile
        self.timestamps = []
        self.perform_epoch = perform_epoch


class HLDataModule(DataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self):
        imgs = load_data(self.data_path, "*_x.npy")
        lbls = load_data(self.data_path, "*_y.npy")

        self.test_imgs, self.kwargs["meta"] = get_test_fnames(self.args, self.data_path, self.kwargs["meta"])
        if self.args.exec_mode != "" or self.args.benchmark:
            if self.args.benchmark:
                train_idx, val_idx = list(self.kfold.split(imgs))[self.args.fold]
            else:
                val_idx, train_idx = list(self.kfold.split(imgs))[self.args.fold]
            self.train_imgs = get_split(imgs, train_idx)
            self.train_lbls = get_split(lbls, train_idx)
            self.val_imgs = get_split(imgs, val_idx)
            self.val_lbls = get_split(lbls, val_idx)
            if is_main_process():
                ntrain, nval = len(self.train_imgs), len(self.val_imgs)
                print(f"Number of examples: Train {ntrain} - Val {nval}")

    def test_dataloader(self):
        if self.kwargs["benchmark"]:
            return fetch_dali_loader(self.train_imgs, self.train_lbls, self.args.val_batch_size, "test", **self.kwargs)
        return fetch_dali_loader(self.test_imgs, None, 1, "test", **self.kwargs)


def fetch_dali_loader(imgs, lbls, batch_size, mode, **kwargs):
    assert len(imgs) > 0, "Got empty list of images"
    if lbls is not None:
        assert len(imgs) == len(lbls), f"Got {len(imgs)} images but {len(lbls)} lables"
    if kwargs["benchmark"]:  # Just to make sure the number of examples is large enough for benchmark run.
        imgs = list(itertools.chain(*(100 * [imgs])))
        lbls = list(itertools.chain(*(100 * [lbls])))
    pipe_kwargs = {
        "imgs": imgs,
        "lbls": lbls,
        "dim": kwargs["dim"],
        "num_device": kwargs["num_device"],
        "seed": kwargs["seed"],
        "meta": kwargs["meta"],
        "patch_size": kwargs["patch_size"],
        "oversampling": kwargs["oversampling"],
    }
    if kwargs["benchmark"]:
        pipeline = BenchmarkPipeline
        output_map = ["image", "label"]
        dynamic_shape = False
        if kwargs["dim"] == 2:
            pipe_kwargs.update({"batch_size_2d": batch_size})
            batch_size = 1
    if kwargs["device"] == "gpu":
        device_id = int(os.getenv("LOCAL_RANK", "0"))
    else:
        device_id = None
    pipe = pipeline(batch_size, kwargs["num_workers"], device_id, **pipe_kwargs)
    return LightningWrapper(
        pipe,
        auto_reset=True,
        reader_name="ReaderX",
        output_map=output_map,
        dynamic_shape=dynamic_shape,
    )


class LoggingCallback(LoggingCallback):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def on_test_batch_start(self, trainer, pl_module:Optional[Any]=None, batch:Optional[int]=0, batch_idx:Optional[int]=0, i=0):
        if trainer.current_epoch == self.perform_epoch:
            self.do_step()


class NNUnetHPUL(NNUnet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_duration = []
        self.output = None

    def test_step(self, batch, batch_idx):
        if self.args.exec_mode == "evaluate":
            return self.validation_step(batch, batch_idx)
        img = batch["image"]
        with torch.autocast(device_type="hpu", dtype=self.args.precision):
            img = img.to(torch.device("hpu"), non_blocking=True)
            img = layout_2d(img, None)
            start = time.time()
            self.output = self.model(img)
            mark_step(self.args.run_lazy_mode)
            elapsed_time = time.time() - start
            self.time_duration.append(elapsed_time)
            if batch_idx == self.args.test_batches -1:
                tuple(_.detach().to("cpu") for _ in self.output)

    def get_total_time(self):
        return sum(self.time_duration)


class NNUnetHPUG(NNUnet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_duration = []
        self.data_set = []
        self.g = ht.hpu.HPUGraph()
        self.s = ht.hpu.Stream()
        self.input = None
        self.output = None
        self.first_run = True

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.warm_up(batch)
        img = batch["image"]
        with torch.autocast(device_type="hpu", dtype=self.args.precision):
            img = img.to(torch.device("hpu"), non_blocking=True)
            img = layout_2d(img, None)
            start = time.time()
            self.input.copy_(img)
            mark_step(self.args.run_lazy_mode)
            self.g.replay()
            elapsed_time = time.time() - start
            self.time_duration.append(elapsed_time)

            if batch_idx == self.args.test_batches -1:
                tuple(_.detach().to("cpu") for _ in self.output)

    def warm_up(self, batch):
        print('Runing HPU Graph record')
        img = batch["image"]
        with torch.autocast(device_type="hpu", dtype=self.args.precision):
            img = img.to(torch.device("hpu"), non_blocking=True)
            img = layout_2d(img, None)
            self.input = img
            with ht.hpu.stream(self.s):
                self.g.capture_begin()
                self.output = self.model(self.input)
                self.g.capture_end()

    def get_total_time(self):
        return sum(self.time_duration)


class NNUnetJIT(NNUnet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input = None
        self.output = None
        self.brand_new = True
        self.jit_model = None
        self.data_set = []
        self.time_duration = []

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.brand_new_run(batch["image"])
        with torch.autocast(device_type="hpu", dtype=self.args.precision):
            img = batch["image"]
            img = img.to(torch.device("hpu"), non_blocking=False)
            img = layout_2d(img, None)
            start = time.time()
            self.jit_model(img)
            mark_step(self.args.run_lazy_mode)
            elapsed_time = time.time() - start
            self.time_duration.append(elapsed_time)

    def brand_new_run(self, img):
        with torch.autocast(device_type="hpu", dtype=self.args.precision):
            self.input = img.to(torch.device("hpu"), non_blocking=False)
            self.input = layout_2d(self.input, None)
            self.jit_model = torch.jit.trace(self.model, (self.input,))
            mark_step(self.args.run_lazy_mode)

    def get_total_time(self):
        return sum(self.time_duration)


def set_env_params(args):
    os.environ['PT_HPU_ENABLE_SYNC_OUTPUT_HOST'] = 'false'
    if args.hpus and not args.run_lazy_mode:
        os.environ["PT_HPU_LAZY_MODE"] = "2"
    if args.hpus > 1:
        torch.distributed._DEFAULT_FIRST_BUCKET_BYTES = 130*1024*1024 #130MB


def load_hpu_library(args):
    import habana_frameworks.torch.core
    import habana_frameworks.torch.distributed.hccl
    import habana_frameworks.torch.hpu


def ptlrun(mode, batch_size, precision, ckpt_path, data_path, res_path, acc=False):
    model_precision = {"bfloat16": torch.bfloat16,
                       "float32": torch.float32}[precision]
    total_samples = len(load_data(data_path, "*_x.npy"))
    os.makedirs(name=res_path, exist_ok=True)

    params = {'affinity':'disabled', 'amp':False, 'attention':False, 'augment':True, 'batch_size':2, 'blend':'gaussian',
              'bucket_cap_mb':125, 'ckpt_path':ckpt_path, 'data':data_path, 'data2d_dim':3, 'deep_supervision':False,
              'dim':2, 'drop_block':False,  'factor':0.3, 'focal':False, 'fold':0, 'framework':'pytorch-lightning',
              'gpus':0, 'gradient_clip':False, 'gradient_clip_norm':12, 'gradient_clip_val':0, 'hmp_bf16':'', 'hmp_fp32':'',
              'hmp_opt_level':'O1', 'hmp_verbose':False, 'hpus':1, 'is_hmp':False, 'learning_rate':0.001, 'logname':'res_log',
              'lr_patience':70, 'max_epochs':2, 'min_epochs':1, 'momentum':0.99, 'negative_slope':0.01, 'nfolds':5, 'norm':'instance',
              'num_workers':8, 'nvol':1, 'optimizer':'adamw', 'overlap':0.5, 'oversampling':0.33, 'patience':100, 'profile':False,
              'residual':False, 'results':res_path, 'resume_training':False, 'run_lazy_mode':True, 'save_ckpt':False,
              'save_preds':False, 'scheduler':'none', 'seed':1, 'set_aug_seed':False, 'skip_first_n_eval':0, 'steps':None,
              'sync_batchnorm':False, 'task':'01', 'test_batches': total_samples, 'train_batches':1, 'tta':False, 'val_batch_size':batch_size,
              'warmup':5, 'weight_decay':0.0001, 'environ':"PTL", "precision": model_precision}

    params.update({"benchmark": True, "exec_mode":"predict"})
    UnetParams = namedtuple('UnetParams', params)
    args = UnetParams(**params)

    if is_main_process():
        print(args)
    set_env_params(args)
    # Load the hpu libs if device is hpu
    if args.hpus:
        load_hpu_library(args)
    deterministic = False
    if args.seed is not None:
        pytorch_lightning.utilities.seed.seed_everything(seed=args.seed)
        torch.backends.cudnn.deterministic = True
        deterministic = True
    seed0 = args.seed
    set_seed(seed0)
    data_module_seed = np.random.randint(0, 1e6)
    model_seed = np.random.randint(0, 1e6)
    seed_everything(seed0)
    set_seed(data_module_seed)
    data_module = HLDataModule(args)
    data_module.prepare_data()
    data_module.setup()
    set_seed(model_seed)

    model_modes = {"lazy": NNUnetHPUL, "trace":NNUnetJIT, "graphs":NNUnetHPUG}
    model = model_modes[mode](args)
    batch_size = args.batch_size if args.exec_mode == "train" else args.val_batch_size
    log_dir = os.path.join(args.results, args.logname if args.logname is not None else "perf.json")
    num_instance = args.gpus if args.gpus else args.hpus
    if not num_instance:
        num_instance = 1
    callbacks = [
        LoggingCallback(
            log_dir=log_dir,
            global_batch_size=batch_size * num_instance,
            mode=args.exec_mode,
            warmup=args.warmup,
            dim=args.dim,
            profile=args.profile
        ),
    ]
    parallel_hpus = [torch.device("hpu")] * args.hpus
    trainer = Trainer(
        logger=False,
        gpus=args.gpus,
        precision=16 if args.amp else 32,
        devices=args.hpus if args.hpus else None,
        accelerator="hpu" if args.hpus else None,
        benchmark=True,
        deterministic=deterministic,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        sync_batchnorm=args.sync_batchnorm,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        default_root_dir=args.results,
        enable_checkpointing=args.save_ckpt,
        strategy=HPUParallelStrategy(parallel_devices=parallel_hpus, bucket_cap_mb=args.bucket_cap_mb,gradient_as_bucket_view=True,static_graph=True) if args.hpus > 1 else None,
        limit_train_batches=1.0 if args.train_batches == 0 else args.train_batches,
        limit_val_batches=1.0 if args.test_batches == 0 else args.test_batches,
        limit_test_batches=1.0 if args.test_batches == 0 else args.test_batches,
        plugins=[HPUPrecisionPlugin(precision=16 if args.is_hmp else 32, opt_level="O1",verbose=False, bf16_file_path=args.hmp_bf16,fp32_file_path=args.hmp_fp32)] if args.hpus else None
    )

    total_sample = args.test_batches * args.val_batch_size
    start = time.time()
    trainer.test(model, dataloaders=data_module.test_dataloader(),
        ckpt_path=args.ckpt_path)
    elapsed = time.time() - start
    total_time = model.get_total_time()

    # total_time metric can only measure host time in some scenarios
    performance = total_sample/elapsed
    latency =  batch_size/performance

    if acc:
        params.update({"benchmark": False, "exec_mode":"evaluate"})
        UnetParams = namedtuple('UnetParams', params)
        args = UnetParams(**params)
        set_env_params(args)
        data_module = HLDataModule(args)
        data_module.prepare_data()
        data_module.setup()

        model = NNUnetHPUL(args)
        trainer.test(model, dataloaders=data_module.val_dataloader(),ckpt_path=args.ckpt_path)

        accuracy = get_accuracy(model.eval_dice)
        accuracy = accuracy["L1"]
    else:
        accuracy = 0
    metrics = {
        'avg_latency (ms)': latency * 1000,
        'performance (img/s)': performance,
        'accuracy' : accuracy
    }
    return metrics
