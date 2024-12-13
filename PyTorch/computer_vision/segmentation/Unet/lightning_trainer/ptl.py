# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
###############################################################################
# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
###############################################################################


import os
import torch
import numpy as np
import time
import datetime

from lightning_utilities import module_available

if module_available('lightning'):
    from lightning.pytorch import Trainer, seed_everything, Callback
    from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
elif module_available('pytorch_lightning'):
    from pytorch_lightning import Trainer, seed_everything, Callback
    from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

from lightning_habana.pytorch import HPUAccelerator
from lightning_habana.pytorch.strategies import HPUDDPStrategy, SingleHPUStrategy

from models.nn_unet import NNUnet
from utils.logger import LoggingCallback
from data_loading.data_module import DataModule
from utils.utils import  is_main_process, log, make_empty_dir, set_cuda_devices, verify_ckpt_path
from utils.utils import set_seed, get_device
from utils.early_stopping_unet import EarlyStopping

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

_KINETO_AVAILABLE = torch.profiler.kineto_available()

class TensorBoardCallback(Callback):
    def __init__(self, tbl, progress_bar_refresh_rate):
        self.tbl = tbl
        self.progress_bar_refresh_rate = progress_bar_refresh_rate
        super().__init__()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.progress_bar_refresh_rate == 0:
            current_time = time.time()
            steps_per_second = self.progress_bar_refresh_rate / (current_time - self.last_time)
            self.last_time = current_time
            self.tbl.log_metrics(metrics = {"steps_per_seconds": steps_per_second}, step = trainer.global_step)

            if "loss" in trainer.callback_metrics:
                loss = trainer.callback_metrics["loss"].item()
                self.tbl.log_metrics(metrics = {"loss": loss}, step = trainer.global_step)

    def on_train_epoch_start(self, trainer, pl_module):
        self.last_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        if "mean_dice" in trainer.callback_metrics:
            mean_dice = trainer.callback_metrics["mean_dice"].item()
            self.tbl.log_metrics(metrics = {"mean_dice": mean_dice}, step = trainer.global_step)

def set_env_params(args):
    if args.hpus:
        if args.run_lazy_mode:
            assert os.getenv('PT_HPU_LAZY_MODE') == '1' or os.getenv('PT_HPU_LAZY_MODE') == None, f"run-lazy-mode == True, but PT_HPU_LAZY_MODE={os.getenv('PT_HPU_LAZY_MODE')}. For run lazy mode, set PT_HPU_LAZY_MODE to 1"
    if args.hpus > 1:
        torch.distributed._DEFAULT_FIRST_BUCKET_BYTES = 130*1024*1024 #130MB
    if args.hpus:
        # Enable hpu dynamic shape
        import habana_frameworks.torch.hpu as hthpu
        if (os.getenv("HPU_DISABLE_DYNAMIC_SHAPE", default='False') in ['True', 'true', '1']):
            hthpu.disable_dynamic_shape()
        else:
            hthpu.enable_dynamic_shape()


def load_hpu_library(args):
    import habana_frameworks.torch.core
    import habana_frameworks.torch.distributed.hccl
    import habana_frameworks.torch.hpu


def ptlrun(args):
    if is_main_process():
        print(args)
    set_env_params(args)

    # Load the hpu libs if device is hpu
    if args.hpus:
        load_hpu_library(args)

    prof = None
    if args.profile:
        if args.gpus:
            import pyprof
            pyprof.init(enable_function_stack=True)
            print("Profiling enabled")
        elif args.hpus:
            if _KINETO_AVAILABLE:
                try:
                    from lightning_habana.pytorch.profiler import HPUProfiler
                    step_words = args.profile_steps.split(":")
                    assert step_words[0] != '' and len(step_words) > 0, "please provide valid profile_steps argument"
                    assert int(step_words[0]) > 0, "please pass starting range greater than 0"
                    warmup_steps = int(step_words[0]) - 1 if int(step_words[0]) > 0 else 0
                    active_steps = 1
                    if len(step_words) == 2:
                        active_steps = int(step_words[1]) - warmup_steps
                    assert active_steps > 0
                    prof = HPUProfiler(dirpath=args.results,
                                       activities=[torch.profiler.ProfilerActivity.CPU],
                                       schedule=torch.profiler.schedule(wait=0, warmup=warmup_steps, active=active_steps),
                                       record_shapes=False,
                                       with_stack=True,
                                       record_module_names=False)
                    """
                    [SW-186602]: record_module_names=True (Argument of HPUProfiler) doesn't work 
                    with torch.compile mode and it'll be set to False (as default value) in 
                    lightning-habana==1.6.0 (& following releases).
                    https://github.com/Lightning-AI/pytorch-lightning/issues/19253
                    """
                except ImportError:
                    print(f"lightning_habana package not installed")
            else:
                print(f"can't use HPUProfiler as kineto not available")

    if args.seed is not None:
        seed_everything(seed=args.seed)
        torch.backends.cudnn.deterministic = True
    seed0 = args.seed
    set_seed(seed0)
    data_module_seed = np.random.randint(0, 1e6)
    train_loader_seed = np.random.randint(0, 1e6)
    model_seed = np.random.randint(0, 1e6)
    trainer_seed = np.random.randint(0, 1e6)

    if args.affinity != "disabled":
        from utils.gpu_affinity import set_affinity
        affinity = set_affinity(os.getenv("LOCAL_RANK", "0"), args.affinity)

    if args.gpus:
        set_cuda_devices(args)

    device = get_device(args)

    seed_everything(seed0)
    set_seed(data_module_seed)
    data_module = DataModule(args)
    data_module.prepare_data()
    data_module.setup()
    ckpt_path = verify_ckpt_path(args)

    callbacks = None
    set_seed(model_seed)
    if args.benchmark:
        model = NNUnet(args)
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
                profile=args.profile and args.gpus,
                measurement_type=args.measurement_type
            ),
            TQDMProgressBar(refresh_rate=args.progress_bar_refresh_rate)
        ]
    elif args.exec_mode == "train":
        model = NNUnet(args)
        batch_size = args.batch_size if args.exec_mode == "train" else args.val_batch_size
        num_instance = args.gpus if args.gpus else args.hpus
        if not num_instance:
            num_instance = 1
        log_dir = os.path.join(args.results, args.logname if args.logname is not None else "perf.json")
        callbacks = [
            LoggingCallback(
                log_dir=log_dir,
                global_batch_size=batch_size * num_instance,
                mode=args.exec_mode,
                warmup=args.warmup,
                dim=args.dim,
                profile=args.profile and args.gpus,
                measurement_type=args.measurement_type
            ),
            EarlyStopping(monitor="dice_sum", patience=args.patience, verbose=True, mode="max"),
            TQDMProgressBar(refresh_rate=args.progress_bar_refresh_rate)
        ]
        if args.save_ckpt:
            model_ckpt = ModelCheckpoint(monitor="dice_sum", mode="max", save_last=True)
            #  in pl 1.4 ModelCheckpoint should be used as a callback
            callbacks.append(model_ckpt)
    else:  # Evaluation or inference
        if ckpt_path is not None:
            model = NNUnet.load_from_checkpoint(ckpt_path)
        else:
            model = NNUnet(args)

    if args.enable_tensorboard_logging:
        tbl = TensorBoardLogger(args.results)
        callbacks.append(TensorBoardCallback(tbl, args.progress_bar_refresh_rate))

    set_seed(trainer_seed)

    parallel_hpus = [torch.device("hpu")] * args.hpus
    trainer = Trainer(
        logger=False,
        profiler=prof,
        precision="bf16-mixed" if args.amp else "32-true",
        devices=args.hpus if args.hpus else None,
        accelerator=HPUAccelerator() if args.hpus else None,
        benchmark=True,
        deterministic=False,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        sync_batchnorm=args.sync_batchnorm,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        default_root_dir=args.results,
        enable_checkpointing=args.save_ckpt,
        strategy=HPUDDPStrategy(parallel_devices=parallel_hpus, bucket_cap_mb=args.bucket_cap_mb,gradient_as_bucket_view=True,static_graph=True) if args.hpus > 1 else SingleHPUStrategy() if args.hpus == 1 else None,
        limit_train_batches=1.0 if args.train_batches == 0 else args.train_batches,
        limit_val_batches=1.0 if args.test_batches == 0 else args.test_batches,
        limit_test_batches=1.0 if args.test_batches == 0 else args.test_batches,
        )

    start_time = time.time()
    if args.benchmark:
        if args.exec_mode == "train":
            if args.profile and args.gpus:
                with torch.autograd.profiler.emit_nvtx():
                    train_dl = data_module.train_dataloader()
                    trainer.fit(model, train_dataloaders=train_dl)
            else:
                train_dl = data_module.train_dataloader()
                trainer.fit(model, train_dataloaders=train_dl)
        else:
            # warmup
            test_dl = data_module.test_dataloader()
            trainer.test(model, dataloaders=test_dl)
            # benchmark run
            for i in range(len(trainer.callbacks)):
                if isinstance(trainer.callbacks[i], LoggingCallback):
                    trainer.callbacks[0].perform_epoch = 0
                    break
            test_dl = data_module.test_dataloader()
            trainer.test(model, dataloaders=test_dl)
    elif args.exec_mode == "train":
        train_dl = data_module.train_dataloader()
        val_dl = data_module.val_dataloader()
        trainer.fit(model, train_dataloaders=train_dl,
                val_dataloaders=val_dl, ckpt_path=args.ckpt_path)
    elif args.exec_mode == "evaluate":
        model.args = args
        eval_dl = data_module.val_dataloader()
        trainer.test(model, dataloaders=eval_dl)
        if is_main_process():
            logname = args.logname if args.logname is not None else "eval_log.json"
            log(logname, model.eval_dice, results=args.results)
    elif args.exec_mode == "predict":
        model.args = args
        if args.save_preds:
            prec = "fp32"
            if args.amp:
                prec = "amp"
            elif args.is_autocast:
                prec = "autocast"
            dir_name = f"preds_task_{args.task}_dim_{args.dim}_fold_{args.fold}_{prec}"
            if args.tta:
                dir_name += "_tta"
            save_dir = os.path.join(args.results, dir_name)
            model.save_dir = save_dir
            make_empty_dir(save_dir)
        test_dl = data_module.test_dataloader()
        trainer.test(model, dataloaders=test_dl)
    print("Training time ", datetime.timedelta(seconds=int(time.time() - start_time)))
