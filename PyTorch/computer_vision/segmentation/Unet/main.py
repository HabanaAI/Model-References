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
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################


import os
import torch
from pytorch_lightning import Trainer, seed_everything
import numpy as np
import time
import datetime

from models.nn_unet import NNUnet
from utils.logger import LoggingCallback
from data_loading.data_module import DataModule
from utils.utils import get_main_args, is_main_process, log, make_empty_dir, set_cuda_devices, verify_ckpt_path
from utils.utils import set_seed, get_device
from pytorch_lightning.plugins import HPUPrecisionPlugin
import pytorch_lightning.utilities.seed
from utils.early_stopping_unet import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import HPUParallelStrategy

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


if __name__ == "__main__":
    args = get_main_args()

    if is_main_process():
        print(args)
    set_env_params(args)

    # Load the hpu libs if device is hpu
    if args.hpus:
        load_hpu_library(args)

    if args.profile:
        assert (not args.hpus), "profiling not supported for HPU devices"
        import pyprof
        pyprof.init(enable_function_stack=True)
        print("Profiling enabled")

    deterministic = False
    if args.seed is not None:
        pytorch_lightning.utilities.seed.seed_everything(seed=args.seed)
        torch.backends.cudnn.deterministic = True
        deterministic = True
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
                profile=args.profile,
            )
        ]
    elif args.exec_mode == "train":
        model = NNUnet(args)
        callbacks = [EarlyStopping(monitor="dice_sum", patience=args.patience, verbose=True, mode="max")]
        if args.save_ckpt:
            model_ckpt = ModelCheckpoint(monitor="dice_sum", mode="max", save_last=True)
            #  in pl 1.4 ModelCheckpoint should be used as a callback
            callbacks.append(model_ckpt)
    else:  # Evaluation or inference
        if ckpt_path is not None:
            model = NNUnet.load_from_checkpoint(ckpt_path)
        else:
            model = NNUnet(args)

    set_seed(trainer_seed)

    parallel_hpus = [torch.device("hpu")] * args.hpus
    trainer = Trainer(
        progress_bar_refresh_rate=args.progress_bar_refresh_rate,
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

    start_time = time.time()
    if args.benchmark:
        if args.exec_mode == "train":
            if args.profile:
                with torch.autograd.profiler.emit_nvtx():
                    trainer.fit(model, train_dataloaders=data_module.train_dataloader())
            else:
                trainer.fit(model, train_dataloaders=data_module.train_dataloader())
        else:
            # warmup
            trainer.test(model, test_dataloaders=data_module.test_dataloader())
            # benchmark run
            trainer.current_epoch = 1
            trainer.test(model, test_dataloaders=data_module.test_dataloader())
    elif args.exec_mode == "train":
        trainer.fit(model, train_dataloaders=data_module.train_dataloader(),
                val_dataloaders=data_module.val_dataloader(), ckpt_path=args.ckpt_path)
    elif args.exec_mode == "evaluate":
        model.args = args
        trainer.test(model, test_dataloaders=data_module.val_dataloader())
        if is_main_process():
            logname = args.logname if args.logname is not None else "eval_log.json"
            log(logname, model.eval_dice, results=args.results)
    elif args.exec_mode == "predict":
        model.args = args
        if args.save_preds:
            prec = "amp" if args.amp else "fp32"
            dir_name = f"preds_task_{args.task}_dim_{args.dim}_fold_{args.fold}_{prec}"
            if args.tta:
                dir_name += "_tta"
            save_dir = os.path.join(args.results, dir_name)
            model.save_dir = save_dir
            make_empty_dir(save_dir)
        trainer.test(model, test_dataloaders=data_module.test_dataloader())
    print("Training time ", datetime.timedelta(seconds=int(time.time() - start_time)))
