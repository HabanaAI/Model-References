# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company

"""run.py:"""
#!/usr/bin/env python
import os
from types import SimpleNamespace
import torch
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from utils.utils import set_seed
from data_loading.data_module import DataModule
from models.nn_unet import NNUnet
from pytorch.trainer import Trainer
from pytorch.trainer import NativeLogging
import habana_frameworks.torch.core
import habana_frameworks.torch.distributed.hccl
import habana_frameworks.torch.hpu
from utils.utils import seed_everything
from utils.logger import LoggingCallback
import time, datetime


def set_env_params(args):
    os.environ["PT_HPU_LAZY_MODE"] = "1"
    if args.hpus and not args.run_lazy_mode:
        os.environ["PT_HPU_LAZY_MODE"] = "2"
    # Enable hpu dynamic shape
    try:
        import habana_frameworks.torch.hpu as hthpu
        if (os.getenv("HPU_DISABLE_DYNAMIC_SHAPE", default='False') in ['True', 'true', '1']):
            hthpu.disable_dynamic_shape()
        else:
            hthpu.enable_dynamic_shape()
    except ImportError:
        print("habana_frameworks could Not be loaded")



def distribute_execution(rank, size, hparams):
    model = hparams.model
    hparams.opt_dict = model.configure_optimizers()

    trainer = hparams.trainer
    if hparams.args.benchmark:
        if hparams.args.exec_mode == "train":
            trainer.benchmark_train(model, hparams)
        else:
            trainer.test(model, hparams)

    elif hparams.args.exec_mode == "train":
        trainer.fit(model, hparams)
    else:
        pass

def init_processes(rank, world_size, hparams, fn, backend='hccl'):
    """ Initialize the distributed environment. """
    if world_size > 1:
        os.environ["ID"] = str(rank)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    dist._DEFAULT_FIRST_BUCKET_BYTES = 200*1024*1024  #200MB
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    set_seed(hparams.args.model_seed)
    hparams.model = NNUnet(hparams.args)
    fn(rank, world_size, hparams)

def nptrun(args):
    print(args)
    set_env_params(args)

    deterministic = False
    if args.seed is not None:
        seed_everything(seed=args.seed)
        torch.backends.cuda.deterministric = True
        torch.use_deterministic_algorithms(True)

    seed0 = args.seed
    set_seed(seed0)
    data_module_seed = np.random.randint(0, 1e6)
    args.model_seed = np.random.randint(0, 1e6)
    trainer_seed = np.random.randint(0, 1e6)

    hparams = SimpleNamespace()
    seed_everything(seed0)
    set_seed(data_module_seed)
    hparams.data_module = DataModule(args)
    hparams.data_module.setup()

    hparams.args = args
    hparams.world_size = args.hpus
    hparams.hpus = args.hpus

    set_seed(trainer_seed)
    hparams.trainer = Trainer(hparams)

    if args.benchmark:
        log_dir = os.path.join(args.results, args.logname if args.logname is not None else "perf.json")
        hparams.log = NativeLogging(
                      log_dir=log_dir,
                      global_batch_size = args.batch_size,
                      mode = args.exec_mode,
                      warmup = args.warmup,
                      dim = args.dim,
                      profile = args.profile,
                      perform_epoch=1
                      )


    start_time = time.time()

    processes = []
    for rank in range(hparams.world_size):
        proc = mp.Process(target=init_processes, args=(rank, hparams.world_size, hparams, distribute_execution))
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()

    end_time = time.time()
    time_interval = end_time - start_time
    print("Total Training time ", datetime.timedelta(seconds=int(time_interval)))
