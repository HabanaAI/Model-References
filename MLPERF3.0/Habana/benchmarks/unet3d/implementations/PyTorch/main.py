###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
import random
from math import ceil

import torch
from mlperf_logging import mllog
from mlperf_logging.mllog import constants

from model.unet3d import Unet3D
from model.losses import DiceCELoss, DiceScore

from data_loading.data_loader import get_data_loaders

from runtime.training import train, warmup
from runtime.inference import evaluate, SlidingwWindowInference
from runtime.arguments import PARSER
from runtime.distributed_utils import init_distributed, get_world_size, get_device, is_main_process, get_rank
from runtime.distributed_utils import seed_everything
from runtime.logging import get_dllogger, mllog_start, mllog_end, mllog_event, mlperf_submission_log
from runtime.logging import mlperf_run_param_log
from runtime.callbacks import get_callbacks, get_profiler

DATASET_SIZE = 168


def main():
    flags = PARSER.parse_args()
    os.makedirs(flags.log_dir, exist_ok=True)

    dllogger = get_dllogger(flags)
    local_rank = flags.local_rank
    device = get_device(flags.device, local_rank)
    is_distributed = init_distributed(flags.device)
    profiler = get_profiler(flags)
    world_size = get_world_size()
    local_rank = get_rank()
    worker_seed = flags.seed if flags.seed != -1 else random.SystemRandom().randint(0, 2**32 - 1)
    seed_everything(worker_seed)

    if is_main_process:
        mllog.config(filename=os.path.join(flags.log_dir, 'result_rank_0.txt'))
        mllogger = mllog.get_mllogger()
        mllogger.logger.propagate = False
        mlperf_submission_log()
        mllog_event(key=constants.CACHE_CLEAR, value=True)
        mllog_start(key=constants.INIT_START)
        mllog_event(key=constants.SEED, value=worker_seed, sync=False)
        mlperf_run_param_log(flags)

    if flags.device == "hpu":
        torch.cuda.current_device = lambda: None
        torch.cuda.set_device = lambda x: None

        # Disable hpu dynamic shapes
        try:
            import habana_frameworks.torch.hpu as hpu
            hpu.disable_dynamic_shape()
        except ImportError:
            print("habana_frameworks could not disable dynamic shapes")
    else:
        torch.backends.cudnn.benchmark = flags.cudnn_benchmark
        torch.backends.cudnn.deterministic = flags.cudnn_deterministic

    callbacks = get_callbacks(flags, dllogger, local_rank, world_size)
    flags.seed = worker_seed

    model = Unet3D(1, 3, normalization=flags.normalization, activation=flags.activation)
    model.to(device)
    model_for_eval = model
    if flags.use_hpu_graphs and flags.device == 'hpu':
        import habana_frameworks.torch.hpu.graphs as htgraphs
        model = htgraphs.wrap_in_hpu_graph(model)

    if is_distributed:
        if flags.device == 'hpu':
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              broadcast_buffers=False,
                                                              gradient_as_bucket_view=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[flags.local_rank],
                                                              output_device=flags.local_rank)

    loss_fn = DiceCELoss(to_onehot_y=True, use_softmax=True,
                         include_background=flags.include_background)
    loss_fn.to(device)
    score_fn = DiceScore(to_onehot_y=True, use_argmax=True,
                         include_background=flags.include_background)

    sw_inference = SlidingwWindowInference(roi_shape=flags.val_input_shape,
                                           overlap=flags.overlap, mode="gaussian", padding_val=-2.2, device=device)
    if flags.enable_device_warmup:
        warmup(flags, model, model_for_eval, loss_fn, score_fn, device, sw_inference=sw_inference)

    mllog_end(key=constants.INIT_STOP, sync=True)
    mllog_start(key=constants.RUN_START, sync=True)

    train_dataloader, val_dataloader = get_data_loaders(flags, num_shards=world_size, local_rank=local_rank)
    samples_per_epoch = world_size * len(train_dataloader) * flags.batch_size
    flags.evaluate_every = flags.evaluate_every or ceil(20 * DATASET_SIZE / samples_per_epoch)
    flags.start_eval_at = flags.start_eval_at or ceil(1000 * DATASET_SIZE / samples_per_epoch)

    mllog_event(key='samples_per_epoch', value=samples_per_epoch, sync=False)
    mllog_event(key=constants.GLOBAL_BATCH_SIZE, value=flags.batch_size * world_size * flags.ga_steps, sync=False)
    mllog_event(key=constants.GRADIENT_ACCUMULATION_STEPS, value=flags.ga_steps)

    if flags.exec_mode == 'train':
        train(flags, model, model_for_eval, train_dataloader, val_dataloader, loss_fn, score_fn, device=device,
              callbacks=callbacks, is_distributed=is_distributed, profiler=profiler, sw_inference=sw_inference)

    elif flags.exec_mode == 'evaluate':
        eval_metrics = evaluate(flags, model_for_eval, val_dataloader, loss_fn,
                                score_fn, device=device, is_distributed=is_distributed, sw_inference=sw_inference)
        if local_rank == 0:
            for key in eval_metrics.keys():
                print(key, eval_metrics[key])
    else:
        print("Invalid exec_mode.")
        pass


if __name__ == '__main__':
    main()
