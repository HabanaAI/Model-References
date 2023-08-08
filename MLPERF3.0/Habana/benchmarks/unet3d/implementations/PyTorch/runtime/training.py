###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################

import time
from tqdm import tqdm

import torch
from torch.optim import Adam, SGD
from torch.cuda.amp import autocast, GradScaler
import habana_frameworks.torch.core as htcore

from runtime.distributed_utils import get_rank, get_world_size
from runtime.inference import evaluate
from runtime.logging import mllog_event, mllog_start, mllog_end, CONSTANTS
from data_loading.data_loader import get_data_loaders


def get_optimizer(params, flags):
    if flags.optimizer == "adam":
        optim = Adam(params, lr=flags.learning_rate, weight_decay=flags.weight_decay)
    elif flags.optimizer == "sgd":
        if flags.device == "hpu":
            from habana_frameworks.torch.hpex.optimizers import FusedSGD
            optim = FusedSGD(params, lr=flags.learning_rate, momentum=flags.momentum, nesterov=True,
                             weight_decay=flags.weight_decay)
        else:
            optim = SGD(params, lr=flags.learning_rate, momentum=flags.momentum, nesterov=True,
                        weight_decay=flags.weight_decay)
    elif flags.optimizer == "lamb":
        import apex
        optim = apex.optimizers.FusedLAMB(params, lr=flags.learning_rate, betas=flags.lamb_betas,
                                          weight_decay=flags.weight_decay)
    else:
        raise ValueError("Optimizer {} unknown.".format(flags.optimizer))
    return optim


def lr_warmup(optimizer, init_lr, lr, current_epoch, warmup_epochs):
    scale = current_epoch / warmup_epochs
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr + (lr - init_lr) * scale


def warmup(flags, model, model_for_eval, loss_fn, score_fn, device, sw_inference=None):
    scaler = GradScaler()
    optimizer = get_optimizer(model.parameters(), flags)
    rank = get_rank()
    world_size = get_world_size()
    dataloader, test_dataloader = get_data_loaders(flags, world_size, rank, warmup=True)
    retain_graph = True if flags.use_hpu_graphs else None

    train_one_epoch(flags, model, dataloader, loss_fn, device,
                    optimizer, scaler, rank, callbacks=[], retain_graph=retain_graph)
    evaluate(flags, model_for_eval, test_dataloader, loss_fn, score_fn, device, sw_inference=sw_inference, warmup=True)


def train_one_epoch(flags, model, train_loader, loss_fn, device, optimizer, scaler, rank, callbacks, profiler=None, retain_graph=True):
    loss_value = None
    for iteration, batch in enumerate(tqdm(train_loader, disable=(rank != 0) or not flags.verbose)):
        image, label = batch["image"], batch["label"]
        image, label = image.to(device), label.to(device)

        for callback in callbacks:
            callback.on_batch_start()

        with torch.autocast(device_type=flags.device, enabled=flags.amp, dtype=torch.bfloat16):
            output = model(image)
            loss_value = loss_fn(output, label)
            loss_value /= flags.ga_steps

        optimizer.zero_grad(set_to_none=True)
        if flags.amp:
            scaler.scale(loss_value).backward(retain_graph=retain_graph)
        else:
            loss_value.backward(retain_graph=retain_graph)

        if flags.device == "hpu":
            htcore.mark_step()

        if (iteration + 1) % flags.ga_steps == 0:
            if flags.amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if flags.device == "hpu":
                htcore.mark_step()
        if profiler is not None:
            profiler.step()


def train(flags, model, model_for_eval, train_loader, val_loader,
          loss_fn, score_fn, device, callbacks, is_distributed, profiler, sw_inference=None):
    rank = get_rank()
    world_size = get_world_size()
    optimizer = get_optimizer(model.parameters(), flags)
    if flags.lr_decay_epochs:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=flags.lr_decay_epochs,
                                                         gamma=flags.lr_decay_factor)
    scaler = GradScaler()

    is_successful = False
    diverged = False
    samples_per_epoch = world_size * len(train_loader) * flags.batch_size
    epoch = 0
    max_cycles = flags.epochs if flags.epochs < flags.evaluate_every else (flags.epochs // flags.evaluate_every + 1)
    retain_graph = True if flags.use_hpu_graphs else None

    model.train()
    for callback in callbacks:
        callback.on_fit_start()

    for cycle in range(1, max_cycles):
        mllog_start(key=CONSTANTS.BLOCK_START, sync=False,
                    metadata={CONSTANTS.FIRST_EPOCH_NUM: epoch + 1, CONSTANTS.EPOCH_COUNT: flags.evaluate_every})
        cycle_start_time = time.time()
        for training_epoch in range(0, flags.evaluate_every):
            epoch += 1
            if epoch <= flags.lr_warmup_epochs and flags.lr_warmup_epochs > 0:
                lr_warmup(optimizer, flags.init_learning_rate, flags.learning_rate, epoch, flags.lr_warmup_epochs)

            if is_distributed and flags.loader != 'dali' and flags.loader != 'media':
                train_loader.sampler.set_epoch(epoch)

            train_one_epoch(flags, model, train_loader, loss_fn, device, optimizer,
                            scaler, rank, callbacks, profiler, retain_graph)

            if flags.lr_decay_epochs:
                scheduler.step()

        mllog_event(key='current_lr', value=optimizer.param_groups[0]['lr'], sync=False)
        throughput = samples_per_epoch * flags.evaluate_every / (time.time() - cycle_start_time)
        mllog_event(key='throughput', value=throughput, sync=False)

        if epoch >= flags.start_eval_at:
            mllog_start(key=CONSTANTS.EVAL_START, value=epoch, metadata={CONSTANTS.EPOCH_NUM: epoch}, sync=False)

            eval_metrics = evaluate(flags, model_for_eval, val_loader, loss_fn,
                                    score_fn, device, epoch, sw_inference=sw_inference)

            mllog_event(key=CONSTANTS.EVAL_ACCURACY,
                        value=eval_metrics["mean_dice"],
                        metadata={CONSTANTS.EPOCH_NUM: epoch},
                        sync=False)
            mllog_end(key=CONSTANTS.EVAL_STOP, metadata={CONSTANTS.EPOCH_NUM: epoch}, sync=False)

            for callback in callbacks:
                callback.on_epoch_end(epoch=epoch, metrics=eval_metrics, model=model_for_eval, optimizer=optimizer)
            model.train()
            if eval_metrics["mean_dice"] >= flags.quality_threshold:
                is_successful = True
            elif eval_metrics["mean_dice"] < 1e-6:
                print("MODEL DIVERGED. ABORTING.")
                diverged = True

        mllog_end(key=CONSTANTS.BLOCK_STOP, sync=False,
                  metadata={CONSTANTS.FIRST_EPOCH_NUM: epoch, CONSTANTS.EPOCH_COUNT: flags.evaluate_every})

        if is_successful or diverged:
            break

    mllog_end(key=CONSTANTS.RUN_STOP, sync=True,
              metadata={CONSTANTS.STATUS: CONSTANTS.SUCCESS if is_successful else CONSTANTS.ABORTED})
    for callback in callbacks:
        callback.on_fit_end()
    if profiler is not None:
        profiler.stop()
