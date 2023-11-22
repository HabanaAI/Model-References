import os
from typing import Any, Dict, Optional, Type

from mlperf_logging import mllog
import mlperf_logging.mllog.constants as mllog_constants

import torch
import torch.distributed as dist

try:
    import lightning.pytorch as pl
    from lightning.pytorch.utilities.types import STEP_OUTPUT
except:
    import pytorch_lightning as pl
    from pytorch_lightning.utilities.types import STEP_OUTPUT


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def barrier():
    if not is_dist_avail_and_initialized():
        return
    torch.distributed.barrier()

class SDLogger:
    def __init__(self, filename=None, default_stack_offset=2):
        self.mllogger = mllog.get_mllogger()
        self.filename = (filename or os.getenv("COMPLIANCE_FILE") or "mlperf_compliance.log")
        mllog.config(default_stack_offset=default_stack_offset, filename=self.filename,
                     root_dir=os.path.normpath(os.path.dirname(os.path.realpath(__file__))))

    @property
    def rank(self):
        return get_rank()

    def event(self, key, value=None, metadata=None, sync=False, time_ms=None):
        if sync:
            barrier()
        if self.rank == 0:
            self.mllogger.event(key=key, value=value, metadata=metadata, time_ms=time_ms)

    def start(self, key, value=None, metadata=None, sync=False, time_ms=None):
        if sync:
            barrier()
        if self.rank == 0:
            self.mllogger.start(key=key, value=value, metadata=metadata, time_ms=time_ms)

    def end(self, key, value=None, metadata=None, sync=False, time_ms=None):
        if sync:
            barrier()
        if self.rank == 0:
            self.mllogger.end(key=key, value=value, metadata=metadata, time_ms=time_ms)

def submission_info():
    """Logs required for a valid MLPerf submission."""
    mllogger.event(key=mllog_constants.SUBMISSION_BENCHMARK, value=mllog_constants.STABLE_DIFFUSION)
    mllogger.event(key=mllog_constants.SUBMISSION_DIVISION, value=mllog_constants.CLOSED)
    mllogger.event(key=mllog_constants.SUBMISSION_ORG, value="reference_implementation")
    mllogger.event(key=mllog_constants.SUBMISSION_PLATFORM, value="Gaudi2")
    mllogger.event(key=mllog_constants.SUBMISSION_POC_NAME, value="")
    mllogger.event(key=mllog_constants.SUBMISSION_POC_EMAIL, value="")
    mllogger.event(key=mllog_constants.SUBMISSION_STATUS, value=mllog_constants.ONPREM)

class MLPerfLoggingCallback(pl.callbacks.Callback):
    def __init__(self, logger, train_log_interval=5, validation_log_interval=1,
                 validation_iter=1, validation_iters=5, validation_timestamp=None,
                 seed=None, gradient_accumulation_steps=None, global_batch_size=None):
        super().__init__()
        self.logger = logger
        self.train_log_interval = train_log_interval
        self.validation_log_interval = validation_log_interval
        self.validation_iter = validation_iter
        self.validation_iters = validation_iters
        self.validation_timestamp = validation_timestamp
        self.seed = seed
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.global_batch_size = global_batch_size

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        submission_info()

        mllogger.event(key=mllog_constants.SEED, value=self.seed)
        # We can't get number of samples without reading the data (which we can't inside the init block), so we hard code them
        self.logger.event(key=mllog_constants.TRAIN_SAMPLES, value=1) # TODO(ahmadki): a placeholder until a dataset is picked
        self.logger.event(key=mllog_constants.EVAL_SAMPLES, value=30000)
        self.logger.event(mllog_constants.GRADIENT_ACCUMULATION_STEPS, value=self.gradient_accumulation_steps)
        self.logger.event(mllog_constants.GLOBAL_BATCH_SIZE, value=self.global_batch_size)
        self.logger.end(mllog_constants.INIT_STOP)
        self.logger.start(mllog_constants.RUN_START)

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                             batch: Any, batch_idx: int) -> None:
        if trainer.global_step % self.train_log_interval == 0:
            self.logger.start(
                key=mllog_constants.BLOCK_START, value="training_step",
                metadata={
                    mllog_constants.EPOCH_COUNT: 1,
                    mllog_constants.FIRST_EPOCH_NUM: 0,
                    mllog_constants.STEP_NUM: trainer.global_step
                }
            )

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                           outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        if trainer.global_step % self.train_log_interval == 0:
            logs = trainer.callback_metrics
            self.logger.event(key="loss", value=logs["train/loss"].item(), metadata={mllog_constants.STEP_NUM: trainer.global_step})
            self.logger.event(key="lr_abs", value=logs["lr_abs"].item(), metadata={mllog_constants.STEP_NUM: trainer.global_step})
            self.logger.end(
                key=mllog_constants.BLOCK_STOP, value="training_step",
                metadata={
                    mllog_constants.FIRST_EPOCH_NUM: 0,
                    mllog_constants.STEP_NUM: trainer.global_step
                }
            )

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.logger.start(
            key=mllog_constants.EVAL_START, value=trainer.global_step, time_ms=self.validation_timestamp,
            metadata={
                mllog_constants.EPOCH_NUM: self.validation_iter,
                mllog_constants.EPOCH_COUNT: self.validation_iters,
            }
        )

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logs = trainer.callback_metrics
        if "validation/fid" in logs:
            self.logger.event(
                key=mllog_constants.EVAL_ACCURACY, value=logs["validation/fid"].item(), time_ms=self.validation_timestamp,
                metadata={
                    mllog_constants.EPOCH_NUM: self.validation_iter,
                    mllog_constants.STEP_NUM: self.validation_iter * 1000, "metric": "FID"
                }
            )
        if "validation/clip" in logs:
            self.logger.event(
                key=mllog_constants.EVAL_ACCURACY, value=logs["validation/clip"].item(), time_ms=self.validation_timestamp,
                metadata={
                    mllog_constants.EPOCH_NUM: self.validation_iter,
                    mllog_constants.STEP_NUM: self.validation_iter * 1000, "metric": "CLIP"
                }
            )
        self.logger.end(key=mllog_constants.EVAL_STOP, value=trainer.global_step,
                        time_ms=self.validation_timestamp, metadata={mllog_constants.EPOCH_NUM: self.validation_iter})

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_validation_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                                  batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if batch_idx % self.validation_log_interval == 0:
            self.logger.start(
                key=mllog_constants.BLOCK_START, value="validation_step", time_ms=self.validation_timestamp,
                metadata={
                    mllog_constants.STEP_NUM: batch_idx,
                    mllog_constants.EPOCH_COUNT: self.validation_iter,
                    mllog_constants.FIRST_EPOCH_NUM: self.validation_iter - 1,
                }
            )

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Optional[STEP_OUTPUT],
                                batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if batch_idx % self.validation_log_interval == 0:
            self.logger.end(
                key=mllog_constants.BLOCK_STOP, value="validation_step", time_ms=self.validation_timestamp,
                metadata={
                    mllog_constants.STEP_NUM: batch_idx,
                    mllog_constants.FIRST_EPOCH_NUM: self.validation_iter - 1,
                }
            )

mllogger = SDLogger()
