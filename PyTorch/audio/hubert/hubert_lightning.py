# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
import os, sys
import math
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
import torchaudio
sys.path.insert(0, os.getcwd())
import models
import models.wav2vec2.components as components
print(os.path.abspath(components.__file__))
from utils import hpu_wav2vec2_params_reshape, hpu_wav2vec2_optimizer_reshape

from dataset import (
    _get_lengths_librilightlimited,
    _get_lengths_librispeech,
    BucketizeBatchSampler,
    CollateFnHubert,
    CollateFnLibriLightLimited,
    DistributedBatchSampler,
    HuBERTDataSet,
)
from loss import hubert_loss, hubert_loss_static
from lightning import LightningModule
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
Batch = Tuple[Tensor, Tensor, Tensor]
Batch_FineTune = Tuple[Tensor, Tensor, Tensor, Tensor]

if os.environ.get("TP_MODEL_PARAM_DUMP_ENABLE",'0') == '1':
    path = os.path.join(os.environ['PYTORCH_MODULES_ROOT_PATH'], 'common')
    tools_path = os.path.join(path, 'tools')
    if os.path.exists(path) is False or os.path.exists(tools_path) is False:
        raise Exception("path for 'tools' NOT found")
    sys.path.insert(0, path)
    from tools import *

class LinearDecayLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear learning rate scheduler with warm up."""
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_updates: int,
        max_updates: int,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.warmup_updates = warmup_updates
        self.max_updates = max_updates
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)
    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            return [self._step_count / self.warmup_updates * base_lr for base_lr in self.base_lrs]
        elif self._step_count >= self.max_updates:
            return [0.0 for _ in self.base_lrs]
        else:
            pct_remaining = (self.max_updates - self._step_count) / (self.max_updates - self.warmup_updates)
            return [base_lr * pct_remaining for base_lr in self.base_lrs]
class TriStageLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear learning rate scheduler with warmup, hold, and decay."""
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_updates: int,
        hold_updates: int,
        decay_updates: int,
        init_lr_scale: float = 0.01,
        final_lr_scale: float = 0.05,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.warmup_updates = warmup_updates
        self.hold_updates = hold_updates
        self.decay_updates = decay_updates
        self.init_lr_scale = init_lr_scale
        self.final_lr_scale = final_lr_scale
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)
    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            return [
                base_lr * (self.init_lr_scale + self._step_count / self.warmup_updates * (1 - self.init_lr_scale))
                for base_lr in self.base_lrs
            ]
        elif self.warmup_updates < self._step_count <= (self.warmup_updates + self.hold_updates):
            return list(self.base_lrs)
        elif self._step_count <= (self.warmup_updates + self.hold_updates + self.decay_updates):
            return [
                base_lr
                * math.exp(
                    math.log(self.final_lr_scale)
                    * (self._step_count - self.warmup_updates - self.hold_updates)
                    / self.decay_updates
                )
                for base_lr in self.base_lrs
            ]
        else:
            return [base_lr * self.final_lr_scale for base_lr in self.base_lrs]
def _compute_accuracy_static(logits: torch.Tensor, mask_m: torch.Tensor, mask_u: torch.Tensor):
    with torch.no_grad():
        def helper(logits, mask):
            overall_max = torch.ones_like(logits) * logits.max()
            overall_min = torch.ones_like(logits) * logits.min()
            # to compute for mask_m, "zero" out mask_u locations with sentinel value
            # sentinel value for max is overall_min

            count_mask = mask.sum()
            mask = mask.flatten().unsqueeze(-1).broadcast_to(logits.shape)
            logits_mask_for_max = torch.where(mask, logits, overall_min)
            max_mask = logits_mask_for_max.argmax(-1) == 0
            logits_mask_for_min = torch.where(mask, logits, overall_max)
            min_mask = logits_mask_for_min.argmin(-1) == 0
            both_mask = max_mask & min_mask
            corr_mask = max_mask.long().sum() - both_mask.long().sum()
            return corr_mask, count_mask
        corr_mask_m, count_mask_m = helper(logits, mask_m)
        corr_mask_u, count_mask_u = helper(logits, mask_u)
        return corr_mask_m, count_mask_m, corr_mask_u, count_mask_u
def _compute_accuracy(logits: torch.Tensor):
    with torch.no_grad():
        max = logits.argmax(-1) == 0
        min = logits.argmin(-1) == 0
        both = max & min
        corr = max.long().sum() - both.long().sum()
        count = max.numel()
    return corr, count
def _reset_stats(device):
    # move these tensors to gpu later once dist is initialized in _step()
    if device == 'gpu':
        device = 'cpu'
    return {
        "train": {
            "correct": torch.tensor(0.0, device=device),
            "count": torch.tensor(0.0, device=device),
        },
        "val": {
            "correct": torch.tensor(0.0,device=device),
            "count": torch.tensor(0.0,device=device),
        },
    }
class HuBERTPreTrainModule(LightningModule):
    def __init__(
        self,
        *,
        model_name: str,
        feature_grad_mult: float,
        num_classes: int,
        dataset: str,
        dataset_path: str,
        feature_type: str,
        seconds_per_batch: float,
        learning_rate: float,
        betas: Tuple[float, float],
        eps: float,
        weight_decay: float,
        clip_norm: Optional[float],
        warmup_updates: int,
        max_updates: int,
        static_logit_generator: bool = False,
        static_indexing: bool = False,
        static_layerdrop: bool = False,
        align_buckets: str = 'none',
        num_buckets: int = 1000,
        accum_steps: int = 1,
        use_conv2d: bool = False,
        use_instancenorm: bool = False,
        all_deterministic: bool = False,
        show_traindataloader_stats: bool = False,
        no_layerdrop: bool = False,
        device: str,
        use_fused_clip: bool = False,
        use_max_sub_softmax_opt: bool = False,
        recompilation_optimization: bool = False,
        split_logits: bool = False,
        optimizer_str:str = "adamw",
        log_every_n_steps = 1,
        use_autocast: bool = False,
        save_1d_ckpt: bool = False,
    ):
        super().__init__()
        kwargs={}
        if no_layerdrop:
            kwargs.update({"encoder_layer_drop":-1.0})
        if all_deterministic:
            kwargs.update({"encoder_projection_dropout":0.0, "encoder_attention_dropout":0.0, \
            "encoder_ff_interm_dropout":0.0, "encoder_dropout":0.0})
        if model_name == "hubert_pretrain_base":
            self.model = models.hubert_pretrain_base(
                feature_grad_mult=feature_grad_mult, num_classes=num_classes, static_logit_generator=static_logit_generator, \
                static_indexing=static_indexing, use_conv2d=use_conv2d, use_instancenorm=use_instancenorm,
                static_layerdrop=static_layerdrop, use_max_sub_softmax_opt=use_max_sub_softmax_opt,
                recompilation_optimization=recompilation_optimization, split_logits=split_logits, **kwargs
            )
        elif model_name == "hubert_pretrain_large":
            self.model = models.hubert_pretrain_large(static_logit_generator=static_logit_generator, \
            static_indexing=static_indexing, use_conv2d=use_conv2d, use_instancenorm=use_instancenorm,
            static_layerdrop=static_layerdrop, use_max_sub_softmax_opt=use_max_sub_softmax_opt,
            recompilation_optimization=recompilation_optimization,  split_logits=split_logits, **kwargs)
        elif model_name == "hubert_pretrain_xlarge":
            self.model = models.hubert_pretrain_xlarge(static_logit_generator=static_logit_generator, \
            static_indexing=static_indexing, use_conv2d=use_conv2d, use_instancenorm=use_instancenorm,
            static_layerdrop=static_layerdrop, use_max_sub_softmax_opt=use_max_sub_softmax_opt,
            recompilation_optimization=recompilation_optimization,  split_logits=split_logits, **kwargs)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        self.automatic_optimization = False
        self.scaler = torch.cuda.amp.GradScaler()
        self.loss = hubert_loss_static if static_logit_generator else hubert_loss
        self.optimizer_str = optimizer_str.lower()
        self.lr = learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.warmup_updates = warmup_updates
        self.max_updates = max_updates
        self.clip_norm = clip_norm
        self.FusedNorm = None
        self.save_1d_ckpt = save_1d_ckpt
        self.ckpt = None
        if self.clip_norm and use_fused_clip:
            try:
                from habana_frameworks.torch.hpex.normalization import FusedClipNorm
            except ImportError:
                raise ImportError("Please install habana_torch.")
            self.FusedNorm = FusedClipNorm(self.model.parameters(), self.clip_norm)
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.feature_type = feature_type
        self.seconds_per_batch = seconds_per_batch
        self.device_str = device
        self.mask_stats = _reset_stats(device)
        self.unmask_stats = _reset_stats(device)
        self.stats_moved_to_device = device != 'gpu'
        self.nan_loss_count = 0.0
        self.static_logit_generator = static_logit_generator
        self.align_buckets = align_buckets
        self.num_buckets = num_buckets
        self.train_data_loader_len = 0
        self.accum_steps = accum_steps
        self.all_deterministic = all_deterministic
        self.show_traindataloader_stats = show_traindataloader_stats
        self.log_every_n_steps = log_every_n_steps
        self.use_autocast = use_autocast
        if os.environ.get("TP_MODEL_PARAM_DUMP_ENABLE",'0') == '1':
            print("SBS Enabled")
            self.trainMetaData = TrainMetaData(self.model, device)

    def _step(self, batch: Batch, batch_idx, step_type):
        if batch is None:
            return None, None
        loss_inf_nan = False
        waveforms, labels, audio_lengths = batch
        if step_type == "val":
            with torch.no_grad():
                ret = self.model(
                    waveforms,
                    labels,
                    audio_lengths,
                )
        else:
            ret = self.model(
                waveforms,
                labels,
                audio_lengths,
            )
        if self.static_logit_generator:
            logit, mask_m, mask_u, feature_penalty = ret
            loss = self.loss(logit, mask_m, mask_u, feature_penalty)
            logit_shape = mask_m.sum()
        else:
            logit_m, logit_u, feature_penalty = ret
            loss = self.loss(logit_m, logit_u, feature_penalty)
            logit_shape = logit_m.size(0)
        if not torch.isinf(loss) and not torch.isnan(loss):
            self.log(f"{step_type}_loss", loss / logit_shape, on_step=False, on_epoch=True)
        else:
            self.nan_loss_count += 1
            self.log("nan_loss_count", self.nan_loss_count, on_step=False, on_epoch=True)
            loss_inf_nan =  True
        # log accuracies of masked and unmasked frames
        if self.static_logit_generator:
            correct_m, count_m, correct_u, count_u = _compute_accuracy_static(logit, mask_m, mask_u)
        else:
            correct_m, count_m = _compute_accuracy(logit_m)
            correct_u, count_u = _compute_accuracy(logit_u)

        if not self.stats_moved_to_device:
            assert self.device_str == 'gpu'
            # in case of gpu, reset_stats did not place the tensors on device, so moving them here
            moveto = torch.distributed.get_rank() if torch.distributed.is_initialized() else 'cuda'
            for k in self.mask_stats[step_type]:
                self.mask_stats[step_type][k] = self.mask_stats[step_type][k].to(moveto)
            for k in self.unmask_stats[step_type]:
                self.unmask_stats[step_type][k] = self.unmask_stats[step_type][k].to(moveto)
        self.mask_stats[step_type]["correct"] += correct_m
        self.mask_stats[step_type]["count"] += count_m
        self.unmask_stats[step_type]["correct"] += correct_u
        self.unmask_stats[step_type]["count"] += count_u
        if (batch_idx % self.log_every_n_steps == 0) or ((batch_idx + 1) == self.train_data_loader_len):
            # self.log has sync_dist=True, which does a allreduce every step for logging
            # hence adding this if to prevent that
            self.log(
                f"{step_type}_masked_accuracy",
                self.mask_stats[step_type]["correct"] / self.mask_stats[step_type]["count"],
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                prog_bar=step_type == "train",
            )
            self.log(
                f"{step_type}_unmasked_accuracy",
                self.unmask_stats[step_type]["correct"] / self.unmask_stats[step_type]["count"],
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                prog_bar=step_type == "train",
            )
        return loss, logit_shape, loss_inf_nan
    def configure_optimizers(self):
        if self.optimizer_str == "adamw":
            self.optimizer = torch.optim.AdamW(
                    self.model.parameters(), lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay)
        elif self.optimizer_str == "fusedadamw":
            from habana_frameworks.torch.hpex.optimizers import FusedAdamW
            self.optimizer = FusedAdamW(self.model.parameters(), lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay)
        else:
            assert False, "optimizer {} not supported".format(self.args.optimizer.lower())

        self.lr_scheduler = LinearDecayLRScheduler(self.optimizer, self.warmup_updates, self.max_updates)
        return (
            [self.optimizer],
            [
                {
                    "scheduler": self.lr_scheduler,
                    "interval": "step",
                },
            ],
        )
    def training_step(self, batch: Batch, batch_idx):
        """Custom training step with loss normalization and automatic mixed precision training.
        By default, DDP does the following on each train step:
        - For each GPU, compute loss and gradient on shard of training data.
        - Sync and average gradients across all GPUs. The final gradient
          is (sum of gradients across all GPUs) / N, where N is the world
          size (total number of GPUs).
        - Update parameters on each GPU.
        Here, we do the following:
        - For k-th GPU, compute loss and scale it by (N / num_frames), where num_frames is
          the sum of masked frames across all GPUs. Compute gradient from scaled loss.
        - Sync and average gradients across all GPUs. The final gradient
          is (sum of gradients across all GPUs) / num_frames.
        - Update parameters on each GPU.
        Doing so allows us to account for the variability in number of masked frames in
        variable-length sequential data.
        """
        #self.log("Train step ", batch_idx, batch[0].shape, batch[1].shape, batch[2].shape)
        if os.environ.get("TP_MODEL_PARAM_DUMP_ENABLE",'0') == '1':
            tp_probe_tensors_iteration_start(self.model, self.device, None, None, self.trainMetaData.ParamsDump, False)
        opt = self.optimizers()
        if batch_idx == 0:
            opt.zero_grad(set_to_none=True)
        device_type = "cuda" if self.device_str == 'gpu' else self.device_str
        dtype = torch.float16 if self.device_str == 'gpu' else torch.bfloat16
        with torch.autocast(device_type=device_type, dtype=dtype, enabled=self.use_autocast):
            loss, num_frame, loss_inf_nan = self._step(batch, batch_idx, "train")
        if loss_inf_nan: #torch.isinf(loss) or torch.isnan(loss):
            opt.zero_grad(set_to_none=True)
            return None
        # normalize the loss based on the sum of num_frame across all GPUs

        if torch.distributed.is_initialized():
            num_frames = self.all_gather(num_frame)
        else:
            num_frames = torch.tensor(num_frame, device =self.device).unsqueeze(-1)
        self.log("Gathered number of frames", num_frames.float().sum(), on_step=False, on_epoch=True)
        loss *= num_frames.size(0) / num_frames.sum()  # world size / num_frames
        # backward the loss and clip the gradients
        loss = self.scaler.scale(loss)
        self.manual_backward(loss)
        if ((batch_idx + 1) % self.accum_steps == 0)  or ((batch_idx + 1) == self.train_data_loader_len):
            self.scaler.unscale_(opt)
            if self.clip_norm is not None:
                if self.FusedNorm is not None:
                    self.FusedNorm.clip_norm(self.model.parameters())
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                #print("After CLIP NORM ", norm1, " for batch ", batch_idx)
            # optimization
            self.scaler.step(opt)
            sch = self.lr_schedulers()
            sch.step()
            self.scaler.update()
            opt.zero_grad(set_to_none=True)
        if os.environ.get("TP_MODEL_PARAM_DUMP_ENABLE",'0') == '1':
            tp_probe_tensors_iteration_end(self.model, self.device, loss, loss.item(), self.trainMetaData.ParamsDump, False)
            self.trainMetaData.increment_train_step()
        return loss
    def validation_step(self, batch: Batch, batch_idx):
        loss = self._step(batch, batch_idx, "val")[0]
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
            loss = loss / torch.distributed.get_world_size()
        return loss
    def on_validation_end(self):
        self.mask_stats = _reset_stats(self.device)
        self.unmask_stats = _reset_stats(self.device)
    def train_dataloader(self):
        dataset = HuBERTDataSet(self.dataset_path, self.dataset, "train")
        sampler = BucketizeBatchSampler(
            dataset.len_list,
            num_buckets=self.num_buckets,
            max_token_count=self.seconds_per_batch * 16000,
            min_len=32000,
            max_len=250000,
            shuffle=False,
            align_buckets=self.align_buckets
        )
        aligner = sampler.gen_bucket_boundary_fn()
        if torch.distributed.is_initialized():
            sampler = DistributedBatchSampler(sampler, shuffle= not self.all_deterministic, epoch=self.current_epoch)
            sampler.set_epoch(self.current_epoch)
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=CollateFnHubert(feature_type=self.feature_type, pad=self.align_buckets!='none', rand_crop=True, aligner=aligner if self.align_buckets != 'none' else None),
            num_workers=10,
        )
        if self.show_traindataloader_stats:
            print('Calculating train dataloader stats...')
            seqlen_vs_bs = {}
            hist_seq = {}
            hist_seq_with_batch = {}
            tot_seq = 0
            actual_tot_len = 0
            from tqdm import tqdm
            for k in tqdm(dataloader):
                shp = [tuple(i.shape) for i in k]
                bs = shp[0][0]
                seqlen = shp[0][1]
                lbllen = shp[1][1]
                k1 = (bs, seqlen, lbllen)
                seqlen_vs_bs[seqlen] = seqlen_vs_bs.get(seqlen, []) + [bs]
                hist_seq[seqlen] = hist_seq.get(seqlen, 0) + 1
                hist_seq_with_batch[(bs, seqlen)] = hist_seq_with_batch.get((bs, seqlen),0) + 1
                actual_tot_len += sum((k[2])).item()
                tot_seq += bs*seqlen
            tot_pad = tot_seq - actual_tot_len

            def print_dict_sorted_by_val(d):
                d_view = [ (d[k],k) for k in d]
                d_view.sort(reverse=True)
                for v,k in d_view:
                    print(k,v)

            print('len seq', len(hist_seq))
            print('len seq with batch', len(hist_seq_with_batch))
            print('\nActual total length', actual_tot_len)
            print('\nTotal padded length', tot_seq)
            print('Padding', tot_pad)
            print('Pad perc', tot_pad / tot_seq)
            print('Unique sequence input shapes, accounting for batches')
            print_dict_sorted_by_val(hist_seq_with_batch)
        self.train_data_loader_len = len(dataloader)
        return dataloader
    def val_dataloader(self):
        # [TODO sasarkar] train_dataloader uses self.align_buckets , do we need it here?
        dataset = HuBERTDataSet(self.dataset_path, self.dataset, "valid")
        sampler = BucketizeBatchSampler(
            dataset.len_list,
            num_buckets=1000,
            max_token_count=self.seconds_per_batch * 16000,
            min_len=32000,
            max_len=250000,
            shuffle=False,
        )
        if torch.distributed.is_initialized():
            sampler = DistributedBatchSampler(sampler, shuffle= False, epoch=self.current_epoch)
            sampler.set_epoch(self.current_epoch)
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=CollateFnHubert(feature_type=self.feature_type, pad=False, rand_crop=True),
            num_workers=10,
        )
        return dataloader
    def on_save_checkpoint(self, checkpoint):
        if torch.distributed.get_rank() == 0 and self.save_1d_ckpt:
            checkpoint['optimizer_states'][0]=hpu_wav2vec2_optimizer_reshape(checkpoint['optimizer_states'][0], True) #change conv2d opt state to conv1d
            checkpoint['state_dict'] = hpu_wav2vec2_params_reshape(checkpoint['state_dict'], True) #change conv2d weights to conv1d
            self.ckpt = checkpoint
    def on_load_checkpoint(self, checkpoint):
        if self.save_1d_ckpt:
            checkpoint['state_dict'] = hpu_wav2vec2_params_reshape(checkpoint['state_dict'], False) #change conv1d weights to conv2d
            checkpoint['optimizer_states'][0] = hpu_wav2vec2_optimizer_reshape(checkpoint['optimizer_states'][0] , False) #change conv1d opt state to conv2d
    def on_train_epoch_end(self):
        if torch.distributed.get_rank() == 0 and self.save_1d_ckpt and self.ckpt is not None:
            self.ckpt['state_dict'] = hpu_wav2vec2_params_reshape(self.ckpt['state_dict'], False) #change conv1d weights to conv2d
            self.ckpt['optimizer_states'][0] = hpu_wav2vec2_optimizer_reshape(self.ckpt['optimizer_states'][0] , False) #change conv1d opt state to conv2d
            self.optimizer.load_state_dict(self.ckpt['optimizer_states'][0])

class HuBERTFineTuneModule(LightningModule):
    def __init__(
        self,
        *,
        model_name: str,
        encoder_projection_dropout: float,
        encoder_attention_dropout: float,
        encoder_ff_interm_dropout: float,
        encoder_dropout: float,
        encoder_layer_drop: float,
        mask_prob: float,
        mask_channel_prob: float,
        mask_channel_length: float,
        num_classes: int,
        aux_num_out: int,
        checkpoint: str,
        dataset_path: str,
        seconds_per_batch: float,
        subset: str,
        learning_rate: float,
        betas: Tuple[float, float],
        adam_eps: float,
        weight_decay: float,
        freeze_encoder_updates: int,
        warmup_updates: int,
        hold_updates: int,
        decay_updates: int,

    ):
        super().__init__()
        if model_name == "hubert_pretrain_base":
            self.model = models.hubert_pretrain_base(
                encoder_projection_dropout=encoder_projection_dropout,
                encoder_attention_dropout=encoder_attention_dropout,
                encoder_ff_interm_dropout=encoder_ff_interm_dropout,
                encoder_dropout=encoder_dropout,
                encoder_layer_drop=encoder_layer_drop,
                mask_prob=mask_prob,
                mask_channel_prob=mask_channel_prob,
                mask_channel_length=mask_channel_length,
                num_classes=num_classes,
            )
        elif model_name == "hubert_large":
            self.model = models.hubert_pretrain_large(
                encoder_projection_dropout=encoder_projection_dropout,
                encoder_attention_dropout=encoder_attention_dropout,
                encoder_ff_interm_dropout=encoder_ff_interm_dropout,
                encoder_dropout=encoder_dropout,
                encoder_layer_drop=encoder_layer_drop,
                mask_prob=mask_prob,
                mask_channel_prob=mask_channel_prob,
                mask_channel_length=mask_channel_length,
                num_classes=num_classes,
            )
        elif model_name == "hubert_xlarge":
            self.model = models.hubert_pretrain_xlarge(
                encoder_projection_dropout=encoder_projection_dropout,
                encoder_attention_dropout=encoder_attention_dropout,
                encoder_ff_interm_dropout=encoder_ff_interm_dropout,
                encoder_dropout=encoder_dropout,
                encoder_layer_drop=encoder_layer_drop,
                mask_prob=mask_prob,
                mask_channel_prob=mask_channel_prob,
                mask_channel_length=mask_channel_length,
                num_classes=num_classes,
            )
        else:
            raise ValueError(f"Unsupported model name: {model_name}.")
        self.aux = torch.nn.Linear(768, aux_num_out)
        self._load_checkpoint(checkpoint)
        for p in self.model.wav2vec2.feature_extractor.parameters():
            p.requires_grad = False
        self.loss_fn = torch.nn.CTCLoss(blank=0, reduction="sum", zero_infinity=True)
        self.optimizer = torch.optim.AdamW(
            list(self.aux.parameters()) + list(self.model.parameters()),
            lr=learning_rate,
            betas=betas,
            eps=adam_eps,
            weight_decay=weight_decay,
        )
        self.freeze_encoder_updates = freeze_encoder_updates
        self.lr_scheduler = TriStageLRScheduler(self.optimizer, warmup_updates, hold_updates, decay_updates)
        self.dataset_path = dataset_path
        self.seconds_per_batch = seconds_per_batch
        self.subset = subset
        self.automatic_optimization = False
        self.scaler = torch.cuda.amp.GradScaler()

    def _load_checkpoint(self, checkpoint):
        # load pretrain model from checkpoint
        state_dict = torch.load(checkpoint, map_location=torch.device("cpu"))
        state_dict = state_dict["state_dict"]
        state_dict = hpu_wav2vec2_params_reshape(state_dict, True) #change conv2d weights to conv1d if needed
        s = {}
        for k in state_dict:
            if "model." in k:
                s[k.replace("model.", "")] = state_dict[k]
        self.model.load_state_dict(s)
    def _step(self, batch: Batch_FineTune, batch_idx, step_type):
        if batch is None:
            return None
        waveforms, labels, audio_lengths, label_lengths = batch
        if self.global_step <= self.freeze_encoder_updates:
            with torch.no_grad():
                x, out_len = self.model.wav2vec2.feature_extractor(waveforms, audio_lengths)
                padding_mask = components._get_padding_mask(x, out_len)
                x, attention_mask = self.model.wav2vec2.encoder._preprocess(x, out_len)
                x, _ = self.model.mask_generator(x, padding_mask)
                x = self.model.wav2vec2.encoder.transformer(x, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                x, out_len = self.model.wav2vec2.feature_extractor(waveforms, audio_lengths)
                padding_mask = components._get_padding_mask(x, out_len)
            x, attention_mask = self.model.wav2vec2.encoder._preprocess(x, out_len)
            x, _ = self.model.mask_generator(x, padding_mask)
            x = self.model.wav2vec2.encoder.transformer(x, attention_mask=attention_mask)
        logits = self.aux(x)
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.transpose(0, 1)
        loss = self.loss_fn(
            log_probs,
            labels,
            out_len,
            label_lengths,
        )
        self.log(f"{step_type}_loss", loss.item() / waveforms.size(0), on_step=True, on_epoch=True)
        return loss
    def configure_optimizers(self):
        return (
            [
                self.optimizer,
            ],
            [
                {"scheduler": self.lr_scheduler, "interval": "step"},
            ],
        )
    def training_step(self, batch: Batch_FineTune, batch_idx):
        """Custom training step with loss normalization and automatic mixed precision training.
        By default, DDP does the following on each train step:
        - For each GPU, compute loss and gradient on shard of training data.
        - Sync and average gradients across all GPUs. The final gradient
          is (sum of gradients across all GPUs) / N, where N is the world
          size (total number of GPUs).
        - Update parameters on each GPU.
        Here, we do the following:
        - For k-th GPU, compute loss and scale it by (N / B_total), where B_total is
          the sum of batch sizes across all GPUs. Compute gradient from scaled loss.
        - Sync and average gradients across all GPUs. The final gradient
          is (sum of gradients across all GPUs) / B_total.
        - Update parameters on each GPU.
        Doing so allows us to account for the variability in batch sizes that
        variable-length sequential data commonly yields.
        """
        opt = self.optimizers()
        opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=False):
            loss = self._step(batch, batch_idx, "train")
        # normalize the loss based on the sum of batch_sie across all GPUs
        batch_size = batch[0].size(0)
        batch_sizes = self.all_gather(batch_size)
        self.log("Gathered batch size", batch_sizes.sum(), on_step=True, on_epoch=True)
        loss *= batch_sizes.size(0) / batch_sizes.sum()  # world size / batch size
        # backward the loss and clip the gradients
        loss = self.scaler.scale(loss)
        self.manual_backward(loss)
        self.scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        # optimization
        self.scaler.step(opt)
        sch = self.lr_schedulers()
        sch.step()
        self.scaler.update()
    def validation_step(self, batch: Batch_FineTune, batch_idx):
        return self._step(batch, batch_idx, "val")
    def train_dataloader(self):
        dataset = torchaudio.datasets.LibriLightLimited(self.dataset_path, self.subset)
        lengths = _get_lengths_librilightlimited(dataset._fileids_paths, dataset._path, dataset._ext_audio)
        sampler = BucketizeBatchSampler(
            lengths, num_buckets=100, max_token_count=self.seconds_per_batch * 16000, shuffle=True
        )
        sampler = DistributedBatchSampler(sampler, shuffle=True)
        sampler.set_epoch(self.global_step)
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=CollateFnLibriLightLimited(),
            num_workers=10,
        )
        return dataloader
    def val_dataloader(self):
        dataset = torchaudio.datasets.LIBRISPEECH(self.dataset_path, "dev-other")
        lengths = _get_lengths_librispeech(dataset._walker, dataset._path, dataset._ext_audio)
        sampler = BucketizeBatchSampler(
            lengths, num_buckets=100, max_token_count=self.seconds_per_batch * 16000, shuffle=False
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=CollateFnLibriLightLimited(),
            num_workers=10,
        )
        return dataloader
