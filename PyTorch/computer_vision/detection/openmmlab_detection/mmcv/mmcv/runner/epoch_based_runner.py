# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
import os
import os.path as osp
import platform
import shutil
import time
import warnings

import torch

import mmcv
import numpy as np
from .base_runner import BaseRunner
from .builder import RUNNERS
from .checkpoint import save_checkpoint
from .utils import get_host_info
from mmcv.runner import get_dist_info

from mmcv.utils import is_hpu_enabled, is_autocast_enabled
dump_enable=os.getenv('TP_MODEL_PARAM_DUMP_ENABLE')
if dump_enable=="1":
    import sys
    sys.path.append(os.environ['PYTORCH_MODULES_ROOT_PATH'])
    from topologies import tools

@RUNNERS.register_module()
class EpochBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """
    def run_iter(self, data_batch, train_mode, **kwargs):
        with torch.autocast(device_type='hpu', dtype=torch.bfloat16, enabled=is_autocast_enabled()):
            if self.batch_processor is not None:
                outputs = self.batch_processor(
                    self.model, data_batch, train_mode=train_mode, **kwargs)
            elif train_mode:
                outputs = self.model.train_step(data_batch, self.optimizer,
                                                **kwargs)
            else:
                outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)

        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])

        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        ctr = 0
        start_t = time.time()
        device = torch.device('cpu')
        if is_hpu_enabled():
            device = torch.device('hpu:0')
        if dump_enable=="1":
            trainMetaData = tools.TrainMetaData(self.model, device)
        target = {}
        batch = 0
        rank, ws  = get_dist_info()
        for i, data_batch in enumerate(self.data_loader):
            from mmcv.utils import dev_mode_epoch_step
            if dev_mode_epoch_step() >= 0 and ctr >= dev_mode_epoch_step():
                break
            ctr+=1
            self._inner_iter = i
            if dump_enable=="1":
                target['gt_bboxes']=data_batch['gt_bboxes']
                target['gt_labels']=data_batch['gt_labels']
                tools.tp_probe_tensors_iteration_start(self.model, device, target, data_batch['img'], trainMetaData.ParamsDump, False)
            #print(i , " STEP START WITH ", data_batch['img'].data[0].shape)
            batch = int(data_batch['img'].data[0].shape[0])
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)

            self.call_hook('after_train_iter')
            self._iter += 1
            #print(i , " STEP DONE WITH ", data_batch['img'].data[0].shape)
            if dump_enable=="1":
                tools.tp_probe_tensors_iteration_end(self.model, device, self.outputs['loss'], self.outputs['loss'].item(), trainMetaData.ParamsDump, False)
                trainMetaData.increment_train_step()
        epoch_time = time.time() - start_t
        epoch_steps = dev_mode_epoch_step() if (dev_mode_epoch_step() >= 0 and  dev_mode_epoch_step() < len(self.data_loader)) \
            else len(self.data_loader)
        epoch_ips = (epoch_steps * batch * ws)/epoch_time
        self.epoch_ips.append(epoch_ips)
        self.epoch_time.append(epoch_time)
        self.call_hook('after_train_epoch')
        epoch_total_time = time.time() - start_t
        self.epoch_total_time.append(epoch_total_time)
        self._epoch += 1
        self.logger.info("Epoch %s Total time: %f, Total train time: %f, Total global steps: %d, Images per second: %f ", \
            self._epoch, epoch_total_time, epoch_time, epoch_steps*ws, epoch_ips)



    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')
        self.epoch_time =[]
        self.epoch_ips = []
        self.epoch_total_time =[]

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

        epoch_total_times = np.array(self.epoch_total_time)
        epoch_times = np.array(self.epoch_time)
        SKIP_EPOCHS = int(os.getenv('DEV_MODE_SKIP_EPOCHS'))
        epoch_ips =  self.epoch_ips[SKIP_EPOCHS:] if SKIP_EPOCHS > 0 and len(self.epoch_ips) > 1 else self.epoch_ips
        self.logger.info("Total Time: %f", epoch_total_times.sum())
        self.logger.info("Total Train Time: %f", epoch_times.sum())
        self.logger.info("Avg Epoch Time: %f", epoch_times.mean(-1))
        self.logger.info("Best Epoch Time: %f", epoch_times[epoch_times.argmin()])
        self.logger.info("Image Per Second: %f ", np.array(epoch_ips).mean(-1))

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)


@RUNNERS.register_module()
class Runner(EpochBasedRunner):
    """Deprecated name of EpochBasedRunner."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            'Runner was deprecated, please use EpochBasedRunner instead',
            DeprecationWarning)
        super().__init__(*args, **kwargs)
