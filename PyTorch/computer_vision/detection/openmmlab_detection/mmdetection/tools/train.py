# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash, register_hpuinfo

from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--hpus',
        type=int,
        default=0,
        choices=[0,1],
        help='0 to indicate non hpu run. 1 indicates hpu run ')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--lazy',
        action='store_true',
        help='Lazy mode for HPU')
    parser.add_argument(
        '--groundtruth-processing-on-hpu',
        action='store_true',
        help='Groundtruth processing on HPU')
    mixed_precision_group = parser.add_mutually_exclusive_group()
    mixed_precision_group.add_argument('--autocast', action='store_true', help='enable autocast mode on Gaudi')
    mixed_precision_group.add_argument('--hmp', action='store_true', help='enable hmp mode')
    parser.add_argument('--hmp-bf16', default='', help='path to bf16 ops list in hmp O1 mode')
    parser.add_argument('--hmp-fp32', default='', help='path to fp32 ops list in hmp O1 mode')
    parser.add_argument('--hmp-opt-level', default='O1', help='choose optimization level for hmp')
    parser.add_argument('--hmp-verbose', action='store_true', help='enable verbose mode for hmp')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--batch-size', default=None, type=int, help='train batch size')
    parser.add_argument('--eval-batch-size', default=None, type=int, help='eval batch size')
    parser.add_argument('--epochs', default=None, type=int, help='number of training epochs')
    parser.add_argument('--steps', default=None, type=int, help='number of training steps per epoch')
    parser.add_argument('--skip-epochs', default=1, type=int, help='number of training epochs to skip for perf calc')
    parser.add_argument('--log-interval', default=50, type=int, required=False, help='log interval')
    parser.add_argument('--bucket-cap-mb', default=250, type=int, required=False, help='bucket size for distributed run')

    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.hpu = {}
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume
    err_msg = 'Seems model is to run in CPU/GPU, but lazy_mode flag was set, which is applicable only for HPU'
    lazy_enabled = False
    hpu_enabled = False
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
        assert not args.lazy, err_msg
    else:
        # cfg.gpu_ids could refer to both hpu_ids or gpu_ids
        if args.hpus==0:
            cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
            assert not args.lazy, err_msg
            hpu_enabled = False
        else:
            assert args.hpus > 0
            cfg.gpu_ids = range(args.hpus)
            hpu_enabled = True
            if args.lazy:
                lazy_enabled = True

    register_hpuinfo(hpu_enabled, lazy_enabled, args.autocast, args.hmp, args.hmp_opt_level, args.hmp_bf16, args.hmp_fp32, args.hmp_verbose, not args.groundtruth_processing_on_hpu)
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(
                f'We treat {cfg.gpu_ids} as gpu-ids, and reset to '
                f'{cfg.gpu_ids[0:1]} as gpu-ids to avoid potential error in '
                'non-distribute training time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(args.launcher, args.bucket_cap_mb, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)


    # override other config option
    if args.batch_size:
        cfg.data.samples_per_gpu = args.batch_size
    if args.eval_batch_size:
        cfg.data.val.samples_per_gpu = args.eval_batch_size
    if args.epochs:
        cfg.runner.max_epochs = args.epochs
    if args.steps:
        os.environ.setdefault('DEV_MODE_EPOCH_STEP', str(args.steps))
    if args.log_interval:
        cfg.log_config.interval = args.log_interval
    if args.bucket_cap_mb:
        cfg.bucket_cap_mb = args.bucket_cap_mb

    os.environ.setdefault('DEV_MODE_SKIP_EPOCHS', str(args.skip_epochs))

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)

if __name__ == '__main__':
    main()
