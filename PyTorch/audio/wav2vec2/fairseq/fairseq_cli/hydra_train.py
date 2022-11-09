#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from fairseq import distributed_utils, metrics
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.initialize import add_defaults, hydra_init
from fairseq.dataclass.utils import omegaconf_no_object_check
from fairseq.utils import reset_logging
from fairseq_cli.train import main as pre_main

logger = logging.getLogger("fairseq_cli.hydra_train")


@hydra.main(config_path=os.path.join("..", "fairseq", "config"), config_name="config")
def hydra_main(cfg: FairseqConfig) -> float:
    _hydra_main(cfg)


def _hydra_main(cfg: FairseqConfig, **kwargs) -> float:
    add_defaults(cfg)

    if cfg.common.reset_logging:
        reset_logging()  # Hydra hijacks logging, fix that
    else:
        # check if directly called or called through hydra_main
        if HydraConfig.initialized():
            with open_dict(cfg):
                # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
                cfg.job_logging_cfg = OmegaConf.to_container(
                    HydraConfig.get().job_logging, resolve=True
                )

    with omegaconf_no_object_check():
        cfg = OmegaConf.create(
            OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
        )
    OmegaConf.set_struct(cfg, True)

    try:
        if cfg.common.profile:
            with torch.cuda.profiler.profile():
                with torch.autograd.profiler.emit_nvtx():
                    distributed_utils.call_main(cfg, pre_main, **kwargs)
        else:
            distributed_utils.call_main(cfg, pre_main, **kwargs)
    except BaseException as e:
        if not cfg.common.suppress_crashes:
            raise
        else:
            logger.error("Crashed! " + str(e))

    # get best val and return - useful for sweepers
    try:
        best_val = metrics.get_smoothed_value(
            "valid", cfg.checkpoint.best_checkpoint_metric
        )
    except:
        best_val = None

    if best_val is None:
        best_val = float("inf")

    return best_val

def remove_local_rank(argv, unknown_args):
    for unknown_arg in unknown_args:
        if str(unknown_arg).startswith('--local_rank'):
            index = argv.index(unknown_arg)
            argv.pop(index)
            while len(argv) > index and not (str(argv[index]).startswith('--') or  str(argv[index]).startswith('-')) and argv[index] in unknown_args:
                argv.pop(index)

    return argv

def cli_main():
    try:
        from hydra._internal.utils import get_args_parser

        parser = get_args_parser()
        _, unknown_args = parser.parse_known_args(None, None)
        if unknown_args:
            parser.add_argument('--local_rank', type=int, default=0, help='local rank')
        args = parser.parse_args(None)

        cfg_name = args.config_name or "config"
        local_rank = '-1'
        if os.getenv('LOCAL_RANK', '-1') != '-1':
            local_rank = os.environ['LOCAL_RANK']
        if unknown_args:
            sys.argv = remove_local_rank(sys.argv, unknown_args)
            local_rank = str(args.local_rank)
        if local_rank != '-1':
            sys.argv.insert(1, 'distributed_training.device_id='+local_rank)
    except:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "config"

    hydra_init(cfg_name)
    hydra_main()


if __name__ == "__main__":
    cli_main()
