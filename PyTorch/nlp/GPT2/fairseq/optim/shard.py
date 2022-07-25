# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict
from fairseq.distributed import utils

try:
    from fairscale.optim import OSS
    _has_fairscale = True
except ImportError:
    OSS = None
    _has_fairscale = False

try:
    from torch.distributed.optim import ZeroRedundancyOptimizer as ZRO
    _has_zero_redundancy_optimizer = True
except ImportError:
    ZRO = None
    _has_zero_redundancy_optimizer = False


def shard_(optimizer, group):
    if not _has_fairscale and not _has_zero_redundancy_optimizer:
        raise ImportError(
            "\n\nPlease install the fairscale package:" "\n\n  pip install fairscale"
            "\n\nOr upgrade PyTorch:" "\n\n  pip install torch --upgrade"
        )

    # for backward compatibility, prefer OSS
    oss_class = OSS if _has_fairscale else ZRO

    class FairseqOSS(oss_class):
        @property
        def disable_mem_eff_fp16_loading_hack(self):
            return True

        def __getattr__(self, name):
            if name.startswith("supports") and hasattr(self.optim, name):
                return getattr(self.optim, name)
            raise AttributeError(
                "'FairseqOSS' object has no attribute {0!r}".format(name)
            )

        def broadcast_global_state_dict(
            self, state_dict: Dict[str, Any]
        ) -> Dict[str, Any]:
            """
            Broadcasts the entire state_dict to all other ranks
            each rank is responsible to load their own partition of data
            """
            return utils.broadcast_object(
                state_dict,
                src_rank=0,
                group=self.group,
            )

    torch_optimizer = optimizer.optimizer
    optim_cls = type(torch_optimizer)

    if _has_fairscale:
        optimizer.optimizer = FairseqOSS(
            torch_optimizer.param_groups,
            optim_cls,
            group=group,
            **optimizer.optimizer_config
        )
    else:
        import inspect
        spec = inspect.getfullargspec(FairseqOSS.__init__)
        if 'group' in spec.args:
            optimizer.optimizer = FairseqOSS(
                list(optimizer.params),
                optim_cls,
                group=group,
                parameters_as_bucket_view=True,
                **optimizer.optimizer_config
            )
        else:
            optimizer.optimizer = FairseqOSS(
                list(optimizer.params),
                optim_cls,
                process_group=group,
                parameters_as_bucket_view=True,
                **optimizer.optimizer_config
            )
