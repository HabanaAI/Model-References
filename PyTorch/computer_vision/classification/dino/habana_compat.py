###############################################################################
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
###############################################################################

import os

is_mark_step_enabled = False

def setup_hpu(args):
    global is_mark_step_enabled
    if args.device == 'hpu':
        if args.lazy_mode:
            is_mark_step_enabled = True
        else:
            os.environ["PT_HPU_LAZY_MODE"] = "2"
        import habana_frameworks.torch.hpu as hthpu
        hthpu.enable_dynamic_shape()
        import habana_frameworks.torch.core as htcore


def mark_step():
    if is_mark_step_enabled:
        import habana_frameworks.torch.core as htcore
        htcore.mark_step()
