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
        import habana_frameworks.torch.core
        if args.precision == "hmp":
            from habana_frameworks.torch.hpex import hmp
            hmp.convert(opt_level=args.hmp_opt_level, bf16_file_path=args.hmp_bf16,
                        fp32_file_path=args.hmp_fp32, isVerbose=args.hmp_verbose)


def mark_step():
    if is_mark_step_enabled:
        import habana_frameworks.torch.core as htcore
        htcore.mark_step()
