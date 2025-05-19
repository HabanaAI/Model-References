# ******************************************************************************
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ******************************************************************************

import os; os.environ["PT_HPU_LAZY_MODE"] = "1"
import torch
import habana_frameworks.torch.core


def test_custom_div_op_function():
    import os
    from pathlib import Path

    my_dir = os.path.realpath(__file__)
    my_len = my_dir.rfind("/")
    base_dir = my_dir[:my_len]

    custom_op_lib_path = str(
        next(
            Path(
                next(Path(os.path.join(base_dir, "build")).glob("lib.linux-x86_64-*"))
            ).glob("hpu_custom_topk.cpython-*-x86_64-linux-gnu.so")
        )
    )
    torch.ops.load_library(custom_op_lib_path)
    print(torch.ops.custom_op.custom_topk)
    a_cpu = torch.rand((6, 6))
    a_hpu = a_cpu.to("hpu")
    a_topk_hpu, a_topk_indices_hpu = torch.ops.custom_op.custom_topk(a_hpu, 3, 1, False)
    a_topk_cpu, a_topk_indices_cpu = a_cpu.topk(3, 1)
    assert torch.equal(a_topk_hpu.detach().cpu(), a_topk_cpu.detach().cpu())
