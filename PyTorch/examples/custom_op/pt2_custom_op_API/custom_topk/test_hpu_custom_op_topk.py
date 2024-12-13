# ******************************************************************************
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ******************************************************************************

import torch
import pytest
import os
from pathlib import Path
import habana_frameworks.torch.core


is_lazy = os.environ.get("PT_HPU_LAZY_MODE", "1") == "1"


@pytest.mark.parametrize("compile", [True, False])
def test_custom_topk_op(compile):
    # torch.compile is not supported in lazy mode
    if is_lazy and compile:
        pytest.skip()

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
    custom_op = torch.ops.custom_op.custom_topk
    if compile:
        custom_op = torch.compile(custom_op, backend="hpu_backend")
    a_cpu = torch.rand((6, 6))
    a_hpu = a_cpu.to("hpu")
    a_topk_hpu, a_topk_indices_hpu = custom_op(a_hpu, 3, 1, False)
    a_topk_cpu, a_topk_indices_cpu = a_cpu.topk(3, 1)
    assert torch.equal(a_topk_hpu.detach().cpu(), a_topk_cpu.detach().cpu())
