import torch
from habana_frameworks.torch.utils.library_loader import load_habana_module
load_habana_module()

def test_custom_div_op_function(custom_op_lib_path):
    torch.ops.load_library(custom_op_lib_path)
    print(torch.ops.custom_op.custom_topk)
    a_cpu = torch.rand((6, 6))
    a_hpu = a_cpu.to('hpu')
    a_topk_hpu, a_topk_indices_hpu = torch.ops.custom_op.custom_topk(a_hpu, 3, 1, False)
    a_topk_cpu, a_topk_indices_cpu = a_cpu.topk(3, 1)
    assert(torch.equal(a_topk_hpu.detach().cpu(), a_topk_cpu.detach().cpu()))