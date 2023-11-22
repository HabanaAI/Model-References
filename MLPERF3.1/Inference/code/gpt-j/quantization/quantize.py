###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################

from quantization.configuration import config as cfg
import torch

_HBQ_STR = 'HB_QUANTIZATION'

def setup_quantization(model, quantization_config_file):
    quant_config = cfg.parse_quant_config(quantization_config_file)
    model._buffers[_HBQ_STR] = dict()
    # set quantization buffer as non-persistent to avoid const marking from handling it
    model._non_persistent_buffers_set.add(_HBQ_STR)
    if quant_config.backoff_factor :
        model._buffers[_HBQ_STR][cfg.CFGS.BACKOFF_FACTOR] = torch.tensor(quant_config.backoff_factor)
    if quant_config.quantization_enabled:
        print("Initializing GPT-J inference with quantization")
        model._buffers[_HBQ_STR][cfg.CFGS.QUANTIZATION] = True
        import habana_frameworks.torch.core as htcore
        htcore.hpu_initialize(model)
        try :
            htcore.quantization._mark_params_as_const(model)
            print("Params were marked as const")
        except AttributeError :
            print("Const marking not supported")
            pass
    return model