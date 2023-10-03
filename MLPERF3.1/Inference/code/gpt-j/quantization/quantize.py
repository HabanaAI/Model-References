###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################

from quantization.configuration import config as cfg

def setup_quantization(model, quantization_config_file):
    quant_config = cfg.parse_quant_config(quantization_config_file)
    if quant_config.quantization_enabled:
        print("Initializing GPT-J inference with quantization")
        import habana_frameworks.torch.core as htcore
        htcore.hpu_initialize(model)
        try :
            htcore.quantization._mark_params_as_const(model)
            print("Params were marked as const")
        except AttributeError :
            print("Const marking not supported")
            pass
    return model