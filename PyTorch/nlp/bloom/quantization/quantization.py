from quantization.configuration import config as cfg
from quantization.measurements import measure_utils as measure_utils
import torch


_HBQ_STR = 'HB_QUANTIZATION'

def parse_configuration(json_file_path : str) -> cfg.QuantConfig :
    return cfg.parse_quant_config(json_file_path)

def apply_quantization(model, quant_config: cfg.QuantConfig) :
    """
    apply quantization configuration into the model
    """
    model._buffers[_HBQ_STR] = dict()
    # set quantization buffer as non-persistent to avoid const marking from handling it
    model._non_persistent_buffers_set.add(_HBQ_STR)
    if quant_config.quantization_enabled :
        model._buffers[_HBQ_STR][cfg.CFGS.QUANTIZATION] = True
    if quant_config.backoff_factor :
        model._buffers[_HBQ_STR][cfg.CFGS.BACKOFF_FACTOR] = torch.tensor(quant_config.backoff_factor)
    if quant_config.measurements_path :
        measure_utils.load_measurements_to_model_from_path(model,
                                                           quant_config.measurements_path)