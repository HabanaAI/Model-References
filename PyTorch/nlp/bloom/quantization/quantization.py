from quantization.configuration import config as cfg
from quantization.measurements import measure_utils as measure_utils

def parse_configuration(json_file_path : str) -> cfg.QuantConfig :
    return cfg.parse_quant_config(json_file_path)

def apply_quantization(model, quant_config: cfg.QuantConfig) :
    if quant_config.measurements_path :
        measure_utils.load_measurements_to_model_from_path(model,
                                                           quant_config.measurements_path)
