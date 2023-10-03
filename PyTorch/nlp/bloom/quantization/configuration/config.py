import json
from os import path

# Configuration Aux strings
class CFGS:
    ON = "on"
    OFF = "off"
    QUANTIZATION = "quantization"
    MEASUREMENTS_PATH = "measurements_path"

# QuantConfig class
class QuantConfig:
    def __init__(self):
        self._quantization_enabled = False
        self._measurements_path = ""

    @property
    def quantization_enabled(self):
        return self._quantization_enabled

    @quantization_enabled.setter
    def quantization_enabled(self, val):
        self._quantization_enabled = val

    @property
    def measurements_path(self):
        return self._measurements_path

    @measurements_path.setter
    def measurements_path(self, path):
        self._measurements_path = path

def parse_quant_config(json_file_path : str) -> QuantConfig:
    quant_config = QuantConfig()
    assert path.isfile(json_file_path), "Quantization configuration file not found. Path - {}".format(
            json_file_path)
    with open(json_file_path, 'r') as f:
        quant_cfg_json = json.load(f)
        if CFGS.QUANTIZATION in quant_cfg_json and quant_cfg_json[CFGS.QUANTIZATION] == CFGS.ON:
            quant_config.quantization_enabled = True
            if CFGS.MEASUREMENTS_PATH in quant_cfg_json:
                measurements_path = quant_cfg_json[CFGS.MEASUREMENTS_PATH]
                if '$' in measurements_path :
                    print("Env var detected in path, expanding it")
                    measurements_path = path.expandvars(measurements_path)
                quant_config.measurements_path = measurements_path

    return quant_config