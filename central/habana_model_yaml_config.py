###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
from typing import Dict, List
import yaml
from central.habana_model_runner_utils import get_canonical_path

# Derive from this class for model-specific handling of yaml config file parameters
class HabanaModelYamlConfig():
    default_num_workers_per_hls = 8
    def __init__(self, modelname, filename):
        self.model = modelname
        self.hb_config = filename
        self.parsed_config = None
        self.env_variables = None
        self.model_parameters = None
        self.model_parameters_store_true = None

        config_path = get_canonical_path(self.hb_config)
        if config_path.is_file() is False:
            raise OSError(f"hb_config has to be existing yaml file, but there is no file {config_path}")

        self.process_config_file(config_path)
        self.add_env_vars({'TF_DISABLE_MKL': '1'})

    def process_config_file(self, config_path):
        try:
            with open(config_path, "r") as hb_config:
                self.parsed_config = yaml.load(hb_config, Loader=yaml.FullLoader)
                print(self.parsed_config['model'])
                print(self.parsed_config.get('env_variables'))
                print(self.parsed_config['parameters'])

                if self.model != self.parsed_config['model']:
                    raise Exception(f"{self.hb_config} has a different model name \"{self.parsed_config['model']}\" than \"{self.model}\"")

                self.env_variables = self.parsed_config['env_variables'] \
                    if self.parsed_config.get('env_variables') is not None else {}

                self.model_parameters = self.parsed_config['parameters']
                if self.model_parameters.get('store_true'):
                    self.model_parameters_store_true = self.model_parameters.pop('store_true')
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} process_config_file()") from exc

    def get_env_vars(self):
        return self.env_variables

    def add_env_vars(self, new_vars:dict):
        self.env_variables.update(new_vars)

    def get_parameters(self):
        return self.model_parameters

    def get_parameter(self, key):
        if isinstance(self.model_parameters, dict):
            return self.model_parameters.get(key)
        return None

    # Derived classes can override how they want yaml sequences that contain list specifications, to be de-serialized
    def add_list_elem(self, key, list_value, command: List[str]):
        value = ","
        value = value.join(str(elem) for elem in list_value)
        command.extend([f"--{key}={value}"])

    # Adds leaf-level yaml config nodes as command-line options
    def add_hier_record(self, record_key, record_value, command: List[str]):
        if record_value is not None:
            if isinstance(record_value, dict):
                for key, value in record_value.items():
                    self.add_hier_record(key, value, command)
            elif isinstance(record_value, list):
                self.add_list_elem(record_key, record_value, command)
            else:
                command.extend([f"--{record_key}={record_value}"])

    def traverse(self, record_key, record_value, command: List[str], fields):
        if fields != []:
            if record_key == fields[0]:
                if len(fields) > 1:
                    fields = fields[1:]
                    if record_value[fields[0]] is not None:
                        if isinstance(record_value[fields[0]], dict) == False:
                            if len(fields) > 1:
                                raise Exception(f"Error in {self.__class__.__name__} traverse(): specified filter does not match yaml config")
                            else:
                                command.extend([f"--{fields[0]}={record_value[fields[0]]}"])
                        else:
                            for key, value in record_value[fields[0]].items():
                                if value is not None:
                                    rfields = fields[1:]
                                    self.traverse(key, value, command, rfields)
                elif record_value is not None:
                    self.add_hier_record(record_key, record_value, command)
            elif record_value is not None:
                self.add_hier_record(record_key, record_value, command)
        elif record_value is not None:
            self.add_hier_record(record_key, record_value, command)

    # Adds all parameters, with the exception of records that don't pass the filter, and top-level excluded keys.
    # filter is a "/" separated field-specifier to zoom into a particular record in the yaml config.
    # All records that this search resolves to, will be added.
    # Example usage: See TensorFlow/nlp/bert/bert_base_default.yaml and the filter 'dataset_parameters/mrpc/data_type_parameters/bf16'
    # used in 'HabanaBertModelRunner' in habana_model_runner.py.
    def add_parameters_with_filter(self, command: List[str], filter: str, exclude_fields: List[str]=None):
        try:
            if self.model_parameters is not None:
                fields = filter.split('/')
                for key, value in self.model_parameters.items():
                    if key not in exclude_fields and value is not None:
                        self.traverse(key, value, command, fields)
            if self.model_parameters_store_true is not None:
                for item in self.model_parameters_store_true:
                    command.extend([f"--{item}"])
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} add_parameters_with_filter()") from exc

    # Derived classes can override how 'use_horovod' along with 'num_workers_per_hls' is passed to the training script
    def add_horovod_parameter(self, command: List[str]):
        try:
            key = 'use_horovod'
            if self.model_parameters[key] == True:
                if self.model_parameters['num_workers_per_hls'] > 1:
                    command.extend([f"--{key}={self.model_parameters['num_workers_per_hls']}"])
                else:
                    command.extend([f"--{key}={default_num_workers_per_hls}"])
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} add_horovod_parameter()") from exc

    # Derive from HabanaModelYamlConfig to override
    def add_all_parameters(self, command: List[str]):
        try:
            # add all parameters to command
            if self.model_parameters is not None:
                for key, value in self.model_parameters.items():
                    if key == 'use_horovod':
                        if value == True:
                            if self.model_parameters['num_workers_per_hls'] > 1:
                                command.extend([f"--{key}={self.model_parameters['num_workers_per_hls']}"])
                            else:
                                command.extend([f"--{key}={default_num_workers_per_hls}"])
                        continue
                    if key == 'num_workers_per_hls':
                        continue
                    if value is not None:
                        self.add_hier_record(key, value, command)
            if self.model_parameters_store_true is not None:
                for item in self.model_parameters_store_true:
                    command.extend([f"--{item}"])
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} add_all_parameters()") from exc

    # 'include_fields' is a "/" separated field specifier to zoom into a particular record in the yaml config
    # All records that this search resolves to, will be added
    def add_specific_parameters(self, command: List[str], include_fields: str):
        try:
            if self.model_parameters is not None:
                fields = include_fields.split('/')
                record = self.model_parameters
                for field in fields:
                    record = record[field]
                for key, value in record.items():
                    if value is not None:
                        self.add_hier_record(key, value, command)
            if self.model_parameters_store_true is not None:
                for item in self.model_parameters_store_true:
                    command.extend([f"--{item}"])
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} add_specific_parameters()") from exc

    # exclude_fields is a list of fields to skip at the top-most parameter level
    def add_parameters_except(self, command: List[str], exclude_fields: List[str]):
        try:
            # add all parameters to command
            if self.model_parameters is not None:
                for key, value in self.model_parameters.items():
                    if key not in exclude_fields:
                        if value is not None:
                            self.add_hier_record(key, value, command)
            if self.model_parameters_store_true is not None:
                for item in self.model_parameters_store_true:
                    command.extend([f"--{item}"])
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} add_parameters_except()") from exc
