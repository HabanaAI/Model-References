###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
###############################################################################

import os
import sys
import argparse
import subprocess
from pathlib import Path

print(f"{__file__} - Enter: PYTHONPATH = {os.environ.get('PYTHONPATH')}")

p1 = Path(f"{__file__}").parent.resolve()
p2 = p1.parent.resolve()
p3 = p1.resolve().parent

if os.environ.get('PYTHONPATH'):
    os.environ['PYTHONPATH'] = os.fspath(p2) + ":" + os.fspath(p3) + ":" + os.fspath(p1) + ":" + os.environ.get('PYTHONPATH')
else:
    os.environ['PYTHONPATH'] = os.fspath(p2) + ":" + os.fspath(p3) + ":" + os.fspath(p1)

print(f"{__file__} - Set up PYTHONPATH = {os.environ.get('PYTHONPATH')}")

from common.habana_model_yaml_config import HabanaModelYamlConfig
from common.habana_model_runner_utils import HabanaEnvVariables, print_env_info, get_script_path, get_canonical_path

class HabanaModelRunner():
    def __init__(self, args):
        try:
            self.args = args
            self.model = args.model
            self.command = []
            self.yaml_config = HabanaModelYamlConfig(self.model, self.args.hb_config)
        except Exception as exc:
            raise RuntimeError("Error constructing HabanaModelRunner") from exc

    def build_command(self):
        """Build full command that will be run as a subprocess."""
        try:
            model_path = get_script_path(self.model)
            self.command = ["python3", str(model_path)]
            self.yaml_config.add_all_parameters(self.command)
        except Exception as exc:
            raise RuntimeError("Error in HabanaModelRunner.build_command()") from exc

    def run(self):
        try:
            self.build_command()
            with HabanaEnvVariables(env_vars_to_set=self.yaml_config.get_env_vars()):
                print_env_info(self.command, self.yaml_config.get_env_vars())
                command_for_proc = " "
                command_for_proc = command_for_proc.join(self.command)
                print('command_for_proc = ', command_for_proc)
                sys.stdout.flush()
                sys.stderr.flush()
                with subprocess.Popen(command_for_proc, shell=True, executable='/bin/bash') as proc:
                    proc.wait()
        except Exception as exc:
            raise RuntimeError("Error in HabanaModelRunner.run()") from exc

class HabanaBertModelRunner(HabanaModelRunner):
    def __init__(self, args):
        super(HabanaBertModelRunner, self).__init__(args)

    def build_command(self):
        """Build full command for BERT that will be run as a subprocess."""
        try:
            model_path = get_script_path(self.model)
            self.command = ["python3", str(model_path)]
            filter = 'dataset_parameters' + '/' + self.yaml_config.get_parameter('test_set') + \
                     '/' + 'data_type_parameters' + '/' + self.yaml_config.get_parameter('data_type')
            exclude_fields = ['use_horovod', 'num_workers_per_hls']
            self.yaml_config.add_parameters_with_filter(self.command, filter, exclude_fields)
            self.yaml_config.add_horovod_parameter(self.command)
        except Exception as exc:
            raise RuntimeError("Error in HabanaBertModelRunner.build_command()") from exc

class HabanaMaskRCNNModelRunner(HabanaModelRunner):

    def build_command(self):
        """Build full command for Mask that will be run as a subprocess."""
        try:
            model_path = get_script_path(self.model)
            self.command = ["python3", str(model_path), "train"]
            self.yaml_config.add_all_parameters(self.command)
        except Exception as exc:
            raise RuntimeError("Error in HabanaMaskRCNNModelRunner.build_command()") from exc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=['bert', 'resnet_estimator', 'resnet_keras', "maskrcnn", "ssd_resnet34"], help="Model name")
    parser.add_argument("--hb_config", type=str, required=True, help="Absolute or relative path to a yaml file with habana config.")
    args = parser.parse_args()
    model_factory = {
        'bert': HabanaBertModelRunner,
        'maskrcnn': HabanaMaskRCNNModelRunner,
    }
    model_runner_class = model_factory.get(args.model, HabanaModelRunner)
    model_runner = model_runner_class(args)
    model_runner.run()

if "__main__" == __name__:
    main()
