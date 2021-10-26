###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
import sys
import argparse
import subprocess
from pathlib import Path
from central.training_run_config import TrainingRunHWConfig

print(f"{__file__} - Enter: PYTHONPATH = {os.environ.get('PYTHONPATH')}")

p1 = Path(f"{__file__}").parent.resolve()
p2 = p1.parent.resolve()
p3 = p1.resolve().parent

if os.environ.get('PYTHONPATH'):
    os.environ['PYTHONPATH'] = os.fspath(p2) + ":" + os.fspath(p3) + ":" + os.fspath(p1) + ":" + os.environ.get('PYTHONPATH')
else:
    os.environ['PYTHONPATH'] = os.fspath(p2) + ":" + os.fspath(p3) + ":" + os.fspath(p1)

print(f"{__file__} - Set up PYTHONPATH = {os.environ.get('PYTHONPATH')}")

from habana_model_yaml_config import HabanaModelYamlConfig
from habana_model_runner_utils import HabanaEnvVariables, print_env_info, get_canonical_path
from script_paths import get_script_path

SUPPORTED_MODELS = ['albert',
                    'bert',
                    'dlrm',
                    'efficientdet',
                    'maskrcnn',
                    'mobilenet_v2',
                    'resnet_estimator',
                    'resnet_keras',
                    'ssd_resnet34',
                    'unet2d',
                    'transformer',
                    'resnet50',
                    'pacman_segnet',
                    'vgg_segnet',
                    'densenet_keras']

class ModelHWConfig():
    def __init__(self, use_horovod, num_workers_per_hls, hls_type, kubernetes_run):
        self.use_horovod = False
        self.num_workers_per_hls = 1
        self.hls_type = "HLS1"
        self.kubernetes_run = False
        if use_horovod is not None:
            self.use_horovod = use_horovod
        if num_workers_per_hls is not None and num_workers_per_hls > 1:
            self.num_workers_per_hls = num_workers_per_hls
        if hls_type is not None:
            self.hls_type = hls_type
        if kubernetes_run is not None:
            self.kubernetes_run = kubernetes_run

class HabanaModelRunner():
    def __init__(self, args, unknown_args):
        try:
            self.args = args
            self.unknown_args = unknown_args
            self.framework = args.framework
            self.model = args.model
            self.command = []
            self.yaml_config = HabanaModelYamlConfig(self.model, self.args.hb_config)
        except Exception as exc:
            raise RuntimeError("Error constructing HabanaModelRunner") from exc

    def build_command(self):
        """Build full command that will be run as a subprocess."""
        try:
            model_path = get_script_path(self.framework, self.model)
            self.command = [sys.executable, str(model_path)]
            self.yaml_config.add_all_parameters(self.command)
            if self.unknown_args:
                for item in self.unknown_args:
                    self.command.append(item)
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

class HabanaResnetModelRunner(HabanaModelRunner):
    def __init__(self, args, unknown_args):
        super(HabanaResnetModelRunner, self).__init__(args, unknown_args)
        from TensorFlow.common.common import setup_preloading, setup_jemalloc
        setup_preloading()
        setup_jemalloc()

    def build_command(self):
        """Build full command that will be run as a subprocess."""
        try:
            model_params=self.yaml_config.get_parameters()
            resnet_hw_args = ModelHWConfig(model_params.get("use_horovod"), model_params.get("num_workers_per_hls"), model_params.get("hls_type"), model_params.get("kubernetes_run"))
            hw_config = TrainingRunHWConfig(resnet_hw_args.use_horovod, resnet_hw_args.num_workers_per_hls, resnet_hw_args.hls_type, resnet_hw_args.kubernetes_run, "demo_resnet")
            self.command = list(hw_config.mpirun_cmd.split(" ")) + [sys.executable, str(get_script_path(self.framework, self.model))]
            exclude_fields = ['use_horovod', 'num_workers_per_hls', 'hls_type', 'kubernetes_run']
            self.yaml_config.add_parameters_except(self.command, exclude_fields)
            if self.unknown_args:
                for item in self.unknown_args:
                    self.command.append(item)
            if self.yaml_config.get_parameter("use_horovod"):
                self.command.append("--use_horovod")
            print('self.command = ', self.command)
        except Exception as exc:
            raise RuntimeError("Error in HabanaResnetModelRunner.build_command()") from exc


class HabanaBertModelRunner(HabanaModelRunner):
    def __init__(self, args, unknown_args):
        super(HabanaBertModelRunner, self).__init__(args, unknown_args)

    def build_command(self):
        """Build full command for BERT that will be run as a subprocess."""
        try:
            model_path = get_script_path(self.framework, self.model)
            self.command = [sys.executable, str(model_path)]
            filter = 'dataset_parameters' + '/' + self.yaml_config.get_parameter('test_set') + \
                     '/' + 'data_type_parameters' + '/' + self.yaml_config.get_parameter('data_type')
            exclude_fields = ['use_horovod', 'num_workers_per_hls']
            self.yaml_config.add_parameters_with_filter(self.command, filter, exclude_fields)
            self.yaml_config.add_horovod_parameter(self.command)
        except Exception as exc:
            raise RuntimeError("Error in HabanaBertModelRunner.build_command()") from exc


class HabanaAlbertModelRunner(HabanaModelRunner):
    def __init__(self, args, unknown_args):
        super(HabanaAlbertModelRunner, self).__init__(args, unknown_args)

    def build_command(self):
        """Build full command for ALBERT that will be run as a subprocess."""
        try:
            model_path = get_script_path(self.framework, self.model)
            self.command = [sys.executable, str(model_path)]
            filter = 'dataset_parameters' + '/' + self.yaml_config.get_parameter('test_set') + \
                     '/' + 'data_type_parameters' + '/' + self.yaml_config.get_parameter('data_type')
            exclude_fields = ['use_horovod', 'num_workers_per_hls']
            self.yaml_config.add_parameters_with_filter(self.command, filter, exclude_fields)
            self.yaml_config.add_horovod_parameter(self.command)
        except Exception as exc:
            raise RuntimeError("Error in HabanaAlbertModelRunner.build_command()") from exc

class HabanaMaskrcnnModelRunner(HabanaModelRunner):
    def __init__(self, args, unknown_args):
        super(HabanaMaskrcnnModelRunner, self).__init__(args, unknown_args)
    def build_command(self):
        """Build full command for Mask RCNN that will be run as a subprocess."""
        try:
            model_path = get_script_path(self.framework, self.model)
            self.command = [sys.executable, str(model_path)]
            if self.yaml_config.get_parameter('mode'):
                self.command.append(self.yaml_config.get_parameter('mode'))
            exclude_fields = ['mode']
            self.yaml_config.add_parameters_except(self.command, exclude_fields)
        except Exception as exc:
            raise RuntimeError("Error in HabanaAlbertModelRunner.build_command()") from exc

class HabanaPTBertModelRunner(HabanaModelRunner):
    def __init__(self, args, unknown_args):
        super(HabanaPTBertModelRunner, self).__init__(args, unknown_args)

    def build_command(self):
        """Build full command for BERT that will be run as a subprocess."""

        try:
            model_path = get_script_path(self.framework, self.model)
            self.command = [sys.executable, str(model_path)]
            # get command value: pretraining or finetuning
            cmd_key = 'command'
            cmd_value = self.yaml_config.get_parameter(cmd_key)
            self.command.extend([cmd_value])
            exclude_fields = [cmd_key]
            # construct args
            filter = 'dataset_parameters' + '/' + self.yaml_config.get_parameter('task_name') + \
                     '/' + 'data_type_parameters' + '/' + self.yaml_config.get_parameter('data_type')
            self.yaml_config.add_parameters_with_filter(self.command, filter, exclude_fields)
            import re
            self.command = [re.sub(r"[=,]", " ", cmd) for cmd in self.command]
        except Exception as exc:
            raise RuntimeError("Error in HabanaPTBertModelRunner.build_command()") from exc


class HabanaSegnetModelRunner(HabanaModelRunner):
    def __init__(self, args, unknown_args):
        super(HabanaSegnetModelRunner, self).__init__(args, unknown_args)

    def build_command(self):
        """Build full command that will be run as a subprocess."""
        try:
            model_path = get_script_path(self.framework, self.model)
            self.command = [sys.executable, str(model_path)]
            if self.unknown_args:
                for item in self.unknown_args:
                    self.command.append(item)
            self.yaml_config.add_all_parameters(self.command)
        except Exception as exc:
            raise RuntimeError("Error in HabanaSegnetModelRunner.build_command()") from exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", type=str, required=True, choices=['tensorflow', 'pytorch'], help="Framework")
    parser.add_argument("--model", type=str, required=True, choices=SUPPORTED_MODELS, help="Model name")
    parser.add_argument("--hb_config", type=str, required=True, help="Absolute or relative path to a yaml file with habana config")
    known_args, unknown_args = parser.parse_known_args()
    model_factory = {
        'tensorflow': {
            'albert': HabanaAlbertModelRunner,
            'bert': HabanaBertModelRunner,
            'resnet_estimator': HabanaResnetModelRunner,
            'resnet_keras': HabanaResnetModelRunner,
            'maskrcnn': HabanaMaskrcnnModelRunner,
            'vgg_segnet': HabanaSegnetModelRunner
        },
        'pytorch': {
            'bert': HabanaPTBertModelRunner
        }
    }
    model_runner_class = model_factory.get(known_args.framework).get(known_args.model, HabanaModelRunner)
    model_runner = model_runner_class(known_args, unknown_args)
    model_runner.run()


if __name__ == "__main__":
    main()
