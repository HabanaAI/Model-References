###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import argparse
import os
import sys
import copy
model_garden_path = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../../.."))
os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + ":" +  model_garden_path
sys.path.append(model_garden_path)
from PyTorch.common.training_runner import TrainingRunner


class BertParams(object):
    """ class for handling BERT parameters and env vars"""

    def __init__(self, **kwargs):
        self._parser = argparse.ArgumentParser(**kwargs)
        self._subparsers = None
        self._d_sub_parsers = {}
        self._opt_arg_dict = {}
        self._args = None
        self._sub_command = None
        self._script_params = {}
        self._script_header = None
        self._script_footer = None
        # collection of env vars specific for bert
        self._env_vars = {}

    def copy(self):
        new_params = copy.copy(self)
        new_params._script_params = self._script_params.copy()
        return new_params

    def set_script_header(self, v):
        self._script_header = v

    def set_script_footer(self, v):
        self._script_footer = v

    # Validate if sub_parser is correctly defined for this sub command
    def check_sub_parser(self, sub_cmd=None):
        sub_cmd = sub_cmd if sub_cmd else self._sub_command
        assert sub_cmd in self._d_sub_parsers, \
            f"FATAL: sub_command={sub_cmd} defined but parser not initialized"

    def set_sub_command(self, cmd):
        self._sub_command = cmd

    # Add a sub command
    def add_sub_command(self, sub_cmd, sub_cmd_help=None):
        sub_cmd_help = f"{sub_cmd}" if sub_cmd_help is None else sub_cmd_help
        if self._subparsers is None:
            self._subparsers = self._parser.add_subparsers()
        self._d_sub_parsers[sub_cmd] = self._subparsers.add_parser(sub_cmd, help=sub_cmd_help)
        self._d_sub_parsers[sub_cmd].set_defaults(which=sub_cmd)
        self._sub_command = sub_cmd

    def get_subparser(self):
        return self._args.which if hasattr(self._args, 'which') else None

    # Add argument for current sub command
    def add_argument(self, *args, **kwargs):
        # Get appropriate parser
        if self._sub_command is None:
            parser_t = self._parser
        else:
            self.check_sub_parser(self._sub_command)
            parser_t = self._d_sub_parsers[self._sub_command]
        # Add argument to the corresponding parser
        act_obj = parser_t.add_argument(*args, **kwargs)
        # Store intermediate data
        self._opt_arg_dict[act_obj.dest] = None
        return act_obj

    # Parse all arguments
    def parse_args(self):
        self._args = self._parser.parse_args()
        # Do some house keeping
        for k_t in self._opt_arg_dict.keys():
            if k_t in dir(self._args):
                setattr(self, k_t, self._args.__dict__[k_t])
        return self._args

    def get(self, key, value=None):
        return value if key not in self.__dict__ else self[key]

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __add__(self, args):
        if isinstance(args, list):
            for item_t in args:
                if isinstance(item_t, list) or isinstance(item_t, tuple):
                    self._script_params[item_t[0]] = item_t[1]
                else:
                    self._script_params[item_t] = None
        elif isinstance(args, tuple):
            self._script_params[args[0]] = args[1]
        elif isinstance(args, dict):
            for k_t, v_t in args.items():
                self._script_params[k_t] = v_t
        elif args != '' and args is not None:
            self._script_params[args] = None
        return self

    def build_cmd_line(self, filters=[]):
        cmd = self._script_header
        for k, v in self._script_params.items():
            if k not in filters:
                cmd += f" {k}={v}" if v is not None else f" {k}"
        return cmd

    def add_env(self, key, value):
        self._env_vars[key] = value

    def get_env_vars(self):
        return self._env_vars


def args_bert_common(params: BertParams):
    """Define BERT common parameters"""

    # required args
    params.add_argument('--model_name_or_path', required=True, type=str,
                        default='base', choices=['base', 'large', 'roberta-base', 'roberta-large', 'albert-large', 'albert-xxlarge', 'electra-large-d'], help='Model name')
    params.add_argument('--data_type', required=True, type=str, default='fp32', choices=['fp32', 'bf16'],
                        help='Data type, possible values: fp32, bf16')
    # optional args
    params.add_argument('-o', '--output_dir', type=str, default='/tmp', help='Output directory')
    params.add_argument('--mode', type=str, default=None, choices=['eager', 'graph', 'lazy'], help='Execution mode')
    params.add_argument('-d', '--device', type=str, default='hpu', help='Device on which to run on')
    params.add_argument('--cache_dir', type=str, default=None, help='Cache directory for bert.')
    params.add_argument('--dist', action='store_true', help='Distribute training')
    params.add_argument('--world_size', type=int, default=1, help='Training device size')
    params.add_argument('--no_mpirun', action='store_true', help='Do not use MPI for distribute training')
    params.add_argument('--process_per_node', type=int, default=0, metavar='N', help='Number of processes per node')
    return params

def args_bert_finetuning(params: BertParams):
    """ Define BERT fine-tuning specific parameters"""

    params.add_argument('--dataset_name', type=str, default='mrpc', help='Dataset name')
    params.add_argument('-t', '--task_name', choices=['mrpc', 'squad'], type=str, default='mrpc', help='Task name')
    params.add_argument('-r', '--learning_rate', type=float, default=2e-5, help='Learning rate')
    params.add_argument('-s', '--max_seq_length', type=int, default=128, help='Max seq length')
    params.add_argument('-b', '--batch_size', type=int, default=8, help='Train batch size per device')
    params.add_argument('-v', '--per_device_eval_batch_size', type=int, default=8, help='Eval batch size per device')
    params.add_argument('-e', '--num_train_epochs', type=int, default=1, help='Number of Training epochs')
    params.add_argument('-l', '--logging_steps', type=int, default=1, help='Number of logging steps')
    params.add_argument('--max_steps', type=int, default=None, help='Maximum training steps')
    params.add_argument('--do_eval', action='store_true', help='Enable evaluation')
    params.add_argument('-ds', '--doc_stride', type=int, default=128, help='Used with SQUAD only')
    params.add_argument('-st', '--save_steps', type=int, default=None, help='Save steps')
    params.add_argument('--seed', type=int, default=42, help='Seed value')
    return params


def build_path_dict():
    real_path = os.path.realpath(__file__)
    demo_dir_path = os.path.dirname(real_path)
    transformer_path = os.path.join(demo_dir_path, 'transformers')

    return {
        'real_path': real_path,
        'demo_dir_path': demo_dir_path,
        'demo_config_path': demo_dir_path,
        'transformer_dir_path': transformer_path,
    }


def build_bert_envs(params, paths):
    # add mpirun env vars
    if not params.get('no_mpirun'):
        params.add_env('MASTER_ADDR', 'localhost')  # modify this with hostfile for multi-hls
        params.add_env('MASTER_PORT', '12345')


def check_world_size(pars):
    world_size = pars.get('world_size', 1)
    if pars.get('dist') and world_size == 1:
        pars.world_size = 8
    return pars.world_size


def bert_finetuning(params, paths):
    """ Construct fine-tuning command """

    check_world_size(params)

    if params.data_type == 'bf16':
        params += ['--hmp']
        if params.model_name_or_path == 'roberta-base':
            params += [('--hmp_bf16', f"{paths['demo_config_path']}/ops_bf16_roberta.txt")]
            params += [('--hmp_fp32', f"{paths['demo_config_path']}/ops_fp32_roberta.txt")]
        elif params.model_name_or_path == 'electra-large-d':
            params += [('--hmp_bf16', f"{paths['demo_config_path']}/ops_bf16_electra.txt")]
            params += [('--hmp_fp32', f"{paths['demo_config_path']}/ops_fp32_electra.txt")]
        else:
            params += [('--hmp_bf16', f"{paths['demo_config_path']}/ops_bf16_bert.txt")]
            params += [('--hmp_fp32', f"{paths['demo_config_path']}/ops_fp32_bert.txt")]

    if params.task_name.lower() == 'mrpc':
        script_path = 'examples/pytorch/text-classification/run_glue.py'
        params += ('--task_name', params.task_name.upper())
    elif params.task_name.lower() == 'squad':
        script_path = 'examples/pytorch/question-answering/run_qa.py'
        params += ('--doc_stride', params.doc_stride)

    output_dir = os.path.join(params.output_dir, params.task_name)
    build_mode_args(params)
    params += ('--per_device_train_batch_size', params.batch_size)
    params += ('--per_device_eval_batch_size', params.per_device_eval_batch_size)
    params += ('--dataset_name', params.dataset_name)
    params += ['--use_fused_adam']
    params += ['--use_fused_clip_norm']
    params += ('--max_steps', params.max_steps) if params.max_steps else None
    params += ('--cache_dir', params.cache_dir) if params.cache_dir else None
    params += ('--save_steps', params.save_steps) if params.save_steps else None
    params += '--use_habana' if params.device == 'hpu' else '--no_cuda'
    params += ('--max_seq_length', params.max_seq_length)
    params += ('--learning_rate', params.learning_rate)
    params += ('--num_train_epochs', params.num_train_epochs)
    params += ('--output_dir', output_dir)
    params += ('--logging_steps', params.logging_steps)
    params += ('--seed', params.seed)
    params += '--overwrite_output_dir'
    params += '--do_train'
    params += '--do_eval' if params.do_eval else None
    if params.model_name_or_path == 'roberta-base':
        model = 'roberta-base'
    elif params.model_name_or_path == 'roberta-large':
        model = 'roberta-large'
    elif params.model_name_or_path == 'albert-large':
        model = 'albert-large-v2'
    elif params.model_name_or_path == 'albert-xxlarge':
        model = 'albert-xxlarge-v1'
    elif params.model_name_or_path == 'electra-large-d':
        model = 'google/electra-large-discriminator'
    elif params.model_name_or_path == 'base':
        model = 'bert-base-uncased'
    else:
        model = 'bert-large-uncased-whole-word-masking'
    params += ('--model_name_or_path', model)

    script_path = os.path.join(paths['transformer_dir_path'], script_path)
    params.set_script_header(f"{script_path}")

    return params


def build_mode_args(pars):
    """build command args for graph or eager mode"""
    if pars.mode:
        pars += '--use_lazy_mode' if pars.mode == 'lazy' else None
    else:
        if pars.model_name_or_path == 'large':
            pars += '--use_lazy_mode'
            pars.mode = 'lazy'
        else:
            pars.mode = 'eager'

if __name__ == '__main__':
    params = BertParams(description="Invoke bert training with finetuning as positional argument.")

    # Define params for finetuning
    params.add_sub_command('finetuning')
    params = args_bert_common(params)
    params = args_bert_finetuning(params)

    # Parse args and divert to appropriate task
    params.parse_args()
    path_dict = build_path_dict()

    print(params.get_env_vars())
    if(params.process_per_node > 0):
        multi_hls = True
    else:
        multi_hls = False

    cmd_list = []
    if params.get_subparser() == 'finetuning':
        params = bert_finetuning(params, path_dict)
        cmd_line = params.build_cmd_line()
        # build bert env vars
        build_bert_envs(params, path_dict)
        print(f"CMD : {cmd_line}")
        print(f"Script parameters : {params._script_params}")
        cmd_list.append(cmd_line)
    else:
        print(f'Invalid subtask: {params.get_subparser()}. Please try -h')
        exit(1)
    # run the training command
    training_runner = TrainingRunner(cmd_list, params.get_env_vars(),
                                     params.get('world_size'), params.get('dist'), mpi_run=not(params.get('no_mpirun')), multi_hls=multi_hls)
    ret_code = training_runner.run()
    sys.exit(ret_code)
