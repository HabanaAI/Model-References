###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import argparse
import os
import sys
import copy
sys.path.append(os.path.realpath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../../..")))
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
                        default='base', choices=['base', 'large'], help='Model name')
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
    return params


def args_bert_pretraining(params: BertParams):
    """Define BRET pre-training specific parameters"""

    params.add_argument('-p', '--data_dir', nargs=2, type=str,
                        default=[None, None], help='Data directory')
    params.add_argument('-t', '--task_name', type=str, default='bookswiki', help='Task name')
    params.add_argument('--config_file', type=str, default=None, help="Optional config file for bert.")
    params.add_argument('-r', '--learning_rate', nargs=2, type=float, default=[6e-3, 4e-3], help='Learning rate')
    params.add_argument('-s', '--max_seq_length', nargs=2, type=int, default=[128, 512], help='Max seq length')
    params.add_argument('-b', '--batch_size', nargs=2, type=int, default=[64, 8],
                        help='Train batch size per device')
    params.add_argument('-st', '--save_steps', nargs=2, type=int, default=[200, 200], help='Save steps')
    params.add_argument('--max_steps', nargs=2, type=int, default=[7038, 1563], help='Maximum training steps')
    params.add_argument('-wp', '--warmup', nargs=2, type=float, default=[0.2843, 0.128],
                        help='Number of warmup steps for each phase of pretraining.')
    params.add_argument('--init_checkpoint', nargs=2, type=str, default=[None, None], help='Initial checkpoints')
    params.add_argument('--create_logfile', action='store_true', help='Enable logfiles')
    params.add_argument('--accumulate_gradients', action='store_true',
                        help='Enable gradient accumulation steps for pre training')
    params.add_argument('--allreduce_post_accumulation', action='store_true',
                        help='Enable allreduces during gradient accumulation steps')
    params.add_argument('--allreduce_post_accumulation_fp16', action='store_true',
                        help='Enable fp16 allreduces post accumulation')
    params.add_argument('--phase', type=int, default=None, choices=[1, 2],
                        help='Phase number to run. By default runs both phase 1 & 2.')
    params.add_argument('--train_batch_size', nargs=2, type=int, default=[8192, 4096],
                        help='Train batch size (gradient accumulation step is calculated based on this)')
    params.add_argument('--tqdm_smoothing', type=float, default=None, help='Smoothing factor for tqdm.')
    params.add_argument('--steps_this_run', nargs=2, type=int, default=[-1, -1],
                        help='If provided, only run this many steps before exiting')
    params.add_argument('--seed', type=int, default=42, help='Seed value')
    return params


def args_bert_finetuning(params: BertParams):
    """ Define BERT fine-tuning specific parameters"""

    params.add_argument('-p', '--data_dir', type=str, default=None, help='Data directory')
    params.add_argument('-t', '--task_name', choices=['mrpc', 'squad'], type=str, default='mrpc', help='Task name')
    params.add_argument('-r', '--learning_rate', type=float, default=2e-5, help='Learning rate')
    params.add_argument('-s', '--max_seq_length', type=int, default=128, help='Max seq length')
    params.add_argument('-b', '--batch_size', type=int, default=8, help='Train batch size per device')
    params.add_argument('-v', '--per_device_eval_batch_size', type=int, default=8, help='Eval batch size per device')
    params.add_argument('-e', '--num_train_epochs', type=int, default=1, help='Number of Training epochs')
    params.add_argument('-l', '--logging_steps', type=int, default=1, help='Number of logging steps')
    params.add_argument('--max_steps', type=int, default=None, help='Maximum training steps')
    params.add_argument('--do_eval', action='store_true', help='Enable evaluation')
    params.add_argument('-tf', '--train_file', type=str, default='train-v1.1.json', help='Used with SQUAD only')
    params.add_argument('-pf', '--predict_file', type=str, default='dev-v1.1.json', help='Used with SQUAD only')
    params.add_argument('-mt', '--model_type', type=str, default='bert', help='Used with SQUAD only')
    params.add_argument('-ds', '--doc_stride', type=int, default=128, help='Used with SQUAD only')
    params.add_argument('-st', '--save_steps', type=int, default=None, help='Save steps')
    params.add_argument('--seed', type=int, default=42, help='Seed value')
    return params


def build_path_dict():
    real_path = os.path.realpath(__file__)
    demo_dir_path = os.path.dirname(real_path)
    fine_tuning_script_path = os.path.join(demo_dir_path, 'finetuning')
    pre_training_script_path = os.path.join(demo_dir_path, 'pretraining')

    return {
        'real_path': real_path,
        'demo_dir_path': demo_dir_path,
        'demo_config_path': demo_dir_path,
        'fine_tuning_script_path': fine_tuning_script_path,
        'pre_training_script_path': pre_training_script_path,
        'transformer_dir_path': fine_tuning_script_path,
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
        params += [('--hmp_bf16', f"{paths['demo_config_path']}/ops_bf16_bert.txt")]
        params += [('--hmp_fp32', f"{paths['demo_config_path']}/ops_fp32_bert.txt")]

    if params.task_name.lower() == 'mrpc':
        script_path = 'examples/text-classification/run_glue.py'
        data_dir = params.get('data_dir', '/software/data/pytorch/transformers/glue_data/MRPC')
        params += ('--task_name', params.task_name.upper())
        params += ('--per_device_train_batch_size', params.batch_size)
        params += ('--per_device_eval_batch_size', params.per_device_eval_batch_size)
    elif params.task_name.lower() == 'squad':
        script_path = 'examples/question-answering/run_squad.py'
        data_dir = params.get('data_dir', '/software/data/pytorch/transformers/Squad')
        params += ['--do_lower_case']
        params += ('--model_type', params.model_type)
        params += ('--train_file', params.train_file)
        params += ('--predict_file', params.predict_file)
        params += ('--doc_stride', params.doc_stride)
        params += ('--per_gpu_train_batch_size', params.batch_size)
        params += ('--per_gpu_eval_batch_size', params.per_device_eval_batch_size)

    output_dir = os.path.join(params.output_dir, params.task_name)
    build_mode_args(params)
    params += ['--use_fused_adam']
    params += ['--use_fused_clip_norm']
    params += ('--max_steps', params.max_steps) if params.max_steps else None
    params += ('--cache_dir', params.cache_dir) if params.cache_dir else None
    params += ('--save_steps', params.save_steps) if params.save_steps else None
    params += '--use_habana' if params.device == 'hpu' else '--no_cuda'
    params += ('--data_dir', data_dir)
    params += ('--max_seq_length', params.max_seq_length)
    params += ('--learning_rate', params.learning_rate)
    params += ('--num_train_epochs', params.num_train_epochs)
    params += ('--output_dir', output_dir)
    params += ('--logging_steps', params.logging_steps)
    params += ('--seed', params.seed)
    params += '--overwrite_output_dir'
    params += '--do_train'
    params += '--do_eval' if params.do_eval else None
    model = 'bert-base-uncased' if params.model_name_or_path == 'base' else 'bert-large-uncased-whole-word-masking'
    params += ('--model_name_or_path', model)

    script_path = os.path.join(paths['transformer_dir_path'], script_path)
    params.set_script_header(f"{script_path}")

    return params


def build_mode_args(pars):
    """build command args for graph or eager mode"""
    if pars.mode:
        pars += '--use_jit_trace' if pars.mode == 'graph' else None
        pars += '--use_lazy_mode' if pars.mode == 'lazy' else None
    else:
        if pars.model_name_or_path == 'large':
            pars += '--use_lazy_mode'
            pars.mode = 'lazy'
        else:
            pars.mode = 'eager'


def bert_pretraining(params, paths):
    """ Construct pre-training command """

    # Default values for train batch size
    p1_train_batch_size = params.train_batch_size[0]
    p2_train_batch_size = params.train_batch_size[1]

    # Default value for communication backend
    communication_backend = "hcl"

    # Default values for max predictions per sequence
    # for phase1 and phase2
    # This is to be removed once the values are read as
    # command line arguments
    p1_max_predicions_per_seq = 20
    p2_max_predicions_per_seq = 80

    # Find the values of grad accumulation steps
    # For phase1 and phase2 if pretraining is enabled
    p1_grad_accum_steps = p1_train_batch_size // params.batch_size[0]
    p2_grad_accum_steps = p2_train_batch_size // params.batch_size[1]

    if params.data_dir[0] and params.data_dir[1]:
        p1_data_dir = params.data_dir[0]
        p2_data_dir = params.data_dir[1]
    else:
        p1_data_dir = os.path.join('/software/data/pytorch/bert_pretraining',
                                   'hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/books_wiki_en_corpus')
        p2_data_dir = os.path.join('/software/data/pytorch/bert_pretraining',
                                   'hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/books_wiki_en_corpus')

    # calculate checkpoint
    if params.init_checkpoint[0] or params.init_checkpoint[1]:
        p1_init_checkpoint = params.get('init_checkpoint')[0]
        p2_init_checkpoint = params.get('init_checkpoint')[1]
        if p1_init_checkpoint:
            if p1_init_checkpoint != 'None':
                p1_checkpoint = f"--resume_from_checkpoint --init_checkpoint={p1_init_checkpoint}"
            else:
                p1_checkpoint = ""
        if p2_init_checkpoint:
            if p2_init_checkpoint != 'None':
                p2_checkpoint = f"--resume_from_checkpoint --init_checkpoint={p2_init_checkpoint}"
            else:
                p2_checkpoint = f"--resume_from_checkpoint --phase1_end_step={params.get('max_steps')[0]}"
    else:
        p1_checkpoint = ""
        p2_checkpoint = f"--resume_from_checkpoint --phase1_end_step={params.get('max_steps')[0]}"

    script_path = os.path.join(paths['pre_training_script_path'], 'run_pretraining.py')
    results_dir = os.path.join(params.output_dir, 'results')
    checkpoints_dir = os.path.join(results_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    model_map_dict = {
        'base': 'bert-base-uncased',
        'large': 'bert-large-uncased',
        'base-cased': 'bert-base-cased',
        'base-multilingual': 'bert-base-multilingual',
        'base-chinese': 'bert-base-chinese',
    }
    assert params.model_name_or_path in model_map_dict, \
        f"FATAL: model_name_or_path should be one of {model_map_dict.keys()}"

    # Construct command args shared by ph1 and ph2 pre-train
    params += '--do_train'
    params += ('--bert_model', model_map_dict[params.model_name_or_path])
    if params.data_type == 'bf16':
        params += ['--hmp']
        params += [('--hmp_bf16', f"{paths['demo_config_path']}/ops_bf16_bert_pt.txt")]
        params += [('--hmp_fp32', f"{paths['demo_config_path']}/ops_fp32_bert_pt.txt")]

    build_mode_args(params)
    params += ('--config_file',
               params.config_file if params.config_file else
               os.path.join(paths['pre_training_script_path'], 'bert_config.json'))
    params += '--use_habana' if params.device == 'hpu' else '--no_cuda'
    params += '--allreduce_post_accumulation' if params.allreduce_post_accumulation else None
    params += '--allreduce_post_accumulation_fp16' if params.allreduce_post_accumulation_fp16 else None
    params += ('--json-summary', os.path.join(results_dir, 'dllogger.json'))
    params += ('--output_dir', checkpoints_dir)
    params += ('--seed', params.seed)
    params += ('--use_fused_lamb')
    params += ('--tqdm_smoothing', params.tqdm_smoothing) if params.tqdm_smoothing else None

    params.set_script_header(f"{script_path}")

    # Construct command args specific for phase1 and phase2
    ph1_params = params
    ph2_params = params.copy()

    ph1_params += ('--input_dir', p1_data_dir)
    ph2_params += ('--input_dir', p2_data_dir)
    ph1_params += ('--train_batch_size', p1_train_batch_size)
    ph2_params += ('--train_batch_size', p2_train_batch_size)
    ph1_params += ('--max_seq_length', params.max_seq_length[0])
    ph2_params += ('--max_seq_length', params.max_seq_length[1])
    ph1_params += ('--max_predictions_per_seq', p1_max_predicions_per_seq)
    ph2_params += ('--max_predictions_per_seq', p2_max_predicions_per_seq)
    ph1_params += ('--max_steps', params.max_steps[0])
    ph2_params += ('--max_steps', params.max_steps[1])
    ph1_params += ('--warmup_proportion', params.warmup[0])
    ph2_params += ('--warmup_proportion', params.warmup[1])
    ph1_params += ('--num_steps_per_checkpoint', params.save_steps[0])
    ph2_params += ('--num_steps_per_checkpoint', params.save_steps[1])
    ph1_params += ('--learning_rate', params.learning_rate[0])
    ph2_params += ('--learning_rate', params.learning_rate[1])
    ph1_params += ('--gradient_accumulation_steps', p1_grad_accum_steps) if params.accumulate_gradients else None
    ph2_params += ('--gradient_accumulation_steps', p2_grad_accum_steps) if params.accumulate_gradients else None
    ph1_params += ('--steps_this_run', params.steps_this_run[0])
    ph2_params += ('--steps_this_run', params.steps_this_run[1])
    ph1_params += p1_checkpoint
    ph2_params += p2_checkpoint
    ph2_params += '--phase2'

    return ph1_params, ph2_params


if __name__ == '__main__':
    params = BertParams(description="Invoke bert training with finetuning or pretraining as positional argument.")

    # Define params for finetuning
    params.add_sub_command('finetuning')
    params = args_bert_common(params)
    params = args_bert_finetuning(params)

    # Define params for pretraining
    params.add_sub_command('pretraining')
    params = args_bert_common(params)
    params = args_bert_pretraining(params)

    # Parse args and divert to appropriate task
    params.parse_args()
    path_dict = build_path_dict()

    print(params.get_env_vars())

    cmd_list = []
    if params.get_subparser() == 'finetuning':
        params = bert_finetuning(params, path_dict)
        cmd_line = params.build_cmd_line()
        # build bert env vars
        build_bert_envs(params, path_dict)
        print(f"CMD : {cmd_line}")
        print(f"Script parameters : {params._script_params}")
        cmd_list.append(cmd_line)
    elif params.get_subparser() == 'pretraining':
        p1_params, p2_params = bert_pretraining(params, path_dict)
        p1_cmd_line = p1_params.build_cmd_line()
        p2_cmd_line = p2_params.build_cmd_line()
        # build bert env vars
        build_bert_envs(params, path_dict)
        if params.phase is None or params.phase == 1:
            print(f"CMD ph1 : {p1_cmd_line}")
            print(f"Script parameters for ph1 : {p1_params._script_params}")
            cmd_list.append(p1_cmd_line)
        if params.phase is None or params.phase == 2:
            print(f"CMD ph2 : {p2_cmd_line}")
            print(f"Script parameters for ph2 : {p2_params._script_params}")
            cmd_list.append(p2_cmd_line)
    else:
        print(f'Invalid subtask: {params.get_subparser()}. Please try -h')
        exit(1)
    # run the training command
    training_runner = TrainingRunner(cmd_list, params.get_env_vars(),
                                     params.get('world_size'), params.get('dist'), mpi_run=not(params.get('no_mpirun')))
    training_runner.run()
