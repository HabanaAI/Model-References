import argparse
import os
import sys
import copy
model_garden_path = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../.."))
os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + ":" +  model_garden_path
sys.path.append(model_garden_path)
from PyTorch.common.training_runner import TrainingRunner

MODEL_NAMES= [
    "distilbert"
]

class DistilBertParams(object):
    """ class for handling DistilBERT parameters and env vars"""

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
        # collection of env vars specific for distilbert
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

def build_distilbert_envs(params, paths):
    # add mpirun env vars
    if not params.get('no_mpirun'):
        params.add_env('MASTER_ADDR', 'localhost')  # modify this with hostfile for multi-hls
        params.add_env('MASTER_PORT', '12345')

def build_path_dict():
    real_path = os.path.realpath(__file__)
    demo_dir_path = ''
    pre_training_script_path = os.path.join(demo_dir_path, 'pretraining')
    transformer_path = ''

    return {
        'real_path': real_path,
        'demo_dir_path': demo_dir_path,
        'demo_config_path': demo_dir_path,
        'transformer_dir_path': transformer_path,
    }



def args_distilbert_common(params: DistilBertParams):
    """Define DistilBERT common parameters"""
    # Required parameters
    params.add_argument(
        "--model_type",
        default='distilbert',
        type=str,
        help="Model type selected in the list: " + ", ".join(MODEL_NAMES),
    )
    params.add_argument(
        "--model_name_or_path",
        default='distilbert-base-uncased',
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    params.add_argument(
        "--output_dir",
        default='/tmp/distbert/tmp_train',
        type=str,
        help="The output directory where the model checkpoints and predictions will be written.",
    )

    # Distillation parameters (optional)
    params.add_argument(
        "--teacher_type",
        default=None,
        type=str,
        help="Teacher type. Teacher tokenizer and student (model) tokenizer must output the same tokenization. Only for distillation.",
    )
    params.add_argument(
        "--teacher_name_or_path",
        default='bert-base-uncased',
        type=str,
        help="Path to the already SQuAD fine-tuned teacher model. Only for distillation.",
    )
    params.add_argument(
        "--alpha_ce", default=0.5, type=float, help="Distillation loss linear weight. Only for distillation."
    )
    params.add_argument(
        "--alpha_squad", default=0.5, type=float, help="True SQuAD loss linear weight. Only for distillation."
    )
    params.add_argument(
        "--temperature", default=2.0, type=float, help="Distillation temperature. Only for distillation."
    )

    # Other parameters
    params.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    params.add_argument(
        "--train_file",
        default='/software/data/pytorch/transformers/Squad/train-v1.1.json',
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    params.add_argument(
        "--predict_file",
        default='/software/data/pytorch/transformers/Squad/dev-v1.1.json',
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    params.add_argument(
        "--config_name", default="./training_configs/distilbert-base-uncased.json", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    params.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    params.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from huggingface.co",
    )

    params.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    params.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )

    params.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    params.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    params.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    params.add_argument("--do_train", action="store_true", help="Whether to run training.")
    params.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    params.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    )
    params.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    params.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    params.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    params.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    params.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    params.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    params.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    params.add_argument("--optimizer", default="FusedAdamW", type=str, choices=["AdamW", "FusedAdamW"], help="type of optimizer.")
    params.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    params.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    params.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    params.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    params.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    params.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    params.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
    )

    params.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    params.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    params.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    params.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    params.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    params.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    params.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    params.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    params.add_argument("--world_size", type=int, default=1, help="Training device size")
    params.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    params.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    params.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    params.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

    params.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
    params.add_argument("--hpu", action="store_true", help="HPU run")
    params.add_argument("--hmp", action="store_true", default=True, help="Enable HMP")
    params.add_argument("--hmp_bf16", type=str, default='./ops_bf16_distilbert_pt.txt', help="List of ops to be run in BF16 for HPU")
    params.add_argument("--hmp_fp32", type=str, default='./ops_fp32_distilbert_pt.txt', help="List of ops to be run in FP32 for HPU")
    params.add_argument("--hmp_opt_level", type=str, default='O1',help="Optimization level for HMP")
    params.add_argument("--hmp_verbose", action="store_true",help="Optimization level for HMP")
    params.add_argument('--mode', type=str, default='eager', choices=['eager', 'lazy'], help='Execution mode')
    params.add_argument('--no_mpirun', action='store_true', help='Do not use MPI for distribute training')
    params.add_argument('--data_type', type=str, choices=["bf16", "fp32"], default='bf16',help="Specify data type to be either bf16 or fp32.")
    return params


def distilbert_finetuning(params, paths):
    """ Construct fine-tuning command """
    real_path = os.path.realpath(__file__)
    demo_dir_path = os.path.dirname(real_path)
    script_path = os.path.join(demo_dir_path, 'run_squad_w_distillation.py')
    output_dir = os.path.join(params.output_dir)
    params += ('--cache_dir', params.cache_dir) if params.cache_dir else None
    params += ('--model_type', params.model_type)
    params += ('--model_name_or_path', params.model_name_or_path)
    if params.teacher_type is not None:
       params += ('--teacher_type', params.teacher_type)
       params += ('--teacher_name_or_path', params.teacher_name_or_path)

    params += ('--config_name', params.config_name)
    params += ('--train_file', params.train_file)
    params += ('--predict_file', params.predict_file)
    params += '--do_eval' if params.do_eval == True else ''
    params += '--do_train'
    params += '--do_lower_case'
    params += ('--output_dir', output_dir)
    params += '--overwrite_output_dir'
    params += ('--hpu')
    params += ('--optimizer', params.optimizer)
    if params.data_type == 'bf16':
       params += "--hmp"
       params += ("--hmp_bf16", params.hmp_bf16)
       params += ("--hmp_fp32", params.hmp_fp32)
    params += ("--hmp_opt_level", params.hmp_opt_level)
    params += "--hmp_verbose" if params.hmp_verbose == True else ''
    params += "--use_lazy_mode" if params.mode == 'lazy' else ''
    params += ('--save_steps', params.save_steps) if params.save_steps else None
    params += ('--num_train_epochs', params.num_train_epochs)
    params += ('--max_steps', params.max_steps) if params.max_steps > 0 else None
    params += ('--world_size', params.world_size)

    script_path = os.path.join(paths['transformer_dir_path'], script_path)
    params.set_script_header(f"{script_path}")
    return params

if __name__ == '__main__':
    params = DistilBertParams(description="Invoke distilBert training:")
    params.add_sub_command('finetuning')
    params = args_distilbert_common(params)
    params.parse_args()

    path_dict = build_path_dict()
    print(params.get_env_vars())
    cmd_list = []
    print(params.get_subparser())
    params = distilbert_finetuning(params,path_dict)
    cmd_line = params.build_cmd_line()
    build_distilbert_envs(params,path_dict)
    print(f"CMD : {cmd_line}")
    print(f"Script parameters : {params._script_params}")
    cmd_list.append(cmd_line)
    print(cmd_list)

    training_runner = TrainingRunner(cmd_list, params.get_env_vars(),
                                     params.get('world_size'), params.get('dist'), mpi_run=not(params.get('no_mpirun')))
    ret_code = training_runner.run()
    sys.exit(ret_code)

