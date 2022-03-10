import sys
import re
import os
import io
import socket
import argparse
import shutil
import subprocess

model_ref_path = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../.."))
python_path = os.getenv('PYTHONPATH') + ":" if os.getenv('PYTHONPATH') else ""
os.environ['PYTHONPATH'] = python_path +  model_ref_path
sys.path.append(model_ref_path)
from PyTorch.common.training_runner import TrainingRunner

class TransformerArgparser(argparse.ArgumentParser):
    """docstring for TransformerArgparser"""
    def __init__(self, **kwargs):
        super(TransformerArgparser, self).__init__(**kwargs)
        self.add_argument('--exec-mode', default='train', choices=['train', 'evaluate', 'predict'], help='Execution mode to run the model')
        self.add_argument('-p', '--data', default="/root/software/data/pytorch/Fairseq/transformer-LT/wmt16_en_de_bpe32k",
                          help='Path to the training dataset')
        self.add_argument('-d', '--device', default='hpu', help='Device on which to run on.')
        self.add_argument('--world_size', type=int, default=1, help='Number of devices to run on.')
        self.add_argument('--learning-rate', required=False, type=float, default=5e-4, help='Learning rate.')
        self.add_argument('--mode', required=False, type=str, default='lazy', choices=['lazy', 'eager'],
                          help='Different modes avaialble. Possible values: lazy, eager. Default: lazy')
        self.add_argument('--data-type', required=False, default='fp32', choices=['fp32', 'bf16'],
                          help='Data Type. Possible values: fp32, bf16. Default: fp32')
        self.add_argument('--max-update', required=True, type=int, help='Maximum number of updates to run training for.')
        self.add_argument('--max-tokens', required=False, type=int, default=4096, help='Maxmium tokens.')
        self.add_argument('--num-batch-buckets', required=False, type=int, default=10, help='Number of batch buckets.')
        self.add_argument('--criterion', required=False, default='label_smoothed_cross_entropy', help='Loss criterion.')
        self.add_argument('--label-smoothing', required=False, default=0.1, help='Label smoothing.')
        self.add_argument('--update-freq', required=False, type=int, default=13, help='Update frequency.')
        self.add_argument('--save-interval-updates', required=False, type=int, default=3000, help='Save checkpoints after this much updates.')
        self.add_argument('--save-interval', required=False, type=int, default=10, help='Save checkpoints.')
        self.add_argument('--validate-interval', required=False, type=int, default=20, help='validate checkpoints.')
        self.add_argument('--do-eval', action='store_true', help='Run Evalation after training.')
        self.add_argument('--arch', required=False, default="transformer_wmt_en_de_big", help='Transformer architecture.')
        self.add_argument('--clip-norm', required=False, type=float, default=0.0, help='Clipnorm.')
        self.add_argument('--dropout', required=False, type=float, default=0.3, help='Dropout value.')
        self.add_argument('--weight-decay', required=False, type=float, default=0.0, help='Weight decay parameter.')
        self.add_argument('--keep-interval-updates', required=False, type=int, default=20, help='Keep these many last checkpoints.')
        self.add_argument('--log-format', required=False, default='simple', help='Logging format.')
        self.add_argument('--log-interval', required=False, default=1, help='Logging interval.')
        self.add_argument('--max-source-positions', required=False, type=int, default=256, help='Maxmimum source positions.')
        self.add_argument('--max-target-positions', required=False, type=int, default=256, help='Maximum target positions.')
        self.add_argument('--warmup-updates', required=False, type=int, default=4000, help='Maximum warmup updates.')
        self.add_argument('--warmup-init-lr', required=False, type=float, default=1e-07, help='Warmup init lr.')
        self.add_argument('--optimizer', required=False, default='adam', help='Training optimizer.')
        self.add_argument('--adam-betas', required=False, default='(0.9, 0.98)', help='Beta parameters for adam (in form of tuple).')
        self.add_argument('--lr-scheduler', required=False, default='inverse_sqrt', help='Learining rate scheduler.')
        self.add_argument('--output_dir', required=False, default=None, help='Output directory for checkpoints, logging etc.')
        self.add_argument('--seed', required=False, default=None, help='Seed value')
        self.add_argument('--eval-bleu', required=False, default=None, help='bleu score validation')
        self.add_argument('--maximize-best-checkpoint-metric', required=False, default=None, help='maximize best checkpoint metric')
        self.add_argument('--eval-bleu-remove-bpe', required=False, default=None, help='bleu score remove bpe')
        self.add_argument('--eval-bleu-print-samples', required=False, default=None, help='print bleu score samples')
        self.add_argument('--eval-bleu-args', required=False, default='{\"beam\":4, \"max_len_a\":1.2, \"max_len_b\":10}', help='bleu score args')
        self.add_argument('--eval-bleu-detok', required=False, default=None, help='bleu detok')


def get_output_dir(args):
    output_dir = f"/tmp/fairseq_{args.arch}" if args.output_dir is None else args.output_dir
    return output_dir

def build_train_command(args, path, train_script):
    """ Constructing training command """

    output_dir = get_output_dir(args)
    save_dir = os.path.join(output_dir, "checkpoint")
    tensorboard_logdir = os.path.join(output_dir, "tensorboard")

    init_command = f"{path + '/' + str(train_script)}"
    command = (
        f"{init_command}"
        f" {args.data}"
        f" --arch={args.arch}"
        f" --lr={args.learning_rate}"
        f" --clip-norm={args.clip_norm}"
        f" --dropout={args.dropout}"
        f" --max-tokens={args.max_tokens}"
        f" --weight-decay={args.weight_decay}"
        f" --criterion={args.criterion}"
        f" --label-smoothing={args.label_smoothing}"
        f" --update-freq={args.update_freq}"
        f" --save-interval-updates={args.save_interval_updates}"
        f" --save-interval={args.save_interval}"
        f" --validate-interval={args.validate_interval}"
        f" --keep-interval-updates={args.keep_interval_updates}"
        f" --log-format={args.log_format}"
        f" --log-interval={args.log_interval}"
        f" --share-all-embeddings"
        f" --num-batch-buckets={args.num_batch_buckets}"
        f" --save-dir={save_dir}"
        f" --tensorboard-logdir={tensorboard_logdir}"
        f" --maximize-best-checkpoint-metric"
        f" --max-source-positions={args.max_source_positions}"
        f" --max-target-positions={args.max_target_positions}"
        f" --max-update={args.max_update}"
        f" --warmup-updates={args.warmup_updates}"
        f" --warmup-init-lr={args.warmup_init_lr}"
        f" --lr-scheduler={args.lr_scheduler}"
        f" --no-epoch-checkpoints"
        )
    if args.data_type == 'bf16':
        command += f" --bf16"

    if args.mode == "lazy":
        command += f" --use-lazy-mode"

    if args.optimizer == "adam":
        command += f" --optimizer=adam --use-fused-adam --adam-betas=\"{args.adam_betas}\""

    if args.device == "hpu":
        command += f" --use-habana"

    if args.world_size > 1:
        command += f" --distributed-world-size={args.world_size} --bucket-cap-mb=230"

    if args.seed:
        command += f" --seed={args.seed}"

    if args.eval_bleu_detok:
        command += (
            f" --eval-bleu-args=\'{args.eval_bleu_args}\'"
            f" --eval-bleu-detok={args.eval_bleu_detok}"
            f" --eval-bleu-remove-bpe"
            f" --eval-bleu-print-samples"
            f" --eval-bleu"
            )
    return command

def build_eval_command(args):
    output_dir = get_output_dir(args)
    checkpt_dir = os.path.join(output_dir, "checkpoint")
    last_ckpt = os.path.join(checkpt_dir, "checkpoint_last.pt")

    beam = 4
    wmt = "wmt14"
    lang_pair = "en-de"
    bpe_code = os.path.join(args.data, "bpe.code")
    src_lang, tgt_lang = lang_pair.split('-')

    # Remove sacrebleu cache dir
    shutil.rmtree(os.path.expanduser('~/.sacrebleu'), ignore_errors=True)

    # Compute BLEU score
    eval_command = (f"sacrebleu -t {wmt} -l {lang_pair} --echo src "
                    f"| fairseq-interactive {args.data} --path {last_ckpt} "
                    f"-s {src_lang} -t {tgt_lang} "
                    f"--batch-size 32 --buffer-size 1024 "
                    f"--beam {beam} --lenpen 0.6 --remove-bpe --max-len-a  1.2 --max-len-b 10 "
                    f"--bpe subword_nmt --bpe-codes {bpe_code} --tokenizer moses --moses-no-dash-splits --use-habana "
                    f"--use-lazy-mode "
                    f"| tee /tmp/.eval_out.txt "
                    f"| grep ^H- | cut -f 3- "
                    f"| sacrebleu -t {wmt} -l {lang_pair} "
                   )

    return eval_command

#Setup the environment variable
def set_env(args):
    env_vars = {}
    if args.world_size > 1:
        env_vars['MASTER_ADDR'] = 'localhost'
        env_vars['MASTER_PORT'] = '12345'
    return env_vars

def set_env_eval(args):
    env_vars = {}
    return env_vars

# NOTE: subprocess.run cribbs if env passed has non-string values
def envs_for_subprocess(env):
    return {x: str(y) for x, y in env.items()}

def main():
    # This formatter class will include default values in --help
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    args = TransformerArgparser(formatter_class=formatter_class).parse_args()

    # Get the path for the train script
    fairseq_path = os.path.abspath(os.path.dirname(__file__))
    train_script = "fairseq_cli/train.py"

    # Set environment variables
    env_vars = set_env(args)

    # Build the command line
    command = build_train_command(args, fairseq_path, train_script)

    training_runner = TrainingRunner([command], env_vars, args.world_size, mpi_run=True, map_by='slot')
    ret_code = training_runner.run()

    # Return if evalaution was not required or training failed for
    # some reason
    if not args.do_eval or ret_code is not None:
        return ret_code

    # Run evaluation
    eval_env_vars = set_env_eval(args)

    # Build eval command line
    eval_command = build_eval_command(args)

    print('Running Evalation.')
    eval_env_vars = envs_for_subprocess({**os.environ, **eval_env_vars})
    eval_ret_code = subprocess.run(eval_command, shell=True, env=eval_env_vars).returncode
    return eval_ret_code

if __name__=="__main__":
  ret = main()
  sys.exit(ret)
