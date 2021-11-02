###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
import sys
from pathlib import Path
import math
import subprocess
from central.habana_model_runner_utils import HabanaEnvVariables, print_env_info, get_canonical_path, get_canonical_path_str, is_valid_multi_node_config, get_multi_node_config_nodes
from central.training_run_config import TrainingRunHWConfig
import data_preprocessing.create_pretraining_data_overfit as create_pretraining_data_overfit
import central.prepare_output_dir as prepare_output_dir
from central.multi_node_utils import run_per_ip

class BertPretrainingOverfit(TrainingRunHWConfig):
    def __init__(self, scaleout, num_workers_per_hls, hls_type, kubernetes_run, args, epochs, p1_steps, p1_warmup, p1_batch_size, p1_max_seq_len, pretrained_model, enable_scoped_allocator):
        super(BertPretrainingOverfit, self).__init__(scaleout, num_workers_per_hls, hls_type, kubernetes_run, "demo_bert_log")
        self.args = args
        self.epochs = epochs
        self.p1_steps = int(p1_steps)
        self.p1_warmup = int(p1_warmup)
        self.p1_batch_size = int(p1_batch_size)
        self.p1_max_seq_len = int(p1_max_seq_len)
        self.pretrained_model = pretrained_model
        self.enable_scoped_allocator = enable_scoped_allocator
        self.eval_batch_size = 8
        self.use_xla = False
        self.dataset_path = self.args.dataset_path

        if self.dataset_path is None:
            self.dataset_path = "./bert-pretraining-overfit-dataset"

        if self.args.output_dir is not None:
            self.results_dir = self.args.output_dir
        else:
            self.results_dir = "./bert-pretraining-overfit-results"

        self.learning_rate = self.args.learning_rate

        self.command = ''
        self.overfit_habana_env_variables = {}

    def create_pretraining_data(self, seq_length, max_pred_per_seq):
        try:
            if self.scaleout and is_valid_multi_node_config() and not self.kubernetes_run:
                create_pt_data_path = Path(__file__).parent.joinpath('data_preprocessing').joinpath('create_pretraining_data_overfit.py')
                run_per_ip(f"{sys.executable} {str(create_pt_data_path)} {self.dataset_path} {self.pretrained_model} {seq_length} {max_pred_per_seq}", ['MULTI_HLS_IPS', 'PYTHONPATH'], False)
            else:
                create_pretraining_data_overfit.create_pretraining_data_overfit_r(self.dataset_path, self.pretrained_model, seq_length, max_pred_per_seq)
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} create_pretraining_data({self.dataset_path} {self.pretrained_model} {seq_length} {max_pred_per_seq})") from exc

    def prepare_results_path(self, results_dir):
        try:
            if self.scaleout and is_valid_multi_node_config() and not self.kubernetes_run:
                prepare_output_dir_path = Path(__file__).parent.parent.parent.parent.joinpath('central').joinpath('prepare_output_dir.py')
                run_per_ip(f"{sys.executable} {str(prepare_output_dir_path)} {results_dir}", ['MULTI_HLS_IPS', 'PYTHONPATH'], False)
            else:
                prepare_output_dir.prepare_output_dir_r(results_dir)
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} prepare_results_path({results_dir})") from exc

    def build_command(self):
        try:
            seq_length = self.p1_max_seq_len
            if seq_length == 128:
                max_pred_per_seq = 20
            elif seq_length == 512:
                max_pred_per_seq = 80
            else:
                print(f"Warning: Unsupported max_sequence_length {seq_length}. Setting max_predictions_per_seq to floor(0.15*max_sequence_length). Please see -s parameter for details")
                max_pred_per_seq = math.floor(0.15 * seq_length)

            # run_per_ip
            self.create_pretraining_data(seq_length, max_pred_per_seq)
            sys.stdout.flush()
            sys.stderr.flush()

            horovod_str = "--horovod" if self.scaleout is True else ""

            # run_per_ip
            self.prepare_results_path(self.results_dir)

            base_lr = 0.006
            num_acc_steps = 1
            learning_rate = float(base_lr * ( self.p1_batch_size * self.num_workers_total * num_acc_steps ) / 65536)
            print(f"learning_rate = {learning_rate}")

            ds_path = str(get_canonical_path(self.dataset_path))
            results_path = get_canonical_path(self.results_dir)
            pretrained_model_path = get_canonical_path(self.pretrained_model)
            bert_config = str(pretrained_model_path.joinpath("bert_config.json"))
            init_checkpoint_path = get_canonical_path(self.args.init_checkpoint_path).joinpath(f"{self.args.model_variant}").joinpath("model.ckpt-0.meta")
            init_checkpoint = str(get_canonical_path(self.args.init_checkpoint_path).joinpath(f"{self.args.model_variant}"))
            init_checkpoint = init_checkpoint + "/" + "model.ckpt-0"
            if os.path.exists(init_checkpoint_path) == False:
                raise Exception(f"Error: init_checkpoint_path {str(init_checkpoint_path)} file or directory missing. Please mount correctly")
            dllog_path = str(results_path.joinpath("bert_dllog.json"))
            run_pretraining_path = Path(__file__).parent.joinpath('run_pretraining.py')

            print(f"{self.__class__.__name__}: self.mpirun_cmd = {self.mpirun_cmd}")
            if self.mpirun_cmd == '':
                init_command = f"time python3 {str(run_pretraining_path)}"
            else:
                init_command = f"time {self.mpirun_cmd} python3 {str(run_pretraining_path)}"
            self.command = (
                f"{init_command}"
                f" --input_files_dir={ds_path}"
                f" --eval_files_dir={ds_path}"
                f" --output_dir={str(results_path)}"
                f" --do_train=True"
                f" --do_eval=True"
                f" --bert_config_file={bert_config}"
                f" --init_checkpoint={init_checkpoint}"
                f" --train_batch_size={self.p1_batch_size}"
                f" --eval_batch_size={self.eval_batch_size}"
                f" --max_seq_length={seq_length}"
                f" --max_predictions_per_seq={max_pred_per_seq}"
                f" --num_train_steps={self.p1_steps}"
                f" --num_accumulation_steps={num_acc_steps}"
                f" --num_warmup_steps={self.p1_warmup}"
                f" --dllog_path={dllog_path}"
                f" --learning_rate={learning_rate}"
                f" {horovod_str}"
                f" --amp=False"
                f" --use_xla=False"
                f" --enable_scoped_allocator={self.enable_scoped_allocator}"
            )
            print('bert_pretraining_overfit_utils build_command(): self.command = ', self.command)
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} build_command()") from exc

    def run(self):
        try:
            run_config_env_vars = self.get_env_vars()
            print('run_config_env_vars = ', run_config_env_vars)
            with HabanaEnvVariables(env_vars_to_set=run_config_env_vars), \
                 HabanaEnvVariables(env_vars_to_set=self.overfit_habana_env_variables):
                self.build_command()
                print_env_info(self.command, run_config_env_vars)
                print_env_info(self.command, self.overfit_habana_env_variables)
                print(f"{self.__class__.__name__} run(): self.command = {self.command}")
                sys.stdout.flush()
                sys.stderr.flush()
                with subprocess.Popen(self.command, shell=True, executable='/bin/bash') as proc:
                    proc.wait()
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} run()") from exc
