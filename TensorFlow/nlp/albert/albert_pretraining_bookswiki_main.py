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
from pathlib import Path
import math
import subprocess
from central.habana_model_runner_utils import HabanaEnvVariables, print_env_info, get_canonical_path, get_canonical_path_str, is_valid_multi_node_config, get_multi_node_config_nodes
from central.training_run_config import TrainingRunHWConfig
import central.prepare_output_dir as prepare_output_dir
from central.multi_node_utils import run_per_ip

class AlbertPretrainingBookswiki(TrainingRunHWConfig):
    def __init__(self, scaleout, num_workers_per_hls, hls_type, kubernetes_run, args, p1_steps, p1_warmup, p1_batch_size, p1_max_seq_len,
                p2_steps, p2_warmup, p2_batch_size, p2_max_seq_len, pretrained_model, enable_scoped_allocator):
        super(AlbertPretrainingBookswiki, self).__init__(scaleout, num_workers_per_hls, hls_type, kubernetes_run, "demo_albert_log")
        self.args = args
        self.p1_steps = int(p1_steps)
        self.p1_warmup = int(p1_warmup)
        self.p1_batch_size = int(p1_batch_size)
        self.p1_max_seq_len = int(p1_max_seq_len)
        self.p2_steps = int(p2_steps)
        self.p2_warmup = int(p2_warmup)
        self.p2_batch_size = int(p2_batch_size)
        self.p2_max_seq_len = int(p2_max_seq_len)
        self.pretrained_model = pretrained_model
        self.enable_scoped_allocator = enable_scoped_allocator
        self.eval_batch_size = 8

        if self.args.dataset_path is not None:
            self.dataset_path = self.args.dataset_path
        else:
            self.dataset_path = "./albert_pretraining_bookswiki_dataset"

        if self.args.output_dir is not None:
            self.results_dir = self.args.output_dir
        else:
            self.results_dir = "./albert_pretraining_bookswiki_results"

        self.learning_rate = self.args.learning_rate

        self.command = ''
        self.overfit_habana_env_variables = {}

    def prepare_results_path(self, results_dir):
        try:
            if self.scaleout and is_valid_multi_node_config() and not self.kubernetes_run:
                prepare_output_dir_path = Path(__file__).parent.parent.parent.parent.parent.joinpath('central').joinpath('prepare_output_dir.py')
                run_per_ip(f"{sys.executable} {str(prepare_output_dir_path)} {results_dir}", ['MULTI_HLS_IPS', 'PYTHONPATH'], False)
            else:
                prepare_output_dir.prepare_output_dir_r(results_dir)
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} prepare_results_path({results_dir})") from exc

    def build_command_phase1(self):
        try:
            horovod_str = "--use_horovod" if self.scaleout is True else ""

            # run_per_ip
            self.prepare_results_path(self.results_dir)
            results_dir_phase1 = self.results_dir + "_phase1"

            seq_len = self.p1_max_seq_len
            dataset_path_phase1 = get_canonical_path(self.args.dataset_path).joinpath(f"seq_len_{seq_len}")
            training_file_path = str(dataset_path_phase1.joinpath("training"))
            eval_file_path = str(dataset_path_phase1.joinpath("test"))

            results_path = get_canonical_path(results_dir_phase1)
            self.prepare_results_path(results_dir_phase1)
            pretrained_model_path = get_canonical_path(self.pretrained_model)
            albert_config = str(pretrained_model_path.joinpath("albert_config.json"))
            init_checkpoint = str(pretrained_model_path.joinpath("model.ckpt-best"))
            run_pretraining_path = Path(__file__).parent.joinpath('run_pretraining.py')

            print(f"{self.__class__.__name__}: self.mpirun_cmd = {self.mpirun_cmd}")
            if self.mpirun_cmd == '':
                print("#####mpirun is none\n")
                init_command = f"time python3 {str(run_pretraining_path)}"
            else:
                print("#####mpirun is not none\n")
                init_command = f"time {self.mpirun_cmd} python3 {str(run_pretraining_path)}"
            self.command = (
                f"{init_command}"
                f" --input_file={training_file_path}/*"
                f" --eval_file={eval_file_path}/*"
                f" --output_dir={str(results_path)}"
                f" --do_train=True"
                f" --do_eval=True"
                f" --albert_config_file={albert_config}"
                f" --init_checkpoint={init_checkpoint}"
                f" --train_batch_size={self.p1_batch_size}"
                f" --eval_batch_size={self.eval_batch_size}"
                f" --max_seq_length={self.p1_max_seq_len}"
                f" --num_train_steps={self.p1_steps}"
                f" --num_warmup_steps={self.p1_warmup}"
                f" --learning_rate={self.learning_rate}"
                f" {horovod_str}"
                f" --enable_scoped_allocator={self.enable_scoped_allocator}"
            )
            print("-------------------------------------------------------------------------\n")
            print("Running the Pre-Training :: Phase 1\n")
            print("-------------------------------------------------------------------------")
            print('albert_pretraining_bookswiki_utils build_command(): self.command for phase1 = ', self.command)
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} build_command_phase1()") from exc

    def build_command_phase2(self):
        try:
            horovod_str = "--use_horovod" if self.scaleout is True else ""

            # run_per_ip
            self.prepare_results_path(self.results_dir)
            results_dir_phase2 = self.results_dir + "_phase2"

            seq_len = self.p2_max_seq_len
            dataset_path_phase2 = get_canonical_path(self.args.dataset_path).joinpath(f"seq_len_{seq_len}")
            training_file_path = str(dataset_path_phase2.joinpath("training"))
            eval_file_path = str(dataset_path_phase2.joinpath("test"))

            results_path = get_canonical_path(results_dir_phase2)
            self.prepare_results_path(results_dir_phase2)
            pretrained_model_path = get_canonical_path(self.pretrained_model)
            albert_config = str(pretrained_model_path.joinpath("albert_config.json"))
            init_checkpoint = str(pretrained_model_path.joinpath("model.ckpt-best"))
            run_pretraining_path = Path(__file__).parent.joinpath('run_pretraining.py')

            print(f"{self.__class__.__name__}: self.mpirun_cmd = {self.mpirun_cmd}")
            if self.mpirun_cmd == '':
                init_command = f"time python3 {str(run_pretraining_path)}"
            else:
                init_command = f"time {self.mpirun_cmd} python3 {str(run_pretraining_path)}"
            self.command = (
                f"{init_command}"
                f" --input_file={training_file_path}/*"
                f" --eval_file={eval_file_path}/*"
                f" --output_dir={str(results_path)}"
                f" --do_train=True"
                f" --do_eval=True"
                f" --albert_config_file={albert_config}"
                f" --init_checkpoint={init_checkpoint}"
                f" --train_batch_size={self.p2_batch_size}"
                f" --eval_batch_size={self.eval_batch_size}"
                f" --max_seq_length={self.p2_max_seq_len}"
                f" --num_train_steps={self.p2_steps}"
                f" --num_warmup_steps={self.p2_warmup}"
                f" --learning_rate={self.learning_rate}"
                f" {horovod_str}"
                f" --enable_scoped_allocator={self.enable_scoped_allocator}"
            )
            print("-------------------------------------------------------------------------\n")
            print("Running the Pre-Training :: Phase 2\n")
            print("-------------------------------------------------------------------------")
            print('albert_pretraining_bookswiki_utils build_command(): self.command for phase1 = ', self.command)
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} build_command_phase1()") from exc

    def run(self):
        try:
            run_config_env_vars = self.get_env_vars()
            print('run_config_env_vars = ', run_config_env_vars)
            with HabanaEnvVariables(env_vars_to_set=run_config_env_vars), \
                 HabanaEnvVariables(env_vars_to_set=self.overfit_habana_env_variables):
                #run phase1
                self.build_command_phase1()
                print_env_info(self.command, run_config_env_vars)
                print_env_info(self.command, self.overfit_habana_env_variables)
                print(f"{self.__class__.__name__} run(): self.command = {self.command}")
                sys.stdout.flush()
                sys.stderr.flush()

                with subprocess.Popen(self.command, shell=True, executable='/bin/bash') as proc:
                    proc.wait()

                sys.stdout.flush()
                sys.stderr.flush()

                 #run phase1
                self.build_command_phase2()
                print_env_info(self.command, run_config_env_vars)
                print_env_info(self.command, self.overfit_habana_env_variables)
                print(f"{self.__class__.__name__} run(): self.command = {self.command}")
                sys.stdout.flush()
                sys.stderr.flush()
                with subprocess.Popen(self.command, shell=True, executable='/bin/bash') as proc:
                    proc.wait()

        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} run()") from exc
