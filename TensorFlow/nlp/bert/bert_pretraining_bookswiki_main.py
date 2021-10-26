###############################################################################
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
#
# Changes:
# - The Python code related to the LAMB optimizer (setup_for_pretraining_lamb,
#   build_for_pretraining_lamb_phase1, build_for_pretraining_lamb_phase2, etc.)
#   are Python implementations of NVIDIA bash code.
# - Removed NVidia container build version message
###############################################################################

import os
import sys
from pathlib import Path
import subprocess
import datetime
from central.habana_model_runner_utils import HabanaEnvVariables, print_env_info, get_canonical_path, get_canonical_path_str, is_valid_multi_node_config, get_multi_node_config_nodes
from central.training_run_config import TrainingRunHWConfig
import central.prepare_output_dir as prepare_output_dir
import central.check_dirs as check_dirs
from central.multi_node_utils import run_per_ip

class BertPretrainingBooksWiki(TrainingRunHWConfig):
    def __init__(self, scaleout, num_workers_per_hls, hls_type, kubernetes_run, args, epochs, p1_steps, p1_warmup, p1_batch_size,
                p1_max_seq_len, p2_steps, p2_warmup, p2_batch_size, p2_max_seq_len, pretrained_model, enable_scoped_allocator):
        super(BertPretrainingBooksWiki, self).__init__(scaleout, num_workers_per_hls, hls_type, kubernetes_run, "demo_bert_log")
        self.args = args
        self.data_type = "fp32"
        self.epochs = epochs
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
        self.use_xla = False
        self.save_checkpoints_steps = args.save_ckpt_steps

        if self.kubernetes_run and self.scaleout and self.args.run_phase2:
            if self.args.output_dir is None:
                raise RuntimeError(f"Error in {self.__class__.__name__}: Please specify a valid output directory using --output_dir option for running pretraining Phase 2. Typically this is where the results from pretraining Phase 1 are stored.")
            else:
                self.check_dirs(self.args.output_dir)

        if self.args.output_dir is not None:
            self.results_dir = self.args.output_dir
        else:
            self.results_dir = "./bert-pretraining-bookswiki-results"

        self.logfile = self.results_dir + "/" + "tf_bert_pretraining_lamb.log"

        self.num_acc_steps_phase1 = 128
        self.num_acc_steps_phase2 = 512
        self.learning_rate_phase1 = self.args.learning_rate
        self.learning_rate_phase2 = self.args.learning_rate
        self.global_batch_size1 = 65536
        self.global_batch_size2 = 32768

        self.command = ''
        self.bookswiki_habana_env_variables = {'TOTAL_DATASET_WORD_COUNT' : '1973248217'}
        self.set_pretraining_parameters()


    def set_pretraining_parameters(self):
        if self.args.no_steps_accumulation == 1:
            self.num_acc_steps_phase1 = 1
            self.num_acc_steps_phase2 = 1
            self.learning_rate_phase1 = 7.5e-4
            self.learning_rate_phase2 = 5.0e-4
        else:
            if self.args.fast_perf_only == 1:
                self.global_batch_size1 = 1600
                self.global_batch_size2 = 200
        print(f"self.global_batch_size1 = {self.global_batch_size1}, self.num_workers_total = {self.num_workers_total}, self.p1_batch_size = {self.p1_batch_size}")
        self.num_acc_steps_phase1 = int(int(self.global_batch_size1) / int(self.num_workers_total) / int(self.p1_batch_size))
        self.num_acc_steps_phase2 = int(int(self.global_batch_size2) / int(self.num_workers_total) / int(self.p2_batch_size))
        print(f"self.num_acc_steps_phase1 = {self.num_acc_steps_phase1}, self.num_acc_steps_phase2 = {self.num_acc_steps_phase2}")

        # Default value of learning rate argument (which is then scaled in run_pretraining.py
        # with the formula: effective_learning_rate = learning_rate * number_of_workers) for:
        # - 1st phase with global batch size = 64Ki and 8 workers is 7.5e-4,
        # - 2nd phase with global batch size = 32Ki and 8 workers is 5.0e-4.
        # According to global_batch_size/learning_rate = const, to compute learning rate of
        # number of workers and global batch size, we first multiply default value by
        # (8 / self.num_workers_total) and then by ($global_batch_size / 65536) for 1st phase, and,
        # respectively for second phase.
        self.learning_rate_phase1 = 0.00075 * float(8 / self.num_workers_total ) * float(self.global_batch_size1 / 65536 )
        self.learning_rate_phase2 = 0.0005  * float(8 / self.num_workers_total ) * float(self.global_batch_size2 / 32768 )

    def prepare_output_dir(self, results_dir):
        try:
            if self.scaleout and is_valid_multi_node_config() and not self.kubernetes_run:
                prepare_output_dir_path = Path(__file__).parent.parent.parent.parent.joinpath('central').joinpath('prepare_output_dir.py')
                run_per_ip(f"{sys.executable} {str(prepare_output_dir_path)} {results_dir}", ['MULTI_HLS_IPS', 'PYTHONPATH'], False)
            else:
                prepare_output_dir.prepare_output_dir_r(results_dir)
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} prepare_output_dir({results_dir})") from exc

    def setup_for_pretraining_lamb(self):
        try:
            GBS1 = self.p1_batch_size * self.num_workers_per_hls * self.num_acc_steps_phase1
            GBS2 = self.p2_batch_size * self.num_workers_per_hls * self.num_acc_steps_phase2
            TAG = f"tf_bert_pretraining_lamb_{self.args.model_variant}_{self.data_type}_gbs1{GBS1}_gbs2{GBS2}"
            c = datetime.datetime.now()
            DATESTAMP = f"{c.year}{c.month}{c.day}{c.hour}{c.minute}{c.second}"
            epochs = int((self.p1_steps * GBS1 * 128 + self.p2_steps * GBS2 * 512) / int(os.environ.get('TOTAL_DATASET_WORD_COUNT')))
            print("Number of epochs: {0:.2f}".format(epochs))

            if self.args.output_dir is None:
                self.results_dir = "./results" + "/" + TAG + "_" + DATESTAMP
            #Edit to save logs & checkpoints in a different directory
            self.logfile = self.results_dir + "/" + TAG + "." + DATESTAMP + ".log"
            if self.kubernetes_run and self.scaleout and self.args.run_phase2:
                pass
            else:
                # run_per_ip
                self.prepare_output_dir(self.results_dir)
            print(f"Saving checkpoints to {self.results_dir}")
            print(f"Logs written to {self.logfile}")
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} setup_for_pretraining_lamb()") from exc

    def set_PREC(self):
        PREC = ""
        if self.data_type == "fp16":
            PREC = "--amp"
        elif self.data_type == "fp32":
            PREC = "--noamp"
        elif self.data_type == "tf32":
            PREC = "--noamp"
        elif self.data_type == "manual_fp16":
            PREC = "--noamp --manual_fp16"
        else:
            raise Exception("Unknown <precision> argument")

        if self.use_xla:
            PREC += " --use_xla"
            print("XLA activated")
        else:
            PREC += " --nouse_xla"

        return PREC

    # Check if all necessary files are available before training
    def check_dirs(self, largs):
        try:
            if self.scaleout and is_valid_multi_node_config() and not self.kubernetes_run:
                check_dirs_path = Path(__file__).parent.parent.parent.parent.joinpath('central').joinpath('check_dirs.py')
                run_per_ip(f"{sys.executable} {str(check_dirs_path)} {largs}", ['MULTI_HLS_IPS', 'PYTHONPATH'], False)
            else:
                check_dirs.check_dirs_r(largs.split())
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} check_dirs(largs)") from exc

    def build_for_pretraining_lamb_phase1(self):
        try:
            pretrained_model_path = get_canonical_path(self.pretrained_model)
            bert_config = str(pretrained_model_path.joinpath("bert_config.json"))
            PREC = self.set_PREC()
            horovod_str = "--horovod" if self.scaleout is True else ""

            #PHASE 1
            gbs_phase1 = self.p1_batch_size * self.num_acc_steps_phase1
            seq_len = self.p1_max_seq_len
            max_pred_per_seq = 20
            results_dir_phase1 = self.results_dir + "/" + "phase_1"
            # run_per_ip
            results_phase1_path = get_canonical_path(results_dir_phase1)
            self.prepare_output_dir(results_dir_phase1)
            input_files_path = get_canonical_path(self.args.dataset_path).joinpath(f"seq_len_{seq_len}").joinpath("books_wiki_en_corpus/")
            # run_per_ip
            dir_list = ""
            dir_list += str(input_files_path)
            dir_list += " "
            dir_list += results_dir_phase1
            dir_list += " "
            dir_list += bert_config
            self.check_dirs(dir_list)

            input_files_dir = str(input_files_path.joinpath("training"))
            eval_files_dir = str(input_files_path.joinpath("test"))
            dllog_path = str(results_phase1_path.joinpath("bert_dllog.json"))
            run_pretraining_path = Path(__file__).parent.joinpath('run_pretraining.py')

            print(f"{self.__class__.__name__}: self.mpirun_cmd = {self.mpirun_cmd}")
            if self.mpirun_cmd == '':
                init_command = f"time python3 {str(run_pretraining_path)}"
            else:
                init_command = f"time {self.mpirun_cmd} python3 {str(run_pretraining_path)}"
            self.command = (
                f"{init_command}"
                f" --input_files_dir={input_files_dir}"
                f" --eval_files_dir={eval_files_dir}"
                f" --output_dir={str(results_phase1_path)}"
                f" --bert_config_file={bert_config}"
                f" --do_train=True"
                f" --do_eval=False"
                f" --train_batch_size={self.p1_batch_size}"
                f" --eval_batch_size={self.eval_batch_size}"
                f" --max_seq_length={seq_len}"
                f" --max_predictions_per_seq={max_pred_per_seq}"
                f" --num_train_steps={self.p1_steps}"
                f" --num_accumulation_steps={self.num_acc_steps_phase1}"
                f" --num_warmup_steps={self.p1_warmup}"
                f" --save_checkpoints_steps={self.save_checkpoints_steps}"
                f" --learning_rate={self.learning_rate_phase1}"
                f" {horovod_str} {PREC}"
                f" --allreduce_post_accumulation=True"
                f" --dllog_path={dllog_path}"
                f" --enable_scoped_allocator={self.enable_scoped_allocator}"
            )
            print("-------------------------------------------------------------------------\n")
            print("Running the Pre-Training :: Phase 1: Masked Language Model\n")
            print("-------------------------------------------------------------------------")
            print('bert_pretraining_bookswiki_utils::self.command for Phase1 = ', self.command)
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} build_for_pretraining_lamb_phase1()") from exc

    def build_for_pretraining_lamb_phase2(self):
        try:
            pretrained_model_path = get_canonical_path(self.pretrained_model)
            bert_config = str(pretrained_model_path.joinpath("bert_config.json"))
            PREC = self.set_PREC()
            horovod_str = "--horovod" if self.scaleout is True else ""

            #PHASE 1 Config
            gbs_phase1 = self.p1_batch_size * self.num_acc_steps_phase1
            PHASE1_CKPT = get_canonical_path(self.results_dir).joinpath("phase_1").joinpath(f"model.ckpt-{self.p1_steps}")

            #PHASE 2
            seq_len = self.p2_max_seq_len
            max_pred_per_seq = 80
            gbs_phase2 = self.p2_batch_size * self.num_acc_steps_phase2

            if self.args.fast_perf_only != 1:
                # Adjust for batch size
                self.p2_steps = int((self.p2_steps * gbs_phase1) / gbs_phase2)

            results_dir_phase2 = self.results_dir + "/" + "phase_2"
            # run_per_ip
            results_phase2_path = get_canonical_path(results_dir_phase2)
            self.prepare_output_dir(results_dir_phase2)
            input_files_path = get_canonical_path(self.args.dataset_path).joinpath(f"seq_len_{seq_len}").joinpath("books_wiki_en_corpus/")
            # run_per_ip
            dir_list = ""
            dir_list += str(input_files_path)
            dir_list += " "
            dir_list += str(results_dir_phase2)
            dir_list += " "
            dir_list += bert_config
            dir_list += " "
            dir_list += f"{str(PHASE1_CKPT)}.meta"
            self.check_dirs(dir_list)

            input_files_dir = str(input_files_path.joinpath("training"))
            eval_files_dir = str(input_files_path.joinpath("test"))
            dllog_path = str(results_phase2_path.joinpath("bert_dllog.json"))
            run_pretraining_path = Path(__file__).parent.joinpath('run_pretraining.py')

            print(f"{self.__class__.__name__}: self.mpirun_cmd = {self.mpirun_cmd}")
            if self.mpirun_cmd == '':
                init_command = f"time python3 {str(run_pretraining_path)}"
            else:
                init_command = f"time {self.mpirun_cmd} python3 {str(run_pretraining_path)}"
            self.command = (
                f"{init_command}"
                f" --input_files_dir={input_files_dir}"
                f" --init_checkpoint={str(PHASE1_CKPT)}"
                f" --eval_files_dir={eval_files_dir}"
                f" --output_dir={str(results_phase2_path)}"
                f" --bert_config_file={bert_config}"
                f" --do_train=True"
                f" --do_eval=False"
                f" --train_batch_size={self.p2_batch_size}"
                f" --eval_batch_size={self.eval_batch_size}"
                f" --max_seq_length={seq_len}"
                f" --max_predictions_per_seq={max_pred_per_seq}"
                f" --num_train_steps={self.p2_steps}"
                f" --num_accumulation_steps={self.num_acc_steps_phase2}"
                f" --num_warmup_steps={self.p2_warmup}"
                f" --save_checkpoints_steps={self.save_checkpoints_steps}"
                f" --learning_rate={self.learning_rate_phase2}"
                f" {horovod_str} {PREC}"
                f" --allreduce_post_accumulation=True"
                f" --dllog_path={dllog_path}"
            )
            print("-------------------------------------------------------------------------\n")
            print("Running the Pre-Training :: Phase 2: Next Sentence Prediction\n")
            print("-------------------------------------------------------------------------")
            print('bert_pretraining_bookswiki_utils::self.command for Phase2 = ', self.command)
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} build_for_pretraining_lamb_phase2()") from exc

    def run_pretraining_lamb_phase1(self):
        try:
            run_config_env_vars = self.get_env_vars()
            print('run_config_env_vars = ', run_config_env_vars)
            with HabanaEnvVariables(env_vars_to_set=run_config_env_vars), \
                 HabanaEnvVariables(env_vars_to_set=self.bookswiki_habana_env_variables):
                self.setup_for_pretraining_lamb()
                # Run phase1
                out_fid = open(self.logfile, 'a')
                self.build_for_pretraining_lamb_phase1()
                print_env_info(self.command, run_config_env_vars)
                print_env_info(self.command, self.bookswiki_habana_env_variables)
                print(f"{self.__class__.__name__} run_pretraining_lamb_phase1(): self.command for Phase1 = {self.command}")
                sys.stdout.flush()
                sys.stderr.flush()
                with subprocess.Popen(self.command, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True) as proc:
                    for line in proc.stdout:
                        sys.stdout.write(line)
                        out_fid.write(line)
                        sys.stdout.flush()
                        out_fid.flush()
                    proc.wait()

                sys.stdout.flush()
                sys.stderr.flush()
                out_fid.flush()
                out_fid.close()
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} run_pretraining_lamb_phase1()") from exc

    def run_pretraining_lamb_phase2(self):
        try:
            run_config_env_vars = self.get_env_vars()
            print('run_config_env_vars = ', run_config_env_vars)
            with HabanaEnvVariables(env_vars_to_set=run_config_env_vars), \
                 HabanaEnvVariables(env_vars_to_set=self.bookswiki_habana_env_variables):
                if self.kubernetes_run and self.scaleout and self.args.run_phase2:
                    self.setup_for_pretraining_lamb()
                # Run phase2
                out_fid = open(self.logfile, 'a')
                self.build_for_pretraining_lamb_phase2()
                print_env_info(self.command, run_config_env_vars)
                print_env_info(self.command, self.bookswiki_habana_env_variables)
                print(f"{self.__class__.__name__} run_pretraining_lamb_phase2(): self.command for Phase2 = {self.command}")
                sys.stdout.flush()
                sys.stderr.flush()
                with subprocess.Popen(self.command, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True) as proc:
                    for line in proc.stdout:
                        sys.stdout.write(line)
                        out_fid.write(line)
                        sys.stdout.flush()
                        out_fid.flush()
                    proc.wait()
                out_fid.close()
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} run_pretraining_lamb_phase2()") from exc

    def run(self):
        try:
            if self.kubernetes_run and self.scaleout and self.args.run_phase2:
                pass
            else:
                self.run_pretraining_lamb_phase1()

            if self.kubernetes_run and self.scaleout and not self.args.run_phase2:
                pass
            else:
                self.run_pretraining_lamb_phase2()
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} run()") from exc
