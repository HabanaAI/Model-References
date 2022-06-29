###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
import sys
from pathlib import Path
import subprocess
from central.training_run_config import TrainingRunHWConfig
from central.multi_node_utils import run_per_ip, is_valid_multi_node_config
import prepare_output_dir_squad

class BertFinetuningSQUAD(TrainingRunHWConfig):
    def __init__(self, scaleout, num_workers_per_hls, hls_type, kubernetes_run, args, steps, epochs, batch_size, max_seq_len, pretrained_model, enable_scoped_allocator):
        super(BertFinetuningSQUAD, self).__init__(scaleout, num_workers_per_hls, hls_type, kubernetes_run, "demo_bert_log")
        self.args = args
        self.steps = steps
        self.iterations_per_loop = args.iterations_per_loop
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.pretrained_model = pretrained_model
        self.enable_scoped_allocator = args.enable_scoped_allocator
        self.iterations_per_loop_str = ''
        self.profiler_str = ''
        self.num_train_steps_str = ''
        if len(args.profile) > 0:
            self.profiler_str = "--profile=" + args.profile
        if self.iterations_per_loop == 1:
            self.iterations_per_loop_str = "--iterations_per_loop=" + str(self.iterations_per_loop)
        if self.steps is not None:
            self.num_train_steps_str = "--num_train_steps="+str(self.steps)
        self.command = ''
        self.squad_habana_env_variables = {}

        tf_bf16_conv_flag = os.environ.get('TF_BF16_CONVERSION')
        if tf_bf16_conv_flag is None or tf_bf16_conv_flag == '0':
            if self.args.model_variant == "large":
                self.squad_habana_env_variables['HABANA_INITIAL_WORKSPACE_SIZE_MB'] = '17257'
            else:
                self.squad_habana_env_variables['HABANA_INITIAL_WORKSPACE_SIZE_MB'] = '13271'
        else:
            if self.args.model_variant == "large":
                self.squad_habana_env_variables['HABANA_INITIAL_WORKSPACE_SIZE_MB'] = '21393'
            else:
                self.squad_habana_env_variables['HABANA_INITIAL_WORKSPACE_SIZE_MB'] = '17371'

    def prepare_output_dir(self):
        try:
            if self.scaleout and is_valid_multi_node_config() and not self.kubernetes_run:
                prepare_output_dir_squad_path = Path(__file__).parent.joinpath('prepare_output_dir_squad.py')
                run_per_ip(f"{sys.executable} {str(prepare_output_dir_squad_path)} {self.args.output_dir} {self.batch_size} {self.max_seq_len}", ['MULTI_HLS_IPS', 'PYTHONPATH'], False)
            else:
                prepare_output_dir_squad.prepare_output_dir_squad_r(self.args.output_dir, self.batch_size, self.max_seq_len)
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} prepare_output_dir()") from exc

    def build_command(self):
        try:
            run_squad_path = Path(__file__).parent.joinpath('run_squad.py')
            pretrained_model_path = Path(self.pretrained_model)
            use_horovod_str = "true" if self.scaleout else "false"
            vocab_path = str(pretrained_model_path.joinpath("vocab.txt"))
            bcfg_path = str(pretrained_model_path.joinpath("bert_config.json"))
            ic_path = str(pretrained_model_path.joinpath("bert_model.ckpt"))
            squad_train_file = Path(self.args.dataset_path).joinpath('train-v1.1.json')
            squad_predict_file = Path(self.args.dataset_path).joinpath('dev-v1.1.json')

            print(f"{self.__class__.__name__}: self.mpirun_cmd = {self.mpirun_cmd}")
            if self.mpirun_cmd == '':
                init_command = f"time {sys.executable} {str(run_squad_path)}"
            else:
                init_command = f"time {self.mpirun_cmd} {sys.executable} {str(run_squad_path)}"
            self.command = (
                f"{init_command}"
                f" --vocab_file={vocab_path}"
                f" --bert_config_file={bcfg_path}"
                f" --init_checkpoint={ic_path}"
                f" --do_train=True"
                f" --train_file={squad_train_file}"
                f" --do_predict=True"
                f" --predict_file={squad_predict_file}"
                f" --do_eval=True"
                f" --train_batch_size={self.batch_size}"
                f" --learning_rate={self.args.learning_rate}"
                f" --num_train_epochs={self.epochs}"
                f" --max_seq_length={self.max_seq_len}"
                f" --doc_stride=128"
                f" --output_dir={self.args.output_dir}"
                f" --use_horovod={use_horovod_str}"
                f" --enable_scoped_allocator={self.enable_scoped_allocator}"
                f" {self.iterations_per_loop_str}"
                f" {self.profiler_str}"
                f" {self.num_train_steps_str}"
            )
            print('bert_squad_utils::self.command = ', self.command)
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} build_command()") from exc

    def run(self):
        try:
            self.prepare_output_dir()
            print("*** Running BERT training...\n\n")

            print("Running with the following env variables: HABANA_INITIAL_WORKSPACE_SIZE_MB={}".format(list(self.squad_habana_env_variables.values())[0]))
            squad_env_vars = os.environ.copy()
            squad_env_vars['HABANA_INITIAL_WORKSPACE_SIZE_MB'] = list(self.squad_habana_env_variables.values())[0]
            self.build_command()
            print(f"{self.__class__.__name__} run(): self.command = {self.command}")
            sys.stdout.flush()
            sys.stderr.flush()
            with subprocess.Popen(self.command, shell=True, executable='/bin/bash', env=squad_env_vars) as proc:
                return_code = proc.wait()
            # catch all error codes other than 0 (succesfull)
            if return_code != 0:
                raise RuntimeError(f"Subprocess exited with return code: {return_code}")
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} run()") from exc
