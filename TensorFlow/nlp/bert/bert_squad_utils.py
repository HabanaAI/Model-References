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
import subprocess
from TensorFlow.common.habana_model_runner_utils import HabanaEnvVariables, print_env_info, get_canonical_path, get_canonical_path_str, is_valid_multi_node_config, get_multi_node_config_nodes
from TensorFlow.common.training_run_config import TrainingRunHWConfig
from TensorFlow.common.multi_node_utils import run_per_ip
import prepare_output_dir_squad

class BertFinetuningSQUAD(TrainingRunHWConfig):
    def __init__(self, args, epochs, batch_size, max_seq_len, pretrained_model):
        super(BertFinetuningSQUAD, self).__init__(args)
        self.args = args
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.pretrained_model = pretrained_model
        self.command = ''
        self.squad_habana_env_variables = {}
        if self.args.use_horovod is not None:
            # By default: HCL Streams:ON, ART:ON, SAO:ON
            self.squad_habana_env_variables = {'HABANA_USE_STREAMS_FOR_HCL' : 'true' ,
                                               'HABANA_USE_PREALLOC_BUFFER_FOR_ALLREDUCE' : 'true',
                                               'TF_DISABLE_SCOPED_ALLOCATOR' : 'true'}

        tf_bf16_conv_flag = os.environ.get('TF_ENABLE_BF16_CONVERSION')
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
            if self.use_horovod and is_valid_multi_node_config():
                prepare_output_dir_squad_path = Path(__file__).parent.joinpath('prepare_output_dir_squad.py')
                run_per_ip(f"python3 {str(prepare_output_dir_squad_path)} {self.args.output_dir} {self.batch_size} {self.max_seq_len}", ['MULTI_HLS_IPS', 'PYTHONPATH'], False)
            else:
                prepare_output_dir_squad.prepare_output_dir_squad_r(self.args.output_dir, self.batch_size, self.max_seq_len)
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} prepare_output_dir()") from exc

    def build_command(self):
        try:
            run_squad_path = Path(__file__).parent.joinpath('run_squad.py')
            pretrained_model_path = get_canonical_path(self.pretrained_model)
            use_horovod_str = "true" if self.use_horovod else "false"
            vocab_path = str(pretrained_model_path.joinpath("vocab.txt"))
            bcfg_path = str(pretrained_model_path.joinpath("bert_config.json"))
            ic_path = str(pretrained_model_path.joinpath("bert_model.ckpt"))
            squad_train_file = Path(__file__).parent.joinpath('train-v1.1.json')
            squad_predict_file = Path(__file__).parent.joinpath('dev-v1.1.json')

            print(f"{self.__class__.__name__}: self.mpirun_cmd = {self.mpirun_cmd}")
            if self.mpirun_cmd == '':
                init_command = f"time python3 {str(run_squad_path)}"
            else:
                init_command = f"time {self.mpirun_cmd} python3 {str(run_squad_path)}"
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
                f" --output_dir={get_canonical_path_str(self.args.output_dir)}"
                f" --use_horovod={use_horovod_str}"
            )
            print('bert_squad_utils::self.command = ', self.command)
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} build_command()") from exc

    def run(self):
        try:
            self.prepare_output_dir()
            print("*** Running BERT training...\n\n")

            run_config_env_vars = self.get_env_vars()
            print('run_config_env_vars = ', run_config_env_vars)
            with HabanaEnvVariables(env_vars_to_set=run_config_env_vars), \
                 HabanaEnvVariables(env_vars_to_set=self.squad_habana_env_variables):
                self.build_command()
                print_env_info(self.command, run_config_env_vars)
                print_env_info(self.command, self.squad_habana_env_variables)
                print(f"{self.__class__.__name__} run(): self.command = {self.command}")
                sys.stdout.flush()
                sys.stderr.flush()
                with subprocess.Popen(self.command, shell=True, executable='/bin/bash') as proc:
                    proc.wait()
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} run()") from exc
