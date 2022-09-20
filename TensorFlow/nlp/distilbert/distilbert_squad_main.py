###############################################################################
# Copyright (C) 2021-2022 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
import sys
import json
import subprocess
from pathlib import Path

import tensorflow as tf
from transformers import TFDistilBertForTokenClassification

from central.training_run_config import TrainingRunHWConfig
from central.multi_node_utils import run_per_ip, is_valid_multi_node_config
import prepare_output_dir_squad

class DistilBertFinetuningSQUAD(TrainingRunHWConfig):
    def __init__(self, scaleout, num_workers_per_hls, hls_type, kubernetes_run, args, epochs, batch_size, max_seq_len, pretrained_model, enable_scoped_allocator):
        super(DistilBertFinetuningSQUAD, self).__init__(scaleout, num_workers_per_hls, hls_type, kubernetes_run, "demo_distilbert_log")
        self.args = args
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.pretrained_model = pretrained_model
        self.enable_scoped_allocator = args.enable_scoped_allocator
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

            # Write distilbert config json file to path
            bcfg_path = str(pretrained_model_path.joinpath("bert_config.json"))
            bert_config = json.load(open(bcfg_path, 'r'))
            bert_config["type_vocab_size"] = 16
            bert_config["num_hidden_layers"] = 6
            distilbcfg_path = str(pretrained_model_path.joinpath("distilbert_config.json"))
            distilbert_file = open(distilbcfg_path, "w")
            json.dump(bert_config, distilbert_file)
            distilbert_file.close()

            # Download huggingface pretrained distilbert_base_uncased model
            if not os.path.isfile(str(pretrained_model_path.joinpath("distilbert-base-uncased.ckpt-1.index"))):
                model = TFDistilBertForTokenClassification.from_pretrained('distilbert-base-uncased')
                model.compile()
                ckpt_prefix = os.path.join(pretrained_model_path, 'distilbert-base-uncased.ckpt')
                checkpoint = tf.train.Checkpoint(model=model)
                checkpoint.save(file_prefix=ckpt_prefix)
            print ("Reusing existing pre-trained model 'distilbert-base-uncased'")
            ic_path = str(pretrained_model_path.joinpath("distilbert-base-uncased.ckpt-1"))

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
                f" --distilbert_config_file={distilbcfg_path}"
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
                f" --deterministic_run={self.args.deterministic_run}"
            )
            if self.args.deterministic_seed:
                self.command += f" --deterministic_seed={self.args.deterministic_seed}"
            print('bert_squad_utils::self.command = ', self.command)
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} build_command()") from exc

    def run(self):
        try:
            self.prepare_output_dir()
            print("*** Running DistilBERT finetuning...\n\n")

            print("Running with the following env variables: HABANA_INITIAL_WORKSPACE_SIZE_MB={}".format(list(self.squad_habana_env_variables.values())[0]))
            squad_env_vars = os.environ.copy()
            squad_env_vars['HABANA_INITIAL_WORKSPACE_SIZE_MB'] = list(self.squad_habana_env_variables.values())[0]
            self.build_command()
            print(f"{self.__class__.__name__} run(): self.command = {self.command}")
            sys.stdout.flush()
            sys.stderr.flush()
            with subprocess.Popen(self.command, shell=True, executable='/bin/bash', env=squad_env_vars) as proc:
                proc.wait()
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} run()") from exc
