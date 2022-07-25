###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
import sys
from pathlib import Path
import socket
import subprocess
from central.training_run_config import TrainingRunHWConfig
from central.multi_node_utils import run_per_ip, is_valid_multi_node_config
import TensorFlow.nlp.bert.download.download_dataset as download_dataset
import central.prepare_output_dir as prepare_output_dir

class BertFinetuningMRPC(TrainingRunHWConfig):
    def __init__(self, scaleout, num_workers_per_hls, hls_type, kubernetes_run, args, epochs, batch_size, max_seq_len, pretrained_model, enable_scoped_allocator):
        super(BertFinetuningMRPC, self).__init__(scaleout, num_workers_per_hls, hls_type, kubernetes_run, "demo_bert_log")
        self.args = args
        self.iterations_per_loop = self.args.iterations_per_loop
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.pretrained_model = pretrained_model
        self.enable_scoped_allocator = enable_scoped_allocator
        self.iterations_per_loop_str = ''
        self.profiler_str = ''
        self.num_train_steps_str = ''
        if len(args.profile) > 0:
            self.profiler_str = "--profile=" + args.profile
        if self.iterations_per_loop > 0:
            self.iterations_per_loop_str = "--iterations_per_loop=" + str(self.iterations_per_loop)

        self.command = ''

    # Download the dataset on all remote IPs
    def download_dataset(self):
        try:
            if self.scaleout and is_valid_multi_node_config() and not self.kubernetes_run:
                download_dataset_path = Path(__file__).parent.joinpath('download').joinpath('download_dataset.py')
                run_per_ip(f"{sys.executable} {str(download_dataset_path)} {self.args.dataset_path}", ['MULTI_HLS_IPS', 'PYTHONPATH'], False)
            else:
                download_dataset.download_dataset_r(self.args.dataset_path)
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} download_dataset()") from exc

    # Prepare the output directory on all remote IPs
    def prepare_output_dir(self):
        try:
            if self.scaleout and is_valid_multi_node_config() and not self.kubernetes_run:
                prepare_output_dir_path = Path(__file__).parent.parent.parent.parent.joinpath('central').joinpath('prepare_output_dir.py')
                run_per_ip(f"{sys.executable} {str(prepare_output_dir_path)} {self.args.output_dir}", ['MULTI_HLS_IPS', 'PYTHONPATH'], False)
            else:
                prepare_output_dir.prepare_output_dir_r(self.args.output_dir)
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} prepare_output_dir()") from exc

    def build_command(self):
        try:
            run_classifier_path = Path(__file__).parent.joinpath('run_classifier.py')
            pretrained_model_path = Path(self.pretrained_model)
            use_horovod_str = "true" if self.scaleout else "false"
            vocab_path = str(pretrained_model_path.joinpath("vocab.txt"))
            bcfg_path = str(pretrained_model_path.joinpath("bert_config.json"))
            ic_path = str(pretrained_model_path.joinpath("bert_model.ckpt"))
            ds_path = self.args.dataset_path
            if not os.path.exists(ds_path + '/train.tsv'):
                if os.path.exists(ds_path + '/MRPC/train.tsv'):
                    ds_path += '/MRPC/'
                else:
                    raise Exception(f"{socket.gethostname()}: Error: {ds_path} does not contain train.tsv file")

            print(f"{self.__class__.__name__}: self.mpirun_cmd = {self.mpirun_cmd}")
            if self.mpirun_cmd == '':
                init_command = f"time {sys.executable} {str(run_classifier_path)}"
            else:
                init_command = f"time {self.mpirun_cmd} {sys.executable} {str(run_classifier_path)}"
            self.command = (
                f"{init_command}"
                f" --task_name=MRPC --do_train=true --do_eval=true --data_dir={ds_path}"
                f" --vocab_file={vocab_path}"
                f" --bert_config_file={bcfg_path}"
                f" --init_checkpoint={ic_path}"
                f" --max_seq_length={self.max_seq_len}"
                f" --train_batch_size={self.batch_size}"
                f" --learning_rate={self.args.learning_rate}"
                f" --num_train_epochs={self.epochs}"
                f" --output_dir={self.args.output_dir}"
                f" --use_horovod={use_horovod_str}"
                f" --enable_scoped_allocator={self.enable_scoped_allocator}"
                f" {self.iterations_per_loop_str}"
                f" {self.profiler_str}"
            )
            print('bert_mrpc_utils::self.command = ', self.command)
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} build_command()") from exc

    def run(self):
        try:
            self.download_dataset()
            self.prepare_output_dir()
            print("*** Running BERT training...\n\n")

            print("Running with the env variable: 'HABANA_INITIAL_WORKSPACE_SIZE_MB'='15437'")
            mrpc_env_vars = os.environ.copy()
            mrpc_env_vars["HABANA_INITIAL_WORKSPACE_SIZE_MB"] = "15437"

            self.build_command()
            print(f"{self.__class__.__name__} run(): self.command = {self.command}")
            sys.stdout.flush()
            sys.stderr.flush()
            with subprocess.Popen(self.command, shell=True, executable='/bin/bash', env=mrpc_env_vars) as proc:
                proc.wait()
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} run()") from exc
