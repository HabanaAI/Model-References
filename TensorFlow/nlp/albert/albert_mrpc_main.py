###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
import sys
from pathlib import Path
import subprocess
from central.training_run_config import TrainingRunHWConfig
from central.multi_node_utils import run_per_ip, is_valid_multi_node_config
import TensorFlow.nlp.bert.download.download_dataset as download_dataset
import central.prepare_output_dir as prepare_output_dir

class AlbertFinetuningMRPC(TrainingRunHWConfig):
    def __init__(self, scaleout, num_workers_per_hls, hls_type, kubernetes_run, args, train_steps, warmup_steps, batch_size, max_seq_len, pretrained_model, enable_scoped_allocator, save_ckpt_steps):
        super(AlbertFinetuningMRPC, self).__init__(scaleout, num_workers_per_hls, hls_type, kubernetes_run, "demo_albert_log")
        self.args = args
        self.train_steps = train_steps
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.pretrained_model = pretrained_model
        self.enable_scoped_allocator = enable_scoped_allocator
        self.save_checkpoints_steps = save_ckpt_steps
        self.command = ''
        self.mrpc_habana_env_variables = {}
        if self.args.dataset_path is None:
            self.args.dataset_path = Path(__file__).parent.joinpath("MRPC")


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
            vocab_path = str(pretrained_model_path.joinpath("30k-clean.vocab"))
            spm_path = str(pretrained_model_path.joinpath("30k-clean.model"))
            bcfg_path = str(pretrained_model_path.joinpath("albert_config.json"))
            ic_path = str(pretrained_model_path.joinpath("model.ckpt-best"))
            data_dir = self.args.dataset_path
            if not os.path.exists(data_dir + '/MRPC/train.tsv'):
                if data_dir[-1] == "/":
                    data_dir = data_dir[:-1]
                if len(data_dir) >= 5 and data_dir[-5:] == "/MRPC" and os.path.exists(data_dir + '/train.tsv'):
                    data_dir = data_dir[:-5]
                else :
                    Exception(f"{socket.gethostname()}: Error: {data_dir} does not contain train.tsv file")

            print(f"{self.__class__.__name__}: self.mpirun_cmd = {self.mpirun_cmd}")
            if self.mpirun_cmd == '':
                init_command = f"time {sys.executable} {str(run_classifier_path)}"
            else:
                init_command = f"time {self.mpirun_cmd} {sys.executable} {str(run_classifier_path)}"
            self.command = (
                f"{init_command}"
                f" --task_name=MRPC --do_train=true --do_eval=true --data_dir={data_dir}"
                f" --vocab_file={vocab_path}"
                f" --albert_config_file={bcfg_path}"
                f" --spm_model_file={spm_path}"
                f" --init_checkpoint={ic_path}"
                f" --max_seq_length={self.max_seq_len}"
                f" --train_batch_size={self.batch_size}"
                f" --learning_rate={self.args.learning_rate}"
                f" --train_step={self.train_steps}"
                f" --warmup_step={self.warmup_steps}"
                f" --output_dir={self.args.output_dir}"
                f" --use_horovod={use_horovod_str}"
                f" --enable_scoped_allocator={self.enable_scoped_allocator}"
                f" --save_checkpoints_steps={self.save_checkpoints_steps}"
            )
            print('albert_mrpc_utils::self.command = ', self.command)
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} build_command()") from exc

    def run(self):
        try:
            self.download_dataset()
            self.prepare_output_dir()
            print("*** Running ALBERT training...\n\n")
            self.build_command()
            print(f"{self.__class__.__name__} run(): self.command = {self.command}")
            sys.stdout.flush()
            sys.stderr.flush()
            with subprocess.Popen(self.command, shell=True, executable='/bin/bash') as proc:
                proc.wait()
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} run()") from exc
