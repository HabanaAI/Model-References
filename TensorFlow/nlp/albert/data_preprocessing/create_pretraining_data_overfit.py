###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
from pathlib import Path
import sys
import socket
import subprocess
from central.habana_model_runner_utils import get_canonical_path

def create_pretraining_data_overfit_r(dataset_path, pretrained_model, seq_length, max_pred_per_seq):
    host_name = socket.gethostname()
    try:
        ds_path = get_canonical_path(dataset_path)
        print("***")
        print(ds_path)
        if not os.path.isdir(ds_path):
            os.makedirs(ds_path, mode=0o777, exist_ok=True)
            run_create_pretraining_path = Path(__file__).parent.joinpath('create_pretraining_data.py')
            input_file_path = Path(__file__).parent.parent.joinpath("data/sample_text.txt")
            output_file_path = ds_path.joinpath("tf_examples.tfrecord")
            meta_data_file_path = ds_path.joinpath("tf_examples_meta_data")
            pretrained_model_path = get_canonical_path(pretrained_model)
            vocab_file_path = pretrained_model_path.joinpath("30k-clean.vocab")
            spm_file_path = pretrained_model_path.joinpath("30k-clean.model")

            command = (
                f"{sys.executable} {str(run_create_pretraining_path)}"
                f" --input_file={str(input_file_path)}"
                f" --output_file={str(output_file_path)}"
                f" --spm_model_file={str(spm_file_path)}"
                f" --vocab_file={str(vocab_file_path)}"
                f" --meta_data_file_path={str(meta_data_file_path)}"
                f" --do_lower_case"
                f" --max_seq_length={seq_length}"
                f" --random_seed=12345"
                f" --dupe_factor=5"
            )
            print(f"{host_name}: {__file__}: create_pretraining_data_overfit_r() command = {command}")
            sys.stdout.flush()
            sys.stderr.flush()
            with subprocess.Popen(command, shell=True, executable='/bin/bash') as proc:
                proc.wait()
    except Exception as exc:
        raise Exception(f"{host_name}: Error in {__file__} create_pretraining_data_overfit_r({dataset_path}, {pretrained_model}, {seq_length}, {max_pred_per_seq})") from exc

if __name__ == "__main__":
    host_name = socket.gethostname()
    print(f"{host_name}: In {sys.argv[0]}")
    print(f"{host_name}: called with arguments: \"{sys.argv[1]} {sys.argv[2]} {sys.argv[3]} {sys.argv[4]}\"")
    dataset_path = sys.argv[1]
    pretrained_model = sys.argv[2]
    seq_length = sys.argv[3]
    max_pred_per_seq = sys.argv[4]
    print(f"{host_name}: MULTI_HLS_IPS = {os.environ.get('MULTI_HLS_IPS')}")
    create_pretraining_data_overfit_r(dataset_path, pretrained_model, seq_length, max_pred_per_seq)
