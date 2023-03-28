###############################################################################
# Copyright (C) 2021-2022 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
import sys
import json
import socket
import zipfile
import urllib.request
from pathlib import Path
import subprocess

import tensorflow as tf
from transformers import TFDistilBertForTokenClassification


def run_cmd_as_subprocess(cmd=str):
    print(cmd)
    sys.stdout.flush()
    sys.stderr.flush()
    with subprocess.Popen(cmd, shell=True, executable='/bin/bash') as proc:
        proc.wait()


def download_pretrained_model_r(pretrained_url, pretrained_model, flatten_archive=False):
    host_name = socket.gethostname()
    this_dir = os.getcwd()
    try:
        os.chdir(Path(__file__).parent.parent)
        if not os.path.isdir(pretrained_model):
            _wget = False
            if os.path.exists(pretrained_model + ".zip") == False:
                _wget = True
            else:
                if os.path.getsize(pretrained_model + ".zip") == 0:
                    print(f"{host_name}: *** Broken file, needs download ...\n\n")
                    _wget = True
            if _wget == True:
                print(f"{host_name}: *** Downloading pre-trained model...\n\n")
                inf = urllib.request.urlopen(pretrained_url + pretrained_model + ".zip")
                with open(pretrained_model + ".zip", "wb") as outf:
                    outf.write(inf.read())

            print(f"{host_name}: *** Extracting pre-trained model...\n\n")
            with zipfile.ZipFile(pretrained_model + ".zip", 'r') as zip_ref:
                if flatten_archive:
                    # large model is zipped with subdirectory, flatten archive tree structure
                    for member in zip_ref.infolist():
                        # skip directories
                        if member.is_dir():
                            continue
                        zip_ref.extract(member)
                else:
                    zip_ref.extractall(pretrained_model)

            if _wget == True:
                cmd = f"rm -f {pretrained_model}.zip"
                run_cmd_as_subprocess(cmd)
        else:
            print(f"{host_name}: Reusing existing pre-trained model directory \'{pretrained_model}\'")
        os.chdir(this_dir)
    except Exception as exc:
        os.chdir(this_dir)
        raise Exception(f"{host_name}: Error in {__file__} download_pretrained_model()") from exc

def download_pretrained_model_distilbert(pretrained_model_bert, pretrained_model_distilbert='distilbert-base-uncased'):
    host_name = socket.gethostname()
    this_dir = os.getcwd()
    pretrained_model_bert_path = Path(pretrained_model_bert)
    try:
        # Write distilbert config json file to path
        print(f"{host_name}: *** Generating distilbert_config.json...")
        bcfg_path = str(pretrained_model_bert_path.joinpath("bert_config.json"))
        bert_config = json.load(open(bcfg_path, 'r'))
        bert_config["type_vocab_size"] = 16
        bert_config["num_hidden_layers"] = 6
        distilbcfg_path = str(pretrained_model_bert_path.joinpath("distilbert_config.json"))
        distilbert_file = open(distilbcfg_path, "w")
        json.dump(bert_config, distilbert_file)
        distilbert_file.close()

        # Download pre-trained distilbert model
        if not os.path.isfile(str(pretrained_model_bert_path.joinpath("distilbert-base-uncased.ckpt-1.index"))):
            print(f"{host_name}: *** Downloading pre-trained distilbert model...")
            model = TFDistilBertForTokenClassification.from_pretrained(pretrained_model_distilbert)
            model.compile()
            ckpt_prefix = os.path.join(pretrained_model_bert_path, 'distilbert-base-uncased.ckpt')
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.save(file_prefix=ckpt_prefix)
        else:
            print(f"{host_name}: *** Reusing existing pre-trained model 'distilbert-base-uncased'")
    except Exception as exc:
        os.chdir(this_dir)
        raise Exception(f"{host_name}: Error in {__file__} download_pretrained_model_distilbert()") from exc

if __name__ == "__main__":
    host_name = socket.gethostname()
    print(f"{host_name}: In {sys.argv[0]}")
    print(f"{host_name}: called with arguments: \"{sys.argv[1]} {sys.argv[2]} {sys.argv[3]} {sys.argv[4]}\"")
    pretrained_url = str(sys.argv[1])
    pretrained_model_bert = str(sys.argv[2])
    pretrained_model_distilbert = str(sys.argv[3]) # need to be 'distilbert-base-uncased'
    flatten_archive_str = str(sys.argv[4])
    if flatten_archive_str == "True":
        flatten_archive = True
    else:
        flatten_archive = False
    print(f"{host_name}: MULTI_HLS_IPS = {os.environ.get('MULTI_HLS_IPS')}")
    download_pretrained_model_r(pretrained_url, pretrained_model_bert, flatten_archive)
    download_pretrained_model_distilbert(pretrained_model_bert, pretrained_model_distilbert)
