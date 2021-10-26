###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
from pathlib import Path
import sys
import socket
import urllib.request
import tarfile
from central.habana_model_runner_utils import get_canonical_path
from central.multi_node_utils import run_cmd_as_subprocess

def download_pretrained_model_r(pretrained_url, pretrained_model):
    host_name = socket.gethostname()
    this_dir = get_canonical_path(os.curdir)
    try:
        os.chdir(Path(__file__).parent.parent)
        if not os.path.isdir(pretrained_model):
            _wget = False
            if os.path.exists(pretrained_model + ".tar.gz") == False:
                _wget = True
            else:
                if os.path.getsize(pretrained_model + ".tar.gz") == 0:
                    print(f"{host_name}: *** Broken file, needs download ...\n\n")
                    _wget = True
            if _wget == True:
                print(f"{host_name}: *** Downloading pre-trained model...\n\n")
                inf = urllib.request.urlopen(pretrained_url + pretrained_model + ".tar.gz")
                with open(pretrained_model + ".tar.gz", "wb") as outf:
                    outf.write(inf.read())

            print(f"{host_name}: *** Extracting pre-trained model...\n\n")
            cmd =f"mkdir {pretrained_model} && tar -xf {pretrained_model}.tar.gz -C {pretrained_model} --strip-components=1"
            run_cmd_as_subprocess(cmd)
            #with tarfile.open(pretrained_model + ".tar.gz", 'r:gz') as tar_ref:
                #tar_ref.extractall(pretrained_model)
                #for member in tar_ref.getmembers():
                    #tar_ref.extractfile(member)
            if _wget == True:
                cmd = f"rm -f {pretrained_model}.tar.gz"
                run_cmd_as_subprocess(cmd)
        else:
            print(f"{host_name}: Reusing existing pre-trained model directory \'{pretrained_model}\'")
        os.chdir(this_dir)
    except Exception as exc:
        os.chdir(this_dir)
        raise Exception(f"{host_name}: Error in {__file__} download_pretrained_model()") from exc

if __name__ == "__main__":
    host_name = socket.gethostname()
    print(f"{host_name}: In {sys.argv[0]}")
    print(f"{host_name}: called with arguments: \"{sys.argv[1]} {sys.argv[2]}\"")
    pretrained_url = sys.argv[1]
    pretrained_model = sys.argv[2]
    print(f"{host_name}: MULTI_HLS_IPS = {os.environ.get('MULTI_HLS_IPS')}")
    download_pretrained_model_r(pretrained_url, pretrained_model)
