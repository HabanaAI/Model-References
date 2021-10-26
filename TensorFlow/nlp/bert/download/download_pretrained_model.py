###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
from pathlib import Path
import sys
import socket
import urllib.request
import zipfile
from central.habana_model_runner_utils import get_canonical_path
from central.multi_node_utils import run_cmd_as_subprocess

def download_pretrained_model_r(pretrained_url, pretrained_model, flatten_archive=False):
    host_name = socket.gethostname()
    this_dir = get_canonical_path(os.curdir)
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

if __name__ == "__main__":
    host_name = socket.gethostname()
    print(f"{host_name}: In {sys.argv[0]}")
    print(f"{host_name}: called with arguments: \"{sys.argv[1]} {sys.argv[2]} {sys.argv[3]}\"")
    pretrained_url = str(sys.argv[1])
    pretrained_model = str(sys.argv[2])
    flatten_archive_str = str(sys.argv[3])
    if flatten_archive_str == "True":
        flatten_archive = True
    else:
        flatten_archive = False
    print(f"{host_name}: MULTI_HLS_IPS = {os.environ.get('MULTI_HLS_IPS')}")
    download_pretrained_model_r(pretrained_url, pretrained_model, flatten_archive)
