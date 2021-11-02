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
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
#
# Changes:
# - Modified path hard-coded for Nvidia container
###############################################################################

import subprocess

class BooksDownloader:
    def __init__(self, save_path):
        self.save_path = save_path
        pass


    def download(self):
        import os
        working_dir = os.environ['BERT_PREP_WORKING_DIR']
        args = '--list ' + working_dir + '/bookcorpus/url_list.jsonl --out'
        bookscorpus_download_command = 'python3 ' + working_dir + '/bookcorpus/download_files.py ' + args
        bookscorpus_download_command += ' ' + self.save_path + '/bookscorpus'
        bookscorpus_download_command += ' --trash-bad-count'
        print("Downloading BookCorpus command: ", bookscorpus_download_command)
        bookscorpus_download_process = subprocess.run(bookscorpus_download_command, shell=True, check=True)