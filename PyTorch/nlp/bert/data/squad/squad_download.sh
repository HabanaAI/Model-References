#!/usr/bin/env bash

###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################
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

echo "Downloading dataset for squad..."

# Download SQuAD

v1="v1.1"
mkdir $v1
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O $v1/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O $v1/dev-v1.1.json
wget https://worksheets.codalab.org/rest/bundles/0xbcd57bee090b421c982906709c8c27e1/contents/blob/ -O $v1/evaluate-v1.1.py

EXP_TRAIN_v1='981b29407e0affa3b1b156f72073b945  -'
EXP_DEV_v1='3e85deb501d4e538b6bc56f786231552  -'
EXP_EVAL_v1='afb04912d18ff20696f7f88eed49bea9  -'
CALC_TRAIN_v1=`cat ${v1}/train-v1.1.json |md5sum`
CALC_DEV_v1=`cat ${v1}/dev-v1.1.json |md5sum`
CALC_EVAL_v1=`cat ${v1}/evaluate-v1.1.py |md5sum`


echo "Squad data download done!"

echo "Verifying Dataset...."

if [ "$EXP_TRAIN_v1" != "$CALC_TRAIN_v1" ]; then
    echo "train-v1.1.json is corrupted! md5sum doesn't match"
fi

if [ "$EXP_DEV_v1" != "$CALC_DEV_v1" ]; then
    echo "dev-v1.1.json is corrupted! md5sum doesn't match"
fi
if [ "$EXP_EVAL_v1" != "$CALC_EVAL_v1" ]; then
    echo "evaluate-v1.1.py is corrupted! md5sum doesn't match"
fi


echo "Complete!"
