#! /bin/bash

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
# - Removed NVidia container build version message
###############################################################################

train_batch_size_phase1=${1:-64}
train_batch_size_phase2=${2:-8}
eval_batch_size=${3:-8}
learning_rate_phase1=${4:-"7.5e-4"}
learning_rate_phase2=${5:-"5e-4"}
precision=${6:-"fp16"}
use_xla=${7:-"true"}
num_hpus=${8:-8}
warmup_steps_phase1=${9:-"2000"}
warmup_steps_phase2=${10:-"200"}
train_steps_phase1=${11:-7038}
train_steps_phase2=${12:-782}
save_checkpoints_steps=${13:-100}
num_accumulation_steps_phase1=${14:-128}
num_accumulation_steps_phase2=${15:-512}
bert_model=${16:-"large"}

#Edit to save logs & checkpoints in a different directory
RESULTS_DIR=${RESULTS_DIR:-/results}

export BERT_CONFIG=${PRETRAINED_MODEL}/bert_config.json

PREC=""
if [ "$precision" = "fp16" ] ; then
   PREC="--amp"
elif [ "$precision" = "fp32" ] ; then
   PREC="--noamp"
elif [ "$precision" = "tf32" ] ; then
   PREC="--noamp"
elif [ "$precision" = "manual_fp16" ] ; then
   PREC="--noamp --manual_fp16"
else
   echo "Unknown <precision> argument"
   exit -2
fi

if [ "$use_xla" = "true" ] ; then
    PREC="$PREC --use_xla"
    echo "XLA activated"
else
    PREC="$PREC --nouse_xla"
fi

horovod_str=""
if [ $USE_HOROVOD == "true" ]; then
   horovod_str="--horovod"
fi

#PHASE 1

gbs_phase1=$(expr $train_batch_size_phase1 \* $num_accumulation_steps_phase1)
seq_len=$P1_MAX_SEQ_LENGTH
max_pred_per_seq=20
RESULTS_DIR_PHASE1=${RESULTS_DIR}/phase_1
run_per_ip mkdir -m 777 -p $RESULTS_DIR_PHASE1

INPUT_FILES_PREFIX=${INPUT_FILES_PREFIX:-/software/data/tf/data/bert/}
INPUT_FILES=${INPUT_FILES_PREFIX}/books_wiki_en_corpus/tfrecord/seq_len_$seq_len/books_wiki_en_corpus/


function check_dirs()
{
   # Check if all necessary files are available before training
   for DIR_or_file in $DATA_DIR $RESULTS_DIR_PHASE1 $BERT_CONFIG; do
   if [ ! -d "$DIR_or_file" ] && [ ! -f "$DIR_or_file" ]; then
      echo "Error! $DIR_or_file file or directory missing. Please mount correctly"
      exit -1
   fi
   done
}

run_per_ip check_dirs || exit -1

echo -------------------------------------------------------------------------
echo "Running the Pre-Training :: Phase 1: Masked Language Model"
echo
echo "python3 $SCRIPT_DIR/run_pretraining.py"
echo "    input_files_dir=$INPUT_FILES/training"
echo "    eval_files_dir=$INPUT_FILES/test"
echo "    output_dir=$RESULTS_DIR_PHASE1"
echo "    bert_config_file=$BERT_CONFIG"
echo "    do_train=True"
echo "    do_eval=False"
echo "    train_batch_size=$train_batch_size_phase1"
echo "    eval_batch_size=$eval_batch_size"
echo "    max_seq_length=$seq_len"
echo "    max_predictions_per_seq=$max_pred_per_seq"
echo "    num_train_steps=$train_steps_phase1"
echo "    num_accumulation_steps=$num_accumulation_steps_phase1"
echo "    num_warmup_steps=$warmup_steps_phase1"
echo "    save_checkpoints_steps=$save_checkpoints_steps"
echo "    learning_rate=$learning_rate_phase1"
echo "    $horovod_str $PREC"
echo "    allreduce_post_accumulation=True"
echo "    dllog_path=$RESULTS_DIR_PHASE1/bert_dllog.json"
echo -------------------------------------------------------------------------

time $MPIRUN_CMD python3 $SCRIPT_DIR/run_pretraining.py \
     --input_files_dir=$INPUT_FILES/training \
     --eval_files_dir=$INPUT_FILES/test \
     --output_dir=$RESULTS_DIR_PHASE1 \
     --bert_config_file=$BERT_CONFIG \
     --do_train=True \
     --do_eval=False \
     --train_batch_size=$train_batch_size_phase1 \
     --eval_batch_size=$eval_batch_size \
     --max_seq_length=$seq_len \
     --max_predictions_per_seq=$max_pred_per_seq \
     --num_train_steps=$train_steps_phase1 \
     --num_accumulation_steps=$num_accumulation_steps_phase1 \
     --num_warmup_steps=$warmup_steps_phase1 \
     --save_checkpoints_steps=$save_checkpoints_steps \
     --learning_rate=$learning_rate_phase1 \
     $horovod_str $PREC \
     --allreduce_post_accumulation=True \
     --dllog_path=$RESULTS_DIR_PHASE1/bert_dllog.json
