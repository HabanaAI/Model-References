#!/usr/bin/env bash

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

train_batch_size_phase2=${1:-14}
global_batch_size_phase2=${2:-448}
block_size_phase2=${3:-150080}
eval_batch_size=${4:-48}
learning_rate_phase2=${5:-"5e-5"}
beta1_phase2=${6:-0.9}
beta2_phase2=${7:-0.999}
precision=${8:-"fp16"}
use_xla=${9:-"true"}
num_hpus=${10:-8}
warmup_steps_phase2=${11:-"0"}
train_steps_phase2=${12:-8103}
num_accumulation_steps_phase2=${13:-4}
bert_model=${14:-"large"}
is_dist_eval_enabled=${15:-"false"}
eval_only=${16:-"false"}
use_lightweight_checkpoint=${17:-"false"}

printf -v TAG "tf_bert_pretraining_lamb_%s_%s_gbs2%d" "$bert_model" "$precision"  $global_batch_size_phase2
DATESTAMP=`date +'%y%m%d%H%M%S'`

samples_between_eval=$block_size_phase2

if [[ $((block_size_phase2 % global_batch_size_phase2)) -ne 0 ]]; then
    echo "Incorrect training configuration - block size ($block_size_phase2) " \
         "must be a multiple of global batch size ($global_batch_size_phase2)"
    exit 1;
fi

save_checkpoints_steps=$(expr $block_size_phase2 / $global_batch_size_phase2)
stop_threshold="0.72"
samples_start_eval=0
epochs=$(echo "( $train_steps_phase2 * $global_batch_size_phase2 ) / $samples_between_eval" | bc -l)
max_eval_steps=$(echo " 10000 / $eval_batch_size" | bc -l)
printf "Number of epochs: %.3f" "$epochs"

#Edit to save logs & checkpoints in a different directory
RESULTS_DIR=${RESULTS_DIR:-/results/${TAG}_${DATESTAMP}}
LOGFILE=$RESULTS_DIR/$TAG.$DATESTAMP.log
run_per_ip mkdir -m 777 -p $RESULTS_DIR
printf "Saving checkpoints to %s\n" "$RESULTS_DIR"
printf "Logs written to %s\n" "$LOGFILE"
export RESULTS_DIR=$RESULTS_DIR

printf -v SCRIPT_ARGS "%d %d %e %f %f %s %s %d %d %d %d %d %s %d %e %d %d %s %s %s" \
                      $train_batch_size_phase2 $eval_batch_size  \
                      $learning_rate_phase2 $beta1_phase2 $beta2_phase2 "$precision" "$use_xla" $num_hpus  \
                      $warmup_steps_phase2  $train_steps_phase2 $save_checkpoints_steps \
                      $num_accumulation_steps_phase2 "$bert_model" $samples_between_eval \
                      $stop_threshold $samples_start_eval $max_eval_steps "$is_dist_eval_enabled" "$eval_only" "$use_lightweight_checkpoint"

echo "learning_rate_phase2=$learning_rate_phase2, warmup_steps_phase2=$warmup_steps_phase2, train_steps_phase2=$train_steps_phase2, num_accumulation_steps_phase2=$num_accumulation_steps_phase2, train_batch_size_phase2=$train_batch_size_phase2"
# RUN PHASE 2
source $SCRIPT_DIR/scripts/run_pretraining_lamb_phase2.sh $SCRIPT_ARGS |& tee -a $LOGFILE
